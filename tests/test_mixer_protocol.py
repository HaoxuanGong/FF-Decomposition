from __future__ import annotations

from collections import Counter
from pathlib import Path
import zipfile

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from MLPMixerBenchmarkSuite import (
    LocalBPMixer,
    build_local_bp_scalers,
    coil100_integrity,
    ensure_download,
    index_sha256,
    stratified_train_validation_indices,
    tinyimagenet_integrity,
)
from medmnist_support import logits_to_scores, prepare_targets, task_loss


def test_mixer_split_is_deterministic_stratified_and_disjoint() -> None:
    labels = np.repeat(np.arange(4), 20)
    train_a, validation_a = stratified_train_validation_indices(labels, 0.25, 7)
    train_b, validation_b = stratified_train_validation_indices(labels, 0.25, 7)
    assert train_a == train_b
    assert validation_a == validation_b
    assert not set(train_a).intersection(validation_a)
    assert Counter(labels[validation_a].tolist()) == Counter({0: 5, 1: 5, 2: 5, 3: 5})


def test_multilabel_medmnist_uses_binary_loss_and_sigmoid_scores() -> None:
    task = "multi-label, binary-class"
    targets = prepare_targets(
        torch.tensor([[0, 1, 0], [1, 0, 1]]), task, torch.device("cpu")
    )
    logits = torch.tensor([[0.0, 1.0, -1.0], [2.0, -2.0, 0.0]])
    loss = task_loss(task)(logits, targets)
    scores = logits_to_scores(logits, task)
    assert torch.isfinite(loss)
    assert targets.dtype == torch.float32
    assert torch.all((0.0 <= scores) & (scores <= 1.0))


def test_local_mixer_detaches_the_upstream_block_for_a_downstream_local_loss() -> None:
    torch.manual_seed(5)
    model = LocalBPMixer(
        in_channels=3,
        num_classes=4,
        image_size=8,
        patch_size=4,
        model_dim=8,
        depth=2,
        token_hidden_dim=4,
        channel_hidden_dim=16,
        dropout=0.0,
    )
    images = torch.randn(3, 3, 8, 8)
    targets = torch.tensor([0, 1, 2])

    first = model.blocks[0](model.patch_embedding(images))
    second = model.blocks[1](first.detach())
    logits = model.local_heads[1](model.local_norms[1](second).mean(dim=1))
    F.cross_entropy(logits, targets).backward()

    upstream = [*model.patch_embedding.parameters(), *model.blocks[0].parameters()]
    downstream = [
        *model.blocks[1].parameters(),
        *model.local_norms[1].parameters(),
        *model.local_heads[1].parameters(),
    ]
    assert all(parameter.grad is None for parameter in upstream)
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in downstream
    )


def test_local_amp_uses_one_independent_scaler_per_optimizer() -> None:
    scalers = build_local_bp_scalers(depth=3, amp_enabled=False)
    assert len(scalers) == 3
    assert len({id(scaler) for scaler in scalers}) == 3


def test_index_hash_is_order_sensitive_and_stable() -> None:
    assert index_sha256([1, 2, 3]) == index_sha256([1, 2, 3])
    assert index_sha256([1, 2, 3]) != index_sha256([3, 2, 1])


def test_tinyimagenet_integrity_validates_structure_and_hashes_used_tree(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tiny-imagenet-200"
    class_ids = ("n0001", "n0002")
    (root / "val" / "images").mkdir(parents=True)
    (root / "wnids.txt").write_text("\n".join(class_ids) + "\n", encoding="utf-8")
    annotations: list[str] = []
    for class_id in class_ids:
        image_dir = root / "train" / class_id / "images"
        image_dir.mkdir(parents=True)
        for image_index in range(2):
            (image_dir / f"{class_id}_{image_index}.JPEG").write_bytes(
                f"train:{class_id}:{image_index}".encode()
            )
        validation_name = f"val_{class_id}.JPEG"
        (root / "val" / "images" / validation_name).write_bytes(
            f"validation:{class_id}".encode()
        )
        annotations.append(f"{validation_name}\t{class_id}\t0\t0\t1\t1")
    (root / "val" / "val_annotations.txt").write_text(
        "\n".join(annotations) + "\n", encoding="utf-8"
    )

    record = tinyimagenet_integrity(
        root,
        tmp_path / "missing.zip",
        expected_class_count=2,
        expected_train_per_class=2,
        expected_validation_per_class=1,
    )
    repeated = tinyimagenet_integrity(
        root,
        tmp_path / "missing.zip",
        expected_class_count=2,
        expected_train_per_class=2,
        expected_validation_per_class=1,
    )
    assert record == repeated
    assert record["tree_file_count"] == 8
    assert len(str(record["tree_sha256"])) == 64
    assert record["archive"] == {"available": False, "sha256": "", "size_bytes": 0}
    assert record["source_published_checksum_available"] is False

    changed_image = root / "train" / class_ids[0] / "images" / f"{class_ids[0]}_0.JPEG"
    changed_image.write_bytes(b"changed")
    changed = tinyimagenet_integrity(
        root,
        tmp_path / "missing.zip",
        expected_class_count=2,
        expected_train_per_class=2,
        expected_validation_per_class=1,
    )
    assert changed["tree_sha256"] != record["tree_sha256"]


def test_coil100_integrity_requires_complete_object_angle_grid(tmp_path: Path) -> None:
    root = tmp_path / "coil-100"
    root.mkdir()
    for object_id in (1, 2):
        for angle in (0, 5):
            (root / f"obj{object_id}__{angle}.png").write_bytes(
                f"coil:{object_id}:{angle}".encode()
            )
    archive = tmp_path / "coil-100.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("source-marker.txt", "fixture")

    record = coil100_integrity(
        root,
        archive,
        expected_object_count=2,
        expected_angles=(0, 5),
    )
    assert record["tree_file_count"] == 4
    assert len(str(record["tree_sha256"])) == 64
    assert record["archive"]["available"] is True
    assert len(record["archive"]["sha256"]) == 64

    (root / "obj1__10.png").write_bytes(b"unexpected angle")
    with pytest.raises(ValueError, match="every expected object"):
        coil100_integrity(
            root,
            archive,
            expected_object_count=2,
            expected_angles=(0, 5),
        )


def test_download_refuses_invalid_existing_archive_without_overwriting(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "dataset.zip"
    original = b"not a ZIP"
    archive.write_bytes(original)
    with pytest.raises(ValueError, match="left untouched"):
        ensure_download("https://example.invalid/dataset.zip", archive)
    assert archive.read_bytes() == original
