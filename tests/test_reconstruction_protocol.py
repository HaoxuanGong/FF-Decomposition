from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from decomposition_core import GoodnessMLP, mark_inputs
from ReconstructionExperimentScheduler import (
    CONDITIONS,
    HIDDEN_DIMS,
    LAYERS,
    SEEDS,
    backbone_command,
    jobs,
    prepare_run_directory,
    probe_command,
)
from ReconstructionProbe import (
    Decoder,
    extract_representations,
    reconstruction_mse,
    train_decoder,
)


def test_reconstruction_matrix_and_commands_are_fixed() -> None:
    matrix = jobs()
    assert CONDITIONS == {
        "normalized": "fc-ff-ge",
        "no-inter-layer-normalization": "fc-nn-ff-ge",
    }
    assert len(matrix) == len(CONDITIONS) * len(SEEDS) == 6
    assert {condition for condition, _method, _seed in matrix} == set(CONDITIONS)
    command = backbone_command(
        "python", Path("project"), Path("run"), "normalized", "fc-ff-ge", 424
    )
    hidden_start = command.index("--hidden-dims") + 1
    assert command[hidden_start : hidden_start + 5] == ["500"] * 5
    assert command[command.index("--validation-size") + 1] == "5000"
    assert command[command.index("--patience") + 1] == "15"
    assert "--scheduler" in command and command[command.index("--scheduler") + 1] == "none"
    probe = probe_command(
        "python", Path("project"), Path("run"), "normalized", "fc-ff-ge", 424
    )
    assert probe[probe.index("--patience") + 1] == "10"
    assert probe[probe.index("--representation-input") + 1] == "true-label"
    no_inter_layer_probe = probe_command(
        "python",
        Path("project"),
        Path("run"),
        "no-inter-layer-normalization",
        "fc-nn-ff-ge",
        424,
    )
    assert "no-inter-layer-normalization" in no_inter_layer_probe


def test_decoder_shape_and_exact_per_pixel_mse() -> None:
    decoder = Decoder(7)
    output = decoder(torch.zeros(3, 7))
    assert output.shape == (3, 784)
    targets = torch.zeros_like(output)
    measured = reconstruction_mse(
        decoder,
        torch.zeros(3, 7),
        targets,
        device=torch.device("cpu"),
        batch_size=2,
    )
    expected = float(output.square().mean())
    assert abs(measured - expected) < 1e-7


def test_normalized_extraction_returns_next_layer_representation() -> None:
    model = GoodnessMLP(12, hidden_dims=(5, 5), normalize=True)
    inputs = torch.randn(4, 12)
    labels = torch.arange(4) % 3
    raw = torch.rand(4, 784)
    loader = DataLoader(TensorDataset(inputs, labels, raw), batch_size=4)
    representations, targets = extract_representations(
        model, loader, device=torch.device("cpu"), encode_true_label=True
    )
    assert len(representations) == 2
    assert targets.shape == (4, 784)
    assert torch.all(representations[0].norm(dim=1) <= 1.0 + 1e-6)


def test_conditions_share_first_layer_normalization_and_probe_actual_outputs() -> None:
    torch.manual_seed(9)
    normalized = GoodnessMLP(
        12,
        hidden_dims=(6, 6),
        normalize=True,
        normalize_first_layer_input=True,
    )
    no_inter_layer = GoodnessMLP(
        12,
        hidden_dims=(6, 6),
        normalize=False,
        normalize_first_layer_input=True,
    )
    no_inter_layer.load_state_dict(normalized.state_dict(), strict=True)
    inputs = torch.randn(4, 12)
    labels = torch.arange(4) % 3
    raw = torch.rand(4, 784)
    marked = mark_inputs(inputs, labels, 10)
    normalized_first = normalized.forward_layer(marked, 0)
    no_inter_layer_first = no_inter_layer.forward_layer(marked, 0)
    assert torch.equal(normalized_first, no_inter_layer_first)
    assert normalized.normalizes_layer_input(0) is True
    assert no_inter_layer.normalizes_layer_input(0) is True
    assert normalized.normalizes_layer_input(1) is True
    assert no_inter_layer.normalizes_layer_input(1) is False

    loader = DataLoader(TensorDataset(inputs, labels, raw), batch_size=4)
    normalized_representations, _targets = extract_representations(
        normalized, loader, device=torch.device("cpu"), encode_true_label=True
    )
    no_inter_layer_representations, _targets = extract_representations(
        no_inter_layer, loader, device=torch.device("cpu"), encode_true_label=True
    )
    expected_normalized_first = normalized_first / (
        normalized_first.norm(dim=1, keepdim=True)
        + normalized.normalization_epsilon
    )
    assert torch.allclose(normalized_representations[0], expected_normalized_first)
    assert torch.equal(no_inter_layer_representations[0], no_inter_layer_first)
    assert torch.equal(
        normalized_representations[1],
        normalized.forward_layer(normalized_first, 1),
    )
    assert torch.equal(
        no_inter_layer_representations[1],
        no_inter_layer.forward_layer(no_inter_layer_first, 1),
    )


def test_decoder_uses_validation_selection_and_restores_checkpoint(tmp_path: Path) -> None:
    torch.manual_seed(5)
    train_features = torch.randn(16, 4)
    train_targets = torch.sigmoid(torch.randn(16, 784))
    validation_features = torch.randn(8, 4)
    validation_targets = torch.sigmoid(torch.randn(8, 784))
    checkpoint = tmp_path / "best_decoder.pt"
    _decoder, selection, history = train_decoder(
        train_features,
        train_targets,
        validation_features,
        validation_targets,
        device=torch.device("cpu"),
        maximum_epochs=3,
        patience=2,
        batch_size=8,
        evaluation_batch_size=8,
        learning_rate=1e-3,
        seed=424,
        checkpoint_path=checkpoint,
    )
    values = [float(row["validation_mse"]) for row in history]
    assert checkpoint.exists()
    assert selection["best_epoch"] == values.index(min(values)) + 1
    assert selection["best_validation_mse"] == min(values)
    assert 1 <= selection["epochs_trained"] <= 3
    assert selection["restored_best_state_verified"] is True
    assert selection["checkpoint_reload_verified"] is True
    assert len(selection["history_sha256"]) == 64


def test_reconstruction_resume_requires_exact_manifest_identity(tmp_path: Path) -> None:
    manifest = {
        "schema_version": 3,
        "project_root": "project",
        "run_dir": str(tmp_path),
        "python": "python",
        "protocol": {"decoder_epochs": 40},
        "source_sha256": {"probe": "a" * 64},
        "jobs": [{"condition": "normalized"}],
        "started_at_utc": "first",
    }
    assert prepare_run_directory(tmp_path, manifest) is False
    assert prepare_run_directory(
        tmp_path, dict(manifest, started_at_utc="second")
    ) is True
    with pytest.raises(RuntimeError, match="source/configuration"):
        prepare_run_directory(
            tmp_path, dict(manifest, source_sha256={"probe": "b" * 64})
        )


def test_reconstruction_accepts_only_launcher_precreated_entries(
    tmp_path: Path,
) -> None:
    manifest = {
        "schema_version": 3,
        "project_root": "project",
        "run_dir": str(tmp_path),
        "python": "python",
        "protocol": {"decoder_epochs": 40},
        "source_sha256": {"probe": "a" * 64},
        "jobs": [{"condition": "normalized"}],
        "started_at_utc": "first",
    }
    (tmp_path / "source").mkdir()
    (tmp_path / "scheduler.log").touch()
    assert prepare_run_directory(tmp_path, manifest) is False
    assert (tmp_path / "manifest.json").is_file()

    unexpected = tmp_path / "unexpected"
    unexpected.mkdir()
    (unexpected / "source").mkdir()
    (unexpected / "partial-result.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="without a manifest"):
        prepare_run_directory(unexpected, dict(manifest, run_dir=str(unexpected)))


def test_declared_protocol_dimensions() -> None:
    assert HIDDEN_DIMS == (500,) * 5
    assert LAYERS == (1, 2, 3, 4, 5)
    assert SEEDS == (424, 425, 426)
