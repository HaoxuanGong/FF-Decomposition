from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

import DatasetBootstrap as bootstrap


class FakeDataset:
    calls: list[dict[str, object]] = []

    def __init__(self, *, root: str, train: bool, download: bool) -> None:
        self.calls.append({"root": root, "train": train, "download": download})
        self.train = train

    def __len__(self) -> int:
        return 12 if self.train else 5


def fake_registry() -> dict[str, type[FakeDataset]]:
    return {name: FakeDataset for name in bootstrap.DEFAULT_DATASETS}


def test_bootstrap_downloads_both_splits_and_writes_manifest(tmp_path: Path) -> None:
    FakeDataset.calls.clear()
    data_dir = tmp_path / "datasets"
    fixed_time = datetime(2026, 10, 4, 1, 2, 3, tzinfo=timezone.utc)

    manifest = bootstrap.bootstrap_datasets(
        data_dir=data_dir,
        dataset_names=["MNIST", "fashion-mnist"],
        dataset_registry=fake_registry(),
        torchvision_version="0.test",
        now=lambda: fixed_time,
    )

    assert FakeDataset.calls == [
        {"root": str(data_dir.resolve()), "train": True, "download": True},
        {"root": str(data_dir.resolve()), "train": False, "download": True},
        {"root": str(data_dir.resolve()), "train": True, "download": True},
        {"root": str(data_dir.resolve()), "train": False, "download": True},
    ]
    assert manifest == {
        "created_at_utc": "2026-10-04T01:02:03Z",
        "data_dir": str(data_dir.resolve()),
        "torchvision_version": "0.test",
        "datasets": [
            {
                "name": "mnist",
                "torchvision_class": "MNIST",
                "train_samples": 12,
                "test_samples": 5,
            },
            {
                "name": "fashionmnist",
                "torchvision_class": "FashionMNIST",
                "train_samples": 12,
                "test_samples": 5,
            },
        ],
    }
    saved = json.loads((data_dir / bootstrap.DEFAULT_MANIFEST_NAME).read_text("utf-8"))
    assert saved == manifest


def test_default_selection_contains_all_four_paper_datasets(tmp_path: Path) -> None:
    FakeDataset.calls.clear()

    manifest = bootstrap.bootstrap_datasets(
        data_dir=tmp_path,
        dataset_registry=fake_registry(),
        torchvision_version="0.test",
    )

    assert [entry["name"] for entry in manifest["datasets"]] == list(
        bootstrap.DEFAULT_DATASETS
    )
    assert len(FakeDataset.calls) == 8


def test_selection_deduplicates_aliases_and_supports_custom_manifest(tmp_path: Path) -> None:
    FakeDataset.calls.clear()
    manifest_path = tmp_path / "metadata" / "datasets.json"

    manifest = bootstrap.bootstrap_datasets(
        data_dir=tmp_path / "data",
        dataset_names=["f_mnist", "FashionMNIST", "CIFAR-10"],
        manifest_path=manifest_path,
        dataset_registry=fake_registry(),
        torchvision_version="0.test",
    )

    assert [entry["name"] for entry in manifest["datasets"]] == ["fashionmnist", "cifar10"]
    assert manifest_path.is_file()
    assert len(FakeDataset.calls) == 4


def test_invalid_dataset_is_rejected_before_any_download(tmp_path: Path) -> None:
    FakeDataset.calls.clear()

    with pytest.raises(ValueError, match="Unsupported dataset"):
        bootstrap.bootstrap_datasets(
            data_dir=tmp_path,
            dataset_names=["imagenet"],
            dataset_registry=fake_registry(),
            torchvision_version="0.test",
        )

    assert FakeDataset.calls == []


def test_empty_download_is_not_recorded_as_verified(tmp_path: Path) -> None:
    class EmptyDataset(FakeDataset):
        def __len__(self) -> int:
            return 0

    registry = fake_registry()
    registry["mnist"] = EmptyDataset

    with pytest.raises(RuntimeError, match="mnist train split is empty"):
        bootstrap.bootstrap_datasets(
            data_dir=tmp_path,
            dataset_names=["mnist"],
            dataset_registry=registry,
            torchvision_version="0.test",
        )

    assert not (tmp_path / bootstrap.DEFAULT_MANIFEST_NAME).exists()


def test_parser_defaults_to_all_datasets() -> None:
    args = bootstrap.build_parser().parse_args([])

    assert args.data_dir == Path("data")
    assert args.datasets == list(bootstrap.DEFAULT_DATASETS)
    assert args.manifest is None
