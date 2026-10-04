"""Download and verify the torchvision datasets used by the MLP and CNN studies."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATASET_CLASS_NAMES: dict[str, str] = {
    "mnist": "MNIST",
    "fashionmnist": "FashionMNIST",
    "cifar10": "CIFAR10",
    "cifar100": "CIFAR100",
}
DEFAULT_DATASETS = tuple(DATASET_CLASS_NAMES)
DEFAULT_MANIFEST_NAME = "torchvision_datasets.json"

DatasetFactory = Callable[..., Any]


def canonical_dataset_name(name: str) -> str:
    """Return the repository's canonical name for a supported dataset."""

    normalized = name.strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "mnist": "mnist",
        "fashionmnist": "fashionmnist",
        "fmnist": "fashionmnist",
        "cifar10": "cifar10",
        "cifar100": "cifar100",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        supported = ", ".join(DEFAULT_DATASETS)
        raise ValueError(f"Unsupported dataset {name!r}. Choose from: {supported}.") from exc


def normalize_dataset_selection(dataset_names: Sequence[str]) -> list[str]:
    """Canonicalize names and remove duplicates while preserving input order."""

    selected: list[str] = []
    for name in dataset_names:
        canonical = canonical_dataset_name(name)
        if canonical not in selected:
            selected.append(canonical)
    if not selected:
        raise ValueError("Select at least one dataset.")
    return selected


def load_torchvision_registry() -> tuple[Mapping[str, DatasetFactory], str]:
    """Load torchvision lazily so tests can inject lightweight dataset factories."""

    import torchvision
    from torchvision import datasets

    registry = {
        name: getattr(datasets, class_name) for name, class_name in DATASET_CLASS_NAMES.items()
    }
    return registry, torchvision.__version__


def _sample_count(dataset: Any, *, dataset_name: str, split: str) -> int:
    try:
        count = int(len(dataset))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Could not verify {dataset_name} {split} split length.") from exc
    if count <= 0:
        raise RuntimeError(f"Downloaded {dataset_name} {split} split is empty.")
    return count


def _format_utc(timestamp: datetime) -> str:
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("The manifest timestamp must be timezone-aware.")
    return timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def write_manifest(manifest: Mapping[str, Any], destination: Path) -> None:
    """Write the manifest atomically so an interrupted run cannot leave partial JSON."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def bootstrap_datasets(
    data_dir: str | Path = "data",
    dataset_names: Sequence[str] = DEFAULT_DATASETS,
    *,
    manifest_path: str | Path | None = None,
    dataset_registry: Mapping[str, DatasetFactory] | None = None,
    torchvision_version: str | None = None,
    now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Download train/test splits, verify them, and write a reproducibility manifest.

    ``dataset_registry``, ``torchvision_version``, and ``now`` are injectable so callers can
    test the complete workflow without network access or real torchvision datasets.
    """

    selected = normalize_dataset_selection(dataset_names)
    destination = Path(data_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    if dataset_registry is None:
        dataset_registry, detected_version = load_torchvision_registry()
        if torchvision_version is None:
            torchvision_version = detected_version
    elif torchvision_version is None:
        torchvision_version = "unknown (injected dataset registry)"

    missing = [name for name in selected if name not in dataset_registry]
    if missing:
        raise ValueError(f"Dataset registry is missing: {', '.join(missing)}.")

    entries: list[dict[str, Any]] = []
    for name in selected:
        factory = dataset_registry[name]
        train_set = factory(root=str(destination), train=True, download=True)
        test_set = factory(root=str(destination), train=False, download=True)
        entries.append(
            {
                "name": name,
                "torchvision_class": DATASET_CLASS_NAMES[name],
                "train_samples": _sample_count(train_set, dataset_name=name, split="train"),
                "test_samples": _sample_count(test_set, dataset_name=name, split="test"),
            }
        )

    clock = now or (lambda: datetime.now(timezone.utc))
    manifest = {
        "created_at_utc": _format_utc(clock()),
        "data_dir": str(destination),
        "torchvision_version": torchvision_version,
        "datasets": entries,
    }
    output_path = (
        Path(manifest_path).expanduser().resolve()
        if manifest_path is not None
        else destination / DEFAULT_MANIFEST_NAME
    )
    write_manifest(manifest, output_path)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download and verify the MNIST, FashionMNIST, CIFAR-10, and CIFAR-100 "
            "datasets used by the paper. This command does not train a model."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Dataset destination (default: data).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        metavar="NAME",
        help="Datasets to prepare (default: all four).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=f"Manifest path (default: DATA_DIR/{DEFAULT_MANIFEST_NAME}).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = bootstrap_datasets(
            data_dir=args.data_dir,
            dataset_names=args.datasets,
            manifest_path=args.manifest,
        )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
