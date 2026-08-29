#!/usr/bin/env python3
"""Run the paper's matched backpropagation and Local BP MLP-Mixer sweep."""

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import random
import re
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.optim import Adam, AdamW, RMSprop, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS

from medmnist_support import (
    MEDMNIST_DATASETS,
    append_medmnist_batches,
    build_medmnist_splits,
    compute_medmnist_metrics,
    prepare_targets,
    task_loss,
)

TINY_URL = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"
COIL_URL = "https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip"
TINY_REFERENCE_URL = "https://cs231n.stanford.edu/2016/project.html"
COIL_REFERENCE_URL = "https://cave.cs.columbia.edu/repository/COIL-100"
MAIN_DATASETS = ("cifar10", "cifar100", "tinyimagenet", "coil100", *MEDMNIST_DATASETS)
PAPER_SEEDS = (41, 42, 43)
PAPER_DEPTHS = (5, 8, 12)
PAPER_DIMS = (256, 512)
MIXER_DROPOUT = 0.1
SCHEDULER_NAME = "cosine_annealing"
RESULT_SCHEMA_VERSION = 2


def source_sha256() -> dict[str, str]:
    """Return the exact source identity required for safe result reuse."""

    root = Path(__file__).resolve().parent
    return {
        "MLPMixerBenchmarkSuite.py": file_sha256(root / "MLPMixerBenchmarkSuite.py"),
        "medmnist_support.py": file_sha256(root / "medmnist_support.py"),
    }


def atomic_json(path: Path, payload: object) -> str:
    """Write JSON atomically and return the resulting SHA-256 digest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)
    return file_sha256(path)


def state_dicts_equal(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> bool:
    """Check exact state restoration without silently tolerating drift."""

    return left.keys() == right.keys() and all(
        torch.equal(left[name].detach().cpu(), right[name].detach().cpu())
        for name in left
    )

SUMMARY_GROUP_FIELDS = (
    "dataset",
    "method",
    "depth",
    "dim",
    "token_dim",
    "channel_dim",
    "patch_size",
    "num_classes",
    "in_channels",
    "image_size",
    "task",
    "metric_primary_name",
    "metric_secondary_name",
    "dataset_source",
    "dataset_version",
    "training_augmentation",
    "final_evaluation_split",
    "epochs_target",
    "early_stop_patience",
    "early_stop_min_delta",
    "validation_fraction",
    "validation_is_official",
    "dropout",
    "scheduler",
    "scheduler_t_max",
    "local_bp_updates_per_block",
    "optimizer",
    "lr",
    "weight_decay",
    "momentum",
    "batch_size",
    "eval_batch_size",
    "train_samples",
    "validation_samples",
    "test_samples",
    "num_workers",
    "device",
    "amp_enabled",
    "torch_version",
    "torchvision_version",
    "cuda_runtime_version",
    "cudnn_version",
    "gpu_name",
)
SUMMARY_METRICS = (
    ("test_primary", "test_primary"),
    ("test_secondary", "test_secondary"),
    ("best_validation_primary", "best_validation_primary"),
    ("peak_train_mem_gb", "peak_train_mem_gb"),
    ("runtime_seconds", "runtime_seconds"),
    ("best_epoch", "best_epoch"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("MLP-Mixer BP vs Local BP benchmark suite")
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        choices=MAIN_DATASETS,
        help="One or more main-table datasets to benchmark.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["bp", "local-bp"],
        choices=["bp", "local-bp"],
    )
    parser.add_argument("--depths", nargs="+", type=int, default=list(PAPER_DEPTHS))
    parser.add_argument("--dims", nargs="+", type=int, default=list(PAPER_DIMS))
    parser.add_argument("--epochs", type=int, default=480)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=10,
        help="Stop after this many epochs without validation-metric improvement.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-4,
        help="Minimum strict improvement in the validation primary metric.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help=(
            "Seed-specific stratified validation fraction for datasets without an "
            "official validation split (default: 0.1)."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--optimizer",
        choices=["adamw", "adam", "sgd", "rmsprop"],
        default="adamw",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--token-dim", type=int, default=0)
    parser.add_argument("--channel-dim", type=int, default=0)
    parser.add_argument("--token-ratio", type=float, default=1.0)
    parser.add_argument("--channel-ratio", type=float, default=4.0)
    parser.add_argument("--patch-size", type=int, default=0, help="0 => dataset default")
    parser.add_argument(
        "--local-bp-updates-per-block",
        type=int,
        default=3,
        help="Number of local optimizer updates per block and minibatch (default: 3).",
    )
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="mlpmixer_suite_results")
    parser.add_argument(
        "--checkpoint-dir",
        default="",
        help="Optional directory in which to save restored best checkpoints.",
    )
    parser.add_argument("--results-csv", default="results.csv")
    parser.add_argument(
        "--summary-csv",
        default="summary.csv",
        help="Seed-aggregated table written alongside the per-run results.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run one seed instead of the paper's three-seed default.",
    )
    seed_group.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Matched seeds for BP and Local BP (default: 41 42 43).",
    )
    parser.add_argument("--download-tinyimagenet", action="store_true")
    parser.add_argument("--download-coil100", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    parser.add_argument("--disable-amp", action="store_true")
    return parser.parse_args(argv)


def resolve_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds is not None:
        seeds = list(args.seeds)
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = list(PAPER_SEEDS)
    if not seeds:
        raise ValueError("at least one seed is required")
    if any(seed < 0 for seed in seeds):
        raise ValueError("seeds cannot be negative")
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be unique")
    return seeds


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for an experiment artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_sha256(indices) -> str:
    """Hash an ordered integer index sequence without platform-dependent bytes."""

    digest = hashlib.sha256()
    for index in indices:
        digest.update(f"{int(index)}\n".encode("ascii"))
    return digest.hexdigest()


def runtime_environment(device: torch.device) -> dict[str, object]:
    """Capture the software and accelerator metadata needed to interpret a run."""

    metadata: dict[str, object] = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_runtime_version": torch.version.cuda or "",
        "cudnn_version": torch.backends.cudnn.version() or "",
        "device": str(device),
    }
    if device.type == "cuda":
        metadata.update(
            {
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_total_memory_bytes": torch.cuda.get_device_properties(
                    device
                ).total_memory,
            }
        )
    else:
        metadata.update({"gpu_name": "", "gpu_total_memory_bytes": 0})
    return metadata


def save_checkpoint_atomic(path: Path, payload: dict[str, object]) -> str:
    """Save a checkpoint atomically and return its SHA-256 digest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return file_sha256(path)


def build_optimizer(optimizer_name, parameters, learning_rate, weight_decay, momentum):
    """Construct an optimizer from the shared command-line settings."""

    if optimizer_name == "adamw":
        return AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "adam":
        return Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return SGD(
            parameters,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    return RMSprop(
        parameters,
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )


def count_topk_correct(logits, targets) -> tuple[int, int]:
    """Count top-1 and top-5 correct predictions in a batch."""

    max_k = min(5, logits.size(1))
    predictions = logits.topk(max_k, dim=1).indices
    correct = predictions.eq(targets.unsqueeze(1))
    top1 = correct[:, :1].any(dim=1).sum().item()
    top5 = correct[:, :max_k].any(dim=1).sum().item()
    return top1, top5


def ensure_download(url: str, destination: Path) -> None:
    """Download a ZIP atomically, without claiming an unpublished source checksum."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        if not zipfile.is_zipfile(destination):
            raise ValueError(
                f"Existing archive is not a valid ZIP and was left untouched: {destination}"
            )
        return
    print(f"Downloading {url}", flush=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        urllib.request.urlretrieve(url, temporary)
        if not zipfile.is_zipfile(temporary):
            raise ValueError(f"Downloaded payload is not a valid ZIP: {url}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def extract_zip_safely(archive: Path, destination: Path) -> None:
    """Extract an archive only when every member remains below destination."""

    destination = destination.resolve()
    with zipfile.ZipFile(archive, "r") as zip_file:
        for member in zip_file.infolist():
            target = (destination / member.filename).resolve()
            if os.path.commonpath((destination, target)) != str(destination):
                raise ValueError(f"Unsafe path in archive {archive}: {member.filename}")
        zip_file.extractall(destination)


def canonical_tree_fingerprint(
    root: Path, files: list[Path]
) -> dict[str, int | str]:
    """Hash relative paths, sizes, and bytes for an explicitly selected file tree."""

    resolved_root = root.resolve()
    unique_files = sorted({Path(os.path.abspath(path)) for path in files})
    digest = hashlib.sha256()
    total_bytes = 0
    for path in unique_files:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Dataset fingerprint requires a regular file: {path}")
        try:
            relative = path.resolve().relative_to(resolved_root).as_posix()
        except ValueError as exc:
            raise ValueError(f"Dataset file escapes its root: {path}") from exc
        size = path.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
        total_bytes += size
    return {
        "tree_sha256": digest.hexdigest(),
        "tree_file_count": len(unique_files),
        "tree_total_bytes": total_bytes,
    }


def archive_fingerprint(archive: Path) -> dict[str, bool | int | str]:
    """Record a retained archive's computed identity, if the archive is available."""

    if not archive.is_file():
        return {"available": False, "sha256": "", "size_bytes": 0}
    if not zipfile.is_zipfile(archive):
        raise ValueError(f"Dataset archive is not a valid ZIP: {archive}")
    return {
        "available": True,
        "sha256": file_sha256(archive),
        "size_bytes": archive.stat().st_size,
    }


def tinyimagenet_integrity(
    dataset_root: Path,
    archive: Path,
    *,
    expected_class_count: int = 200,
    expected_train_per_class: int = 500,
    expected_validation_per_class: int = 50,
) -> dict[str, object]:
    """Validate Tiny ImageNet's published layout and compute local fingerprints."""

    wnids_path = dataset_root / "wnids.txt"
    annotation_path = dataset_root / "val" / "val_annotations.txt"
    validation_image_dir = dataset_root / "val" / "images"
    required_files = (wnids_path, annotation_path)
    if any(not path.is_file() for path in required_files):
        raise ValueError("Tiny ImageNet is missing wnids.txt or val_annotations.txt")
    wnids = [
        line.strip()
        for line in wnids_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(wnids) != expected_class_count or len(set(wnids)) != len(wnids):
        raise ValueError(
            f"Tiny ImageNet must contain {expected_class_count} unique class IDs"
        )
    train_root = dataset_root / "train"
    observed_class_dirs = {
        path.name for path in train_root.iterdir() if path.is_dir()
    }
    if observed_class_dirs != set(wnids):
        raise ValueError("Tiny ImageNet training directories do not match wnids.txt")

    selected_files = [wnids_path, annotation_path]
    train_counts: dict[str, int] = {}
    for class_id in wnids:
        class_root = train_root / class_id
        image_dir = class_root / "images"
        if class_root.is_symlink() or image_dir.is_symlink():
            raise ValueError("Tiny ImageNet class directories cannot be symbolic links")
        class_images = sorted(
            path
            for path in class_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS
        )
        if any(
            path.parent != image_dir or path.suffix != ".JPEG"
            for path in class_images
        ):
            raise ValueError("Tiny ImageNet contains an unexpected training-image path")
        train_counts[class_id] = len(class_images)
        selected_files.extend(class_images)
    if set(train_counts.values()) != {expected_train_per_class}:
        raise ValueError(
            "Tiny ImageNet training layout does not match the published "
            f"{expected_train_per_class} images per class"
        )

    validation_images = sorted(
        path for path in validation_image_dir.iterdir() if path.is_file()
    )
    if any(path.suffix != ".JPEG" for path in validation_images):
        raise ValueError("Tiny ImageNet contains an unexpected validation-image path")
    expected_validation_total = expected_class_count * expected_validation_per_class
    if len(validation_images) != expected_validation_total:
        raise ValueError(
            f"Tiny ImageNet must contain {expected_validation_total} validation images"
        )
    validation_names = {path.name for path in validation_images}
    validation_counts = {class_id: 0 for class_id in wnids}
    annotated_names: set[str] = set()
    for line in annotation_path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) < 2:
            raise ValueError("Malformed Tiny ImageNet validation annotation")
        filename, class_id = fields[0], fields[1]
        if filename in annotated_names or filename not in validation_names:
            raise ValueError("Tiny ImageNet validation annotations are not one-to-one")
        if class_id not in validation_counts:
            raise ValueError(f"Unknown Tiny ImageNet validation class: {class_id}")
        annotated_names.add(filename)
        validation_counts[class_id] += 1
    if annotated_names != validation_names or set(validation_counts.values()) != {
        expected_validation_per_class
    }:
        raise ValueError(
            "Tiny ImageNet validation layout does not match the published "
            f"{expected_validation_per_class} images per class"
        )
    selected_files.extend(validation_images)
    tree = canonical_tree_fingerprint(dataset_root, selected_files)
    return {
        "dataset": "tinyimagenet",
        "status": "expected_structure_validated",
        "source_url": TINY_URL,
        "reference_url": TINY_REFERENCE_URL,
        "source_published_checksum_available": False,
        "fingerprint_scope": (
            "wnids.txt, all training JPEGs, validation annotations, and all "
            "validation JPEGs"
        ),
        "expected_structure": {
            "classes": expected_class_count,
            "training_images_per_class": expected_train_per_class,
            "validation_images_per_class": expected_validation_per_class,
            "training_images": expected_class_count * expected_train_per_class,
            "validation_images": expected_validation_total,
        },
        "archive": archive_fingerprint(archive),
        **tree,
    }


def ensure_tiny(data_root: Path, allow_download: bool) -> tuple[Path, dict[str, object]]:
    dataset_root = data_root / "tiny-imagenet-200"
    if (dataset_root / "train").exists() and (dataset_root / "val").exists():
        archive_path = data_root / "tiny-imagenet-200.zip"
        return dataset_root, tinyimagenet_integrity(dataset_root, archive_path)
    if not allow_download:
        raise FileNotFoundError(f"{dataset_root} missing. Use --download-tinyimagenet")
    archive_path = data_root / "tiny-imagenet-200.zip"
    ensure_download(TINY_URL, archive_path)
    extract_zip_safely(archive_path, data_root)
    return dataset_root, tinyimagenet_integrity(dataset_root, archive_path)


class TinyImageNetValidation(Dataset):
    """Tiny ImageNet validation split using its annotation file."""

    def __init__(self, root: Path, class_to_idx: dict[str, int], transform):
        annotation_path = root / "val" / "val_annotations.txt"
        image_dir = root / "val" / "images"
        self.transform = transform
        samples = []
        with annotation_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                fields = line.strip().split("\t")
                if len(fields) < 2:
                    continue
                filename, class_id = fields[0], fields[1]
                if class_id not in class_to_idx:
                    continue
                image_path = image_dir / filename
                if image_path.exists():
                    samples.append((image_path, class_to_idx[class_id]))
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), label


def coil100_integrity(
    dataset_root: Path,
    archive: Path,
    *,
    expected_object_count: int = 100,
    expected_angles: tuple[int, ...] = tuple(range(0, 360, 5)),
) -> dict[str, object]:
    """Validate COIL-100's object/angle layout and compute local fingerprints."""

    filename_pattern = re.compile(r"obj(\d+)__(\d+)\.png", re.IGNORECASE)
    images = sorted(dataset_root.glob("obj*__*.png"))
    observed: dict[int, set[int]] = {}
    for image in images:
        match = filename_pattern.fullmatch(image.name)
        if match is None:
            raise ValueError(f"Malformed COIL-100 filename: {image.name}")
        object_id, angle = int(match.group(1)), int(match.group(2))
        if angle in observed.setdefault(object_id, set()):
            raise ValueError(f"Duplicate COIL-100 object/angle pair: {image.name}")
        observed[object_id].add(angle)
    expected_objects = set(range(1, expected_object_count + 1))
    expected_angle_set = set(expected_angles)
    if set(observed) != expected_objects or any(
        angles != expected_angle_set for angles in observed.values()
    ):
        raise ValueError(
            "COIL-100 layout must contain every expected object at every expected angle"
        )
    tree = canonical_tree_fingerprint(dataset_root, images)
    return {
        "dataset": "coil100",
        "status": "expected_structure_validated",
        "source_url": COIL_URL,
        "reference_url": COIL_REFERENCE_URL,
        "source_published_checksum_available": False,
        "fingerprint_scope": "all processed COIL-100 PNG files",
        "expected_structure": {
            "objects": expected_object_count,
            "angles_degrees": list(expected_angles),
            "images_per_object": len(expected_angles),
            "images": expected_object_count * len(expected_angles),
        },
        "archive": archive_fingerprint(archive),
        **tree,
    }


def validate_dataset_integrity_record(
    dataset: str, record: object
) -> dict[str, object]:
    """Validate the internally recorded provenance for non-library datasets."""

    if dataset not in {"tinyimagenet", "coil100"}:
        if record in (None, {}):
            return {}
        if isinstance(record, dict):
            return record
        raise ValueError(f"Invalid dataset-integrity record for {dataset}")
    if not isinstance(record, dict):
        raise ValueError(f"Missing dataset-integrity record for {dataset}")
    expected = {
        "tinyimagenet": {
            "source_url": TINY_URL,
            "reference_url": TINY_REFERENCE_URL,
            "tree_file_count": 110002,
        },
        "coil100": {
            "source_url": COIL_URL,
            "reference_url": COIL_REFERENCE_URL,
            "tree_file_count": 7200,
        },
    }[dataset]
    for key, value in {
        "dataset": dataset,
        "status": "expected_structure_validated",
        "source_published_checksum_available": False,
        **expected,
    }.items():
        if record.get(key) != value:
            raise ValueError(f"Invalid {dataset} integrity field {key!r}")
    digest = record.get("tree_sha256", "")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"Invalid {dataset} tree SHA-256")
    if not isinstance(record.get("tree_total_bytes"), int) or record["tree_total_bytes"] <= 0:
        raise ValueError(f"Invalid {dataset} tree byte count")
    archive = record.get("archive")
    if not isinstance(archive, dict) or not isinstance(archive.get("available"), bool):
        raise ValueError(f"Invalid {dataset} archive provenance")
    if archive["available"]:
        archive_digest = archive.get("sha256", "")
        if (
            not isinstance(archive_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", archive_digest) is None
            or not isinstance(archive.get("size_bytes"), int)
            or archive["size_bytes"] <= 0
        ):
            raise ValueError(f"Invalid retained {dataset} archive fingerprint")
    elif archive.get("sha256") != "" or archive.get("size_bytes") != 0:
        raise ValueError(f"Invalid absent-archive record for {dataset}")
    return record


def ensure_coil(data_root: Path, allow_download: bool) -> tuple[Path, dict[str, object]]:
    dataset_root = data_root / "coil-100"
    if dataset_root.exists() and any(dataset_root.glob("obj*__*.png")):
        archive_path = data_root / "coil-100.zip"
        return dataset_root, coil100_integrity(dataset_root, archive_path)
    if not allow_download:
        raise FileNotFoundError(f"{dataset_root} missing. Use --download-coil100")
    archive_path = data_root / "coil-100.zip"
    ensure_download(COIL_URL, archive_path)
    extract_zip_safely(archive_path, data_root)
    return dataset_root, coil100_integrity(dataset_root, archive_path)


class ImagePathDataset(Dataset):
    """Load labelled images from an explicit list of paths."""

    def __init__(self, samples: list[tuple[Path, int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), label


def build_coil_splits(
    root: Path,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    filename_pattern = re.compile(r"obj(\d+)__(\d+)\.png", re.IGNORECASE)
    samples_by_class: dict[int, list[tuple[int, Path]]] = {}
    for image_path in sorted(root.glob("obj*__*.png")):
        match = filename_pattern.match(image_path.name)
        if not match:
            continue
        class_index = int(match.group(1)) - 1
        angle = int(match.group(2))
        samples_by_class.setdefault(class_index, []).append((angle, image_path))
    train_samples, test_samples = [], []
    for class_index in sorted(samples_by_class):
        class_samples = sorted(samples_by_class[class_index], key=lambda sample: sample[0])
        train_count = max(1, int(0.8 * len(class_samples)))
        class_train, class_test = class_samples[:train_count], class_samples[train_count:]
        if not class_test:
            class_train, class_test = class_samples[:-1], class_samples[-1:]
        train_samples += [(image_path, class_index) for _, image_path in class_train]
        test_samples += [(image_path, class_index) for _, image_path in class_test]
    return train_samples, test_samples


def build_dataset(dataset_name: str, data_root: Path, args):
    dataset_name = dataset_name.lower()
    if dataset_name == "cifar10":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            ]
        )
        return (
            CIFAR10(str(data_root), train=True, download=True, transform=train_transform),
            CIFAR10(str(data_root), train=True, download=True, transform=test_transform),
            CIFAR10(str(data_root), train=False, download=True, transform=test_transform),
            dict(
                num_classes=10,
                in_channels=3,
                image_size=32,
                patch_default=4,
                task="multi-class",
                metric_primary_name="top1",
                metric_secondary_name="top5",
                is_medmnist=False,
                validation_is_official=False,
                dataset_source="torchvision.datasets.CIFAR10",
                dataset_version=torchvision.__version__,
                training_augmentation="random_crop_padding4+random_horizontal_flip",
                final_evaluation_split="official_test",
            ),
        )
    if dataset_name == "cifar100":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
            ]
        )
        return (
            CIFAR100(str(data_root), train=True, download=True, transform=train_transform),
            CIFAR100(str(data_root), train=True, download=True, transform=test_transform),
            CIFAR100(str(data_root), train=False, download=True, transform=test_transform),
            dict(
                num_classes=100,
                in_channels=3,
                image_size=32,
                patch_default=4,
                task="multi-class",
                metric_primary_name="top1",
                metric_secondary_name="top5",
                is_medmnist=False,
                validation_is_official=False,
                dataset_source="torchvision.datasets.CIFAR100",
                dataset_version=torchvision.__version__,
                training_augmentation="random_crop_padding4+random_horizontal_flip",
                final_evaluation_split="official_test",
            ),
        )
    if dataset_name == "tinyimagenet":
        dataset_root, dataset_integrity = ensure_tiny(
            data_root, args.download_tinyimagenet
        )
        channel_mean, channel_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(channel_mean, channel_std),
            ]
        )
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(channel_mean, channel_std)]
        )
        train_set = ImageFolder(dataset_root / "train", transform=train_transform)
        validation_pool = ImageFolder(dataset_root / "train", transform=test_transform)
        test_set = TinyImageNetValidation(dataset_root, train_set.class_to_idx, test_transform)
        return (
            train_set,
            validation_pool,
            test_set,
            dict(
                num_classes=200,
                in_channels=3,
                image_size=64,
                patch_default=8,
                task="multi-class",
                metric_primary_name="top1",
                metric_secondary_name="top5",
                is_medmnist=False,
                validation_is_official=False,
                dataset_source="Tiny ImageNet 200",
                dataset_version="tiny-imagenet-200",
                dataset_integrity=dataset_integrity,
                training_augmentation="random_crop_padding4+random_horizontal_flip",
                final_evaluation_split="official_validation_with_labels",
            ),
        )
    if dataset_name == "coil100":
        dataset_root, dataset_integrity = ensure_coil(data_root, args.download_coil100)
        train_samples, test_samples = build_coil_splits(dataset_root)
        channel_mean, channel_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        train_transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(channel_mean, channel_std),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(channel_mean, channel_std),
            ]
        )
        return (
            ImagePathDataset(train_samples, train_transform),
            ImagePathDataset(train_samples, test_transform),
            ImagePathDataset(test_samples, test_transform),
            dict(
                num_classes=100,
                in_channels=3,
                image_size=128,
                patch_default=16,
                task="multi-class",
                metric_primary_name="top1",
                metric_secondary_name="top5",
                is_medmnist=False,
                validation_is_official=False,
                dataset_source="Columbia Object Image Library COIL-100",
                dataset_version="coil-100",
                dataset_integrity=dataset_integrity,
                training_augmentation="resize128+random_horizontal_flip",
                final_evaluation_split="deterministic_angle_holdout",
            ),
        )
    if dataset_name in MEDMNIST_DATASETS:
        return build_medmnist_splits(dataset_name, data_root)
    raise ValueError(dataset_name)


def dataset_targets(dataset: Dataset) -> np.ndarray:
    """Return one integer class label per example for stratified splitting."""

    if hasattr(dataset, "targets"):
        targets = np.asarray(dataset.targets)
    elif hasattr(dataset, "samples"):
        targets = np.asarray([label for _path, label in dataset.samples])
    else:  # pragma: no cover - all non-MedMNIST datasets use one of the branches above
        raise TypeError(f"Cannot extract class labels from {type(dataset).__name__}")
    return targets.reshape(-1).astype(np.int64, copy=False)


def stratified_train_validation_indices(
    labels: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Create a deterministic, class-stratified train/validation partition."""

    generator = np.random.default_rng(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for class_label in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_label)
        if class_indices.size < 2:
            raise ValueError(
                f"Class {class_label} has fewer than two training examples; "
                "a disjoint validation split is impossible"
            )
        class_indices = generator.permutation(class_indices)
        validation_count = int(round(class_indices.size * validation_fraction))
        validation_count = min(max(validation_count, 1), class_indices.size - 1)
        validation_indices.extend(class_indices[:validation_count].tolist())
        train_indices.extend(class_indices[validation_count:].tolist())
    generator.shuffle(train_indices)
    generator.shuffle(validation_indices)
    return train_indices, validation_indices


def make_loaders(
    train_set,
    validation_set,
    test_set,
    metadata,
    args,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, object]]:
    """Build disjoint train, validation, and test loaders for one seed."""

    pin_memory = torch.device(args.device).type == "cuda"
    persistent_workers = args.num_workers > 0
    train_generator = torch.Generator().manual_seed(seed)
    validation_generator = torch.Generator().manual_seed(seed + 1)
    test_generator = torch.Generator().manual_seed(seed + 2)
    if metadata["validation_is_official"]:
        train_indices = list(range(len(train_set)))
        validation_indices = list(range(len(validation_set)))
        selected_train_set = train_set
        selected_validation_set = validation_set
        split_protocol = "official_train_validation"
    else:
        if len(train_set) != len(validation_set):
            raise ValueError("training and validation-view datasets must be index-aligned")
        training_labels = dataset_targets(train_set)
        validation_labels = dataset_targets(validation_set)
        if not np.array_equal(training_labels, validation_labels):
            raise ValueError("training and validation-view labels are not index-aligned")
        train_indices, validation_indices = stratified_train_validation_indices(
            training_labels,
            args.validation_fraction,
            seed,
        )
        selected_train_set = Subset(train_set, train_indices)
        selected_validation_set = Subset(validation_set, validation_indices)
        split_protocol = "seed_specific_stratified_training_holdout"
    train_loader = DataLoader(
        selected_train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        generator=train_generator,
        worker_init_fn=seed_worker,
    )
    validation_loader = DataLoader(
        selected_validation_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        generator=validation_generator,
        worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        generator=test_generator,
        worker_init_fn=seed_worker,
    )
    split_sizes = {
        "train_samples": len(selected_train_set),
        "validation_samples": len(selected_validation_set),
        "test_samples": len(test_set),
        "split_seed": seed,
        "split_protocol": split_protocol,
        "train_index_sha256": index_sha256(train_indices),
        "validation_index_sha256": index_sha256(validation_indices),
        "test_index_sha256": index_sha256(range(len(test_set))),
    }
    return train_loader, validation_loader, test_loader, split_sizes


class ImageToPatchSequence(nn.Module):
    """Convert a patch-embedding grid from BCHW to a token sequence."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.flatten(2).transpose(1, 2)


class TransposeTokenChannelAxes(nn.Module):
    """Swap token and channel axes for token mixing, and back again."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.transpose(1, 2)


class FeedForward(nn.Module):
    """Two-layer MLP used for token or channel mixing."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class MixerBlock(nn.Module):
    """Residual token-mixing and channel-mixing MLPs."""

    def __init__(
        self,
        model_dim: int,
        num_patches: int,
        token_hidden_dim: int,
        channel_hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.token_mixing = nn.Sequential(
            nn.LayerNorm(model_dim),
            TransposeTokenChannelAxes(),
            FeedForward(num_patches, token_hidden_dim, dropout),
            TransposeTokenChannelAxes(),
        )
        self.channel_mixing = nn.Sequential(
            nn.LayerNorm(model_dim),
            FeedForward(model_dim, channel_hidden_dim, dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs + self.token_mixing(inputs)
        return hidden + self.channel_mixing(hidden)


class BackpropMixer(nn.Module):
    """MLP-Mixer trained end to end with backpropagation."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        image_size: int,
        patch_size: int,
        model_dim: int,
        depth: int,
        token_hidden_dim: int,
        channel_hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, model_dim, kernel_size=patch_size, stride=patch_size),
            ImageToPatchSequence(),
        )
        self.blocks = nn.ModuleList(
            [
                MixerBlock(
                    model_dim,
                    num_patches,
                    token_hidden_dim,
                    channel_hidden_dim,
                    dropout,
                )
                for _ in range(depth)
            ]
        )
        self.output_norm = nn.LayerNorm(model_dim)
        self.classifier = nn.Linear(model_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.patch_embedding(inputs)
        for block in self.blocks:
            hidden = block(hidden)
        return self.classifier(self.output_norm(hidden).mean(dim=1))


class LocalBPMixer(nn.Module):
    """MLP-Mixer with a local classifier and optimizer at every block."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        image_size: int,
        patch_size: int,
        model_dim: int,
        depth: int,
        token_hidden_dim: int,
        channel_hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.depth = depth
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, model_dim, kernel_size=patch_size, stride=patch_size),
            ImageToPatchSequence(),
        )
        self.blocks = nn.ModuleList(
            [
                MixerBlock(
                    model_dim,
                    num_patches,
                    token_hidden_dim,
                    channel_hidden_dim,
                    dropout,
                )
                for _ in range(depth)
            ]
        )
        self.local_norms = nn.ModuleList([nn.LayerNorm(model_dim) for _ in range(depth)])
        self.local_heads = nn.ModuleList([nn.Linear(model_dim, num_classes) for _ in range(depth)])

    def layer_logits(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        logits_by_layer = []
        hidden = self.patch_embedding(inputs)
        for layer_index in range(self.depth):
            hidden = self.blocks[layer_index](hidden)
            normalized = self.local_norms[layer_index](hidden).mean(dim=1)
            logits_by_layer.append(self.local_heads[layer_index](normalized))
        return logits_by_layer


def train_bp(model, loader, optimizer, scaler, device, amp_enabled, metadata):
    """Train the end-to-end baseline for one epoch."""

    criterion = task_loss(metadata["task"])
    model.train()
    total_samples = 0
    loss_sum = primary_sum = secondary_sum = 0
    y_true_batches: list[np.ndarray] = []
    y_score_batches: list[np.ndarray] = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = prepare_targets(targets, metadata["task"], device)
        optimizer.zero_grad(set_to_none=True)
        if amp_enabled:
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
        batch_size = targets.size(0)
        total_samples += batch_size
        loss_sum += loss.item() * batch_size
        if metadata["is_medmnist"]:
            append_medmnist_batches(
                y_true_batches,
                y_score_batches,
                logits,
                targets,
                metadata["task"],
            )
        else:
            primary_correct, secondary_correct = count_topk_correct(logits.detach(), targets)
            primary_sum += primary_correct
            secondary_sum += secondary_correct
    if metadata["is_medmnist"]:
        primary_metric, secondary_metric = compute_medmnist_metrics(
            y_true_batches, y_score_batches, metadata["task"]
        )
        return loss_sum / total_samples, primary_metric, secondary_metric
    return loss_sum / total_samples, primary_sum / total_samples, secondary_sum / total_samples


@torch.no_grad()
def eval_bp(model, loader, device, metadata):
    """Evaluate the end-to-end baseline."""

    criterion = task_loss(metadata["task"])
    model.eval()
    total_samples = 0
    loss_sum = primary_sum = secondary_sum = 0
    y_true_batches: list[np.ndarray] = []
    y_score_batches: list[np.ndarray] = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = prepare_targets(targets, metadata["task"], device)
        logits = model(images)
        loss = criterion(logits, targets)
        batch_size = targets.size(0)
        total_samples += batch_size
        loss_sum += loss.item() * batch_size
        if metadata["is_medmnist"]:
            append_medmnist_batches(
                y_true_batches,
                y_score_batches,
                logits,
                targets,
                metadata["task"],
            )
        else:
            primary_correct, secondary_correct = count_topk_correct(logits, targets)
            primary_sum += primary_correct
            secondary_sum += secondary_correct
    if metadata["is_medmnist"]:
        primary_metric, secondary_metric = compute_medmnist_metrics(
            y_true_batches, y_score_batches, metadata["task"]
        )
        return loss_sum / total_samples, primary_metric, secondary_metric
    return loss_sum / total_samples, primary_sum / total_samples, secondary_sum / total_samples


def build_local_bp_optimizers(model: LocalBPMixer, args):
    """Build one optimizer and scheduler for each locally trained block."""

    optimizers, schedulers = [], []
    for layer_index in range(model.depth):
        parameters = []
        if layer_index == 0:
            parameters += list(model.patch_embedding.parameters())
        parameters += (
            list(model.blocks[layer_index].parameters())
            + list(model.local_norms[layer_index].parameters())
            + list(model.local_heads[layer_index].parameters())
        )
        optimizer = build_optimizer(
            args.optimizer,
            parameters,
            args.lr,
            args.weight_decay,
            args.momentum,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        optimizers.append(optimizer)
        schedulers.append(scheduler)
    return optimizers, schedulers


def build_local_bp_scalers(depth: int, amp_enabled: bool):
    """Use one gradient scaler per independently stepped local optimizer."""

    return [torch.amp.GradScaler("cuda", enabled=amp_enabled) for _ in range(depth)]


def train_local_bp(
    model,
    loader,
    optimizers,
    scalers,
    device,
    updates_per_block,
    amp_enabled,
    metadata,
):
    """Train every Mixer block with a detached local cross-entropy objective."""

    if len(optimizers) != model.depth or len(scalers) != model.depth:
        raise ValueError("Local BP requires one optimizer and gradient scaler per block")
    criterion = task_loss(metadata["task"])
    model.train()
    total_samples = 0
    primary_sum = secondary_sum = 0
    layer_loss_sums = [0.0 for _ in range(model.depth)]
    y_true_batches: list[np.ndarray] = []
    y_score_batches: list[np.ndarray] = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = prepare_targets(targets, metadata["task"], device)
        batch_size = targets.size(0)
        total_samples += batch_size
        logits_by_layer = []
        detached_activation: torch.Tensor | None = None
        for layer_index in range(model.depth):
            optimizer = optimizers[layer_index]
            scaler = scalers[layer_index]
            local_loss_sum = 0.0
            last_activation: torch.Tensor | None = None
            last_logits: torch.Tensor | None = None
            for _ in range(updates_per_block):
                if (
                    layer_index > 0 and detached_activation is None
                ):  # pragma: no cover - guarded by loop order
                    raise RuntimeError("missing detached activation for Mixer block")
                if amp_enabled:
                    with torch.autocast("cuda", dtype=torch.float16):
                        layer_input = (
                            model.patch_embedding(images)
                            if layer_index == 0
                            else detached_activation
                        )
                        activation = model.blocks[layer_index](layer_input)
                        normalized = model.local_norms[layer_index](activation).mean(dim=1)
                        logits = model.local_heads[layer_index](normalized)
                        loss = criterion(logits, targets)
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    layer_input = (
                        model.patch_embedding(images) if layer_index == 0 else detached_activation
                    )
                    activation = model.blocks[layer_index](layer_input)
                    normalized = model.local_norms[layer_index](activation).mean(dim=1)
                    logits = model.local_heads[layer_index](normalized)
                    loss = criterion(logits, targets)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                local_loss_sum += loss.item()
                last_activation = activation.detach()
                last_logits = logits.detach()
            # Streaming Local BP: forward the activation produced during the
            # local update, rather than recomputing the block once more.
            if last_activation is None or last_logits is None:  # pragma: no cover - validated CLI
                raise RuntimeError("Local BP update produced no activation")
            logits_by_layer.append(last_logits)
            detached_activation = last_activation
            layer_loss_sums[layer_index] += (local_loss_sum / updates_per_block) * batch_size
        combined_logits = torch.stack(logits_by_layer).sum(dim=0)
        if metadata["is_medmnist"]:
            append_medmnist_batches(
                y_true_batches,
                y_score_batches,
                combined_logits,
                targets,
                metadata["task"],
            )
        else:
            primary_correct, secondary_correct = count_topk_correct(combined_logits, targets)
            primary_sum += primary_correct
            secondary_sum += secondary_correct
    mean_layer_losses = [loss_sum / total_samples for loss_sum in layer_loss_sums]
    mean_loss = float(np.mean(mean_layer_losses))
    if metadata["is_medmnist"]:
        primary_metric, secondary_metric = compute_medmnist_metrics(
            y_true_batches, y_score_batches, metadata["task"]
        )
        return mean_loss, primary_metric, secondary_metric
    return mean_loss, primary_sum / total_samples, secondary_sum / total_samples


@torch.no_grad()
def eval_local_bp(model, loader, device, metadata):
    """Evaluate a Local BP Mixer by summing all local logits."""

    criterion = task_loss(metadata["task"])
    model.eval()
    total_samples = 0
    loss_sum = primary_sum = secondary_sum = 0
    y_true_batches: list[np.ndarray] = []
    y_score_batches: list[np.ndarray] = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = prepare_targets(targets, metadata["task"], device)
        logits_by_layer = model.layer_logits(images)
        combined_logits = torch.stack(logits_by_layer).sum(dim=0)
        loss = criterion(combined_logits, targets)
        batch_size = targets.size(0)
        total_samples += batch_size
        loss_sum += loss.item() * batch_size
        if metadata["is_medmnist"]:
            append_medmnist_batches(
                y_true_batches,
                y_score_batches,
                combined_logits,
                targets,
                metadata["task"],
            )
        else:
            primary_correct, secondary_correct = count_topk_correct(combined_logits, targets)
            primary_sum += primary_correct
            secondary_sum += secondary_correct
    if metadata["is_medmnist"]:
        primary_metric, secondary_metric = compute_medmnist_metrics(
            y_true_batches, y_score_batches, metadata["task"]
        )
        return loss_sum / total_samples, primary_metric, secondary_metric
    return loss_sum / total_samples, primary_sum / total_samples, secondary_sum / total_samples


def csv_header():
    return [
        "result_schema_version",
        "suite_source_sha256",
        "support_source_sha256",
        "timestamp_utc",
        "status",
        "error",
        "run_signature",
        "dataset",
        "method",
        "depth",
        "dim",
        "token_dim",
        "channel_dim",
        "patch_size",
        "num_classes",
        "in_channels",
        "image_size",
        "task",
        "metric_primary_name",
        "metric_secondary_name",
        "dataset_source",
        "dataset_version",
        "training_augmentation",
        "final_evaluation_split",
        "seed",
        "epochs_target",
        "early_stop_patience",
        "early_stop_min_delta",
        "validation_fraction",
        "validation_is_official",
        "dropout",
        "scheduler",
        "scheduler_t_max",
        "epochs_ran",
        "early_stopped",
        "local_bp_updates_per_block",
        "optimizer",
        "lr",
        "weight_decay",
        "momentum",
        "batch_size",
        "eval_batch_size",
        "train_samples",
        "validation_samples",
        "test_samples",
        "split_seed",
        "split_protocol",
        "train_index_sha256",
        "validation_index_sha256",
        "test_index_sha256",
        "num_workers",
        "device",
        "amp_enabled",
        "best_epoch",
        "best_validation_primary",
        "best_validation_secondary",
        "best_validation_loss",
        "test_primary",
        "test_secondary",
        "test_loss",
        "test_evaluations",
        "last_train_loss",
        "last_train_primary",
        "last_train_secondary",
        "peak_train_mem_gb",
        "runtime_seconds",
        "num_params",
        "checkpoint_path",
        "checkpoint_sha256",
        "history_path",
        "history_sha256",
        "terminal_bad_epochs",
        "selection_rule",
        "restored_best_state_verified",
        "checkpoint_reload_verified",
        "python_version",
        "torch_version",
        "torchvision_version",
        "cuda_runtime_version",
        "cudnn_version",
        "gpu_name",
        "gpu_total_memory_bytes",
        "notes_json",
    ]


def _load_checkpoint(path: Path) -> dict[str, object]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid checkpoint payload: {path}")
    return payload


def validate_reusable_row(row: dict[str, str]) -> None:
    """Fail closed unless a successful row is fully provenance/audit verified."""

    sources = source_sha256()
    expected = {
        "result_schema_version": str(RESULT_SCHEMA_VERSION),
        "suite_source_sha256": sources["MLPMixerBenchmarkSuite.py"],
        "support_source_sha256": sources["medmnist_support.py"],
        "test_evaluations": "1",
        "selection_rule": "validation_primary_strict_improvement",
        "restored_best_state_verified": "1",
        "checkpoint_reload_verified": "1",
    }
    mismatches = [key for key, value in expected.items() if row.get(key) != value]
    if mismatches:
        raise ValueError(f"Reusable result metadata mismatch: {mismatches}")
    numeric_fields = (
        "best_validation_primary",
        "best_validation_secondary",
        "best_validation_loss",
        "test_primary",
        "test_secondary",
        "test_loss",
        "last_train_loss",
        "last_train_primary",
        "last_train_secondary",
    )
    if not all(math.isfinite(float(row[field])) for field in numeric_fields):
        raise ValueError("Reusable result contains a non-finite metric")
    try:
        notes = json.loads(row["notes_json"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Reusable result has invalid notes metadata") from exc
    if not isinstance(notes, dict):
        raise ValueError("Reusable result notes must be an object")
    dataset_integrity = validate_dataset_integrity_record(
        row.get("dataset", ""), notes.get("dataset_integrity")
    )
    history_path = Path(row["history_path"])
    if not history_path.is_file() or file_sha256(history_path) != row["history_sha256"]:
        raise ValueError(f"History provenance check failed: {history_path}")
    history_payload = json.loads(history_path.read_text(encoding="utf-8"))
    if history_payload.get("run_signature") != row.get("run_signature"):
        raise ValueError(f"History signature mismatch: {history_path}")
    if history_payload.get("dataset_integrity", {}) != dataset_integrity:
        raise ValueError(f"History dataset provenance mismatch: {history_path}")
    history = history_payload.get("epochs")
    if not isinstance(history, list) or len(history) != int(row["epochs_ran"]):
        raise ValueError(f"History length mismatch: {history_path}")
    best = -float("inf")
    best_epoch = 0
    bad_epochs = 0
    minimum_delta = float(row["early_stop_min_delta"])
    patience = int(row["early_stop_patience"])
    for expected_epoch, item in enumerate(history, start=1):
        if int(item["epoch"]) != expected_epoch:
            raise ValueError(f"Non-contiguous history: {history_path}")
        values = [
            float(item[field])
            for field in (
                "learning_rate",
                "train_loss",
                "train_primary",
                "train_secondary",
                "validation_loss",
                "validation_primary",
                "validation_secondary",
            )
        ]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite history value: {history_path}")
        improved = values[5] > best + minimum_delta
        if bool(item["improved"]) != improved:
            raise ValueError(f"Improvement flag mismatch: {history_path}")
        if improved:
            best = values[5]
            best_epoch = expected_epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
        if int(item["epochs_without_improvement"]) != bad_epochs:
            raise ValueError(f"Patience history mismatch: {history_path}")
        if not math.isclose(
            float(item["best_validation_primary"]),
            best,
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or int(item["best_epoch"]) != best_epoch:
            raise ValueError(f"Stored best history mismatch: {history_path}")
        if bad_epochs == patience and expected_epoch != len(history):
            raise ValueError(f"Training continued beyond patience: {history_path}")
    if best_epoch != int(row["best_epoch"]) or not math.isclose(
        best, float(row["best_validation_primary"]), rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(f"Strict best-selection mismatch: {history_path}")
    if bad_epochs != int(row["terminal_bad_epochs"]):
        raise ValueError(f"Terminal patience mismatch: {history_path}")
    epochs_ran = int(row["epochs_ran"])
    epochs_target = int(row["epochs_target"])
    stopped = row["early_stopped"] == "1"
    if stopped and bad_epochs != patience:
        raise ValueError(f"Invalid patience termination: {history_path}")
    if not stopped and epochs_ran != epochs_target:
        raise ValueError(f"Unexplained early termination: {history_path}")
    if not stopped and bad_epochs >= patience:
        raise ValueError(f"Missing patience termination: {history_path}")
    checkpoint_path = Path(row["checkpoint_path"])
    if (
        not checkpoint_path.is_file()
        or file_sha256(checkpoint_path) != row["checkpoint_sha256"]
    ):
        raise ValueError(f"Checkpoint provenance check failed: {checkpoint_path}")
    checkpoint = _load_checkpoint(checkpoint_path)
    if (
        int(checkpoint.get("schema_version", 0)) != RESULT_SCHEMA_VERSION
        or checkpoint.get("run_signature") != row.get("run_signature")
        or int(checkpoint.get("best_epoch", 0)) != best_epoch
        or not math.isclose(
            float(checkpoint.get("best_validation_primary", float("nan"))),
            best,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or checkpoint.get("history_sha256") != row.get("history_sha256")
        or checkpoint.get("source_sha256") != sources
        or checkpoint.get("dataset_integrity", {}) != dataset_integrity
        or not isinstance(checkpoint.get("state_dict"), dict)
    ):
        raise ValueError(f"Checkpoint audit metadata mismatch: {checkpoint_path}")


def load_done(csv_path: Path):
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    done = set()
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != csv_header():
            raise ValueError(
                f"Results schema mismatch in {csv_path}; use a new --results-csv "
                "instead of appending to a legacy file"
            )
        for r in reader:
            if r.get("status") != "ok":
                continue
            validate_reusable_row(r)
            signature = r.get("run_signature", "").strip()
            if signature:
                done.add(signature)
    return done


def run_signature(
    dataset: str,
    method: str,
    depth: int,
    dim: int,
    token_dim: int,
    channel_dim: int,
    patch_size: int,
    seed: int,
    args,
    dataset_integrity: dict[str, object],
) -> str:
    """Return a stable identity for every setting that affects a run."""

    payload = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "source_sha256": source_sha256(),
        "dataset_integrity": dataset_integrity,
        "dataset": dataset,
        "method": method,
        "depth": depth,
        "dim": dim,
        "token_dim": token_dim,
        "channel_dim": channel_dim,
        "patch_size": patch_size,
        "seed": seed,
        "epochs": args.epochs,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "validation_fraction": args.validation_fraction,
        "dropout": MIXER_DROPOUT,
        "scheduler": SCHEDULER_NAME,
        "scheduler_t_max": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "momentum": args.momentum,
        "local_bp_updates_per_block": (
            args.local_bp_updates_per_block if method == "local-bp" else 0
        ),
        "amp": torch.device(args.device).type == "cuda" and not args.disable_amp,
        "device": str(torch.device(args.device)),
        "num_workers": args.num_workers,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def append_row(csv_path: Path, row: dict[str, object]):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_head = not csv_path.exists() or csv_path.stat().st_size == 0
    head = csv_header()
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with csv_path.open("r", newline="", encoding="utf-8") as existing:
            if next(csv.reader(existing), None) != head:
                raise ValueError(f"Results schema mismatch in {csv_path}; use a new --results-csv")
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=head)
        if write_head:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in head})


def summary_header() -> list[str]:
    fields = [*SUMMARY_GROUP_FIELDS, "num_runs"]
    for output_name, _raw_name in SUMMARY_METRICS:
        fields.extend((f"{output_name}_mean", f"{output_name}_std"))
    return fields


def _mean_and_sample_std(values: list[float]) -> tuple[float, float | str]:
    mean = float(np.mean(values))
    if len(values) == 1:
        return mean, ""
    return mean, float(np.std(values, ddof=1))


def write_summary(results_csv: Path, summary_csv: Path) -> None:
    """Aggregate successful seed runs without mixing distinct configurations."""

    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    successful_rows: list[dict[str, str]] = []
    if results_csv.exists():
        with results_csv.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != csv_header():
                raise ValueError(
                    f"Results schema mismatch in {results_csv}; use a new --results-csv"
                )
            successful_rows = [row for row in reader if row.get("status") == "ok"]

    # Rerunning with --no-skip-existing appends a new row for the same run.
    # Preserve only the latest successful result for each stable signature.
    latest_by_signature: dict[str, dict[str, str]] = {}
    rows_without_signature: list[dict[str, str]] = []
    for row in successful_rows:
        signature = row.get("run_signature", "").strip()
        if signature:
            latest_by_signature[signature] = row
        else:
            rows_without_signature.append(row)

    for row in [*rows_without_signature, *latest_by_signature.values()]:
        key = tuple(row[field] for field in SUMMARY_GROUP_FIELDS)
        groups.setdefault(key, []).append(row)

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = summary_csv.with_suffix(f"{summary_csv.suffix}.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_header())
        writer.writeheader()
        for key in sorted(groups):
            rows = groups[key]
            summary_row: dict[str, str | int | float] = dict(
                zip(SUMMARY_GROUP_FIELDS, key, strict=True)
            )
            summary_row["num_runs"] = len(rows)
            for output_name, raw_name in SUMMARY_METRICS:
                try:
                    values = [float(row[raw_name]) for row in rows]
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid {raw_name!r} value in successful rows of {results_csv}"
                    ) from exc
                mean, sample_std = _mean_and_sample_std(values)
                summary_row[f"{output_name}_mean"] = mean
                summary_row[f"{output_name}_std"] = sample_std
            writer.writerow(summary_row)
    temporary_path.replace(summary_csv)


def run_one(
    dataset_name,
    method,
    depth,
    model_dim,
    token_dim,
    channel_dim,
    patch_size,
    train_loader,
    validation_loader,
    test_loader,
    split_sizes,
    metadata,
    args,
    run_seed,
    signature,
):
    """Train one run, select on validation data, then evaluate test once."""

    started_at = time.perf_counter()
    set_seed(run_seed)
    device = torch.device(args.device)
    amp_enabled = device.type == "cuda" and not args.disable_amp
    environment = runtime_environment(device)
    model_kwargs = dict(
        in_channels=metadata["in_channels"],
        num_classes=metadata["num_classes"],
        image_size=metadata["image_size"],
        patch_size=patch_size,
        model_dim=model_dim,
        depth=depth,
        token_hidden_dim=token_dim,
        channel_hidden_dim=channel_dim,
        dropout=MIXER_DROPOUT,
    )
    if method == "bp":
        model = BackpropMixer(**model_kwargs).to(device)
        optimizer = build_optimizer(
            args.optimizer,
            model.parameters(),
            args.lr,
            args.weight_decay,
            args.momentum,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    else:
        model = LocalBPMixer(**model_kwargs).to(device)
        optimizers, schedulers = build_local_bp_optimizers(model, args)
        scalers = build_local_bp_scalers(model.depth, amp_enabled)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    best_epoch = 0
    best_validation_primary = -float("inf")
    best_validation_secondary = -float("inf")
    best_validation_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    epochs_ran, peak_memory_bytes = 0, 0
    stopped_for_patience = False
    history: list[dict[str, object]] = []
    last_train_loss = last_train_primary = last_train_secondary = float("nan")

    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if method == "bp":
            current_lr = float(optimizer.param_groups[0]["lr"])
            train_loss, train_primary, train_secondary = train_bp(
                model,
                train_loader,
                optimizer,
                scaler,
                device,
                amp_enabled,
                metadata,
            )
            scheduler.step()
        else:
            current_lr = float(optimizers[0].param_groups[0]["lr"])
            train_loss, train_primary, train_secondary = train_local_bp(
                model,
                train_loader,
                optimizers,
                scalers,
                device,
                args.local_bp_updates_per_block,
                amp_enabled,
                metadata,
            )
            for local_scheduler in schedulers:
                local_scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_memory_bytes = max(
                peak_memory_bytes,
                torch.cuda.max_memory_allocated(device),
            )
        epochs_ran = epoch
        if method == "bp":
            validation_loss, validation_primary, validation_secondary = eval_bp(
                model, validation_loader, device, metadata
            )
        else:
            validation_loss, validation_primary, validation_secondary = eval_local_bp(
                model, validation_loader, device, metadata
            )
        last_train_loss = train_loss
        last_train_primary = train_primary
        last_train_secondary = train_secondary
        epoch_metrics = (
            train_loss,
            train_primary,
            train_secondary,
            validation_loss,
            validation_primary,
            validation_secondary,
            current_lr,
        )
        if not all(np.isfinite(value) for value in epoch_metrics):
            raise FloatingPointError(f"Non-finite epoch metrics at epoch {epoch}")
        improved = validation_primary > (
            best_validation_primary + args.early_stop_min_delta
        )
        if improved:
            best_epoch = epoch
            best_validation_primary = validation_primary
            best_validation_secondary = validation_secondary
            best_validation_loss = validation_loss
            best_state = {
                name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        history.append(
            {
                "epoch": epoch,
                "learning_rate": current_lr,
                "train_loss": train_loss,
                "train_primary": train_primary,
                "train_secondary": train_secondary,
                "validation_loss": validation_loss,
                "validation_primary": validation_primary,
                "validation_secondary": validation_secondary,
                "improved": improved,
                "best_validation_primary": best_validation_primary,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
            }
        )
        metric_primary = metadata["metric_primary_name"]
        metric_secondary = metadata["metric_secondary_name"]
        print(
            f"[{dataset_name}|{method}|d={depth}|dim={model_dim}] "
            f"Epoch {epoch:03d}/{args.epochs} | LR {current_lr:.6e} | "
            f"Train loss {train_loss:.4f} "
            f"{metric_primary} {train_primary * 100:.2f}% "
            f"{metric_secondary} {train_secondary * 100:.2f}% | "
            f"Validation loss {validation_loss:.4f} "
            f"{metric_primary} {validation_primary * 100:.2f}% "
            f"{metric_secondary} {validation_secondary * 100:.2f}%",
            flush=True,
        )
        if epochs_without_improvement >= args.early_stop_patience:
            stopped_for_patience = True
            print(
                f"Early stop at epoch {epoch} ({args.early_stop_patience} bad epochs).",
                flush=True,
            )
            break

    if best_state is None:
        raise RuntimeError("Training finished without a valid validation checkpoint")
    model.load_state_dict(best_state, strict=True)
    restored_best_state_verified = state_dicts_equal(model.state_dict(), best_state)
    if not restored_best_state_verified:
        raise RuntimeError("Best validation state was not restored exactly")
    artifact_key = hashlib.sha256(signature.encode("utf-8")).hexdigest()
    history_file = (
        Path(args.output_dir) / "artifacts" / f"{artifact_key}.history.json"
    ).resolve()
    history_sha256 = atomic_json(
        history_file,
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "run_signature": signature,
            "source_sha256": source_sha256(),
            "dataset_integrity": metadata.get("dataset_integrity", {}),
            "epochs": history,
        },
    )
    checkpoint_root = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else Path(args.output_dir) / "checkpoints"
    )
    checkpoint = (
        checkpoint_root
        / dataset_name
        / method
        / f"d{depth}_w{model_dim}"
        / f"seed_{run_seed}_{artifact_key[:16]}.pt"
    ).resolve()
    checkpoint_payload: dict[str, object] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "state_dict": best_state,
        "run_signature": signature,
        "best_epoch": best_epoch,
        "best_validation_primary": best_validation_primary,
        "history_sha256": history_sha256,
        "source_sha256": source_sha256(),
        "dataset_integrity": metadata.get("dataset_integrity", {}),
    }
    checkpoint_sha256 = save_checkpoint_atomic(checkpoint, checkpoint_payload)
    reloaded_checkpoint = _load_checkpoint(checkpoint)
    checkpoint_reload_verified = state_dicts_equal(
        reloaded_checkpoint["state_dict"], best_state
    )
    if not checkpoint_reload_verified:
        raise RuntimeError("Serialized checkpoint does not match the restored best state")
    checkpoint_path = str(checkpoint)
    test_evaluations = 0
    test_evaluations += 1
    if method == "bp":
        test_loss, test_primary, test_secondary = eval_bp(model, test_loader, device, metadata)
    else:
        test_loss, test_primary, test_secondary = eval_local_bp(
            model, test_loader, device, metadata
        )
    if test_evaluations != 1:
        raise RuntimeError("The official test split must be evaluated exactly once")
    final_metrics = (
        test_loss,
        test_primary,
        test_secondary,
        best_validation_primary,
        best_validation_secondary,
        best_validation_loss,
    )
    if not all(np.isfinite(value) for value in final_metrics):
        raise FloatingPointError("Non-finite final metric")
    runtime_seconds = time.perf_counter() - started_at

    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "suite_source_sha256": source_sha256()["MLPMixerBenchmarkSuite.py"],
        "support_source_sha256": source_sha256()["medmnist_support.py"],
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "status": "ok",
        "error": "",
        "run_signature": signature,
        "dataset": dataset_name,
        "method": method,
        "depth": depth,
        "dim": model_dim,
        "token_dim": token_dim,
        "channel_dim": channel_dim,
        "patch_size": patch_size,
        "num_classes": metadata["num_classes"],
        "in_channels": metadata["in_channels"],
        "image_size": metadata["image_size"],
        "task": metadata["task"],
        "metric_primary_name": metadata["metric_primary_name"],
        "metric_secondary_name": metadata["metric_secondary_name"],
        "dataset_source": metadata["dataset_source"],
        "dataset_version": metadata["dataset_version"],
        "training_augmentation": metadata["training_augmentation"],
        "final_evaluation_split": metadata["final_evaluation_split"],
        "seed": run_seed,
        "epochs_target": args.epochs,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "validation_fraction": args.validation_fraction,
        "validation_is_official": int(metadata["validation_is_official"]),
        "dropout": MIXER_DROPOUT,
        "scheduler": SCHEDULER_NAME,
        "scheduler_t_max": args.epochs,
        "epochs_ran": epochs_ran,
        "early_stopped": int(stopped_for_patience),
        "local_bp_updates_per_block": (
            args.local_bp_updates_per_block if method == "local-bp" else 0
        ),
        "optimizer": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "momentum": args.momentum,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        **split_sizes,
        "num_workers": args.num_workers,
        "device": str(device),
        "amp_enabled": int(amp_enabled),
        "best_epoch": best_epoch,
        "best_validation_primary": best_validation_primary,
        "best_validation_secondary": best_validation_secondary,
        "best_validation_loss": best_validation_loss,
        "test_primary": test_primary,
        "test_secondary": test_secondary,
        "test_loss": test_loss,
        "test_evaluations": test_evaluations,
        "last_train_loss": last_train_loss,
        "last_train_primary": last_train_primary,
        "last_train_secondary": last_train_secondary,
        "peak_train_mem_gb": (peak_memory_bytes / (1024**3) if device.type == "cuda" else 0.0),
        "runtime_seconds": runtime_seconds,
        "num_params": parameter_count,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "history_path": str(history_file),
        "history_sha256": history_sha256,
        "terminal_bad_epochs": epochs_without_improvement,
        "selection_rule": "validation_primary_strict_improvement",
        "restored_best_state_verified": int(restored_best_state_verified),
        "checkpoint_reload_verified": int(checkpoint_reload_verified),
        "python_version": environment["python_version"],
        "torch_version": environment["torch_version"],
        "torchvision_version": environment["torchvision_version"],
        "cuda_runtime_version": environment["cuda_runtime_version"],
        "cudnn_version": environment["cudnn_version"],
        "gpu_name": environment["gpu_name"],
        "gpu_total_memory_bytes": environment["gpu_total_memory_bytes"],
        "notes_json": json.dumps(
            {
                "amp": amp_enabled,
                "amp_gradient_scaler": (
                    "one_per_local_optimizer" if method == "local-bp" else "one_global"
                ),
                "environment": environment,
                "dataset_integrity": metadata.get("dataset_integrity", {}),
                "selection": "validation_primary_strict_improvement",
                "test_evaluated_once_after_best_checkpoint_restore": True,
            },
            sort_keys=True,
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for field_name in ("epochs", "batch_size", "eval_batch_size"):
        if getattr(args, field_name) <= 0:
            raise ValueError(f"--{field_name.replace('_', '-')} must be positive")
    if not args.depths or any(value <= 0 for value in args.depths):
        raise ValueError("--depths must contain positive integers")
    if not args.dims or any(value <= 0 for value in args.dims):
        raise ValueError("--dims must contain positive integers")
    if args.token_dim < 0 or args.channel_dim < 0 or args.patch_size < 0:
        raise ValueError("explicit token, channel, and patch dimensions cannot be negative")
    if args.token_ratio <= 0 or args.channel_ratio <= 0:
        raise ValueError("--token-ratio and --channel-ratio must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.lr <= 0 or args.weight_decay < 0 or args.momentum < 0:
        raise ValueError("optimizer settings must be nonnegative and --lr must be positive")
    if args.local_bp_updates_per_block <= 0:
        raise ValueError("--local-bp-updates-per-block must be > 0")
    if args.early_stop_patience <= 0:
        raise ValueError("--early-stop-patience must be > 0")
    if args.early_stop_min_delta < 0:
        raise ValueError("--early-stop-min-delta cannot be negative")
    if not 0 < args.validation_fraction < 1:
        raise ValueError("--validation-fraction must lie strictly between 0 and 1")
    matched_seeds = resolve_seeds(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    set_seed(matched_seeds[0])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / args.results_csv
    summary_path = output_dir / args.summary_csv
    if results_path.resolve() == summary_path.resolve():
        raise ValueError("--results-csv and --summary-csv must name different files")
    completed_signatures = load_done(results_path) if args.skip_existing else set()
    data_root = Path(args.data_dir)
    run_index = 0
    failures: list[str] = []

    print(f"CSV output: {results_path}", flush=True)
    for dataset_name in args.datasets:
        print(f"\nPreparing dataset: {dataset_name}", flush=True)
        try:
            train_set, validation_set, test_set, metadata = build_dataset(
                dataset_name, data_root, args
            )
            dataset_integrity = metadata.get("dataset_integrity", {})
            validate_dataset_integrity_record(dataset_name, dataset_integrity)
            if dataset_integrity:
                atomic_json(
                    output_dir / "dataset_provenance" / f"{dataset_name}.json",
                    {
                        "dataset": dataset_name,
                        "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "integrity": dataset_integrity,
                    },
                )
            patch_size = args.patch_size if args.patch_size > 0 else metadata["patch_default"]
            if metadata["image_size"] % patch_size != 0:
                raise ValueError(
                    f"{dataset_name}: image_size {metadata['image_size']} "
                    f"not divisible by patch {patch_size}"
                )
        except Exception as exc:
            failures.append(f"{dataset_name}: {exc!r}")
            print(f"FAILED to set up {dataset_name}: {exc!r}", flush=True)
            append_row(
                results_path,
                {
                    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                    "status": "failed",
                    "error": repr(exc),
                    "dataset": dataset_name,
                    "notes_json": json.dumps({"run": "dataset setup failed"}),
                },
            )
            continue
        for depth in args.depths:
            for model_dim in args.dims:
                token_dim = (
                    args.token_dim
                    if args.token_dim > 0
                    else int(round(model_dim * args.token_ratio))
                )
                channel_dim = (
                    args.channel_dim
                    if args.channel_dim > 0
                    else int(round(model_dim * args.channel_ratio))
                )
                for seed in matched_seeds:
                    for method in args.methods:
                        run_index += 1
                        signature = run_signature(
                            dataset_name,
                            method,
                            depth,
                            model_dim,
                            token_dim,
                            channel_dim,
                            patch_size,
                            seed,
                            args,
                            dataset_integrity,
                        )
                        if signature in completed_signatures:
                            print(
                                f"Skip existing: {dataset_name}/{method}/"
                                f"d{depth}/w{model_dim}/seed{seed}",
                                flush=True,
                            )
                            continue
                        print(
                            f"\nRun {run_index}: dataset={dataset_name} method={method} "
                            f"depth={depth} dim={model_dim} token_dim={token_dim} "
                            f"channel_dim={channel_dim} patch={patch_size} seed={seed}",
                            flush=True,
                        )
                        try:
                            (
                                train_loader,
                                validation_loader,
                                test_loader,
                                split_sizes,
                            ) = make_loaders(
                                train_set,
                                validation_set,
                                test_set,
                                metadata,
                                args,
                                seed,
                            )
                            result_row = run_one(
                                dataset_name,
                                method,
                                depth,
                                model_dim,
                                token_dim,
                                channel_dim,
                                patch_size,
                                train_loader,
                                validation_loader,
                                test_loader,
                                split_sizes,
                                metadata,
                                args,
                                seed,
                                signature,
                            )
                            append_row(results_path, result_row)
                            completed_signatures.add(signature)
                            print(
                                f"Done: test_{result_row['metric_primary_name']}="
                                f"{result_row['test_primary'] * 100:.2f}% "
                                f"test_{result_row['metric_secondary_name']}="
                                f"{result_row['test_secondary'] * 100:.2f}% "
                                f"| peak_mem={result_row['peak_train_mem_gb']:.2f}GB",
                                flush=True,
                            )
                        except Exception as exc:
                            failures.append(
                                f"{dataset_name}/{method}/d{depth}/w{model_dim}/seed{seed}: {exc!r}"
                            )
                            print(f"FAILED: {repr(exc)}", flush=True)
                            append_row(
                                results_path,
                                {
                                    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(
                                        timespec="seconds"
                                    ),
                                    "status": "failed",
                                    "error": repr(exc),
                                    "run_signature": signature,
                                    "dataset": dataset_name,
                                    "method": method,
                                    "depth": depth,
                                    "dim": model_dim,
                                    "token_dim": token_dim,
                                    "channel_dim": channel_dim,
                                    "patch_size": patch_size,
                                    "num_classes": metadata["num_classes"],
                                    "in_channels": metadata["in_channels"],
                                    "image_size": metadata["image_size"],
                                    "task": metadata["task"],
                                    "metric_primary_name": metadata["metric_primary_name"],
                                    "metric_secondary_name": metadata["metric_secondary_name"],
                                    "seed": seed,
                                    "epochs_target": args.epochs,
                                    "early_stop_patience": args.early_stop_patience,
                                    "early_stop_min_delta": args.early_stop_min_delta,
                                    "validation_fraction": args.validation_fraction,
                                    "validation_is_official": int(
                                        metadata["validation_is_official"]
                                    ),
                                    "epochs_ran": 0,
                                    "early_stopped": 0,
                                    "local_bp_updates_per_block": (
                                        args.local_bp_updates_per_block
                                        if method == "local-bp"
                                        else 0
                                    ),
                                    "optimizer": args.optimizer,
                                    "lr": args.lr,
                                    "weight_decay": args.weight_decay,
                                    "momentum": args.momentum,
                                    "batch_size": args.batch_size,
                                    "eval_batch_size": args.eval_batch_size,
                                    "train_samples": 0,
                                    "validation_samples": 0,
                                    "test_samples": len(test_set),
                                    "num_workers": args.num_workers,
                                    "device": str(device),
                                    "amp_enabled": int(
                                        device.type == "cuda" and not args.disable_amp
                                    ),
                                    "test_evaluations": 0,
                                    "notes_json": json.dumps({"run": "failed"}),
                                },
                            )
    write_summary(results_path, summary_path)
    print(f"Summary output: {summary_path}", flush=True)
    if failures:
        print(f"Finished with {len(failures)} failed run(s).", flush=True)
        return 1
    print("All runs finished.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
