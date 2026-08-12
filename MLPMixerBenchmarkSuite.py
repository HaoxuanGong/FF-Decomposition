#!/usr/bin/env python3
"""Run the paper's matched Backpropagation and Mono-Forward MLP-Mixer sweep."""

import argparse
import csv
import datetime as dt
import json
import os
import random
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.optim import Adam, AdamW, RMSprop, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

from medmnist_support import (
    MEDMNIST_DATASETS,
    append_medmnist_batches,
    build_medmnist_splits,
    compute_medmnist_metrics,
    prepare_targets,
    task_loss,
)

TINY_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
COIL_URL = "http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip"
MAIN_DATASETS = ("cifar10", "cifar100", "tinyimagenet", "coil100", *MEDMNIST_DATASETS)
PAPER_SEEDS = (41, 42, 43)
PAPER_DEPTHS = (5, 8, 12)
PAPER_DIMS = (256, 512)

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
    "epochs_target",
    "early_stop_patience",
    "early_stop_min_delta",
    "mf_local_iters",
    "optimizer",
    "lr",
    "weight_decay",
    "momentum",
    "batch_size",
    "eval_batch_size",
    "num_workers",
    "device",
    "amp_enabled",
)
SUMMARY_METRICS = (
    ("best_primary", "best_test_top1"),
    ("best_secondary", "best_test_top5"),
    ("peak_train_mem_gb", "peak_train_mem_gb"),
    ("best_epoch", "best_epoch"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("MLP-Mixer BP vs MF benchmark suite")
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        choices=MAIN_DATASETS,
        help="One or more main-table datasets to benchmark.",
    )
    parser.add_argument("--methods", nargs="+", default=["bp", "mf"], choices=["bp", "mf"])
    parser.add_argument("--depths", nargs="+", type=int, default=list(PAPER_DEPTHS))
    parser.add_argument("--dims", nargs="+", type=int, default=list(PAPER_DIMS))
    parser.add_argument("--epochs", type=int, default=480)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
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
    parser.add_argument("--mf-local-iters", type=int, default=3)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="mlpmixer_suite_results")
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
        help="Matched seeds for BP and MF (default: 41 42 43).",
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
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return
    print(f"Downloading {url}", flush=True)
    urllib.request.urlretrieve(url, destination)


def extract_zip_safely(archive: Path, destination: Path) -> None:
    """Extract an archive only when every member remains below destination."""

    destination = destination.resolve()
    with zipfile.ZipFile(archive, "r") as zip_file:
        for member in zip_file.infolist():
            target = (destination / member.filename).resolve()
            if os.path.commonpath((destination, target)) != str(destination):
                raise ValueError(f"Unsafe path in archive {archive}: {member.filename}")
        zip_file.extractall(destination)


def ensure_tiny(data_root: Path, allow_download: bool) -> Path:
    dataset_root = data_root / "tiny-imagenet-200"
    if (dataset_root / "train").exists() and (dataset_root / "val").exists():
        return dataset_root
    if not allow_download:
        raise FileNotFoundError(f"{dataset_root} missing. Use --download-tinyimagenet")
    archive_path = data_root / "tiny-imagenet-200.zip"
    ensure_download(TINY_URL, archive_path)
    extract_zip_safely(archive_path, data_root)
    return dataset_root


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


def ensure_coil(data_root: Path, allow_download: bool) -> Path:
    dataset_root = data_root / "coil-100"
    if dataset_root.exists() and any(dataset_root.glob("obj*__*.png")):
        return dataset_root
    if not allow_download:
        raise FileNotFoundError(f"{dataset_root} missing. Use --download-coil100")
    archive_path = data_root / "coil-100.zip"
    ensure_download(COIL_URL, archive_path)
    extract_zip_safely(archive_path, data_root)
    return dataset_root


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
            ),
        )
    if dataset_name == "tinyimagenet":
        dataset_root = ensure_tiny(data_root, args.download_tinyimagenet)
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
        test_set = TinyImageNetValidation(dataset_root, train_set.class_to_idx, test_transform)
        return (
            train_set,
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
            ),
        )
    if dataset_name == "coil100":
        dataset_root = ensure_coil(data_root, args.download_coil100)
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
            ),
        )
    if dataset_name in MEDMNIST_DATASETS:
        return build_medmnist_splits(dataset_name, data_root)
    raise ValueError(dataset_name)


def make_loaders(train_set, test_set, args, seed: int) -> tuple[DataLoader, DataLoader]:
    pin_memory = torch.device(args.device).type == "cuda"
    persistent_workers = args.num_workers > 0
    train_generator = torch.Generator().manual_seed(seed)
    eval_generator = torch.Generator().manual_seed(seed + 1)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        generator=train_generator,
        worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        generator=eval_generator,
        worker_init_fn=seed_worker,
    )
    return train_loader, test_loader


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


class MonoForwardMixer(nn.Module):
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


def build_mf_optimizers(model: MonoForwardMixer, args):
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


def train_mf(model, loader, optimizers, scaler, device, local_iters, amp_enabled, metadata):
    """Train every Mixer block with its local Mono-Forward objective."""

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
            local_loss_sum = 0.0
            last_activation: torch.Tensor | None = None
            last_logits: torch.Tensor | None = None
            for _ in range(local_iters):
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
            # Streaming MF: forward the activation already produced during the
            # local update, rather than recomputing the block once more.
            if last_activation is None or last_logits is None:  # pragma: no cover - validated CLI
                raise RuntimeError("MF local update produced no activation")
            logits_by_layer.append(last_logits)
            detached_activation = last_activation
            layer_loss_sums[layer_index] += (local_loss_sum / local_iters) * batch_size
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
def eval_mf(model, loader, device, metadata):
    """Evaluate a Mono-Forward Mixer by summing all local logits."""

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
        "seed",
        "epochs_target",
        "early_stop_patience",
        "early_stop_min_delta",
        "epochs_ran",
        "early_stopped",
        "mf_local_iters",
        "optimizer",
        "lr",
        "weight_decay",
        "momentum",
        "batch_size",
        "eval_batch_size",
        "num_workers",
        "device",
        "amp_enabled",
        "best_epoch",
        "best_test_top1",
        "best_test_top5",
        "best_test_loss",
        "final_test_top1",
        "final_test_top5",
        "final_test_loss",
        "best_train_loss",
        "final_train_loss",
        "final_train_top1",
        "final_train_top5",
        "peak_train_mem_gb",
        "num_params",
        "notes_json",
    ]


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
) -> str:
    """Return a stable identity for every setting that affects a run."""

    payload = {
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
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "momentum": args.momentum,
        "mf_local_iters": args.mf_local_iters if method == "mf" else 0,
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
    test_loader,
    metadata,
    args,
    run_seed,
    signature,
):
    """Train and evaluate one dataset, method, architecture, and seed."""

    set_seed(run_seed)
    device = torch.device(args.device)
    amp_enabled = device.type == "cuda" and not args.disable_amp
    model_kwargs = dict(
        in_channels=metadata["in_channels"],
        num_classes=metadata["num_classes"],
        image_size=metadata["image_size"],
        patch_size=patch_size,
        model_dim=model_dim,
        depth=depth,
        token_hidden_dim=token_dim,
        channel_hidden_dim=channel_dim,
        dropout=0.1,
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
    else:
        model = MonoForwardMixer(**model_kwargs).to(device)
        optimizers, schedulers = build_mf_optimizers(model, args)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    best_epoch, best_primary, best_secondary, best_test_loss = 0, -1.0, -1.0, float("inf")
    best_train_loss, epochs_without_improvement = float("inf"), 0
    epochs_ran, peak_memory_bytes = 0, 0
    final_metrics = {}

    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if method == "bp":
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
            current_lr = optimizer.param_groups[0]["lr"]
        else:
            train_loss, train_primary, train_secondary = train_mf(
                model,
                train_loader,
                optimizers,
                scaler,
                device,
                args.mf_local_iters,
                amp_enabled,
                metadata,
            )
            for local_scheduler in schedulers:
                local_scheduler.step()
            current_lr = optimizers[0].param_groups[0]["lr"]
        if device.type == "cuda":
            peak_memory_bytes = max(
                peak_memory_bytes,
                torch.cuda.max_memory_allocated(device),
            )
        epochs_ran = epoch
        if method == "bp":
            test_loss, test_primary, test_secondary = eval_bp(model, test_loader, device, metadata)
        else:
            test_loss, test_primary, test_secondary = eval_mf(model, test_loader, device, metadata)
        final_metrics = dict(
            train_loss=train_loss,
            train_primary=train_primary,
            train_secondary=train_secondary,
            test_loss=test_loss,
            test_primary=test_primary,
            test_secondary=test_secondary,
        )
        if test_primary > best_primary:
            best_epoch = epoch
            best_primary = test_primary
            best_secondary = test_secondary
            best_test_loss = test_loss
        if train_loss < (best_train_loss - args.early_stop_min_delta):
            best_train_loss = train_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        metric_primary = metadata["metric_primary_name"]
        metric_secondary = metadata["metric_secondary_name"]
        print(
            f"[{dataset_name}|{method}|d={depth}|dim={model_dim}] "
            f"Epoch {epoch:03d}/{args.epochs} | LR {current_lr:.6e} | "
            f"Train loss {train_loss:.4f} "
            f"{metric_primary} {train_primary * 100:.2f}% "
            f"{metric_secondary} {train_secondary * 100:.2f}% | "
            f"Test {metric_primary} {test_primary * 100:.2f}% "
            f"{metric_secondary} {test_secondary * 100:.2f}%",
            flush=True,
        )
        if epochs_without_improvement >= args.early_stop_patience:
            print(
                f"Early stop at epoch {epoch} ({args.early_stop_patience} bad epochs).",
                flush=True,
            )
            break

    return {
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
        "seed": run_seed,
        "epochs_target": args.epochs,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "epochs_ran": epochs_ran,
        "early_stopped": int(epochs_ran < args.epochs),
        "mf_local_iters": args.mf_local_iters if method == "mf" else 0,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "momentum": args.momentum,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "device": str(device),
        "amp_enabled": int(amp_enabled),
        "best_epoch": best_epoch,
        "best_test_top1": best_primary,
        "best_test_top5": best_secondary,
        "best_test_loss": best_test_loss,
        "final_test_top1": final_metrics["test_primary"],
        "final_test_top5": final_metrics["test_secondary"],
        "final_test_loss": final_metrics["test_loss"],
        "best_train_loss": best_train_loss,
        "final_train_loss": final_metrics["train_loss"],
        "final_train_top1": final_metrics["train_primary"],
        "final_train_top5": final_metrics["train_secondary"],
        "peak_train_mem_gb": (peak_memory_bytes / (1024**3) if device.type == "cuda" else 0.0),
        "num_params": parameter_count,
        "notes_json": json.dumps({"amp": amp_enabled}),
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
    if args.mf_local_iters <= 0:
        raise ValueError("--mf-local-iters must be > 0")
    if args.early_stop_patience <= 0:
        raise ValueError("--early-stop-patience must be > 0")
    if args.early_stop_min_delta < 0:
        raise ValueError("--early-stop-min-delta cannot be negative")
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
            train_set, test_set, metadata = build_dataset(dataset_name, data_root, args)
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
                            train_loader, test_loader = make_loaders(
                                train_set, test_set, args, seed
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
                                test_loader,
                                metadata,
                                args,
                                seed,
                                signature,
                            )
                            append_row(results_path, result_row)
                            completed_signatures.add(signature)
                            print(
                                f"Done: best_{result_row['metric_primary_name']}="
                                f"{result_row['best_test_top1'] * 100:.2f}% "
                                f"best_{result_row['metric_secondary_name']}="
                                f"{result_row['best_test_top5'] * 100:.2f}% "
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
                                    "epochs_ran": 0,
                                    "early_stopped": 0,
                                    "mf_local_iters": (
                                        args.mf_local_iters if method == "mf" else 0
                                    ),
                                    "optimizer": args.optimizer,
                                    "lr": args.lr,
                                    "weight_decay": args.weight_decay,
                                    "momentum": args.momentum,
                                    "batch_size": args.batch_size,
                                    "eval_batch_size": args.eval_batch_size,
                                    "num_workers": args.num_workers,
                                    "device": str(device),
                                    "amp_enabled": int(
                                        device.type == "cuda" and not args.disable_amp
                                    ),
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
