#!/usr/bin/env python3
"""Compare three cross-entropy controls on a matched four-block CNN.

All methods use the same Conv--BatchNorm--ReLU--MaxPool backbone. BP applies
one cross-entropy loss to the final classifier and propagates its gradient
through the complete backbone. Local BP attaches a classifier to every block,
optimizes each block and classifier with a local cross-entropy loss, and
detaches the activation passed between blocks. CE (Global, Multi-Head) uses
the same per-block classifiers, losses, and summed-logit prediction rule as
Local BP, but removes the detachments and performs one global update from the
sum of the four cross-entropy losses.

For every seed, a deterministic stratified subset of the official training
split is reserved for validation. Validation accuracy selects the checkpoint
and controls early stopping. The selected checkpoint is restored before the
official test split is evaluated exactly once. Transforms perform tensor
conversion and normalization only: there is no augmentation, dropout, or
weight decay.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import platform
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
import uuid

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, MNIST
from torchvision.transforms import Compose, Normalize, ToTensor


@dataclass(frozen=True)
class DatasetSpec:
    dataset_class: type
    input_channels: int
    num_classes: int
    mean: tuple[float, ...]
    std: tuple[float, ...]


DATASETS = {
    "mnist": DatasetSpec(MNIST, 1, 10, (0.1307,), (0.3081,)),
    "fashionmnist": DatasetSpec(FashionMNIST, 1, 10, (0.2860,), (0.3530,)),
    "cifar10": DatasetSpec(
        CIFAR10,
        3,
        10,
        (0.4914, 0.4822, 0.4465),
        (0.2471, 0.2435, 0.2616),
    ),
    "cifar100": DatasetSpec(
        CIFAR100,
        3,
        100,
        (0.5071, 0.4865, 0.4409),
        (0.2673, 0.2564, 0.2762),
    ),
}

METHODS = ("bp", "local-bp", "ce-matched-ge")
BACKBONE_WIDTHS = (64, 128, 256, 512)
WEIGHT_DECAY = 0.0
DROPOUT = 0.0
SGD_MOMENTUM = 0.9


@dataclass
class RunConfig:
    method: str
    dataset: str
    epochs: int
    patience: int
    minimum_delta: float
    validation_size: int
    optimizer: str
    optimizer_parameters: str
    learning_rate: float
    scheduler: str
    scheduler_parameters: str
    step_size: int
    step_gamma: float
    batch_size: int
    eval_batch_size: int
    seeds: list[int]
    num_workers: int
    data_dir: str
    output_dir: str
    device: str
    download: bool
    save_checkpoints: bool
    provenance_run_id: str
    overwrite: bool
    defer_test: bool
    heartbeat_file: str
    terminal_classifier_bias: bool = False
    backbone_widths: tuple[int, ...] = BACKBONE_WIDTHS
    validation_split: str = "seed-specific-stratified"
    test_evaluation_policy: str = "once-after-best-checkpoint-restoration"
    augmentation: str = "none"
    dropout: float = DROPOUT
    weight_decay: float = WEIGHT_DECAY
    local_bp_prediction: str = "sum-local-logits"
    precision: str = "float32"
    checkpoint_selection: str = "strict-validation-accuracy-improvement"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument(
        "--method",
        choices=METHODS,
        required=True,
        help=(
            "Use terminal CE, detached per-block CE, or globally propagated "
            "per-block CE on the matched backbone."
        ),
    )
    parser.add_argument("--epochs", type=int, default=200, help="Maximum number of epochs.")
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Stop after this many epochs without a validation-accuracy improvement.",
    )
    parser.add_argument(
        "--minimum-delta",
        type=float,
        default=0.0,
        help="Minimum validation-accuracy increase counted as an improvement.",
    )
    parser.add_argument(
        "--validation-size",
        type=int,
        default=5000,
        help="Number of examples reserved from the official training split.",
    )
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="adam")
    parser.add_argument(
        "--lr", "--learning-rate", dest="learning_rate", type=float, default=0.001
    )
    parser.add_argument("--scheduler", choices=("cosine", "step"), default="cosine")
    parser.add_argument("--step-size", type=int, default=30)
    parser.add_argument("--step-gamma", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[424, 425, 426],
        help="Independent run and train/validation split seeds.",
    )
    parser.add_argument("--num-workers", "--workers", dest="num_workers", type=int, default=4)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Result directory (default: cnn_<method>_<dataset>).",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument(
        "--defer-test",
        action="store_true",
        help="Train and save validation-selected checkpoints without evaluating the test set.",
    )
    parser.add_argument(
        "--heartbeat-file",
        default="",
        help="Optional JSON heartbeat path updated after every epoch.",
    )
    parser.add_argument(
        "--provenance-run-id",
        default="standalone",
        help="Opaque scheduler-issued identifier recorded in every run artifact.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing files for this run instead of refusing to start.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.patience <= 0:
        raise ValueError("--patience must be positive")
    if args.minimum_delta < 0:
        raise ValueError("--minimum-delta cannot be negative")
    if args.validation_size <= 0:
        raise ValueError("--validation-size must be positive")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.step_size <= 0:
        raise ValueError("--step-size must be positive")
    if not 0 < args.step_gamma <= 1:
        raise ValueError("--step-gamma must be in (0, 1]")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if not args.seeds:
        raise ValueError("--seeds must contain at least one seed")
    if any(seed < 0 for seed in args.seeds):
        raise ValueError("--seeds must contain non-negative integers")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must not contain duplicates")
    if not args.provenance_run_id or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for character in args.provenance_run_id
    ):
        raise ValueError(
            "--provenance-run-id must contain only letters, digits, '.', '_', or '-'"
        )

    data_dir = Path(args.data_dir).expanduser()
    if data_dir.exists() and not data_dir.is_dir():
        raise ValueError("--data-dir must name a directory")
    if args.output_dir is not None:
        output_dir = Path(args.output_dir).expanduser()
        if output_dir.exists() and not output_dir.is_dir():
            raise ValueError("--output-dir must name a directory")

    try:
        device = torch.device(args.device)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"Invalid --device value: {args.device!r}") from error
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index {device.index} is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")


def config_from_args(args: argparse.Namespace) -> RunConfig:
    method_slug = args.method.replace("-", "_")
    output_dir = args.output_dir or f"cnn_{method_slug}_{args.dataset}"
    optimizer_parameters = (
        f"momentum={SGD_MOMENTUM},dampening=0,nesterov=false"
        if args.optimizer == "sgd"
        else "betas=(0.9,0.999),eps=1e-8"
    )
    scheduler = "cosine-annealing" if args.scheduler == "cosine" else "step"
    scheduler_parameters = (
        f"T_max={args.epochs},eta_min=0"
        if args.scheduler == "cosine"
        else f"step_size={args.step_size},gamma={args.step_gamma}"
    )
    return RunConfig(
        method=args.method,
        dataset=args.dataset,
        epochs=args.epochs,
        patience=args.patience,
        minimum_delta=args.minimum_delta,
        validation_size=args.validation_size,
        optimizer=args.optimizer,
        optimizer_parameters=optimizer_parameters,
        learning_rate=args.learning_rate,
        scheduler=scheduler,
        scheduler_parameters=scheduler_parameters,
        step_size=args.step_size,
        step_gamma=args.step_gamma,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        seeds=list(args.seeds),
        num_workers=args.num_workers,
        data_dir=str(Path(args.data_dir).expanduser()),
        output_dir=str(Path(output_dir).expanduser()),
        device=str(torch.device(args.device)),
        download=args.download,
        save_checkpoints=args.save_checkpoints,
        provenance_run_id=args.provenance_run_id,
        overwrite=args.overwrite,
        defer_test=args.defer_test,
        heartbeat_file=str(Path(args.heartbeat_file).expanduser()) if args.heartbeat_file else "",
        terminal_classifier_bias=False,
        test_evaluation_policy=(
            "deferred-until-profile-selection"
            if args.defer_test
            else "once-after-best-checkpoint-restoration"
        ),
    )


def intended_output_paths(config: RunConfig) -> list[Path]:
    """Return every file the command can replace for this configuration."""
    output_dir = Path(config.output_dir)
    paths = [
        output_dir / "config.json",
        output_dir / "history.csv",
        output_dir / "per_seed_results.csv",
        output_dir / "aggregate_results.csv",
        output_dir / "run_metadata.json",
    ]
    if config.save_checkpoints:
        paths.extend(output_dir / f"seed_{seed}_best.pt" for seed in config.seeds)
    return paths


def preflight_outputs(config: RunConfig) -> None:
    """Refuse to replace a prior run unless the caller explicitly opts in."""
    existing = [path for path in intended_output_paths(config) if path.exists()]
    directories = [path for path in existing if path.is_dir()]
    if directories:
        rendered = ", ".join(str(path) for path in directories)
        raise ValueError(f"Output paths name existing directories: {rendered}")
    if existing and not config.overwrite:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Output files already exist: {rendered}. Use --overwrite to replace them."
        )
    if config.overwrite:
        for path in existing:
            path.unlink()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_transform(dataset: str) -> Compose:
    """Return normalization-only preprocessing, with no augmentation."""
    spec = DATASETS[dataset]
    return Compose([ToTensor(), Normalize(spec.mean, spec.std)])


def dataset_targets(dataset: Dataset) -> np.ndarray:
    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise TypeError(f"{type(dataset).__name__} does not expose class targets")
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()
    array = np.asarray(targets, dtype=np.int64)
    if array.ndim != 1 or len(array) != len(dataset):
        raise ValueError("Dataset targets must be a one-dimensional array")
    return array


def stratified_split_indices(
    targets: np.ndarray,
    validation_size: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Create an exact-size, seed-specific stratified split without sklearn."""
    num_examples = len(targets)
    if validation_size >= num_examples:
        raise ValueError(
            f"--validation-size ({validation_size}) must be smaller than the "
            f"training split ({num_examples})"
        )

    classes, counts = np.unique(targets, return_counts=True)
    if np.any(counts < 2):
        raise ValueError("Every class needs at least two examples for a train/validation split")
    if validation_size < len(classes):
        raise ValueError(
            f"--validation-size must be at least the number of classes ({len(classes)})"
        )

    ideal = validation_size * counts.astype(np.float64) / num_examples
    quotas = np.floor(ideal).astype(np.int64)
    quotas = np.minimum(quotas, counts - 1)
    remaining = validation_size - int(quotas.sum())
    order = sorted(
        range(len(classes)),
        key=lambda index: (-(ideal[index] - np.floor(ideal[index])), int(classes[index])),
    )
    while remaining > 0:
        allocated = False
        for index in order:
            if quotas[index] < counts[index] - 1:
                quotas[index] += 1
                remaining -= 1
                allocated = True
                if remaining == 0:
                    break
        if not allocated:
            raise ValueError("Validation size leaves too few examples for the training split")

    rng = np.random.default_rng(seed)
    validation_parts = []
    for class_value, quota in zip(classes, quotas):
        class_indices = np.flatnonzero(targets == class_value)
        rng.shuffle(class_indices)
        validation_parts.append(class_indices[: int(quota)])

    validation_indices = np.sort(np.concatenate(validation_parts)).astype(np.int64)
    validation_mask = np.zeros(num_examples, dtype=bool)
    validation_mask[validation_indices] = True
    train_indices = np.flatnonzero(~validation_mask).astype(np.int64)
    if len(validation_indices) != validation_size:
        raise RuntimeError("Internal error: stratified split has the wrong validation size")
    return train_indices.tolist(), validation_indices.tolist()


def split_sha256(validation_indices: list[int]) -> str:
    encoded = np.asarray(sorted(validation_indices), dtype="<i8").tobytes()
    return hashlib.sha256(encoded).hexdigest()


def build_loaders(
    config: RunConfig,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader, str]:
    spec = DATASETS[config.dataset]
    transform = build_transform(config.dataset)
    full_train_dataset = spec.dataset_class(
        config.data_dir,
        train=True,
        transform=transform,
        download=config.download,
    )
    test_dataset = spec.dataset_class(
        config.data_dir,
        train=False,
        transform=transform,
        download=config.download,
    )
    train_indices, validation_indices = stratified_split_indices(
        dataset_targets(full_train_dataset),
        config.validation_size,
        seed,
    )
    train_dataset = Subset(full_train_dataset, train_indices)
    validation_dataset = Subset(full_train_dataset, validation_indices)

    pin_memory = torch.device(config.device).type == "cuda"
    common = {
        "num_workers": config.num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": config.num_workers > 0,
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        **common,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        generator=torch.Generator().manual_seed(seed + 1),
        **common,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        generator=torch.Generator().manual_seed(seed + 2),
        **common,
    )
    return train_loader, validation_loader, test_loader, split_sha256(validation_indices)


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class ClassificationHead(nn.Module):
    def __init__(self, input_channels: int, num_classes: int, *, bias: bool = True):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(input_channels, num_classes, bias=bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.flatten(self.pool(features), 1))


def make_backbone(input_channels: int) -> nn.ModuleList:
    channels = (input_channels, *BACKBONE_WIDTHS)
    return nn.ModuleList(
        ConvBlock(channels[index], channels[index + 1])
        for index in range(len(BACKBONE_WIDTHS))
    )


class BPCNN(nn.Module):
    """Matched backbone trained from one terminal cross-entropy objective."""

    def __init__(
        self, input_channels: int, num_classes: int, *, classifier_bias: bool = False
    ):
        super().__init__()
        self.blocks = make_backbone(input_channels)
        self.head = ClassificationHead(
            BACKBONE_WIDTHS[-1], num_classes, bias=classifier_bias
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = inputs
        for block in self.blocks:
            features = block(features)
        return self.head(features)


class LocalBPCNN(nn.Module):
    """Matched backbone with one detached cross-entropy objective per block."""

    def __init__(self, input_channels: int, num_classes: int):
        super().__init__()
        self.blocks = make_backbone(input_channels)
        self.heads = nn.ModuleList(
            ClassificationHead(width, num_classes) for width in BACKBONE_WIDTHS
        )

    def forward_local(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        logits = []
        features = inputs
        for index, (block, head) in enumerate(zip(self.blocks, self.heads)):
            features = block(features.detach() if index else features)
            logits.append(head(features))
        return logits

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.stack(self.forward_local(inputs)).sum(dim=0)


class GlobalMultiHeadCECNN(nn.Module):
    """Matched per-block CE heads with gradients propagated through all blocks."""

    def __init__(self, input_channels: int, num_classes: int):
        super().__init__()
        self.blocks = make_backbone(input_channels)
        self.heads = nn.ModuleList(
            ClassificationHead(width, num_classes) for width in BACKBONE_WIDTHS
        )

    def forward_heads(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        logits = []
        features = inputs
        for block, head in zip(self.blocks, self.heads):
            features = block(features)
            logits.append(head(features))
        return logits

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.stack(self.forward_heads(inputs)).sum(dim=0)


def build_model(config: RunConfig) -> nn.Module:
    spec = DATASETS[config.dataset]
    if config.method == "bp":
        return BPCNN(
            spec.input_channels,
            spec.num_classes,
            classifier_bias=config.terminal_classifier_bias,
        )
    if config.method == "local-bp":
        return LocalBPCNN(spec.input_channels, spec.num_classes)
    if config.method == "ce-matched-ge":
        return GlobalMultiHeadCECNN(spec.input_channels, spec.num_classes)
    raise ValueError(f"Unsupported method: {config.method}")


def build_optimizer(
    name: str,
    parameters: Iterable[torch.nn.Parameter],
    learning_rate: float,
) -> torch.optim.Optimizer:
    if name == "sgd":
        return SGD(
            parameters,
            lr=learning_rate,
            momentum=SGD_MOMENTUM,
            dampening=0.0,
            weight_decay=WEIGHT_DECAY,
            nesterov=False,
        )
    if name == "adam":
        return Adam(
            parameters,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=WEIGHT_DECAY,
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def build_optimizers(
    model: nn.Module,
    config: RunConfig,
) -> list[torch.optim.Optimizer]:
    if isinstance(model, BPCNN):
        return [build_optimizer(config.optimizer, model.parameters(), config.learning_rate)]
    if isinstance(model, LocalBPCNN):
        return [
            build_optimizer(
                config.optimizer,
                list(block.parameters()) + list(head.parameters()),
                config.learning_rate,
            )
            for block, head in zip(model.blocks, model.heads)
        ]
    if isinstance(model, GlobalMultiHeadCECNN):
        return [build_optimizer(config.optimizer, model.parameters(), config.learning_rate)]
    raise TypeError(f"Unsupported model type: {type(model).__name__}")


def build_schedulers(
    optimizers: list[torch.optim.Optimizer],
    config: RunConfig,
) -> list[CosineAnnealingLR | StepLR]:
    if config.scheduler == "cosine-annealing":
        return [
            CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=0)
            for optimizer in optimizers
        ]
    if config.scheduler == "step":
        return [
            StepLR(optimizer, step_size=config.step_size, gamma=config.step_gamma)
            for optimizer in optimizers
        ]
    raise ValueError(f"Unsupported scheduler: {config.scheduler}")


def require_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    """Fail immediately instead of recording a numerically invalid run."""
    if not torch.isfinite(tensor).all().item():
        raise FloatingPointError(f"Non-finite {name} detected")


def require_finite_gradients(parameters: Iterable[torch.nn.Parameter], context: str) -> None:
    for parameter in parameters:
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
            raise FloatingPointError(f"Non-finite gradient detected in {context}")


def validation_improved(candidate: float, best: float, minimum_delta: float) -> bool:
    """Treat ties as non-improvements, as required by the fixed protocol."""
    return candidate > best + minimum_delta


def train_bp_epoch(
    model: BPCNN,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, object]:
    model.train()
    loss_sum = 0.0
    correct = 0
    total_samples = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        require_finite_tensor("BP logits", logits)
        loss = criterion(logits, labels)
        require_finite_tensor("BP loss", loss)
        loss.backward()
        require_finite_gradients(model.parameters(), "BP")
        optimizer.step()

        batch_size = labels.size(0)
        total_samples += batch_size
        loss_sum += loss.item() * batch_size
        correct += logits.detach().argmax(dim=1).eq(labels).sum().item()
    if total_samples == 0:
        raise ValueError("The training loader is empty")
    return {
        "loss": loss_sum / total_samples,
        "accuracy": correct / total_samples,
        "layer_losses": [loss_sum / total_samples],
    }


def train_local_bp_epoch(
    model: LocalBPCNN,
    loader: DataLoader,
    criterion: nn.Module,
    optimizers: list[torch.optim.Optimizer],
    device: torch.device,
) -> dict[str, object]:
    model.train()
    layer_loss_sums = [0.0] * len(model.blocks)
    correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = labels.size(0)
        total_samples += batch_size

        features = images
        detached_logits = []
        for index, (block, head, optimizer) in enumerate(
            zip(model.blocks, model.heads, optimizers)
        ):
            optimizer.zero_grad(set_to_none=True)
            block_output = block(features.detach() if index else features)
            logits = head(block_output)
            require_finite_tensor(f"Local BP layer {index + 1} logits", logits)
            loss = criterion(logits, labels)
            require_finite_tensor(f"Local BP layer {index + 1} loss", loss)
            loss.backward()
            require_finite_gradients(
                (*block.parameters(), *head.parameters()),
                f"Local BP layer {index + 1}",
            )
            optimizer.step()
            layer_loss_sums[index] += loss.item() * batch_size

            # Reuse the activation from this local update so BatchNorm observes
            # each minibatch exactly once in every block.
            features = block_output.detach()
            detached_logits.append(logits.detach())

        prediction_logits = torch.stack(detached_logits).sum(dim=0)
        correct += prediction_logits.argmax(dim=1).eq(labels).sum().item()

    if total_samples == 0:
        raise ValueError("The training loader is empty")
    layer_losses = [value / total_samples for value in layer_loss_sums]
    return {
        "loss": statistics.mean(layer_losses),
        "accuracy": correct / total_samples,
        "layer_losses": layer_losses,
    }


def train_global_multihead_ce_epoch(
    model: GlobalMultiHeadCECNN,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, object]:
    """Train matched per-block CE heads with one globally propagated update."""
    model.train()
    layer_loss_sums = [0.0] * len(model.blocks)
    correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = labels.size(0)
        total_samples += batch_size

        optimizer.zero_grad(set_to_none=True)
        logits_by_head = model.forward_heads(images)
        losses = []
        for index, logits in enumerate(logits_by_head):
            require_finite_tensor(f"Global multi-head CE layer {index + 1} logits", logits)
            loss = criterion(logits, labels)
            require_finite_tensor(f"Global multi-head CE layer {index + 1} loss", loss)
            losses.append(loss)
            layer_loss_sums[index] += loss.item() * batch_size

        # Summation preserves the full per-head gradient scale used by the
        # existing local implementation, where each head performs its own CE
        # update. The only intended change is the removal of stop-gradients.
        total_loss = torch.stack(losses).sum()
        total_loss.backward()
        require_finite_gradients(model.parameters(), "Global multi-head CE")
        optimizer.step()

        prediction_logits = torch.stack([logits.detach() for logits in logits_by_head]).sum(
            dim=0
        )
        correct += prediction_logits.argmax(dim=1).eq(labels).sum().item()

    if total_samples == 0:
        raise ValueError("The training loader is empty")
    layer_losses = [value / total_samples for value in layer_loss_sums]
    return {
        "loss": statistics.mean(layer_losses),
        "accuracy": correct / total_samples,
        "layer_losses": layer_losses,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizers: list[torch.optim.Optimizer],
    device: torch.device,
) -> dict[str, object]:
    if isinstance(model, BPCNN):
        return train_bp_epoch(model, loader, criterion, optimizers[0], device)
    if isinstance(model, LocalBPCNN):
        return train_local_bp_epoch(model, loader, criterion, optimizers, device)
    if isinstance(model, GlobalMultiHeadCECNN):
        return train_global_multihead_ce_epoch(
            model, loader, criterion, optimizers[0], device
        )
    raise TypeError(f"Unsupported model type: {type(model).__name__}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total_samples = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        require_finite_tensor("evaluation logits", logits)
        total_samples += labels.size(0)
        correct += logits.argmax(dim=1).eq(labels).sum().item()
    if total_samples == 0:
        raise ValueError("The evaluation loader is empty")
    return correct / total_samples


def state_dict_on_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    """Hash tensor names, types, shapes, and bytes in a serialization-independent form."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def source_sha256() -> dict[str, str]:
    project_root = Path(__file__).resolve().parent
    name = "LocalBPCNNBenchmark.py"
    return {name: file_sha256(project_root / name)}


def load_checkpoint(path: Path) -> dict[str, object]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older supported PyTorch
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid checkpoint payload in {path}")
    return payload


def measure_training_peak(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    torch.cuda.synchronize(device)
    return int(torch.cuda.max_memory_allocated(device))


def runtime_environment(device: torch.device) -> dict[str, object]:
    """Capture enough software and hardware detail to audit a remote run."""
    gpu: dict[str, object] | None = None
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        gpu = {
            "index": index,
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": f"{properties.major}.{properties.minor}",
        }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
        "gpu": gpu,
        "precision": "float32",
    }


def write_csv(path: Path, fieldnames: Iterable[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_seed(
    seed: int,
    split_hash: str,
    config: RunConfig,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    seed_started_at_utc = datetime.now(timezone.utc).isoformat()
    seed_start = time.perf_counter()
    set_seed(seed)
    device = torch.device(config.device)
    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizers = build_optimizers(model, config)
    schedulers = build_schedulers(optimizers, config)

    best_validation_accuracy = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    peak_training_memory_bytes: int | None = None
    history: list[dict[str, object]] = []

    for epoch in range(1, config.epochs + 1):
        learning_rate = optimizers[0].param_groups[0]["lr"]
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizers, device)
        train_seconds = time.perf_counter() - start
        epoch_peak = measure_training_peak(device)
        if epoch_peak is not None:
            peak_training_memory_bytes = max(peak_training_memory_bytes or 0, epoch_peak)

        validation_accuracy = evaluate(model, validation_loader, device)
        if not math.isfinite(validation_accuracy):
            raise FloatingPointError("Non-finite validation accuracy detected")
        improved = validation_improved(
            validation_accuracy,
            best_validation_accuracy,
            config.minimum_delta,
        )
        if improved:
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch
            best_state = state_dict_on_cpu(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        row: dict[str, object] = {
            "method": config.method,
            "dataset": config.dataset,
            "seed": seed,
            "split_sha256": split_hash,
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_accuracy": validation_accuracy,
            "best_validation_accuracy": best_validation_accuracy,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "train_seconds": train_seconds,
            "epoch_peak_gpu_training_memory_bytes": epoch_peak,
        }
        layer_losses = list(train_metrics["layer_losses"])
        for index in range(4):
            row[f"train_layer{index + 1}_loss"] = (
                layer_losses[index] if index < len(layer_losses) else ""
            )
        history.append(row)

        for scheduler in schedulers:
            scheduler.step()
        memory_text = "n/a"
        if epoch_peak is not None:
            memory_text = f"{epoch_peak / (1024**2):.1f} MiB"
        print(
            f"[{config.method}][{config.dataset}][seed {seed}] "
            f"epoch {epoch:03d}/{config.epochs} | lr {learning_rate:.6g} | "
            f"train {float(train_metrics['accuracy']) * 100:.2f}% | "
            f"validation {validation_accuracy * 100:.2f}% | "
            f"best validation {best_validation_accuracy * 100:.2f}% "
            f"(epoch {best_epoch}) | peak train memory {memory_text} | "
            f"{train_seconds:.1f}s"
        )
        if config.heartbeat_file:
            write_json(
                Path(config.heartbeat_file),
                {
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "method": config.method,
                    "dataset": config.dataset,
                    "seed": seed,
                    "epoch": epoch,
                    "maximum_epochs": config.epochs,
                    "best_epoch": best_epoch,
                    "best_validation_accuracy": best_validation_accuracy,
                    "epochs_without_improvement": epochs_without_improvement,
                },
            )
        if epochs_without_improvement >= config.patience:
            print(
                f"Early stopping seed {seed} after epoch {epoch}: validation accuracy "
                f"did not improve for {config.patience} epochs."
            )
            break

    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")
    early_stopping_triggered = epochs_without_improvement >= config.patience
    termination_reason = "patience" if early_stopping_triggered else "maximum_epochs"

    checkpoint_file = ""
    checkpoint_sha256 = ""
    checkpoint_bytes: int | str = ""
    best_state_sha256 = state_dict_sha256(best_state)
    restored_state_sha256 = ""
    if config.save_checkpoints:
        checkpoint_path = Path(config.output_dir) / f"seed_{seed}_best.pt"
        checkpoint_file = checkpoint_path.name
        checkpoint = {
            "schema_version": 1,
            "provenance_run_id": config.provenance_run_id,
            "method": config.method,
            "dataset": config.dataset,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation_accuracy": best_validation_accuracy,
            "split_sha256": split_hash,
            "model_state_sha256": best_state_sha256,
            "model_state_dict": best_state,
            "config": json.loads(json.dumps(asdict(config))),
        }
        temporary_checkpoint = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
        torch.save(checkpoint, temporary_checkpoint)
        temporary_checkpoint.replace(checkpoint_path)
        checkpoint_sha256 = file_sha256(checkpoint_path)
        checkpoint_bytes = checkpoint_path.stat().st_size

        persisted = load_checkpoint(checkpoint_path)
        persisted_state = persisted.get("model_state_dict")
        if not isinstance(persisted_state, dict):
            raise RuntimeError("Persisted checkpoint does not contain a model state")
        if persisted.get("model_state_sha256") != best_state_sha256:
            raise RuntimeError("Persisted checkpoint state-hash marker is inconsistent")
        if state_dict_sha256(persisted_state) != best_state_sha256:
            raise RuntimeError("Persisted checkpoint tensor state failed hash verification")
        model.load_state_dict(persisted_state)
    else:
        model.load_state_dict(best_state)
    restored_state_sha256 = state_dict_sha256(state_dict_on_cpu(model))
    best_checkpoint_restored = restored_state_sha256 == best_state_sha256
    if not best_checkpoint_restored:
        raise RuntimeError("Restored model state does not match the selected checkpoint")

    test_evaluations = 0
    test_accuracy: float | None = None
    if not config.defer_test:
        # This is the only call that evaluates the official test split.
        test_accuracy = evaluate(model, test_loader, device)
        test_evaluations += 1
        if test_evaluations != 1 or not math.isfinite(test_accuracy):
            raise RuntimeError("The official test split must produce one finite evaluation")
    peak_mib = (
        peak_training_memory_bytes / (1024**2)
        if peak_training_memory_bytes is not None
        else None
    )
    summary: dict[str, object] = {
        "method": config.method,
        "dataset": config.dataset,
        "seed": seed,
        "split_sha256": split_hash,
        "optimizer": config.optimizer,
        "optimizer_parameters": config.optimizer_parameters,
        "learning_rate": config.learning_rate,
        "scheduler": config.scheduler,
        "scheduler_parameters": config.scheduler_parameters,
        "batch_size": config.batch_size,
        "eval_batch_size": config.eval_batch_size,
        "train_samples": len(train_loader.dataset),
        "validation_samples": len(validation_loader.dataset),
        "test_samples": len(test_loader.dataset),
        "validation_size": config.validation_size,
        "validation_split": config.validation_split,
        "maximum_epochs": config.epochs,
        "patience": config.patience,
        "minimum_delta": config.minimum_delta,
        "epochs_trained": len(history),
        "validation_evaluations": len(history),
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_validation_accuracy,
        "early_stopping_triggered": early_stopping_triggered,
        "termination_reason": termination_reason,
        "final_epochs_without_improvement": epochs_without_improvement,
        "test_accuracy": test_accuracy,
        "test_evaluations": test_evaluations,
        "test_evaluation_policy": config.test_evaluation_policy,
        "best_checkpoint_restored": best_checkpoint_restored,
        "checkpoint_file": checkpoint_file,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_bytes": checkpoint_bytes,
        "model_state_sha256": best_state_sha256,
        "restored_state_sha256": restored_state_sha256,
        "provenance_run_id": config.provenance_run_id,
        "peak_gpu_training_memory_bytes": peak_training_memory_bytes,
        "peak_gpu_training_memory_mib": peak_mib,
        "mean_train_epoch_seconds": statistics.mean(
            float(row["train_seconds"]) for row in history
        ),
        "augmentation": config.augmentation,
        "dropout": config.dropout,
        "weight_decay": config.weight_decay,
        "precision": config.precision,
        "terminal_classifier_bias": config.terminal_classifier_bias,
        "backbone_widths": json.dumps(config.backbone_widths),
        "finite": True,
        "seed_started_at_utc": seed_started_at_utc,
        "seed_finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed_wall_seconds": time.perf_counter() - seed_start,
    }

    test_text = (
        f"test {test_accuracy * 100:.2f}% (one evaluation) | "
        if test_accuracy is not None
        else "test deferred until profile selection | "
    )
    print(
        f"[{config.method}][{config.dataset}][seed {seed}] selected epoch {best_epoch} | "
        f"validation {best_validation_accuracy * 100:.2f}% | "
        f"{test_text}"
        f"peak GPU training memory "
        f"{f'{peak_mib:.1f} MiB' if peak_mib is not None else 'n/a on CPU'}"
    )
    return history, summary


def mean_and_sample_std(
    rows: list[dict[str, object]],
    field: str,
) -> tuple[float | None, float | None]:
    values = [float(row[field]) for row in rows if row[field] not in (None, "")]
    if not values:
        return None, None
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate_results(seed_rows: list[dict[str, object]]) -> dict[str, object]:
    if not seed_rows:
        raise ValueError("At least one seed result is required")
    validation_mean, validation_std = mean_and_sample_std(
        seed_rows, "best_validation_accuracy"
    )
    test_mean, test_std = mean_and_sample_std(seed_rows, "test_accuracy")
    epoch_mean, epoch_std = mean_and_sample_std(seed_rows, "best_epoch")
    memory_mean, memory_std = mean_and_sample_std(
        seed_rows, "peak_gpu_training_memory_mib"
    )
    time_mean, time_std = mean_and_sample_std(seed_rows, "mean_train_epoch_seconds")
    first = seed_rows[0]
    return {
        "method": first["method"],
        "dataset": first["dataset"],
        "optimizer": first["optimizer"],
        "optimizer_parameters": first["optimizer_parameters"],
        "learning_rate": first["learning_rate"],
        "scheduler": first["scheduler"],
        "scheduler_parameters": first["scheduler_parameters"],
        "batch_size": first["batch_size"],
        "eval_batch_size": first["eval_batch_size"],
        "validation_size": first["validation_size"],
        "validation_split": first["validation_split"],
        "maximum_epochs": first["maximum_epochs"],
        "patience": first["patience"],
        "minimum_delta": first["minimum_delta"],
        "num_seeds": len(seed_rows),
        "best_validation_accuracy_mean": validation_mean,
        "best_validation_accuracy_sample_std": validation_std,
        "test_accuracy_mean": test_mean,
        "test_accuracy_sample_std": test_std,
        "best_epoch_mean": epoch_mean,
        "best_epoch_sample_std": epoch_std,
        "peak_gpu_training_memory_mib_mean": memory_mean,
        "peak_gpu_training_memory_mib_sample_std": memory_std,
        "train_epoch_seconds_mean": time_mean,
        "train_epoch_seconds_sample_std": time_std,
        "test_evaluations_per_seed": int(first["test_evaluations"]),
        "test_evaluation_policy": first["test_evaluation_policy"],
        "all_best_checkpoints_restored": all(
            bool(row["best_checkpoint_restored"]) for row in seed_rows
        ),
        "augmentation": first["augmentation"],
        "dropout": first["dropout"],
        "weight_decay": first["weight_decay"],
        "precision": first["precision"],
        "terminal_classifier_bias": first["terminal_classifier_bias"],
        "backbone_widths": first["backbone_widths"],
        "all_finite": all(bool(row["finite"]) for row in seed_rows),
    }


HISTORY_FIELDS = [
    "method",
    "dataset",
    "seed",
    "split_sha256",
    "epoch",
    "learning_rate",
    "train_loss",
    "train_accuracy",
    "validation_accuracy",
    "best_validation_accuracy",
    "best_epoch",
    "epochs_without_improvement",
    "train_seconds",
    "epoch_peak_gpu_training_memory_bytes",
    *(f"train_layer{index}_loss" for index in range(1, 5)),
]

SEED_RESULT_FIELDS = [
    "method",
    "dataset",
    "seed",
    "split_sha256",
    "optimizer",
    "optimizer_parameters",
    "learning_rate",
    "scheduler",
    "scheduler_parameters",
    "batch_size",
    "eval_batch_size",
    "train_samples",
    "validation_samples",
    "test_samples",
    "validation_size",
    "validation_split",
    "maximum_epochs",
    "patience",
    "minimum_delta",
    "epochs_trained",
    "validation_evaluations",
    "best_epoch",
    "best_validation_accuracy",
    "early_stopping_triggered",
    "termination_reason",
    "final_epochs_without_improvement",
    "test_accuracy",
    "test_evaluations",
    "test_evaluation_policy",
    "best_checkpoint_restored",
    "checkpoint_file",
    "checkpoint_sha256",
    "checkpoint_bytes",
    "model_state_sha256",
    "restored_state_sha256",
    "provenance_run_id",
    "peak_gpu_training_memory_bytes",
    "peak_gpu_training_memory_mib",
    "mean_train_epoch_seconds",
    "augmentation",
    "dropout",
    "weight_decay",
    "precision",
    "terminal_classifier_bias",
    "backbone_widths",
    "finite",
    "seed_started_at_utc",
    "seed_finished_at_utc",
    "seed_wall_seconds",
]

AGGREGATE_FIELDS = [
    "method",
    "dataset",
    "optimizer",
    "optimizer_parameters",
    "learning_rate",
    "scheduler",
    "scheduler_parameters",
    "batch_size",
    "eval_batch_size",
    "validation_size",
    "validation_split",
    "maximum_epochs",
    "patience",
    "minimum_delta",
    "num_seeds",
    "best_validation_accuracy_mean",
    "best_validation_accuracy_sample_std",
    "test_accuracy_mean",
    "test_accuracy_sample_std",
    "best_epoch_mean",
    "best_epoch_sample_std",
    "peak_gpu_training_memory_mib_mean",
    "peak_gpu_training_memory_mib_sample_std",
    "train_epoch_seconds_mean",
    "train_epoch_seconds_sample_std",
    "test_evaluations_per_seed",
    "test_evaluation_policy",
    "all_best_checkpoints_restored",
    "augmentation",
    "dropout",
    "weight_decay",
    "precision",
    "terminal_classifier_bias",
    "backbone_widths",
    "all_finite",
]


def main(argv: list[str] | None = None) -> None:
    run_started_at_utc = datetime.now(timezone.utc).isoformat()
    run_start = time.perf_counter()
    process_instance_id = uuid.uuid4().hex
    torch.set_default_dtype(torch.float32)
    args = parse_args(argv)
    validate_args(args)
    config = config_from_args(args)
    output_dir = Path(config.output_dir)
    preflight_outputs(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", asdict(config))
    environment = runtime_environment(torch.device(config.device))
    source_hashes = source_sha256()
    write_json(
        output_dir / "run_metadata.json",
        {
            "schema_version": 1,
            "status": "running",
            "started_at_utc": run_started_at_utc,
            "config": asdict(config),
            "environment": environment,
            "source_sha256": source_hashes,
            "provenance_run_id": config.provenance_run_id,
            "process_instance_id": process_instance_id,
        },
    )

    print("Run configuration:")
    print(json.dumps(asdict(config), indent=2))
    all_history: list[dict[str, object]] = []
    seed_results: list[dict[str, object]] = []
    for seed in config.seeds:
        set_seed(seed)
        train_loader, validation_loader, test_loader, split_hash = build_loaders(config, seed)
        history, seed_result = run_seed(
            seed,
            split_hash,
            config,
            train_loader,
            validation_loader,
            test_loader,
        )
        all_history.extend(history)
        seed_results.append(seed_result)
        write_csv(output_dir / "history.csv", HISTORY_FIELDS, all_history)
        write_csv(output_dir / "per_seed_results.csv", SEED_RESULT_FIELDS, seed_results)

    aggregate = aggregate_results(seed_results)
    write_csv(output_dir / "aggregate_results.csv", AGGREGATE_FIELDS, [aggregate])
    write_json(
        output_dir / "run_metadata.json",
        {
            "schema_version": 1,
            "status": "train_complete" if config.defer_test else "complete",
            "started_at_utc": run_started_at_utc,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.perf_counter() - run_start,
            "config": asdict(config),
            "environment": environment,
            "source_sha256": source_hashes,
            "provenance_run_id": config.provenance_run_id,
            "process_instance_id": process_instance_id,
            "split_sha256_by_seed": {
                str(row["seed"]): row["split_sha256"] for row in seed_results
            },
            "checkpoint_sha256_by_seed": {
                str(row["seed"]): row["checkpoint_sha256"] for row in seed_results
            },
            "model_state_sha256_by_seed": {
                str(row["seed"]): row["model_state_sha256"] for row in seed_results
            },
            "test_evaluations_total": sum(
                int(row["test_evaluations"]) for row in seed_results
            ),
            "nonfinite_values_detected": False,
            "best_checkpoints_restored": all(
                bool(row["best_checkpoint_restored"]) for row in seed_results
            ),
        },
    )
    print("Aggregate result (mean ± sample standard deviation):")
    print(
        "Validation accuracy: "
        f"{float(aggregate['best_validation_accuracy_mean']) * 100:.2f}% ± "
        f"{float(aggregate['best_validation_accuracy_sample_std']) * 100:.2f}%"
    )
    if aggregate["test_accuracy_mean"] is None:
        print("Test accuracy: deferred until optimizer-profile selection")
    else:
        print(
            "Test accuracy: "
            f"{float(aggregate['test_accuracy_mean']) * 100:.2f}% ± "
            f"{float(aggregate['test_accuracy_sample_std']) * 100:.2f}%"
        )
    memory_mean = aggregate["peak_gpu_training_memory_mib_mean"]
    memory_std = aggregate["peak_gpu_training_memory_mib_sample_std"]
    if memory_mean is not None and memory_std is not None:
        print(f"Peak GPU training memory: {float(memory_mean):.1f} ± {float(memory_std):.1f} MiB")
    else:
        print("Peak GPU training memory: not available for a non-CUDA run")
    print(
        "Official test evaluations: deferred"
        if config.defer_test
        else "Official test evaluations: exactly one per seed"
    )
    print(f"Results written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
