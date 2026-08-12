#!/usr/bin/env python3
"""Run the paper's Mono-Forward CNN accuracy experiment.

The model has four strictly local Conv-BN-ReLU-MaxPool blocks.  Each block
owns a classifier and optimizer, and its output is detached before it is
passed to the next block.  Accuracy is reported for both the cumulative
(sum of local logits) prediction and the final local classifier.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
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


@dataclass
class RunConfig:
    dataset: str
    epochs: int
    optimizer: str
    learning_rate: float
    batch_size: int
    eval_batch_size: int
    seeds: list[int]
    selection_metric: str
    num_workers: int
    data_dir: str
    output_dir: str
    device: str
    download: bool
    save_checkpoints: bool
    overwrite: bool


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="sgd")
    parser.add_argument("--lr", "--learning-rate", dest="learning_rate", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[41, 42, 43],
        help="Final-run seeds (the paper uses 41, 42, and 43).",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("cumulative", "final"),
        default="cumulative",
        help="Metric used only to select an optional saved checkpoint.",
    )
    parser.add_argument("--num-workers", "--workers", dest="num_workers", type=int, default=4)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Result directory (default: mf_cnn_<dataset>).",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing files for this run instead of refusing to start.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
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
    output_dir = args.output_dir or f"mf_cnn_{args.dataset}"
    return RunConfig(
        dataset=args.dataset,
        epochs=args.epochs,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        seeds=list(args.seeds),
        selection_metric=args.selection_metric,
        num_workers=args.num_workers,
        data_dir=str(Path(args.data_dir).expanduser()),
        output_dir=str(Path(output_dir).expanduser()),
        device=str(torch.device(args.device)),
        download=args.download,
        save_checkpoints=args.save_checkpoints,
        overwrite=args.overwrite,
    )


def intended_output_paths(config: RunConfig) -> list[Path]:
    """Return every file the command can replace for this configuration."""
    output_dir = Path(config.output_dir)
    paths = [
        output_dir / "config.json",
        output_dir / "history.csv",
        output_dir / "per_seed_results.csv",
        output_dir / "aggregate_results.csv",
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


def build_transforms(dataset: str) -> tuple[Compose, Compose]:
    """Return the paper transforms: normalization only, with no augmentation."""
    spec = DATASETS[dataset]
    transform = Compose([ToTensor(), Normalize(spec.mean, spec.std)])
    return transform, transform


def build_loaders(config: RunConfig, seed: int) -> tuple[DataLoader, DataLoader]:
    spec = DATASETS[config.dataset]
    train_transform, test_transform = build_transforms(config.dataset)
    train_dataset = spec.dataset_class(
        config.data_dir,
        train=True,
        transform=train_transform,
        download=config.download,
    )
    test_dataset = spec.dataset_class(
        config.data_dir,
        train=False,
        transform=test_transform,
        download=config.download,
    )

    train_generator = torch.Generator().manual_seed(seed)
    test_generator = torch.Generator().manual_seed(seed + 1)
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
        generator=train_generator,
        **common,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        generator=test_generator,
        **common,
    )
    return train_loader, test_loader


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


class LocalHead(nn.Module):
    def __init__(self, input_channels: int, num_classes: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(input_channels, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = torch.flatten(self.pool(features), 1)
        return self.classifier(pooled)


class MFCNN(nn.Module):
    widths = (64, 128, 256, 512)

    def __init__(self, input_channels: int, num_classes: int):
        super().__init__()
        channels = (input_channels, *self.widths)
        self.blocks = nn.ModuleList(
            ConvBlock(channels[index], channels[index + 1]) for index in range(len(self.widths))
        )
        self.heads = nn.ModuleList(LocalHead(width, num_classes) for width in self.widths)

    @torch.no_grad()
    def predict_logits(
        self, inputs: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        local_logits = []
        cumulative_logits = []
        features = inputs
        running_logits = None
        for block, head in zip(self.blocks, self.heads):
            features = block(features)
            logits = head(features)
            local_logits.append(logits)
            running_logits = logits if running_logits is None else running_logits + logits
            cumulative_logits.append(running_logits)
        return local_logits, cumulative_logits, local_logits[-1]


def build_optimizer(
    name: str,
    parameters: Iterable[torch.nn.Parameter],
    learning_rate: float,
) -> torch.optim.Optimizer:
    if name == "sgd":
        return SGD(parameters, lr=learning_rate)
    if name == "adam":
        return Adam(parameters, lr=learning_rate)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_local_optimizers(model: MFCNN, config: RunConfig) -> list[torch.optim.Optimizer]:
    return [
        build_optimizer(
            config.optimizer,
            list(block.parameters()) + list(head.parameters()),
            config.learning_rate,
        )
        for block, head in zip(model.blocks, model.heads)
    ]


def train_one_epoch(
    model: MFCNN,
    loader: DataLoader,
    criterion: nn.Module,
    optimizers: list[torch.optim.Optimizer],
    device: torch.device,
) -> dict[str, object]:
    model.train()
    layer_loss_sums = [0.0] * len(model.blocks)
    cumulative_correct = 0
    final_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = labels.size(0)
        total_samples += batch_size

        features = images
        local_logits = []
        for index, (block, head, optimizer) in enumerate(
            zip(model.blocks, model.heads, optimizers)
        ):
            optimizer.zero_grad(set_to_none=True)
            block_output = block(features.detach() if index else features)
            logits = head(block_output)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            layer_loss_sums[index] += loss.item() * batch_size

            # Stream the activation already used by this local objective to
            # the next block. Re-running the block after the optimizer step
            # would update BatchNorm twice for the same minibatch.
            features = block_output.detach()
            local_logits.append(logits.detach())

        cumulative_logits = torch.stack(local_logits).sum(dim=0)
        cumulative_correct += cumulative_logits.argmax(dim=1).eq(labels).sum().item()
        final_correct += local_logits[-1].argmax(dim=1).eq(labels).sum().item()

    if total_samples == 0:
        raise ValueError("The training loader is empty")
    return {
        "layer_losses": [value / total_samples for value in layer_loss_sums],
        "cumulative_accuracy": cumulative_correct / total_samples,
        "final_accuracy": final_correct / total_samples,
    }


@torch.no_grad()
def evaluate(model: MFCNN, loader: DataLoader, device: torch.device) -> dict[str, object]:
    model.eval()
    local_correct = [0] * len(model.blocks)
    cumulative_correct = [0] * len(model.blocks)
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        local_logits, cumulative_logits, _ = model.predict_logits(images)
        total_samples += labels.size(0)
        for index, logits in enumerate(local_logits):
            local_correct[index] += logits.argmax(dim=1).eq(labels).sum().item()
        for index, logits in enumerate(cumulative_logits):
            cumulative_correct[index] += logits.argmax(dim=1).eq(labels).sum().item()

    if total_samples == 0:
        raise ValueError("The evaluation loader is empty")
    local_accuracies = [value / total_samples for value in local_correct]
    cumulative_accuracies = [value / total_samples for value in cumulative_correct]
    return {
        "local_accuracies": local_accuracies,
        "cumulative_accuracies": cumulative_accuracies,
        "cumulative_accuracy": cumulative_accuracies[-1],
        "final_accuracy": local_accuracies[-1],
    }


def write_csv(path: Path, fieldnames: Iterable[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_seed(
    seed: int,
    config: RunConfig,
    train_loader: DataLoader,
    test_loader: DataLoader,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    set_seed(seed)
    device = torch.device(config.device)
    spec = DATASETS[config.dataset]
    model = MFCNN(spec.input_channels, spec.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizers = build_local_optimizers(model, config)
    schedulers = [CosineAnnealingLR(optimizer, T_max=config.epochs) for optimizer in optimizers]

    history = []
    best_cumulative_accuracy = -1.0
    best_cumulative_epoch = 0
    best_final_accuracy = -1.0
    best_final_epoch = 0
    checkpoint_accuracy = -1.0
    checkpoint_epoch = 0
    best_state = None
    for epoch_index in range(config.epochs):
        epoch = epoch_index + 1
        learning_rate = optimizers[0].param_groups[0]["lr"]
        start = time.perf_counter()
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizers, device)
        train_seconds = time.perf_counter() - start
        test_metrics = evaluate(model, test_loader, device)
        cumulative_accuracy = float(test_metrics["cumulative_accuracy"])
        final_accuracy = float(test_metrics["final_accuracy"])
        if cumulative_accuracy > best_cumulative_accuracy:
            best_cumulative_accuracy = cumulative_accuracy
            best_cumulative_epoch = epoch
        if final_accuracy > best_final_accuracy:
            best_final_accuracy = final_accuracy
            best_final_epoch = epoch

        if config.save_checkpoints:
            selected_accuracy = float(test_metrics[f"{config.selection_metric}_accuracy"])
            if selected_accuracy > checkpoint_accuracy:
                checkpoint_accuracy = selected_accuracy
                checkpoint_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

        row: dict[str, object] = {
            "seed": seed,
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_cumulative_accuracy": train_metrics["cumulative_accuracy"],
            "train_final_accuracy": train_metrics["final_accuracy"],
            "test_cumulative_accuracy": test_metrics["cumulative_accuracy"],
            "test_final_accuracy": test_metrics["final_accuracy"],
            "best_test_cumulative_accuracy": best_cumulative_accuracy,
            "best_test_cumulative_epoch": best_cumulative_epoch,
            "best_test_final_accuracy": best_final_accuracy,
            "best_test_final_epoch": best_final_epoch,
            "train_seconds": train_seconds,
        }
        for index, loss in enumerate(train_metrics["layer_losses"], start=1):
            row[f"train_layer{index}_loss"] = loss
        for index, accuracy in enumerate(test_metrics["local_accuracies"], start=1):
            row[f"test_local_layer{index}_accuracy"] = accuracy
        for index, accuracy in enumerate(test_metrics["cumulative_accuracies"], start=1):
            row[f"test_cumulative_layer{index}_accuracy"] = accuracy
        history.append(row)

        for scheduler in schedulers:
            scheduler.step()
        print(
            f"[{config.dataset}][seed {seed}] epoch {epoch:03d}/{config.epochs} | "
            f"lr {learning_rate:.6g} | "
            f"train cumulative {float(train_metrics['cumulative_accuracy']) * 100:.2f}% | "
            f"test cumulative {float(test_metrics['cumulative_accuracy']) * 100:.2f}% | "
            f"test final {float(test_metrics['final_accuracy']) * 100:.2f}% | "
            f"best cumulative {best_cumulative_accuracy * 100:.2f}% | "
            f"best final {best_final_accuracy * 100:.2f}% | {train_seconds:.1f}s"
        )

    final_row = history[-1]
    summary: dict[str, object] = {
        "dataset": config.dataset,
        "seed": seed,
        "optimizer": config.optimizer,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "best_test_cumulative_accuracy": best_cumulative_accuracy,
        "best_test_cumulative_epoch": best_cumulative_epoch,
        "best_test_final_accuracy": best_final_accuracy,
        "best_test_final_epoch": best_final_epoch,
        "final_epoch_cumulative_accuracy": final_row["test_cumulative_accuracy"],
        "final_epoch_final_accuracy": final_row["test_final_accuracy"],
        "mean_train_epoch_seconds": statistics.mean(float(row["train_seconds"]) for row in history),
    }
    if best_state is not None:
        checkpoint = {
            "seed": seed,
            "epoch": checkpoint_epoch,
            "selection_metric": config.selection_metric,
            "best_accuracy": checkpoint_accuracy,
            "model_state_dict": best_state,
            "config": asdict(config),
        }
        torch.save(checkpoint, Path(config.output_dir) / f"seed_{seed}_best.pt")
    return history, summary


def aggregate_results(seed_rows: list[dict[str, object]]) -> dict[str, object]:
    if not seed_rows:
        raise ValueError("At least one seed result is required")

    def mean_and_std(field: str) -> tuple[float, float]:
        values = [float(row[field]) for row in seed_rows]
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        return mean, std

    best_cumulative_mean, best_cumulative_std = mean_and_std("best_test_cumulative_accuracy")
    best_cumulative_epoch_mean, best_cumulative_epoch_std = mean_and_std(
        "best_test_cumulative_epoch"
    )
    best_final_mean, best_final_std = mean_and_std("best_test_final_accuracy")
    best_final_epoch_mean, best_final_epoch_std = mean_and_std("best_test_final_epoch")
    final_epoch_cumulative_mean, final_epoch_cumulative_std = mean_and_std(
        "final_epoch_cumulative_accuracy"
    )
    final_epoch_final_mean, final_epoch_final_std = mean_and_std("final_epoch_final_accuracy")
    time_mean, time_std = mean_and_std("mean_train_epoch_seconds")
    first = seed_rows[0]
    return {
        "dataset": first["dataset"],
        "optimizer": first["optimizer"],
        "learning_rate": first["learning_rate"],
        "batch_size": first["batch_size"],
        "num_seeds": len(seed_rows),
        "best_test_cumulative_accuracy_mean": best_cumulative_mean,
        "best_test_cumulative_accuracy_std": best_cumulative_std,
        "best_test_cumulative_epoch_mean": best_cumulative_epoch_mean,
        "best_test_cumulative_epoch_std": best_cumulative_epoch_std,
        "best_test_final_accuracy_mean": best_final_mean,
        "best_test_final_accuracy_std": best_final_std,
        "best_test_final_epoch_mean": best_final_epoch_mean,
        "best_test_final_epoch_std": best_final_epoch_std,
        "final_epoch_cumulative_accuracy_mean": final_epoch_cumulative_mean,
        "final_epoch_cumulative_accuracy_std": final_epoch_cumulative_std,
        "final_epoch_final_accuracy_mean": final_epoch_final_mean,
        "final_epoch_final_accuracy_std": final_epoch_final_std,
        "train_epoch_seconds_mean": time_mean,
        "train_epoch_seconds_std": time_std,
    }


HISTORY_FIELDS = [
    "seed",
    "epoch",
    "learning_rate",
    "train_cumulative_accuracy",
    "train_final_accuracy",
    "test_cumulative_accuracy",
    "test_final_accuracy",
    "best_test_cumulative_accuracy",
    "best_test_cumulative_epoch",
    "best_test_final_accuracy",
    "best_test_final_epoch",
    "train_seconds",
    *(f"train_layer{index}_loss" for index in range(1, 5)),
    *(f"test_local_layer{index}_accuracy" for index in range(1, 5)),
    *(f"test_cumulative_layer{index}_accuracy" for index in range(1, 5)),
]

SEED_RESULT_FIELDS = [
    "dataset",
    "seed",
    "optimizer",
    "learning_rate",
    "batch_size",
    "best_test_cumulative_accuracy",
    "best_test_cumulative_epoch",
    "best_test_final_accuracy",
    "best_test_final_epoch",
    "final_epoch_cumulative_accuracy",
    "final_epoch_final_accuracy",
    "mean_train_epoch_seconds",
]

AGGREGATE_FIELDS = [
    "dataset",
    "optimizer",
    "learning_rate",
    "batch_size",
    "num_seeds",
    "best_test_cumulative_accuracy_mean",
    "best_test_cumulative_accuracy_std",
    "best_test_cumulative_epoch_mean",
    "best_test_cumulative_epoch_std",
    "best_test_final_accuracy_mean",
    "best_test_final_accuracy_std",
    "best_test_final_epoch_mean",
    "best_test_final_epoch_std",
    "final_epoch_cumulative_accuracy_mean",
    "final_epoch_cumulative_accuracy_std",
    "final_epoch_final_accuracy_mean",
    "final_epoch_final_accuracy_std",
    "train_epoch_seconds_mean",
    "train_epoch_seconds_std",
]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    config = config_from_args(args)
    output_dir = Path(config.output_dir)
    preflight_outputs(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)

    print(json.dumps(asdict(config), indent=2))
    all_history: list[dict[str, object]] = []
    seed_results: list[dict[str, object]] = []
    for seed in config.seeds:
        set_seed(seed)
        train_loader, test_loader = build_loaders(config, seed)
        history, seed_result = run_seed(seed, config, train_loader, test_loader)
        all_history.extend(history)
        seed_results.append(seed_result)
        write_csv(output_dir / "history.csv", HISTORY_FIELDS, all_history)
        write_csv(output_dir / "per_seed_results.csv", SEED_RESULT_FIELDS, seed_results)

    aggregate = aggregate_results(seed_results)
    write_csv(output_dir / "aggregate_results.csv", AGGREGATE_FIELDS, [aggregate])
    print(
        "Best cumulative accuracy: "
        f"{float(aggregate['best_test_cumulative_accuracy_mean']) * 100:.2f}% ± "
        f"{float(aggregate['best_test_cumulative_accuracy_std']) * 100:.2f}% "
        f"(epoch {float(aggregate['best_test_cumulative_epoch_mean']):.1f} ± "
        f"{float(aggregate['best_test_cumulative_epoch_std']):.1f})"
    )
    print(
        "Best final accuracy: "
        f"{float(aggregate['best_test_final_accuracy_mean']) * 100:.2f}% ± "
        f"{float(aggregate['best_test_final_accuracy_std']) * 100:.2f}% "
        f"(epoch {float(aggregate['best_test_final_epoch_mean']):.1f} ± "
        f"{float(aggregate['best_test_final_epoch_std']):.1f})"
    )
    print(f"Results written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
