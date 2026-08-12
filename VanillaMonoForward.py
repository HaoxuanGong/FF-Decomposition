"""Run the paper's Mono-Forward (MF) fully connected accuracy experiment."""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, MNIST
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm


PAPER_SEEDS = (424, 425, 426)

DATASET_SPECS = {
    "mnist": (MNIST, 28 * 28, 10, (0.1307,), (0.3081,)),
    "fashionmnist": (FashionMNIST, 28 * 28, 10, (0.2860,), (0.3530,)),
    "cifar10": (
        CIFAR10,
        32 * 32 * 3,
        10,
        (0.4914, 0.4822, 0.4465),
        (0.2471, 0.2435, 0.2616),
    ),
    "cifar100": (
        CIFAR100,
        32 * 32 * 3,
        100,
        (0.5071, 0.4865, 0.4409),
        (0.2673, 0.2564, 0.2762),
    ),
}

PAPER_HIDDEN_DIMS = {
    "mnist": [1000, 1000],
    "fashionmnist": [1000, 1000],
    "cifar10": [2000, 2000, 2000],
    "cifar100": [2000, 2000, 2000],
}


class Flatten:
    """Pickle-safe flatten transform for multi-worker data loading."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.flatten()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["train", "test"],
        default="train",
        help="Train a paper run or evaluate one checkpoint.",
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_SPECS,
        default="cifar10",
        help="Dataset to use.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--test-batch-size", type=int, default=512, help="Evaluation batch size.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate for every local optimizer."
    )
    parser.add_argument(
        "--optimizer",
        choices=["adam", "sgd"],
        default="adam",
        help="Optimizer used by each layer.",
    )
    architecture = parser.add_mutually_exclusive_group()
    architecture.add_argument(
        "--hidden-dims",
        nargs="+",
        type=int,
        default=None,
        help="Hidden-layer widths. The dataset input width is inferred automatically.",
    )
    architecture.add_argument(
        "--architecture",
        nargs="+",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=20,
        help="Epoch interval for the StepLR scheduler.",
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum.")
    parser.add_argument(
        "--l2-lambda",
        type=float,
        default=0,
        help="L2 coefficient for the local classifier weights.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            f"Run one seed instead of the paper's matched seeds {', '.join(map(str, PAPER_SEEDS))}."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device.",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help=(
            "History CSV path. Multi-seed runs add a seed suffix; summary CSVs use the same stem."
        ),
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help=(
            "Optional checkpoint path. Training adds a seed suffix for a multi-seed run; "
            "test mode loads this exact path."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing training outputs. This option has no effect in test mode.",
    )
    parser.add_argument("--data-dir", default="data", help="Dataset cache directory.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    args = parser.parse_args(argv)

    input_dim = DATASET_SPECS[args.dataset][1]
    if args.architecture is not None:
        if len(args.architecture) < 2 or args.architecture[0] != input_dim:
            parser.error(
                f"--architecture must begin with the inferred input width {input_dim}; "
                "prefer --hidden-dims for new runs"
            )
        args.hidden_dims = args.architecture[1:]
    elif args.hidden_dims is None:
        args.hidden_dims = list(PAPER_HIDDEN_DIMS[args.dataset])
    del args.architecture
    return args


@dataclass
class Config:
    mode: str
    dataset: str
    batch_size: int
    test_batch_size: int
    epochs: int
    lr: float
    optimizer: str
    hidden_dims: list[int]
    step_size: int
    momentum: float
    l2_lambda: float
    seed: int | None
    device: str
    save_path: str | None
    checkpoint_path: str | None = None
    data_dir: str = "data"
    num_workers: int = 0
    overwrite: bool = False


class Layer(nn.Module):
    def __init__(self, in_features, out_features, cfg: Config, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.mask_weights = nn.Parameter(torch.randn(num_classes, out_features))
        self.activation = nn.ReLU()
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = torch.device(cfg.device)
        self.to(self.device)
        params = list(self.parameters())
        if cfg.optimizer == "adam":
            self.optimizer = Adam(params, lr=cfg.lr)
        else:
            self.optimizer = SGD(params, lr=cfg.lr, momentum=cfg.momentum)
        self.scheduler = StepLR(self.optimizer, step_size=cfg.step_size, gamma=0.1)
        nn.init.kaiming_uniform_(self.mask_weights, a=math.sqrt(5))

    def forward(self, x):
        return self.activation(self.linear(x))

    def train_layer(self, x, labels, cfg: Config):
        x = x.to(self.device)
        labels = labels.to(self.device)
        self.optimizer.zero_grad()
        out = self.forward(x)
        goodness = torch.matmul(out, self.mask_weights.t())
        loss = self.loss_fn(goodness, labels) + cfg.l2_lambda * torch.sum(self.mask_weights**2)
        loss.backward()
        self.optimizer.step()
        return out.detach(), loss.item()


class MonoForward(nn.Module):
    """The original layer-local MLP, with cumulative and local predictions."""

    def __init__(self, cfg: Config, num_classes):
        super().__init__()
        _, input_dim, _, _, _ = DATASET_SPECS[cfg.dataset]
        dims = [input_dim, *cfg.hidden_dims]
        self.layers = nn.ModuleList(
            [Layer(dims[i], dims[i + 1], cfg, num_classes) for i in range(len(dims) - 1)]
        )
        self.cfg = cfg
        self.test_accuracies = {f"Layer_{i + 1}": [] for i in range(len(self.layers))}
        self.test_accuracies["Cumulative"] = []
        self.history: list[dict[str, int | float]] = []

    def train_network(self, train_loader, test_loader) -> dict[str, int | float]:
        best_cumulative_accuracy = -1.0
        best_cumulative_epoch = 0
        best_final_accuracy = -1.0
        best_final_epoch = 0

        for epoch in range(1, self.cfg.epochs + 1):
            self.train()
            train_correct, train_total = 0, 0
            layer_losses = [[] for _ in self.layers]
            learning_rate = float(self.layers[0].optimizer.param_groups[0]["lr"])
            for data, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
                x = data.view(data.size(0), -1)
                for idx, layer in enumerate(self.layers):
                    x, loss = layer.train_layer(x, labels, self.cfg)
                    layer_losses[idx].append(loss)
                with torch.no_grad():
                    preds = self.predict(data.view(data.size(0), -1))
                train_correct += int((preds.cpu() == labels).sum())
                train_total += labels.size(0)

            if train_total == 0:
                raise ValueError("The training loader is empty")
            train_cumulative_accuracy = train_correct / train_total
            mean_layer_losses = [float(np.mean(losses)) for losses in layer_losses]
            loss_text = ", ".join(
                f"L{index} loss {loss:.4f}" for index, loss in enumerate(mean_layer_losses, start=1)
            )
            print(
                f"[{self.cfg.dataset}][seed {self.cfg.seed}] "
                f"epoch {epoch:03d}/{self.cfg.epochs} | lr {learning_rate:.6g} | "
                f"train cumulative {train_cumulative_accuracy * 100:.2f}% | {loss_text}"
            )

            layer_accuracies, cumulative_accuracy = self.test_network(test_loader)
            final_accuracy = layer_accuracies[-1]
            if cumulative_accuracy > best_cumulative_accuracy:
                best_cumulative_accuracy = cumulative_accuracy
                best_cumulative_epoch = epoch
            if final_accuracy > best_final_accuracy:
                best_final_accuracy = final_accuracy
                best_final_epoch = epoch

            row: dict[str, int | float] = {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_cumulative_accuracy": train_cumulative_accuracy,
                "test_cumulative_accuracy": cumulative_accuracy,
                "test_final_layer_accuracy": final_accuracy,
                "best_test_cumulative_accuracy": best_cumulative_accuracy,
                "best_test_cumulative_epoch": best_cumulative_epoch,
                "best_test_final_layer_accuracy": best_final_accuracy,
                "best_test_final_layer_epoch": best_final_epoch,
            }
            for index, loss in enumerate(mean_layer_losses, start=1):
                row[f"train_layer_{index}_loss"] = loss
            for index, accuracy in enumerate(layer_accuracies, start=1):
                row[f"test_layer_{index}_accuracy"] = accuracy
                self.test_accuracies[f"Layer_{index}"].append(accuracy)
            self.test_accuracies["Cumulative"].append(cumulative_accuracy)
            self.history.append(row)

            test_text = ", ".join(
                f"L{index} {accuracy * 100:.2f}%"
                for index, accuracy in enumerate(layer_accuracies, start=1)
            )
            print(
                f"[{self.cfg.dataset}][seed {self.cfg.seed}] test | {test_text} | "
                f"final layer {final_accuracy * 100:.2f}% | "
                f"cumulative {cumulative_accuracy * 100:.2f}% | "
                f"best final {best_final_accuracy * 100:.2f}% | "
                f"best cumulative {best_cumulative_accuracy * 100:.2f}%"
            )
            for layer in self.layers:
                layer.scheduler.step()

        self.save_all_accuracies_to_csv()
        return summarize_history(self.history)

    def test_network(self, loader):
        was_training = self.training
        self.eval()
        per_layer_correct = [0 for _ in self.layers]
        cumulative_correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in loader:
                x = data.view(data.size(0), -1).to(self.layers[0].device)
                cumulative_logits = None
                for idx, layer in enumerate(self.layers):
                    x = layer(x)
                    logits = torch.matmul(x, layer.mask_weights.t())
                    predictions = logits.argmax(1)
                    per_layer_correct[idx] += int((predictions.cpu() == labels).sum())
                    cumulative_logits = (
                        logits if cumulative_logits is None else cumulative_logits + logits
                    )
                cumulative_correct += int((cumulative_logits.argmax(1).cpu() == labels).sum())
                total += labels.size(0)
        if was_training:
            self.train()
        if total == 0:
            raise ValueError("The evaluation loader is empty")
        return [correct / total for correct in per_layer_correct], cumulative_correct / total

    def predict(self, x):
        x = x.to(self.layers[0].device)
        summed_logits = None
        for layer in self.layers:
            x = layer(x)
            logits = torch.matmul(x, layer.mask_weights.t())
            summed_logits = logits if summed_logits is None else summed_logits + logits
        return summed_logits.argmax(1)

    def save_all_accuracies_to_csv(self):
        if self.cfg.save_path is None:
            raise ValueError("A history CSV path is required before training")
        write_rows(Path(self.cfg.save_path), self.history)


def summarize_history(history: list[dict[str, int | float]]) -> dict[str, int | float]:
    """Summarize the two Table 2 metrics without coupling their best epochs."""
    if not history:
        raise ValueError("Training history is empty")
    best_cumulative = max(history, key=lambda row: float(row["test_cumulative_accuracy"]))
    best_final = max(history, key=lambda row: float(row["test_final_layer_accuracy"]))
    final = history[-1]
    summary: dict[str, int | float] = {
        "best_test_cumulative_accuracy": float(best_cumulative["test_cumulative_accuracy"]),
        "best_test_cumulative_epoch": int(best_cumulative["epoch"]),
        "best_test_final_layer_accuracy": float(best_final["test_final_layer_accuracy"]),
        "best_test_final_layer_epoch": int(best_final["epoch"]),
        "final_epoch_train_cumulative_accuracy": float(final["train_cumulative_accuracy"]),
        "final_epoch_test_cumulative_accuracy": float(final["test_cumulative_accuracy"]),
        "final_epoch_test_final_layer_accuracy": float(final["test_final_layer_accuracy"]),
    }
    for key, value in final.items():
        if key.startswith("train_layer_") or key.startswith("test_layer_"):
            summary[f"final_epoch_{key}"] = float(value)
    return summary


def aggregate_results(seed_rows: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate paper runs, using sample standard deviation across seeds."""
    if not seed_rows:
        raise ValueError("At least one seed result is required")

    first = seed_rows[0]
    aggregate: dict[str, object] = {
        "dataset": first["dataset"],
        "optimizer": first["optimizer"],
        "learning_rate": first["learning_rate"],
        "batch_size": first["batch_size"],
        "epochs": first["epochs"],
        "architecture": first["architecture"],
        "num_seeds": len(seed_rows),
    }
    metric_names = [
        key for key in first if key.startswith("best_test_") or key.startswith("final_epoch_")
    ]
    for name in metric_names:
        values = [float(row[name]) for row in seed_rows]
        aggregate[f"{name}_mean"] = statistics.mean(values)
        aggregate[f"{name}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return aggregate


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _add_seed_suffix(path: Path, seed: int) -> Path:
    extension = path.suffix or ".csv"
    stem = path.stem if path.suffix else path.name
    return path.with_name(f"{stem}_seed{seed}{extension}")


def history_path(
    requested_path: str | None,
    dataset: str,
    seed: int,
    multiple_seeds: bool,
) -> Path:
    if requested_path is None:
        return Path("results") / f"mf_mlp_{dataset}_seed{seed}.csv"
    path = Path(requested_path).expanduser()
    return _add_seed_suffix(path, seed) if multiple_seeds else path


def checkpoint_path(
    requested_path: str | None,
    seed: int,
    multiple_seeds: bool,
) -> Path | None:
    if requested_path is None:
        return None
    path = Path(requested_path).expanduser()
    if not multiple_seeds:
        return path
    extension = path.suffix or ".pt"
    stem = path.stem if path.suffix else path.name
    return path.with_name(f"{stem}_seed{seed}{extension}")


def summary_paths(requested_path: str | None, dataset: str) -> tuple[Path, Path]:
    if requested_path is None:
        base = Path("results") / f"mf_mlp_{dataset}"
    else:
        path = Path(requested_path).expanduser()
        base = path.with_suffix("") if path.suffix else path
    return (
        base.with_name(f"{base.name}_per_seed_results.csv"),
        base.with_name(f"{base.name}_aggregate_results.csv"),
    )


def training_seeds(seed_override: int | None) -> list[int]:
    return [seed_override] if seed_override is not None else list(PAPER_SEEDS)


def plan_training_outputs(
    args: argparse.Namespace,
    seeds: list[int],
) -> tuple[dict[int, Path], dict[int, Path | None], Path, Path]:
    """Resolve and validate every path that a training command may write."""
    multiple_seeds = len(seeds) > 1
    histories = {
        seed: history_path(args.save_path, args.dataset, seed, multiple_seeds) for seed in seeds
    }
    checkpoints = {
        seed: checkpoint_path(args.checkpoint_path, seed, multiple_seeds) for seed in seeds
    }
    per_seed_path, aggregate_path = summary_paths(args.save_path, args.dataset)

    intended_paths = [
        *histories.values(),
        *(path for path in checkpoints.values() if path is not None),
        per_seed_path,
        aggregate_path,
    ]
    canonical_paths: dict[str, Path] = {}
    for path in intended_paths:
        key = str(path.resolve()).casefold()
        if key in canonical_paths:
            raise ValueError(f"Training output paths collide: {canonical_paths[key]} and {path}")
        canonical_paths[key] = path

    return histories, checkpoints, per_seed_path, aggregate_path


def preflight_training_outputs(paths: list[Path], overwrite: bool) -> None:
    """Reject all output conflicts before the first dataset or model is created."""
    directories = [path for path in paths if path.is_dir()]
    if directories:
        names = ", ".join(str(path) for path in directories)
        raise ValueError(f"Training output path names an existing directory: {names}")

    conflicts = [path for path in paths if path.exists()]
    if conflicts and not overwrite:
        names = ", ".join(str(path) for path in conflicts)
        raise FileExistsError(
            f"Refusing to overwrite existing training output(s): {names}. "
            "Pass --overwrite to replace them."
        )
    if overwrite:
        for path in conflicts:
            path.unlink()


def validate_args(args: argparse.Namespace) -> None:
    if not args.hidden_dims or any(width <= 0 for width in args.hidden_dims):
        raise ValueError("--hidden-dims must contain positive integers")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    for name in ("epochs", "batch_size", "test_batch_size", "step_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")
    if args.seed is not None and args.seed < 0:
        raise ValueError("--seed cannot be negative")
    if args.l2_lambda < 0:
        raise ValueError("--l2-lambda cannot be negative")
    if args.optimizer == "sgd" and args.momentum < 0:
        raise ValueError("--momentum cannot be negative")

    data_dir = Path(args.data_dir).expanduser()
    if data_dir.exists() and not data_dir.is_dir():
        raise ValueError("--data-dir must name a directory")
    for name in ("save_path", "checkpoint_path"):
        value = getattr(args, name)
        if value is not None and Path(value).expanduser().is_dir():
            raise ValueError(f"--{name.replace('_', '-')} must name a file")

    try:
        device = torch.device(args.device)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"Invalid --device value: {args.device!r}") from error
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index {device.index} is not available")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise ValueError("MPS was requested but is not available")
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("--device must be cpu, cuda, cuda:N, or mps")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def get_loaders(cfg: Config):
    if cfg.seed is None:
        raise ValueError("A concrete seed is required to build data loaders")
    dataset_cls, _, _, mean, std = DATASET_SPECS[cfg.dataset]
    transform = Compose([ToTensor(), Normalize(mean, std), Flatten()])
    train = dataset_cls(cfg.data_dir, train=True, download=True, transform=transform)
    test = dataset_cls(cfg.data_dir, train=False, download=True, transform=transform)
    loader_kwargs = {
        "num_workers": cfg.num_workers,
        "pin_memory": torch.device(cfg.device).type == "cuda",
        "persistent_workers": cfg.num_workers > 0,
        "worker_init_fn": seed_worker,
    }
    train_generator = torch.Generator().manual_seed(cfg.seed)
    test_generator = torch.Generator().manual_seed(cfg.seed + 1)
    return (
        DataLoader(
            train,
            batch_size=cfg.batch_size,
            shuffle=True,
            generator=train_generator,
            **loader_kwargs,
        ),
        DataLoader(
            test,
            batch_size=cfg.test_batch_size,
            shuffle=False,
            generator=test_generator,
            **loader_kwargs,
        ),
    )


def config_for_seed(
    args: argparse.Namespace,
    seed: int,
    run_history_path: Path,
    run_checkpoint_path: Path | None,
) -> Config:
    values = vars(args).copy()
    values.update(
        seed=seed,
        save_path=str(run_history_path),
        checkpoint_path=str(run_checkpoint_path) if run_checkpoint_path else None,
    )
    return Config(**values)


def save_checkpoint(
    path: Path,
    model: MonoForward,
    cfg: Config,
    num_classes: int,
    summary: dict[str, int | float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "dataset": cfg.dataset,
                "architecture": [DATASET_SPECS[cfg.dataset][1], *cfg.hidden_dims],
                "seed": cfg.seed,
                "optimizer": cfg.optimizer,
                "lr": cfg.lr,
                "momentum": cfg.momentum,
                "l2_lambda": cfg.l2_lambda,
            },
            "num_classes": num_classes,
            "metrics": summary,
        },
        path,
    )


def run_seed(
    args: argparse.Namespace,
    seed: int,
    run_history_path: Path,
    run_checkpoint_path: Path | None,
) -> dict[str, object]:
    cfg = config_for_seed(args, seed, run_history_path, run_checkpoint_path)
    set_seed(seed)
    train_loader, test_loader = get_loaders(cfg)
    num_classes = DATASET_SPECS[cfg.dataset][2]
    model = MonoForward(cfg, num_classes)
    metrics = model.train_network(train_loader, test_loader)
    if run_checkpoint_path is not None:
        save_checkpoint(run_checkpoint_path, model, cfg, num_classes, metrics)

    return {
        "dataset": cfg.dataset,
        "seed": seed,
        "optimizer": cfg.optimizer,
        "learning_rate": cfg.lr,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs,
        "architecture": "-".join(map(str, [DATASET_SPECS[cfg.dataset][1], *cfg.hidden_dims])),
        **metrics,
    }


def print_main_table_rows(aggregate: dict[str, object]) -> None:
    cumulative_mean = float(aggregate["best_test_cumulative_accuracy_mean"])
    cumulative_std = float(aggregate["best_test_cumulative_accuracy_std"])
    final_mean = float(aggregate["best_test_final_layer_accuracy_mean"])
    final_std = float(aggregate["best_test_final_layer_accuracy_std"])
    print(f"MF cumulative: {cumulative_mean * 100:.2f}% ± {cumulative_std * 100:.2f}%")
    print(f"MF (Final layer): {final_mean * 100:.2f}% ± {final_std * 100:.2f}%")


def evaluate_checkpoint(args: argparse.Namespace) -> None:
    if not args.checkpoint_path:
        raise ValueError("--checkpoint-path is required in test mode")
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device, weights_only=True)
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        metadata = checkpoint["config"]
        checkpoint_dataset = metadata.get("dataset")
        architecture = metadata.get("architecture")
        if checkpoint_dataset in DATASET_SPECS:
            args.dataset = checkpoint_dataset
        if architecture:
            expected_input = DATASET_SPECS[args.dataset][1]
            if int(architecture[0]) != expected_input:
                raise ValueError(
                    "Checkpoint architecture is incompatible with its dataset metadata"
                )
            args.hidden_dims = [int(width) for width in architecture[1:]]
        checkpoint_seed = metadata.get("seed")
        if args.seed is None and checkpoint_seed is not None:
            args.seed = int(checkpoint_seed)

    seed = args.seed if args.seed is not None else PAPER_SEEDS[0]
    run_history_path = history_path(args.save_path, args.dataset, seed, False)
    cfg = config_for_seed(args, seed, run_history_path, Path(args.checkpoint_path))
    set_seed(seed)
    _, test_loader = get_loaders(cfg)
    num_classes = DATASET_SPECS[cfg.dataset][2]
    model = MonoForward(cfg, num_classes)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    layer_accuracies, cumulative_accuracy = model.test_network(test_loader)
    final_accuracy = layer_accuracies[-1]
    print(
        "Test accuracies -> "
        + ", ".join(
            f"L{index}: {accuracy * 100:.2f}%"
            for index, accuracy in enumerate(layer_accuracies, start=1)
        )
        + f", Final layer: {final_accuracy * 100:.2f}%, "
        + f"Cumulative: {cumulative_accuracy * 100:.2f}%"
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    if args.mode == "test":
        evaluate_checkpoint(args)
        return

    seeds = training_seeds(args.seed)
    histories, checkpoints, per_seed_path, aggregate_path = plan_training_outputs(args, seeds)
    preflight_training_outputs(
        [
            *histories.values(),
            *(path for path in checkpoints.values() if path is not None),
            per_seed_path,
            aggregate_path,
        ],
        args.overwrite,
    )
    seed_results: list[dict[str, object]] = []
    for seed in seeds:
        run_history_path = histories[seed]
        run_checkpoint_path = checkpoints[seed]
        seed_result = run_seed(args, seed, run_history_path, run_checkpoint_path)
        seed_results.append(seed_result)
        write_rows(per_seed_path, seed_results)

    aggregate = aggregate_results(seed_results)
    write_rows(aggregate_path, [aggregate])
    print_main_table_rows(aggregate)
    print(f"Per-seed results written to {per_seed_path.resolve()}")
    print(f"Aggregate results written to {aggregate_path.resolve()}")


if __name__ == "__main__":
    main()
