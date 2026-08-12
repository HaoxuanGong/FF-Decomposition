#!/usr/bin/env python3
"""Run the fixed main-text Forward-Forward decomposition experiment.

The five methods in this file differ only in the two factors studied in the
main-text decomposition: error propagation/comparison scope and inter-layer
normalization.  All methods use the same data, two 1,000-unit ReLU hidden
layers, optimizer, learning rate, batch size, epoch budget, and random seeds.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import statistics
import time
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, FashionMNIST, MNIST
from torchvision.transforms import Compose, Lambda, Normalize, ToTensor


DATASET_NAMES = ("mnist", "fashionmnist", "cifar10")
METHOD_NAMES = ("ff", "ff-ge", "fc-ff-ge", "fc-nn-ff-ge", "bp")
DEFAULT_SEEDS = (424, 425, 426)
HIDDEN_DIMS = (1000, 1000)
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
DEFAULT_EPOCHS = 200
GOODNESS_THRESHOLD = 2.0
NORMALIZATION_EPSILON = 1e-4


@dataclass(frozen=True)
class DatasetSpec:
    dataset_class: type[Dataset]
    shape: tuple[int, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]

    @property
    def input_dim(self) -> int:
        return int(np.prod(self.shape))


DATASETS = {
    "mnist": DatasetSpec(MNIST, (1, 28, 28), (0.1307,), (0.3081,)),
    "fashionmnist": DatasetSpec(
        FashionMNIST,
        (1, 28, 28),
        (0.2860,),
        (0.3530,),
    ),
    "cifar10": DatasetSpec(
        CIFAR10,
        (3, 32, 32),
        (0.4914, 0.4822, 0.4465),
        (0.2471, 0.2435, 0.2616),
    ),
}


@dataclass(frozen=True)
class DatasetBundle:
    train: Dataset
    test: Dataset
    input_dim: int
    num_classes: int


@dataclass(frozen=True)
class Loaders:
    train: DataLoader
    train_evaluation: DataLoader
    test: DataLoader


def _flatten_image(image: torch.Tensor) -> torch.Tensor:
    return torch.flatten(image)


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


def load_dataset(
    name: str,
    data_dir: Path,
    *,
    download: bool,
) -> DatasetBundle:
    """Load one normalized, flattened main-text dataset."""
    spec = DATASETS[name]
    transform = Compose(
        [
            ToTensor(),
            Normalize(spec.mean, spec.std),
            Lambda(_flatten_image),
        ]
    )
    train = spec.dataset_class(
        root=str(data_dir),
        train=True,
        transform=transform,
        download=download,
    )
    test = spec.dataset_class(
        root=str(data_dir),
        train=False,
        transform=transform,
        download=download,
    )
    classes = getattr(train, "classes", None)
    if classes is None:
        raise RuntimeError(f"{name} does not expose its class labels")
    return DatasetBundle(
        train=train,
        test=test,
        input_dim=spec.input_dim,
        num_classes=len(classes),
    )


def make_loaders(
    bundle: DatasetBundle,
    *,
    seed: int,
    evaluation_batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Loaders:
    generator = torch.Generator().manual_seed(seed)
    common = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "persistent_workers": num_workers > 0,
    }
    train = DataLoader(
        bundle.train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        **common,
    )
    train_evaluation = DataLoader(
        bundle.train,
        batch_size=evaluation_batch_size,
        shuffle=False,
        **common,
    )
    test = DataLoader(
        bundle.test,
        batch_size=evaluation_batch_size,
        shuffle=False,
        **common,
    )
    return Loaders(train=train, train_evaluation=train_evaluation, test=test)


class GoodnessMLP(nn.Module):
    """Two-layer ReLU MLP whose layer activations define goodness scores."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dims: Sequence[int] = HIDDEN_DIMS,
        normalize: bool,
    ) -> None:
        super().__init__()
        dims = (input_dim, *hidden_dims)
        self.layers = nn.ModuleList(
            nn.Linear(dims[index], dims[index + 1]) for index in range(len(dims) - 1)
        )
        self.normalize = normalize

    def forward_layer(self, inputs: torch.Tensor, layer_index: int) -> torch.Tensor:
        if self.normalize:
            inputs = inputs / (inputs.norm(p=2, dim=1, keepdim=True) + NORMALIZATION_EPSILON)
        return F.relu(self.layers[layer_index](inputs))

    def layer_goodness(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        goodness = []
        for layer_index in range(len(self.layers)):
            inputs = self.forward_layer(inputs, layer_index)
            goodness.append(inputs.square().mean(dim=1))
        return goodness

    def goodness(self, inputs: torch.Tensor, aggregation: str) -> torch.Tensor:
        layer_scores = self.layer_goodness(inputs)
        if aggregation == "final":
            return layer_scores[-1]
        if aggregation == "sum":
            return torch.stack(layer_scores, dim=1).sum(dim=1)
        raise ValueError(f"Unknown goodness aggregation: {aggregation}")


class BackpropMLP(nn.Module):
    """Clean supervised baseline with the same two ReLU hidden layers."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        hidden_dims: Sequence[int] = HIDDEN_DIMS,
    ) -> None:
        super().__init__()
        dims = (input_dim, *hidden_dims)
        modules: list[nn.Module] = []
        for index in range(len(dims) - 1):
            modules.extend((nn.Linear(dims[index], dims[index + 1]), nn.ReLU()))
        self.features = nn.Sequential(*modules)
        self.classifier = nn.Linear(dims[-1], num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def build_model(method: str, input_dim: int, num_classes: int) -> nn.Module:
    if method == "bp":
        return BackpropMLP(input_dim, num_classes)
    if method in {"ff", "ff-ge", "fc-ff-ge"}:
        return GoodnessMLP(input_dim, normalize=True)
    if method == "fc-nn-ff-ge":
        return GoodnessMLP(input_dim, normalize=False)
    raise ValueError(f"Unknown method: {method}")


def mark_inputs(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Replace the first class-sized input region with one-hot labels."""
    if inputs.ndim != 2 or labels.ndim != 1 or inputs.size(0) != labels.size(0):
        raise ValueError("inputs must be [batch, features] and labels must be [batch]")
    if inputs.size(1) < num_classes:
        raise ValueError("input width must be at least the number of classes")
    marked = inputs.clone()
    marked[:, :num_classes] = 0.0
    marked.scatter_(1, labels.unsqueeze(1), 1.0)
    return marked


def sample_wrong_labels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Sample one wrong class uniformly without a rejection loop."""
    offsets = torch.randint(
        1,
        num_classes,
        labels.shape,
        device=labels.device,
    )
    return (labels + offsets) % num_classes


def softplus_goodness_loss(
    positive_goodness: torch.Tensor,
    negative_goodness: torch.Tensor,
) -> torch.Tensor:
    terms = torch.cat(
        (
            GOODNESS_THRESHOLD - positive_goodness,
            negative_goodness - GOODNESS_THRESHOLD,
        )
    )
    return F.softplus(terms).mean()


def candidate_goodness(
    model: GoodnessMLP,
    inputs: torch.Tensor,
    num_classes: int,
    *,
    aggregation: str,
    chunk_size: int,
) -> torch.Tensor:
    """Score every class label while bounding candidate-expanded memory."""
    batch_size, input_dim = inputs.shape
    score_chunks = []
    for start in range(0, num_classes, chunk_size):
        stop = min(start + chunk_size, num_classes)
        candidates = torch.arange(start, stop, device=inputs.device)
        candidates = candidates.unsqueeze(0).expand(batch_size, -1)
        expanded = inputs.unsqueeze(1).expand(-1, stop - start, -1).reshape(-1, input_dim)
        marked = mark_inputs(expanded, candidates.reshape(-1), num_classes)
        scores = model.goodness(marked, aggregation=aggregation)
        score_chunks.append(scores.view(batch_size, stop - start))
    return torch.cat(score_chunks, dim=1)


def _move_batch(
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs, labels = batch
    non_blocking = device.type == "cuda"
    return (
        inputs.to(device, non_blocking=non_blocking),
        labels.to(device, non_blocking=non_blocking),
    )


def train_vanilla_ff_epoch(
    model: GoodnessMLP,
    loader: DataLoader,
    optimizers: Sequence[torch.optim.Optimizer],
    *,
    device: torch.device,
    num_classes: int,
) -> float:
    """Train each layer locally, detaching representations between layers."""
    model.train()
    total_loss = 0.0
    observations = 0

    for layer_index, optimizer in enumerate(optimizers):
        for batch in loader:
            inputs, labels = _move_batch(batch, device)
            wrong_labels = sample_wrong_labels(labels, num_classes)
            positive = mark_inputs(inputs, labels, num_classes)
            negative = mark_inputs(inputs, wrong_labels, num_classes)

            with torch.no_grad():
                for previous_index in range(layer_index):
                    positive = model.forward_layer(positive, previous_index)
                    negative = model.forward_layer(negative, previous_index)

            positive_output = model.forward_layer(positive, layer_index)
            negative_output = model.forward_layer(negative, layer_index)
            loss = softplus_goodness_loss(
                positive_output.square().mean(dim=1),
                negative_output.square().mean(dim=1),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            observations += batch_size
    return total_loss / observations


def train_pairwise_global_epoch(
    model: GoodnessMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    num_classes: int,
) -> float:
    """Train FF+GE with one wrong label and final-layer global goodness."""
    model.train()
    total_loss = 0.0
    observations = 0
    for batch in loader:
        inputs, labels = _move_batch(batch, device)
        positive = mark_inputs(inputs, labels, num_classes)
        negative = mark_inputs(inputs, sample_wrong_labels(labels, num_classes), num_classes)
        loss = softplus_goodness_loss(
            model.goodness(positive, aggregation="final"),
            model.goodness(negative, aggregation="final"),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        observations += batch_size
    return total_loss / observations


def train_full_comparison_epoch(
    model: GoodnessMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    num_classes: int,
    candidate_chunk: int,
) -> float:
    """Train against every class using summed goodness and a global error."""
    model.train()
    total_loss = 0.0
    observations = 0
    for batch in loader:
        inputs, labels = _move_batch(batch, device)
        scores = candidate_goodness(
            model,
            inputs,
            num_classes,
            aggregation="sum",
            chunk_size=candidate_chunk,
        )
        loss = F.cross_entropy(scores, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        observations += batch_size
    return total_loss / observations


def train_backprop_epoch(
    model: BackpropMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    observations = 0
    for batch in loader:
        inputs, labels = _move_batch(batch, device)
        loss = F.cross_entropy(model(inputs), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        observations += batch_size
    return total_loss / observations


def train_one_epoch(
    method: str,
    model: nn.Module,
    loader: DataLoader,
    optimizers: Sequence[torch.optim.Optimizer],
    *,
    device: torch.device,
    num_classes: int,
    candidate_chunk: int,
) -> float:
    if method == "ff":
        if not isinstance(model, GoodnessMLP):
            raise TypeError("ff requires GoodnessMLP")
        return train_vanilla_ff_epoch(
            model,
            loader,
            optimizers,
            device=device,
            num_classes=num_classes,
        )
    optimizer = optimizers[0]
    if method == "ff-ge":
        if not isinstance(model, GoodnessMLP):
            raise TypeError("ff-ge requires GoodnessMLP")
        return train_pairwise_global_epoch(
            model,
            loader,
            optimizer,
            device=device,
            num_classes=num_classes,
        )
    if method in {"fc-ff-ge", "fc-nn-ff-ge"}:
        if not isinstance(model, GoodnessMLP):
            raise TypeError(f"{method} requires GoodnessMLP")
        return train_full_comparison_epoch(
            model,
            loader,
            optimizer,
            device=device,
            num_classes=num_classes,
            candidate_chunk=candidate_chunk,
        )
    if method == "bp":
        if not isinstance(model, BackpropMLP):
            raise TypeError("bp requires BackpropMLP")
        return train_backprop_epoch(model, loader, optimizer, device=device)
    raise ValueError(f"Unknown method: {method}")


@torch.no_grad()
def classification_accuracy(
    method: str,
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    num_classes: int,
    candidate_chunk: int,
) -> float:
    model.eval()
    correct = 0
    observations = 0
    for batch in loader:
        inputs, labels = _move_batch(batch, device)
        if method == "bp":
            if not isinstance(model, BackpropMLP):
                raise TypeError("bp requires BackpropMLP")
            predictions = model(inputs).argmax(dim=1)
        else:
            if not isinstance(model, GoodnessMLP):
                raise TypeError(f"{method} requires GoodnessMLP")
            aggregation = "final" if method == "ff-ge" else "sum"
            predictions = candidate_goodness(
                model,
                inputs,
                num_classes,
                aggregation=aggregation,
                chunk_size=candidate_chunk,
            ).argmax(dim=1)
        correct += predictions.eq(labels).sum().item()
        observations += labels.size(0)
    if observations == 0:
        raise RuntimeError("Cannot evaluate an empty dataset")
    return 100.0 * correct / observations


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def protocol_config(dataset: str, method: str, seed: int, epochs: int) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "epochs": epochs,
        "hidden_dims": list(HIDDEN_DIMS),
        "activation": "relu",
        "optimizer": "adam",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "goodness_threshold": GOODNESS_THRESHOLD,
    }


def _make_optimizers(method: str, model: nn.Module) -> list[torch.optim.Optimizer]:
    if method == "ff":
        if not isinstance(model, GoodnessMLP):
            raise TypeError("ff requires GoodnessMLP")
        return [Adam(layer.parameters(), lr=LEARNING_RATE) for layer in model.layers]
    return [Adam(model.parameters(), lr=LEARNING_RATE)]


def run_seed(
    *,
    dataset_name: str,
    method: str,
    seed: int,
    epochs: int,
    bundle: DatasetBundle,
    output_dir: Path,
    device: torch.device,
    evaluation_batch_size: int,
    candidate_chunk: int,
    num_workers: int,
    resume: bool,
) -> dict[str, Any]:
    run_dir = output_dir / dataset_name / method / f"seed_{seed}"
    result_path = run_dir / "run.json"
    config = protocol_config(dataset_name, method, seed, epochs)

    if result_path.exists():
        if not resume:
            raise FileExistsError(
                f"Completed output already exists at {result_path}; use --resume to reuse it"
            )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("config") != config:
            raise RuntimeError(f"Existing run has a different protocol: {result_path}")
        return result
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(
            f"Incomplete output exists at {run_dir}; move it aside before rerunning this seed"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    loaders = make_loaders(
        bundle,
        seed=seed,
        evaluation_batch_size=evaluation_batch_size,
        num_workers=num_workers,
        device=device,
    )
    model = build_model(method, bundle.input_dim, bundle.num_classes).to(device)
    optimizers = _make_optimizers(method, model)
    history: list[dict[str, Any]] = []
    history_fields = ("epoch", "train_loss", "learning_rate", "seconds")

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_loss = train_one_epoch(
            method,
            model,
            loaders.train,
            optimizers,
            device=device,
            num_classes=bundle.num_classes,
            candidate_chunk=candidate_chunk,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rate": LEARNING_RATE,
            "seconds": time.perf_counter() - started,
        }
        history.append(row)
        _atomic_write_csv(run_dir / "history.csv", history, history_fields)
        print(
            f"{dataset_name:12s} {method:12s} seed={seed} "
            f"epoch={epoch:3d}/{epochs} loss={train_loss:.6f}"
        )

    # The test set is read exactly once, after the final optimizer update.  It is
    # never used for model selection, checkpointing, or early stopping.
    train_accuracy = classification_accuracy(
        method,
        model,
        loaders.train_evaluation,
        device=device,
        num_classes=bundle.num_classes,
        candidate_chunk=candidate_chunk,
    )
    test_accuracy = classification_accuracy(
        method,
        model,
        loaders.test,
        device=device,
        num_classes=bundle.num_classes,
        candidate_chunk=candidate_chunk,
    )
    result = {
        "schema_version": 1,
        "status": "complete",
        "config": config,
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "final_train_accuracy": train_accuracy,
        "final_test_accuracy": test_accuracy,
        "history_file": "history.csv",
    }
    _atomic_write_json(result_path, result)
    print(
        f"FINAL {dataset_name:12s} {method:12s} seed={seed}: "
        f"train={train_accuracy:.2f}% test={test_accuracy:.2f}%"
    )
    return result


def aggregate_results(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for result in results:
        config = result["config"]
        key = (config["dataset"], config["method"], config["epochs"])
        grouped.setdefault(key, []).append(result)

    rows = []
    for (dataset, method, epochs), group in sorted(grouped.items()):
        train_values = [float(run["final_train_accuracy"]) for run in group]
        test_values = [float(run["final_test_accuracy"]) for run in group]
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "epochs": epochs,
                "seeds": ",".join(str(run["config"]["seed"]) for run in group),
                "runs": len(group),
                "final_train_accuracy_mean": statistics.mean(train_values),
                "final_train_accuracy_std": (
                    statistics.stdev(train_values) if len(train_values) > 1 else 0.0
                ),
                "final_test_accuracy_mean": statistics.mean(test_values),
                "final_test_accuracy_std": (
                    statistics.stdev(test_values) if len(test_values) > 1 else 0.0
                ),
            }
        )
    return rows


def _result_key(result: dict[str, Any]) -> tuple[str, str, int, int] | None:
    """Return a stable key for a complete, structurally valid run result."""
    if result.get("status") != "complete" or not isinstance(result.get("config"), dict):
        return None
    config = result["config"]
    dataset = config.get("dataset")
    method = config.get("method")
    epochs = config.get("epochs")
    seed = config.get("seed")
    if (
        dataset not in DATASET_NAMES
        or method not in METHOD_NAMES
        or not isinstance(epochs, int)
        or epochs <= 0
        or not isinstance(seed, int)
    ):
        return None
    required = ("final_train_accuracy", "final_test_accuracy", "parameter_count", "device")
    if any(field not in result for field in required):
        return None
    return dataset, method, epochs, seed


def completed_results(
    output_dir: Path,
    current_results: Sequence[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Collect and deduplicate every valid completed run below ``output_dir``."""
    by_key: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for result in current_results:
        key = _result_key(result)
        if key is not None:
            by_key[key] = result

    for dataset in DATASET_NAMES:
        for method in METHOD_NAMES:
            method_dir = output_dir / dataset / method
            for result_path in sorted(method_dir.glob("seed_*/run.json")):
                try:
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(result, dict):
                    continue
                key = _result_key(result)
                if key is None:
                    continue
                expected_dir = f"seed_{key[3]}"
                if key[:2] != (dataset, method) or result_path.parent.name != expected_dir:
                    continue
                by_key[key] = result
    return [by_key[key] for key in sorted(by_key)]


def write_summaries(output_dir: Path, results: Sequence[dict[str, Any]]) -> None:
    results = completed_results(output_dir, results)
    run_rows = []
    for result in results:
        config = result["config"]
        run_rows.append(
            {
                "dataset": config["dataset"],
                "method": config["method"],
                "epochs": config["epochs"],
                "seed": config["seed"],
                "final_train_accuracy": result["final_train_accuracy"],
                "final_test_accuracy": result["final_test_accuracy"],
                "parameter_count": result["parameter_count"],
                "device": result["device"],
            }
        )
    _atomic_write_csv(
        output_dir / "runs.csv",
        run_rows,
        (
            "dataset",
            "method",
            "epochs",
            "seed",
            "final_train_accuracy",
            "final_test_accuracy",
            "parameter_count",
            "device",
        ),
    )

    summary_rows = aggregate_results(results)
    summary_fields = (
        "dataset",
        "method",
        "epochs",
        "seeds",
        "runs",
        "final_train_accuracy_mean",
        "final_train_accuracy_std",
        "final_test_accuracy_mean",
        "final_test_accuracy_std",
    )
    _atomic_write_csv(output_dir / "summary.csv", summary_rows, summary_fields)
    _atomic_write_json(
        output_dir / "summary.json",
        {
            "schema_version": 1,
            "accuracy_unit": "percent",
            "standard_deviation_ddof": 1,
            "protocol": {
                "datasets": list(DATASET_NAMES),
                "methods": list(METHOD_NAMES),
                "default_seeds": list(DEFAULT_SEEDS),
                "hidden_dims": list(HIDDEN_DIMS),
                "activation": "relu",
                "optimizer": "adam",
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "default_epochs": DEFAULT_EPOCHS,
                "test_policy": "one evaluation after the final training epoch",
            },
            "aggregate": summary_rows,
            "runs": run_rows,
        },
    )


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    try:
        device = torch.device(name)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"Invalid device {name!r}") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise ValueError("MPS was requested but is not available")
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("--device must be auto, cpu, cuda, cuda:N, or mps")
    return device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed main-text Forward-Forward decomposition table."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_NAMES,
        default=list(DATASET_NAMES),
        help="datasets to run (default: all main-text datasets)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHOD_NAMES,
        default=list(METHOD_NAMES),
        help="methods to run (default: all five decomposition methods)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="run seeds (default: 424 425 426)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/ff_decomposition"))
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:N, or mps",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--evaluation-batch-size", type=int, default=1024)
    parser.add_argument("--candidate-chunk", type=int, default=10)
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="allow torchvision to download missing datasets",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse completed runs after verifying their fixed protocol",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def validate_args(args: argparse.Namespace) -> tuple[Path, Path, torch.device]:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.evaluation_batch_size <= 0:
        raise ValueError("--evaluation-batch-size must be positive")
    if args.candidate_chunk <= 0:
        raise ValueError("--candidate-chunk must be positive")
    if any(seed < 0 for seed in args.seeds):
        raise ValueError("--seeds must contain non-negative integers")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds cannot contain duplicates")
    if len(set(args.datasets)) != len(args.datasets):
        raise ValueError("--datasets cannot contain duplicates")
    if len(set(args.methods)) != len(args.methods):
        raise ValueError("--methods cannot contain duplicates")

    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if data_dir == output_dir:
        raise ValueError("--data-dir and --output-dir must be different directories")
    return data_dir, output_dir, resolve_device(args.device)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir, output_dir, device = validate_args(args)
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "Fixed protocol: hidden=1000,1000 ReLU | Adam lr=0.001 | "
        f"batch=128 | epochs={args.epochs} | device={device}"
    )
    results = []
    for dataset_name in args.datasets:
        bundle = load_dataset(dataset_name, data_dir, download=args.download)
        for method in args.methods:
            for seed in args.seeds:
                result = run_seed(
                    dataset_name=dataset_name,
                    method=method,
                    seed=seed,
                    epochs=args.epochs,
                    bundle=bundle,
                    output_dir=output_dir,
                    device=device,
                    evaluation_batch_size=args.evaluation_batch_size,
                    candidate_chunk=args.candidate_chunk,
                    num_workers=args.num_workers,
                    resume=args.resume,
                )
                results.append(result)
                write_summaries(output_dir, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
