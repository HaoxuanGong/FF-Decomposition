#!/usr/bin/env python3
"""Run one paper MLP decomposition benchmark."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

from mlp_components import (
    BackpropMLP,
    GOODNESS_THRESHOLD,
    GoodnessMLP,
    LocalBPMLP,
    NORMALIZATION_EPSILON,
    candidate_goodness,
    mark_inputs,
    seed_worker,
    set_seed,
    train_backprop_epoch,
    train_local_bp_epoch,
    train_matched_ce_epoch,
    train_matched_ff_epoch,
    train_pairwise_global_epoch,
    train_vanilla_ff_epoch,
)


PRIMARY_METHODS = (
    "ff",
    "ff-matched-ge",
    "ff-ge",
    "nn-ff-ge",
    "fc-ff",
    "fc-ff-matched-ge",
    "fc-ff-ge",
    "fc-nn-ff-ge",
    "local-bp",
    "ce-matched-ge",
    "bp",
)
DATASETS = ("mnist", "fashionmnist", "cifar10", "cifar100")
DATASET_LABELS = {
    "mnist": "MNIST",
    "fashionmnist": "F-MNIST",
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
}
ARCHITECTURES = {
    "mnist": (1000, 1000),
    "fashionmnist": (1000, 1000),
    "cifar10": (2000, 2000, 2000),
    "cifar100": (2000, 2000, 2000),
}
FC_METHODS = {"fc-ff", "fc-ff-matched-ge", "fc-ff-ge", "fc-nn-ff-ge"}
MATCHED_FF_METHODS = {"ff-matched-ge"}
THRESHOLDED_FF_METHODS = {"ff", "ff-ge", "nn-ff-ge"} | MATCHED_FF_METHODS
MATCHED_CE_METHODS = {"ce-matched-ge"}
MATCHED_LOCALITY_METHODS = MATCHED_FF_METHODS | MATCHED_CE_METHODS
NO_INTER_LAYER_NORMALIZATION_METHODS = {"nn-ff-ge", "fc-nn-ff-ge"}
NORMALIZED_FF_METHODS = {
    "ff",
    "ff-ge",
    "ff-matched-ge",
    "fc-ff",
    "fc-ff-matched-ge",
    "fc-ff-ge",
}
DEFAULT_BATCH_SIZE = 128
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_EPOCHS = 200
DEFAULT_VALIDATION_SIZE = 5000
DEFAULT_PATIENCE = 15
DEFAULT_STEP_SIZE = 30
DEFAULT_STEP_GAMMA = 0.1


class Flatten:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.flatten()


DATASET_SPECS = {
    "mnist": {
        "class": MNIST,
        "input_dim": 28 * 28,
        "classes": 10,
        "mean": (0.1307,),
        "std": (0.3081,),
    },
    "fashionmnist": {
        "class": FashionMNIST,
        "input_dim": 28 * 28,
        "classes": 10,
        "mean": (0.2860,),
        "std": (0.3530,),
    },
    "cifar10": {
        "class": CIFAR10,
        "input_dim": 32 * 32 * 3,
        "classes": 10,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2471, 0.2435, 0.2616),
        "archive": "cifar-10-python.tar.gz",
        "archive_md5": "c58f30108f718f92721af3b95e74349a",
    },
    "cifar100": {
        "class": CIFAR100,
        "input_dim": 32 * 32 * 3,
        "classes": 100,
        "mean": (0.5071, 0.4865, 0.4409),
        "std": (0.2673, 0.2564, 0.2762),
        "archive": "cifar-100-python.tar.gz",
        "archive_md5": "eb9058c3a382ffc7106e4002c42a8d85",
    },
}


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def dataset_manifest(data_dir: Path, dataset: str) -> dict[str, Any]:
    spec = DATASET_SPECS[dataset]
    archive_name = spec.get("archive")
    if archive_name:
        archive_path = data_dir / str(archive_name)
        actual_md5 = file_digest(archive_path, "md5") if archive_path.exists() else None
        expected_md5 = str(spec["archive_md5"])
        if actual_md5 is not None and actual_md5 != expected_md5:
            raise RuntimeError(f"{dataset} archive failed MD5 verification")
        return {
            "archive": archive_name,
            "actual_md5": actual_md5,
            "expected_md5": expected_md5,
        }
    raw_dir = data_dir / ("MNIST" if dataset == "mnist" else "FashionMNIST") / "raw"
    files = []
    for path in sorted(raw_dir.glob("*.gz")):
        files.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "md5": file_digest(path, "md5"),
            }
        )
    return {"raw_gzip_files": files}


def load_dataset(
    dataset: str, data_dir: Path, download: bool
) -> tuple[Dataset, Dataset]:
    spec = DATASET_SPECS[dataset]
    transform = Compose(
        [
            ToTensor(),
            Normalize(tuple(spec["mean"]), tuple(spec["std"])),
            Flatten(),
        ]
    )
    dataset_class = spec["class"]
    train = dataset_class(
        str(data_dir), train=True, transform=transform, download=download
    )
    test = dataset_class(
        str(data_dir), train=False, transform=transform, download=download
    )
    return train, test


def targets_array(dataset: Dataset) -> np.ndarray:
    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise TypeError("Dataset does not expose targets for a stratified split")
    if isinstance(targets, torch.Tensor):
        values = targets.detach().cpu().numpy()
    else:
        values = np.asarray(targets)
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    if len(values) != len(dataset):
        raise RuntimeError("Dataset target count does not match dataset length")
    return values


def stratified_split_indices(
    dataset: Dataset,
    *,
    seed: int,
    num_classes: int,
    validation_size: int,
) -> tuple[list[int], list[int]]:
    if validation_size <= 0:
        raise ValueError("--validation-size must be positive")
    if validation_size % num_classes != 0:
        raise ValueError("--validation-size must be divisible by the number of classes")
    validation_per_class = validation_size // num_classes
    targets = targets_array(dataset)
    observed_classes = np.unique(targets)
    expected_classes = np.arange(num_classes, dtype=np.int64)
    if not np.array_equal(observed_classes, expected_classes):
        raise RuntimeError(
            f"Expected classes 0..{num_classes - 1}, observed {observed_classes.tolist()}"
        )
    generator = np.random.default_rng(seed)
    training_indices: list[int] = []
    validation_indices: list[int] = []
    for class_index in range(num_classes):
        class_indices = np.flatnonzero(targets == class_index)
        if len(class_indices) <= validation_per_class:
            raise ValueError(
                f"Class {class_index} has {len(class_indices)} examples, which is not enough "
                f"for a validation allocation of {validation_per_class}"
            )
        generator.shuffle(class_indices)
        validation_indices.extend(class_indices[:validation_per_class].tolist())
        training_indices.extend(class_indices[validation_per_class:].tolist())
    generator.shuffle(training_indices)
    generator.shuffle(validation_indices)
    if len(validation_indices) != validation_size:
        raise RuntimeError("Stratified validation split has an unexpected size")
    if set(training_indices).intersection(validation_indices):
        raise RuntimeError("Training and validation indices overlap")
    if len(training_indices) + len(validation_indices) != len(dataset):
        raise RuntimeError("Training and validation indices do not cover the dataset")
    return training_indices, validation_indices


def make_loaders(
    train: Dataset,
    test: Dataset,
    *,
    seed: int,
    num_classes: int,
    validation_size: int,
    device: torch.device,
    evaluation_batch_size: int,
    batch_size: int,
    num_workers: int,
    train_limit: int | None,
    test_limit: int | None,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Any]]:
    training_indices, validation_indices = stratified_split_indices(
        train,
        seed=seed,
        num_classes=num_classes,
        validation_size=validation_size,
    )
    if train_limit is not None:
        training_indices = training_indices[: min(train_limit, len(training_indices))]
    train = Subset(train, training_indices)
    validation = Subset(train.dataset, validation_indices)
    if test_limit is not None:
        test = Subset(test, list(range(min(test_limit, len(test)))))
    common = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        **common,
    )
    validation_loader = DataLoader(
        validation,
        batch_size=evaluation_batch_size,
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(
        test,
        batch_size=evaluation_batch_size,
        shuffle=False,
        **common,
    )
    split_manifest = {
        "scheme": "seeded stratified holdout from the original training split",
        "seed": seed,
        "validation_size": len(validation_indices),
        "validation_per_class": validation_size // num_classes,
        "training_size_before_limit": len(targets_array(train.dataset))
        - len(validation_indices),
        "training_size_used": len(training_indices),
        "validation_index_sha256": hashlib.sha256(
            np.asarray(validation_indices, dtype=np.int64).tobytes()
        ).hexdigest(),
    }
    return train_loader, validation_loader, test_loader, split_manifest


def build_model(
    method: str,
    dataset: str,
    *,
    hidden_dims: Sequence[int] | None = None,
) -> nn.Module:
    spec = DATASET_SPECS[dataset]
    input_dim = int(spec["input_dim"])
    num_classes = int(spec["classes"])
    resolved_hidden_dims = tuple(hidden_dims or ARCHITECTURES[dataset])
    if method == "bp":
        return BackpropMLP(input_dim, num_classes, hidden_dims=resolved_hidden_dims)
    elif method == "local-bp" or method in MATCHED_CE_METHODS:
        return LocalBPMLP(input_dim, num_classes, hidden_dims=resolved_hidden_dims)
    return GoodnessMLP(
        input_dim,
        hidden_dims=resolved_hidden_dims,
        normalize=method in NORMALIZED_FF_METHODS,
        normalize_first_layer_input=(
            method in NORMALIZED_FF_METHODS | NO_INTER_LAYER_NORMALIZATION_METHODS
        ),
    )


def expand_candidates(
    inputs: torch.Tensor,
    num_classes: int,
    start: int,
    stop: int,
) -> tuple[torch.Tensor, int]:
    batch_size, input_dim = inputs.shape
    count = stop - start
    labels = (
        torch.arange(start, stop, device=inputs.device)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    expanded = inputs.unsqueeze(1).expand(-1, count, -1).reshape(-1, input_dim)
    return mark_inputs(expanded, labels.reshape(-1), num_classes), count


def train_local_full_comparison_epoch(
    model: GoodnessMLP,
    loader: DataLoader,
    optimizers: Sequence[torch.optim.Optimizer],
    *,
    device: torch.device,
    num_classes: int,
    candidate_chunk: int,
) -> float:
    """Train FC-FF: all classes at each layer, with detached inter-layer credit."""
    model.train()
    total_loss = 0.0
    observations = 0
    for layer_index, optimizer in enumerate(optimizers):
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            score_chunks = []
            for start in range(0, num_classes, candidate_chunk):
                stop = min(start + candidate_chunk, num_classes)
                marked, count = expand_candidates(inputs, num_classes, start, stop)
                with torch.no_grad():
                    for previous_index in range(layer_index):
                        marked = model.forward_layer(marked, previous_index)
                outputs = model.forward_layer(marked, layer_index)
                goodness = outputs.square().mean(dim=1)
                score_chunks.append(goodness.view(labels.size(0), count))
            loss = F.cross_entropy(torch.cat(score_chunks, dim=1), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            observations += labels.size(0)
    return total_loss / observations


def full_comparison_layer_scores(
    model: GoodnessMLP,
    inputs: torch.Tensor,
    *,
    num_classes: int,
    candidate_chunk: int,
    detach_between_layers: bool,
) -> list[torch.Tensor]:
    """Return one all-class goodness matrix per hidden layer.

    Each result has shape ``[batch, classes]``.  Detachment changes only the
    backward graph; it leaves the class-conditioned forward scores unchanged.
    """
    layer_chunks: list[list[torch.Tensor]] = [[] for _ in model.layers]
    for start in range(0, num_classes, candidate_chunk):
        stop = min(start + candidate_chunk, num_classes)
        marked, count = expand_candidates(inputs, num_classes, start, stop)
        for layer_index in range(len(model.layers)):
            if layer_index > 0 and detach_between_layers:
                marked = marked.detach()
            marked = model.forward_layer(marked, layer_index)
            goodness = model.activation_goodness(marked)
            layer_chunks[layer_index].append(goodness.view(inputs.size(0), count))
    return [torch.cat(chunks, dim=1) for chunks in layer_chunks]


def train_global_multihead_full_comparison_epoch(
    model: GoodnessMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    num_classes: int,
    candidate_chunk: int,
) -> float:
    """Train globally connected FC-FF with one goodness objective per layer."""
    model.train()
    total_loss = 0.0
    observations = 0
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        layer_scores = full_comparison_layer_scores(
            model,
            inputs,
            num_classes=num_classes,
            candidate_chunk=candidate_chunk,
            detach_between_layers=False,
        )
        loss = torch.stack(
            [F.cross_entropy(scores, labels) for scores in layer_scores]
        ).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        observations += labels.size(0)
    return total_loss / observations


def train_global_full_comparison_epoch(
    model: GoodnessMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    num_classes: int,
    candidate_chunk: int,
) -> float:
    """Train FC-FF+GE with all-class CE on terminal-layer goodness."""
    model.train()
    total_loss = 0.0
    observations = 0
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        scores = candidate_goodness(
            model,
            inputs,
            num_classes,
            aggregation="final",
            chunk_size=candidate_chunk,
        )
        loss = F.cross_entropy(scores, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        observations += labels.size(0)
    return total_loss / observations


def make_optimizer(
    parameters: Any,
    *,
    optimizer_name: str,
    learning_rate: float,
    momentum: float,
) -> torch.optim.Optimizer:
    if optimizer_name == "adam":
        return Adam(
            parameters,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )
    if optimizer_name == "sgd":
        return SGD(
            parameters,
            lr=learning_rate,
            momentum=momentum,
            dampening=0.0,
            weight_decay=0.0,
            nesterov=False,
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def make_optimizers(
    method: str,
    model: nn.Module,
    *,
    optimizer_name: str,
    learning_rate: float,
    momentum: float,
) -> list[torch.optim.Optimizer]:
    if method in {"ff", "fc-ff"}:
        if not isinstance(model, GoodnessMLP):
            raise TypeError(f"{method} requires GoodnessMLP")
        parameter_groups = [layer.parameters() for layer in model.layers]
    elif method == "local-bp":
        if not isinstance(model, LocalBPMLP):
            raise TypeError("local-bp requires LocalBPMLP")
        parameter_groups = [layer.parameters() for layer in model.layers]
    else:
        parameter_groups = [model.parameters()]
    return [
        make_optimizer(
            parameters,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            momentum=momentum,
        )
        for parameters in parameter_groups
    ]


def validate_optimizer_coverage(
    model: nn.Module,
    optimizers: Sequence[torch.optim.Optimizer],
) -> None:
    expected = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    observed = [
        id(parameter)
        for optimizer in optimizers
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    counts = Counter(observed)
    if set(observed) != expected:
        raise RuntimeError("Optimizers do not cover exactly all trainable parameters")
    duplicates = [parameter_id for parameter_id, count in counts.items() if count != 1]
    if duplicates:
        raise RuntimeError("A trainable parameter appears in more than one optimizer")


def make_schedulers(
    optimizers: Sequence[torch.optim.Optimizer],
    *,
    scheduler_name: str,
    epochs: int,
    step_size: int,
    step_gamma: float,
) -> list[Any]:
    """Create one scheduler per optimizer; callers step them once per epoch."""
    if scheduler_name == "none":
        return []
    if scheduler_name == "step":
        return [
            StepLR(optimizer, step_size=step_size, gamma=step_gamma)
            for optimizer in optimizers
        ]
    if scheduler_name == "cosine":
        return [
            CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
            for optimizer in optimizers
        ]
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def positive_goodness_threshold(value: str) -> float:
    threshold = float(value)
    if not math.isfinite(threshold) or threshold <= 0:
        raise argparse.ArgumentTypeError(
            "--goodness-threshold must be finite and positive"
        )
    return threshold


def validate_goodness_threshold(method: str, threshold: float) -> float:
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("--goodness-threshold must be finite and positive")
    if method not in THRESHOLDED_FF_METHODS and threshold != GOODNESS_THRESHOLD:
        raise ValueError(f"--goodness-threshold does not apply to method {method!r}")
    return threshold


def train_epoch(
    method: str,
    model: nn.Module,
    loader: DataLoader,
    optimizers: Sequence[torch.optim.Optimizer],
    *,
    device: torch.device,
    num_classes: int,
    candidate_chunk: int,
    goodness_threshold: float = GOODNESS_THRESHOLD,
) -> float:
    goodness_threshold = validate_goodness_threshold(method, goodness_threshold)
    if method == "bp":
        return train_backprop_epoch(model, loader, optimizers[0], device=device)  # type: ignore[arg-type]
    if method == "local-bp":
        return train_local_bp_epoch(model, loader, optimizers, device=device)  # type: ignore[arg-type]
    if method in MATCHED_CE_METHODS:
        return train_matched_ce_epoch(
            model,  # type: ignore[arg-type]
            loader,
            optimizers[0],
            device=device,
            detach_between_layers=False,
        )
    if method == "ff":
        return train_vanilla_ff_epoch(
            model,
            loader,
            optimizers,
            device=device,
            num_classes=num_classes,  # type: ignore[arg-type]
            threshold=goodness_threshold,
        )
    if method in {"ff-ge", "nn-ff-ge"}:
        return train_pairwise_global_epoch(
            model,
            loader,
            optimizers[0],
            device=device,
            num_classes=num_classes,  # type: ignore[arg-type]
            threshold=goodness_threshold,
        )
    if method == "ff-matched-ge":
        return train_matched_ff_epoch(
            model,  # type: ignore[arg-type]
            loader,
            optimizers[0],
            device=device,
            num_classes=num_classes,
            detach_between_layers=False,
            threshold=goodness_threshold,
        )
    if method == "fc-ff":
        return train_local_full_comparison_epoch(
            model,  # type: ignore[arg-type]
            loader,
            optimizers,
            device=device,
            num_classes=num_classes,
            candidate_chunk=candidate_chunk,
        )
    if method == "fc-ff-matched-ge":
        return train_global_multihead_full_comparison_epoch(
            model,  # type: ignore[arg-type]
            loader,
            optimizers[0],
            device=device,
            num_classes=num_classes,
            candidate_chunk=candidate_chunk,
        )
    if method in {"fc-ff-ge", "fc-nn-ff-ge"}:
        return train_global_full_comparison_epoch(
            model,  # type: ignore[arg-type]
            loader,
            optimizers[0],
            device=device,
            num_classes=num_classes,
            candidate_chunk=candidate_chunk,
        )
    raise ValueError(method)


@torch.no_grad()
def evaluate(
    method: str,
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    num_classes: int,
    candidate_chunk: int,
) -> dict[str, float | int]:
    model.eval()
    correct = 0
    final_correct = 0
    total = 0
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        if method == "bp":
            predictions = model(inputs).argmax(dim=1)  # type: ignore[operator]
        elif method == "local-bp" or method in MATCHED_CE_METHODS:
            logits = model.class_logits(inputs)  # type: ignore[union-attr]
            predictions = torch.stack(logits, dim=0).sum(dim=0).argmax(dim=1)
            final_correct += logits[-1].argmax(dim=1).eq(labels).sum().item()
        else:
            aggregation = (
                "final"
                if method in {"ff-ge", "nn-ff-ge", "fc-ff-ge", "fc-nn-ff-ge"}
                else "sum"
            )
            scores = candidate_goodness(
                model,  # type: ignore[arg-type]
                inputs,
                num_classes,
                aggregation=aggregation,
                chunk_size=candidate_chunk,
            )
            predictions = scores.argmax(dim=1)
        correct += predictions.eq(labels).sum().item()
        total += labels.size(0)
    result: dict[str, float | int] = {
        "correct": correct,
        "total": total,
        "accuracy": 100.0 * correct / total,
    }
    if method == "local-bp" or method in MATCHED_CE_METHODS:
        result["final_layer_accuracy"] = 100.0 * final_correct / total
    return result


def configuration(args: argparse.Namespace) -> dict[str, Any]:
    goodness_threshold = validate_goodness_threshold(
        args.method, getattr(args, "goodness_threshold", GOODNESS_THRESHOLD)
    )
    local_schedule = (
        "all local layers updated per minibatch with detached representations"
        if args.method == "local-bp"
        else "one complete dataset pass per layer per nominal epoch"
        if args.method in {"ff", "fc-ff"}
        else "one matched multi-loss update per minibatch"
        if args.method in MATCHED_LOCALITY_METHODS | {"fc-ff-matched-ge"}
        else "one end-to-end update per minibatch"
    )
    config = {
        "dataset": args.dataset,
        "method": args.method,
        "seed": args.seed,
        "epochs": args.epochs,
        "hidden_dims": list(args.hidden_dims or ARCHITECTURES[args.dataset]),
        "activation": "relu",
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "adam_betas": [0.9, 0.999] if args.optimizer == "adam" else None,
        "adam_epsilon": 1e-8 if args.optimizer == "adam" else None,
        "sgd_momentum": args.momentum if args.optimizer == "sgd" else None,
        "sgd_dampening": 0.0 if args.optimizer == "sgd" else None,
        "sgd_nesterov": False if args.optimizer == "sgd" else None,
        "batch_size": args.batch_size,
        "evaluation_batch_size": args.evaluation_batch_size,
        "scheduler": args.scheduler,
        "step_size": args.step_size if args.scheduler == "step" else None,
        "step_gamma": args.step_gamma if args.scheduler == "step" else None,
        "weight_decay": 0.0,
        "dropout": 0.0,
        "precision": "float32",
        "amp": False,
        "early_stopping": True,
        "early_stopping_patience": args.patience,
        "early_stopping_minimum_epochs": args.minimum_epochs,
        "early_stopping_monitor": "validation accuracy",
        "early_stopping_mode": "max",
        "early_stopping_min_delta": 0.0,
        "early_stopping_tie_policy": "ties count as non-improvement",
        "restore_best_checkpoint": True,
        "validation_size": args.validation_size,
        "validation_split": "seeded stratified holdout from the original training split",
        "model_selection": "restore the checkpoint with the highest validation accuracy",
        "test_policy": (
            "deferred until a separate evaluation after validation-only configuration selection"
            if args.defer_test
            else "one evaluation after restoring the best validation checkpoint"
        ),
        "local_update_schedule": local_schedule,
        "inter_layer_normalization": args.method in NORMALIZED_FF_METHODS,
        "first_layer_input_normalization": (
            args.method in NORMALIZED_FF_METHODS | NO_INTER_LAYER_NORMALIZATION_METHODS
        ),
        "normalization_scope": (
            "input to every FF layer, including the encoded input to layer 1"
            if args.method in NORMALIZED_FF_METHODS
            else "encoded input to layer 1 only"
            if args.method in NO_INTER_LAYER_NORMALIZATION_METHODS
            else "none"
        ),
        "normalization_epsilon": (
            NORMALIZATION_EPSILON
            if args.method
            in NORMALIZED_FF_METHODS | NO_INTER_LAYER_NORMALIZATION_METHODS
            else None
        ),
        "goodness_definition": (
            None
            if args.method in {"bp", "local-bp"} | MATCHED_CE_METHODS
            else "mean squared activation"
        ),
        "goodness_threshold": (
            goodness_threshold if args.method in THRESHOLDED_FF_METHODS else None
        ),
        "matched_locality_control": args.method in MATCHED_LOCALITY_METHODS,
        "matched_loss_placement": (
            "equal-weight mean of one thresholded FF loss at every layer"
            if args.method in MATCHED_FF_METHODS
            else "equal-weight mean of one all-class goodness loss at every layer"
            if args.method == "fc-ff-matched-ge"
            else "equal-weight mean of one cross-entropy loss at every layer"
            if args.method in MATCHED_CE_METHODS
            else None
        ),
        "detach_between_layers": (
            True
            if args.method == "local-bp"
            else False
            if args.method in MATCHED_LOCALITY_METHODS | {"fc-ff-matched-ge"}
            else None
        ),
        "prediction": (
            "cumulative local logits"
            if args.method == "local-bp" or args.method in MATCHED_CE_METHODS
            else "final-layer goodness"
            if args.method in {"ff-ge", "nn-ff-ge", "fc-ff-ge", "fc-nn-ff-ge"}
            else "summed layer goodness"
            if args.method in {"ff", "ff-matched-ge", "fc-ff", "fc-ff-matched-ge"}
            else "classifier logits"
        ),
        "terminal_classifier_bias": False if args.method == "bp" else None,
        "candidate_chunk": args.candidate_chunk,
        "num_workers": args.num_workers,
        "download": args.download,
        "requested_device": args.device,
        "negative_sampling": (
            "one uniformly sampled incorrect class per example per update"
            if args.method
            in {
                "ff",
                "ff-ge",
                "nn-ff-ge",
                "ff-matched-ge",
            }
            else "all classes"
            if args.method in FC_METHODS
            else None
        ),
        "label_encoding": (
            "replace the first C normalized input entries with a 0/1 one-hot label"
            if args.method
            in NORMALIZED_FF_METHODS | NO_INTER_LAYER_NORMALIZATION_METHODS
            else None
        ),
        "dnc_policy": "report every finite completed accuracy numerically; DNC only on run failure",
        "train_limit": args.train_limit,
        "test_limit": args.test_limit,
    }
    if args.defer_test or args.scheduler == "cosine":
        config.update(
            {
                "cosine_t_max": args.epochs if args.scheduler == "cosine" else None,
                "cosine_eta_min": 0.0 if args.scheduler == "cosine" else None,
                "scheduler_step_unit": "one step after each nominal training epoch",
                "defer_test": args.defer_test,
            }
        )
    return config


def run(args: argparse.Namespace) -> None:
    if args.dataset == "cifar100" and args.method in FC_METHODS:
        raise ValueError("FC methods are intentionally not run on CIFAR-100")
    device = torch.device(args.device)
    run_dir = args.output_dir / args.dataset / args.method / f"seed_{args.seed}"
    result_path = run_dir / "run.json"
    config = configuration(args)
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("config") != config:
            raise RuntimeError(
                f"Existing result uses a different configuration: {result_path}"
            )
        allowed_statuses = (
            {"train_complete", "complete"} if args.defer_test else {"complete"}
        )
        if existing.get("status") not in allowed_statuses:
            raise RuntimeError(f"Existing run is not complete: {result_path}")
        print(f"REUSE {result_path}", flush=True)
        return
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"Incomplete run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / "config.json", config)

    set_seed(args.seed)
    train_data, test_data = load_dataset(args.dataset, args.data_dir, args.download)
    num_classes = int(DATASET_SPECS[args.dataset]["classes"])
    train_loader, validation_loader, test_loader, split_manifest = make_loaders(
        train_data,
        test_data,
        seed=args.seed,
        num_classes=num_classes,
        validation_size=args.validation_size,
        device=device,
        evaluation_batch_size=args.evaluation_batch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
    )
    model = build_model(
        args.method,
        args.dataset,
        hidden_dims=tuple(int(width) for width in config["hidden_dims"]),
    ).to(device)
    optimizers = make_optimizers(
        args.method,
        model,
        optimizer_name=args.optimizer,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
    )
    validate_optimizer_coverage(model, optimizers)
    schedulers = make_schedulers(
        optimizers,
        scheduler_name=args.scheduler,
        epochs=args.epochs,
        step_size=args.step_size,
        step_gamma=args.step_gamma,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history: list[dict[str, Any]] = []
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    checkpoint_path = run_dir / "best_model.pt"
    best_validation_accuracy = float("-inf")
    best_validation_metrics: dict[str, float | int] | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    early_stopped = False
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        learning_rates = [
            float(optimizer.param_groups[0]["lr"]) for optimizer in optimizers
        ]
        loss = train_epoch(
            args.method,
            model,
            train_loader,
            optimizers,
            device=device,
            num_classes=num_classes,
            candidate_chunk=args.candidate_chunk,
            goodness_threshold=getattr(args, "goodness_threshold", GOODNESS_THRESHOLD),
        )
        if not math.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite training loss at epoch {epoch}: {loss}"
            )
        validation_metrics = evaluate(
            args.method,
            model,
            validation_loader,
            device=device,
            num_classes=num_classes,
            candidate_chunk=args.candidate_chunk,
        )
        validation_accuracy = float(validation_metrics["accuracy"])
        if not math.isfinite(validation_accuracy):
            raise FloatingPointError(
                f"Non-finite validation accuracy at epoch {epoch}: {validation_accuracy}"
            )
        improved = validation_accuracy > best_validation_accuracy
        if improved:
            best_validation_accuracy = validation_accuracy
            best_validation_metrics = dict(validation_metrics)
            best_epoch = epoch
            epochs_without_improvement = 0
            temporary_checkpoint = checkpoint_path.with_name(
                f".{checkpoint_path.name}.tmp"
            )
            torch.save(
                {
                    "state_dict": {
                        name: value.detach().cpu()
                        for name, value in model.state_dict().items()
                    },
                    "config": config,
                    "best_epoch": best_epoch,
                    "best_validation_metrics": best_validation_metrics,
                },
                temporary_checkpoint,
            )
            os.replace(temporary_checkpoint, checkpoint_path)
        else:
            epochs_without_improvement += 1
        row = {
            "epoch": epoch,
            "train_loss": loss,
            "validation_accuracy": validation_accuracy,
            "best_validation_accuracy": best_validation_accuracy,
            "improved": improved,
            "epochs_without_improvement": epochs_without_improvement,
            "learning_rate": learning_rates[0],
            "all_learning_rates": json.dumps(learning_rates),
            "finite": True,
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        atomic_csv(
            run_dir / "history.csv",
            history,
            (
                "epoch",
                "train_loss",
                "validation_accuracy",
                "best_validation_accuracy",
                "improved",
                "epochs_without_improvement",
                "learning_rate",
                "all_learning_rates",
                "finite",
                "seconds",
            ),
        )
        print(
            f"{args.dataset} {args.method} seed={args.seed} "
            f"epoch={epoch}/{args.epochs} loss={loss:.6f} "
            f"val={validation_accuracy:.2f}% best={best_validation_accuracy:.2f}% "
            f"patience={epochs_without_improvement}/{args.patience}",
            flush=True,
        )
        for scheduler in schedulers:
            scheduler.step()
        atomic_json(
            run_dir / "heartbeat.json",
            {
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "epoch": epoch,
                "validation_accuracy": validation_accuracy,
                "best_validation_accuracy": best_validation_accuracy,
            },
        )
        if epoch >= args.minimum_epochs and epochs_without_improvement >= args.patience:
            early_stopped = True
            break

    if (
        best_epoch == 0
        or best_validation_metrics is None
        or not checkpoint_path.exists()
    ):
        raise RuntimeError("No best validation checkpoint was created")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    test_metrics = None
    if not args.defer_test:
        test_metrics = evaluate(
            args.method,
            model,
            test_loader,
            device=device,
            num_classes=num_classes,
            candidate_chunk=args.candidate_chunk,
        )
    result = {
        "schema_version": 2,
        "status": "train_complete" if args.defer_test else "complete",
        "config": config,
        "dataset": {
            "name": DATASET_LABELS[args.dataset],
            "training_samples": len(train_loader.dataset),
            "validation_samples": len(validation_loader.dataset),
            "test_samples": len(test_loader.dataset),
            "original_training_samples": len(train_data),
            "original_test_samples": len(test_data),
            "normalization_mean": list(DATASET_SPECS[args.dataset]["mean"]),
            "normalization_std": list(DATASET_SPECS[args.dataset]["std"]),
            "augmentation": "none",
            "split": split_manifest,
            "source_manifest": dataset_manifest(args.data_dir, args.dataset),
        },
        "selection": {
            "best_epoch": best_epoch,
            "epochs_trained": len(history),
            "maximum_epochs": args.epochs,
            "patience": args.patience,
            "early_stopped": early_stopped,
            "best_validation_accuracy": best_validation_accuracy,
            "best_validation_metrics": best_validation_metrics,
        },
        "test": test_metrics,
        "test_evaluations": 0 if args.defer_test else 1,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": time.perf_counter() - started,
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": file_digest(checkpoint_path, "sha256"),
        "software": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_arch_list": torch.cuda.get_arch_list(),
            "cudnn": torch.backends.cudnn.version(),
        },
        "numerics": {
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "hardware": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
            "cuda_capability": list(torch.cuda.get_device_capability(device))
            if device.type == "cuda"
            else None,
        },
        "source_sha256": {
            name: file_digest(Path(__file__).resolve().with_name(name), "sha256")
            for name in ("mlp_experiment.py", "mlp_components.py")
        },
        "command": [sys.executable, *sys.argv],
    }
    atomic_json(result_path, result)
    print(
        f"FINAL {args.dataset} {args.method} seed={args.seed}: "
        f"best_epoch={best_epoch} val={best_validation_accuracy:.2f}% "
        + (
            "test=DEFERRED"
            if test_metrics is None
            else f"test={float(test_metrics['accuracy']):.2f}%"
        ),
        flush=True,
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--dataset", choices=DATASETS, required=True)
    run_parser.add_argument("--method", choices=PRIMARY_METHODS, required=True)
    run_parser.add_argument("--seed", type=int, required=True)
    run_parser.add_argument(
        "--hidden-dims",
        nargs="+",
        type=int,
        default=None,
        help="override the dataset's default hidden-layer widths",
    )
    run_parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    run_parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    run_parser.add_argument(
        "--minimum-epochs",
        type=int,
        default=0,
        help="minimum epochs before patience can terminate training",
    )
    run_parser.add_argument(
        "--validation-size", type=int, default=DEFAULT_VALIDATION_SIZE
    )
    run_parser.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    run_parser.add_argument(
        "--learning-rate", type=float, default=DEFAULT_LEARNING_RATE
    )
    run_parser.add_argument(
        "--goodness-threshold",
        type=positive_goodness_threshold,
        default=GOODNESS_THRESHOLD,
        help="positive FF goodness threshold; applies to vanilla, global, and matched FF",
    )
    run_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    run_parser.add_argument("--momentum", type=float, default=0.9)
    run_parser.add_argument(
        "--scheduler", choices=("none", "step", "cosine"), default="none"
    )
    run_parser.add_argument("--step-size", type=int, default=DEFAULT_STEP_SIZE)
    run_parser.add_argument("--step-gamma", type=float, default=DEFAULT_STEP_GAMMA)
    run_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--device", default="cuda")
    run_parser.add_argument("--evaluation-batch-size", type=int, default=256)
    run_parser.add_argument("--candidate-chunk", type=int, default=10)
    run_parser.add_argument("--num-workers", type=int, default=0)
    run_parser.add_argument(
        "--download", action=argparse.BooleanOptionalAction, default=False
    )
    run_parser.add_argument(
        "--defer-test",
        action="store_true",
        help="save the validation-selected model without evaluating the test split",
    )
    run_parser.add_argument(
        "--train-limit", type=int, default=None, help=argparse.SUPPRESS
    )
    run_parser.add_argument(
        "--test-limit", type=int, default=None, help=argparse.SUPPRESS
    )
    run_parser.set_defaults(function=run)
    return root


def main() -> None:
    args = parser().parse_args()
    if getattr(args, "epochs", 1) <= 0:
        raise ValueError("--epochs must be positive")
    if getattr(args, "patience", 1) <= 0:
        raise ValueError("--patience must be positive")
    if getattr(args, "minimum_epochs", 0) < 0:
        raise ValueError("--minimum-epochs cannot be negative")
    if getattr(args, "minimum_epochs", 0) > getattr(args, "epochs", 1):
        raise ValueError("--minimum-epochs cannot exceed --epochs")
    if getattr(args, "validation_size", 1) <= 0:
        raise ValueError("--validation-size must be positive")
    if getattr(args, "evaluation_batch_size", 1) <= 0:
        raise ValueError("--evaluation-batch-size must be positive")
    if getattr(args, "candidate_chunk", 1) <= 0:
        raise ValueError("--candidate-chunk must be positive")
    if getattr(args, "learning_rate", 1.0) <= 0:
        raise ValueError("--learning-rate must be positive")
    if getattr(args, "batch_size", 1) <= 0:
        raise ValueError("--batch-size must be positive")
    if getattr(args, "momentum", 0.0) < 0:
        raise ValueError("--momentum cannot be negative")
    if getattr(args, "step_size", 1) <= 0:
        raise ValueError("--step-size must be positive")
    if not 0 < getattr(args, "step_gamma", 1.0) <= 1:
        raise ValueError("--step-gamma must be in (0, 1]")
    if getattr(args, "train_limit", None) is not None and args.train_limit <= 0:
        raise ValueError("--train-limit must be positive")
    if getattr(args, "test_limit", None) is not None and args.test_limit <= 0:
        raise ValueError("--test-limit must be positive")
    if getattr(args, "hidden_dims", None) is not None and any(
        width <= 0 for width in args.hidden_dims
    ):
        raise ValueError("--hidden-dims values must be positive")
    args.function(args)


if __name__ == "__main__":
    main()
