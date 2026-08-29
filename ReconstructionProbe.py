#!/usr/bin/env python3
"""Train validation-selected reconstruction probes on a frozen FF backbone.

One invocation handles one independently trained FashionMNIST backbone and one
experimental seed. The official test split is evaluated once per hidden-layer
decoder, only after the best validation-MSE checkpoint has been restored.
"""

from __future__ import annotations

import argparse
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
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
import torchvision
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Normalize, ToTensor

from decomposition_core import (
    NORMALIZATION_EPSILON,
    GoodnessMLP,
    mark_inputs,
    seed_worker,
    set_seed,
)
from MLPBenchmarkSuite import dataset_manifest, file_digest, stratified_split_indices


FASHION_MEAN = (0.2860,)
FASHION_STD = (0.3530,)
EXPECTED_HIDDEN_DIMS = (500, 500, 500, 500, 500)
EXPECTED_METHODS = {
    "normalized": "fc-ff-ge",
    "no-inter-layer-normalization": "fc-nn-ff-ge",
}
CONDITION_LABELS = {
    "normalized": "inter-layer L2 normalization",
    "no-inter-layer-normalization": "no inter-layer L2 normalization",
}
RESULT_SCHEMA_VERSION = 3


def source_sha256() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        name: file_digest(root / name, "sha256")
        for name in (
            "ReconstructionProbe.py",
            "decomposition_core.py",
            "MLPBenchmarkSuite.py",
        )
    }


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def state_dicts_equal(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> bool:
    return left.keys() == right.keys() and all(
        torch.equal(left[name].detach().cpu(), right[name].detach().cpu())
        for name in left
    )


def load_torch_payload(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint payload: {path}")
    return payload


class ReconstructionDataset(Dataset):
    """Return normalized model input, integer label, and raw [0, 1] target."""

    def __init__(self, root: Path, *, train: bool, download: bool) -> None:
        self.dataset = FashionMNIST(str(root), train=train, download=download)
        self.targets = self.dataset.targets
        self.to_tensor = ToTensor()
        self.normalize = Normalize(FASHION_MEAN, FASHION_STD)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image, label = self.dataset[index]
        raw = self.to_tensor(image)
        normalized = self.normalize(raw.clone())
        return normalized.flatten(), torch.tensor(label), raw.flatten()


class Decoder(nn.Module):
    """One-hidden-layer, 500-unit image reconstruction probe."""

    def __init__(self, representation_dim: int, *, hidden_dim: int = 500) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(representation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 28 * 28),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


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


def load_backbone(
    path: Path,
    *,
    condition: str,
    seed: int,
    device: torch.device,
) -> tuple[GoodnessMLP, dict[str, Any], dict[str, Any]]:
    result_path = path.with_name("run.json")
    if not result_path.exists():
        raise FileNotFoundError(f"Backbone result is missing: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "complete" or int(result.get("test_evaluations", 0)) != 1:
        raise ValueError(f"Backbone result is not a verified completed run: {result_path}")
    if file_digest(path, "sha256") != result.get("checkpoint_sha256"):
        raise ValueError(f"Backbone checkpoint checksum mismatch: {path}")
    provenance_path = path.with_name("reconstruction_provenance.json")
    if not provenance_path.is_file():
        raise ValueError(f"Backbone provenance sidecar is missing: {provenance_path}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parent
    expected_sources = {
        name: file_digest(root / name, "sha256")
        for name in ("MLPBenchmarkSuite.py", "decomposition_core.py")
    }
    provenance_requirements = {
        "schema": provenance.get("schema_version") == 1,
        "sources": provenance.get("source_sha256") == expected_sources,
        "result": provenance.get("run_json_sha256")
        == file_digest(result_path, "sha256"),
        "checkpoint": provenance.get("checkpoint_sha256")
        == file_digest(path, "sha256"),
        "history": provenance.get("history_sha256")
        == file_digest(path.with_name("history.csv"), "sha256"),
    }
    failed_provenance = [
        name for name, passed in provenance_requirements.items() if not passed
    ]
    if failed_provenance:
        raise ValueError(
            f"Backbone provenance verification failed ({failed_provenance}): {path}"
        )
    checkpoint = load_torch_payload(path)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    config = checkpoint.get("config")
    if not isinstance(config, dict) or config != result.get("config"):
        raise ValueError(f"Backbone checkpoint/result configuration mismatch: {path}")
    expected_method = EXPECTED_METHODS[condition]
    expected_inter_layer_normalization = condition == "normalized"
    requirements = {
        "dataset": config.get("dataset") == "fashionmnist",
        "method": config.get("method") == expected_method,
        "seed": int(config.get("seed", -1)) == seed,
        "hidden_dims": tuple(config.get("hidden_dims", ())) == EXPECTED_HIDDEN_DIMS,
        "inter_layer_normalization": bool(config.get("inter_layer_normalization"))
        == expected_inter_layer_normalization,
        "common_first_layer_input_normalization": bool(
            config.get("first_layer_input_normalization")
        ),
        "common_normalization_epsilon": math.isclose(
            float(config.get("normalization_epsilon", float("nan"))),
            NORMALIZATION_EPSILON,
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        "validation_selection": config.get("early_stopping_monitor")
        == "validation accuracy",
        "restore_best": bool(config.get("restore_best_checkpoint")),
    }
    failed = [name for name, passed in requirements.items() if not passed]
    if failed:
        raise ValueError(f"Backbone does not match reconstruction protocol ({failed}): {path}")
    model = GoodnessMLP(
        28 * 28,
        hidden_dims=EXPECTED_HIDDEN_DIMS,
        normalize=expected_inter_layer_normalization,
        normalize_first_layer_input=True,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, config, result


@torch.no_grad()
def extract_representations(
    model: GoodnessMLP,
    loader: DataLoader,
    *,
    device: torch.device,
    encode_true_label: bool,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Extract the representation available to the following layer.

    Both conditions L2-normalize the label-encoded input before layer 1. For
    the normalized condition, layers 1--4 expose the post-activation vector
    after the same L2 operation used at the next layer's input. The final-layer
    vector is left pre-normalized because the trained network applies no
    subsequent inter-layer operation there.
    """

    per_layer: list[list[torch.Tensor]] = [[] for _ in model.layers]
    targets: list[torch.Tensor] = []
    for normalized, labels, raw in loader:
        normalized = normalized.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        activations = (
            mark_inputs(normalized, labels, 10) if encode_true_label else normalized
        )
        for layer_index, layer in enumerate(model.layers):
            layer_inputs = activations
            if model.normalizes_layer_input(layer_index):
                denominator = layer_inputs.norm(p=2, dim=1, keepdim=True)
                layer_inputs = layer_inputs / (
                    denominator + model.normalization_epsilon
                )
            activations = F.relu(layer(layer_inputs))
            representation = activations
            if (
                layer_index < len(model.layers) - 1
                and model.normalizes_layer_input(layer_index + 1)
            ):
                denominator = activations.norm(p=2, dim=1, keepdim=True)
                representation = activations / (
                    denominator + model.normalization_epsilon
                )
            per_layer[layer_index].append(representation.cpu())
        targets.append(raw.cpu())
    return [torch.cat(chunks) for chunks in per_layer], torch.cat(targets)


@torch.no_grad()
def reconstruction_mse(
    decoder: Decoder,
    representations: torch.Tensor,
    targets: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> float:
    """Return mean squared error per raw input pixel."""

    decoder.eval()
    loader = DataLoader(
        TensorDataset(representations, targets),
        batch_size=batch_size,
        shuffle=False,
    )
    squared_error = 0.0
    elements = 0
    for features, images in loader:
        features = features.to(device, non_blocking=device.type == "cuda")
        images = images.to(device, non_blocking=device.type == "cuda")
        prediction = decoder(features)
        squared_error += F.mse_loss(prediction, images, reduction="sum").item()
        elements += images.numel()
    value = squared_error / elements
    if not math.isfinite(value):
        raise FloatingPointError(f"Non-finite reconstruction MSE: {value}")
    return value


def train_decoder(
    train_representations: torch.Tensor,
    train_targets: torch.Tensor,
    validation_representations: torch.Tensor,
    validation_targets: torch.Tensor,
    *,
    device: torch.device,
    maximum_epochs: int,
    patience: int,
    batch_size: int,
    evaluation_batch_size: int,
    learning_rate: float,
    seed: int,
    checkpoint_path: Path,
    audit_context: dict[str, Any] | None = None,
) -> tuple[Decoder, dict[str, Any], list[dict[str, Any]]]:
    """Fit a decoder and restore the strict best-validation-MSE checkpoint."""

    set_seed(seed)
    decoder = Decoder(train_representations.size(1)).to(device)
    optimizer = Adam(
        decoder.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    loader = DataLoader(
        TensorDataset(train_representations, train_targets),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    best_validation_mse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    early_stopped = False
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, maximum_epochs + 1):
        decoder.train()
        total_loss = 0.0
        observations = 0
        for features, images in loader:
            features = features.to(device, non_blocking=device.type == "cuda")
            images = images.to(device, non_blocking=device.type == "cuda")
            loss = F.mse_loss(decoder(features), images)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite decoder loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            observations += images.size(0)
        validation_mse = reconstruction_mse(
            decoder,
            validation_representations,
            validation_targets,
            device=device,
            batch_size=evaluation_batch_size,
        )
        improved = validation_mse < best_validation_mse
        if improved:
            best_validation_mse = validation_mse
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in decoder.state_dict().items()
            }
        else:
            epochs_without_improvement += 1
        train_mse = total_loss / observations
        if not math.isfinite(train_mse):
            raise FloatingPointError(f"Non-finite decoder train MSE at epoch {epoch}")
        history.append(
            {
                "epoch": epoch,
                "train_mse": train_mse,
                "validation_mse": validation_mse,
                "best_validation_mse": best_validation_mse,
                "improved": improved,
                "epochs_without_improvement": epochs_without_improvement,
            }
        )
        if epochs_without_improvement >= patience:
            early_stopped = True
            break
    if best_epoch == 0 or best_state is None:
        raise RuntimeError("Decoder did not produce a validation-selected checkpoint")
    decoder.load_state_dict(best_state, strict=True)
    restored_best_state_verified = state_dicts_equal(decoder.state_dict(), best_state)
    if not restored_best_state_verified:
        raise RuntimeError("Decoder best state was not restored exactly")
    history_path = checkpoint_path.with_name("history.csv")
    atomic_csv(
        history_path,
        history,
        (
            "epoch",
            "train_mse",
            "validation_mse",
            "best_validation_mse",
            "improved",
            "epochs_without_improvement",
        ),
    )
    history_sha256 = file_digest(history_path, "sha256")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
    checkpoint_payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "state_dict": best_state,
        "best_epoch": best_epoch,
        "best_validation_mse": best_validation_mse,
        "seed": seed,
        "representation_dim": train_representations.size(1),
        "maximum_epochs": maximum_epochs,
        "patience": patience,
        "terminal_bad_epochs": epochs_without_improvement,
        "history_sha256": history_sha256,
        "source_sha256": source_sha256(),
        "audit_context": audit_context or {},
    }
    torch.save(checkpoint_payload, temporary)
    os.replace(temporary, checkpoint_path)
    checkpoint = load_torch_payload(checkpoint_path)
    checkpoint_reload_verified = state_dicts_equal(
        checkpoint["state_dict"], best_state
    )
    if not checkpoint_reload_verified:
        raise RuntimeError("Reloaded decoder checkpoint differs from the best state")
    selection = {
        "best_epoch": best_epoch,
        "epochs_trained": len(history),
        "maximum_epochs": maximum_epochs,
        "patience": patience,
        "early_stopped": early_stopped,
        "best_validation_mse": best_validation_mse,
        "tie_policy": "ties count as non-improvement",
        "terminal_bad_epochs": epochs_without_improvement,
        "history_sha256": history_sha256,
        "restored_best_state_verified": restored_best_state_verified,
        "checkpoint_reload_verified": checkpoint_reload_verified,
    }
    return decoder, selection, history


def make_split_loaders(
    dataset: ReconstructionDataset,
    test_dataset: ReconstructionDataset,
    *,
    seed: int,
    validation_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Any]]:
    training_indices, validation_indices = stratified_split_indices(
        dataset,
        seed=seed,
        num_classes=10,
        validation_size=validation_size,
    )
    common = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "persistent_workers": num_workers > 0,
    }
    split = {
        "scheme": "seeded stratified holdout from the original training split",
        "seed": seed,
        "training_size": len(training_indices),
        "validation_size": len(validation_indices),
        "validation_per_class": validation_size // 10,
        "validation_index_sha256": hashlib.sha256(
            np.asarray(validation_indices, dtype=np.int64).tobytes()
        ).hexdigest(),
    }
    return (
        DataLoader(Subset(dataset, training_indices), **common),
        DataLoader(Subset(dataset, validation_indices), **common),
        DataLoader(test_dataset, **common),
        split,
    )


def probe_configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "condition": args.condition,
        "condition_label": CONDITION_LABELS[args.condition],
        "seed": args.seed,
        "dataset": "fashionmnist",
        "hidden_dims": list(EXPECTED_HIDDEN_DIMS),
        "representation_input": args.representation_input,
        "representation_point": (
            "post-ReLU activation after inter-layer L2 for layers 1..L-1; "
            "raw post-ReLU activation at layer L"
            if args.condition == "normalized"
            else "raw post-ReLU activation at every layer"
        ),
        "first_layer_input_normalization": True,
        "inter_layer_normalization": args.condition == "normalized",
        "normalization_scope": (
            "label-encoded input before layer 1 and every hidden-to-hidden input"
            if args.condition == "normalized"
            else "label-encoded input before layer 1 only"
        ),
        "decoder": ["linear", 500, "relu", "linear", 784, "sigmoid"],
        "decoder_optimizer": "adam",
        "decoder_learning_rate": args.learning_rate,
        "decoder_batch_size": args.batch_size,
        "decoder_evaluation_batch_size": args.evaluation_batch_size,
        "representation_extraction_batch_size": args.extraction_batch_size,
        "decoder_maximum_epochs": args.epochs,
        "decoder_early_stopping_patience": args.patience,
        "decoder_selection": "strict minimum validation per-pixel MSE",
        "decoder_weight_decay": 0.0,
        "decoder_dropout": 0.0,
        "validation_size": args.validation_size,
        "num_workers": args.num_workers,
        "requested_device": args.device,
        "data_dir": str(args.data_dir.resolve()),
        "download": args.download,
        "precision": "float32",
        "target_scaling": "raw input pixels in [0, 1]",
        "test_policy": "once per decoder after best validation checkpoint restoration",
    }


def probe_run_signature(
    config: dict[str, Any], backbone_checkpoint: Path
) -> str:
    return canonical_sha256(
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "config": config,
            "source_sha256": source_sha256(),
            "backbone_checkpoint": str(backbone_checkpoint.resolve()),
            "backbone_checkpoint_sha256": file_digest(backbone_checkpoint, "sha256"),
        }
    )


def _csv_history(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_completed_run(
    result_path: Path,
    *,
    expected_config: dict[str, Any],
    expected_backbone_checkpoint: Path,
) -> dict[str, Any]:
    """Verify a completed probe strongly enough to permit safe reuse."""

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    expected_signature = probe_run_signature(
        expected_config, expected_backbone_checkpoint
    )
    expected = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete",
        "config": expected_config,
        "source_sha256": source_sha256(),
        "run_signature": expected_signature,
        "test_evaluations_per_decoder": 1,
    }
    mismatches = [key for key, value in expected.items() if payload.get(key) != value]
    if mismatches:
        raise ValueError(f"Stale or incompatible probe result ({mismatches}): {result_path}")
    config_path = result_path.with_name("config.json")
    if (
        not config_path.is_file()
        or file_digest(config_path, "sha256") != payload.get("config_sha256")
        or json.loads(config_path.read_text(encoding="utf-8"))
        != {
            "config": expected_config,
            "run_signature": expected_signature,
            "source_sha256": source_sha256(),
        }
    ):
        raise ValueError(f"Probe config artifact mismatch: {config_path}")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_HIDDEN_DIMS):
        raise ValueError(f"Unexpected decoder rows: {result_path}")
    for expected_layer, row in enumerate(rows, start=1):
        if int(row.get("layer", 0)) != expected_layer:
            raise ValueError(f"Unexpected layer order: {result_path}")
        if int(row.get("test_evaluations", 0)) != 1:
            raise ValueError(f"Unexpected test-evaluation count: {result_path}")
        finite_fields = (
            "test_mse",
            "best_validation_mse",
        )
        if not all(math.isfinite(float(row[field])) for field in finite_fields):
            raise ValueError(f"Non-finite decoder result: {result_path}")
        history_path = result_path.parent / row["decoder_history"]
        if (
            not history_path.is_file()
            or file_digest(history_path, "sha256")
            != row.get("decoder_history_sha256")
        ):
            raise ValueError(f"Decoder history checksum mismatch: {history_path}")
        history = _csv_history(history_path)
        if len(history) != int(row["epochs_trained"]):
            raise ValueError(f"Decoder history length mismatch: {history_path}")
        best = float("inf")
        best_epoch = 0
        bad_epochs = 0
        for epoch, item in enumerate(history, start=1):
            if int(item["epoch"]) != epoch:
                raise ValueError(f"Non-contiguous decoder history: {history_path}")
            train_mse = float(item["train_mse"])
            validation_mse = float(item["validation_mse"])
            stored_best = float(item["best_validation_mse"])
            if not all(math.isfinite(value) for value in (train_mse, validation_mse, stored_best)):
                raise ValueError(f"Non-finite decoder history: {history_path}")
            improved = validation_mse < best
            if (item["improved"].lower() == "true") != improved:
                raise ValueError(f"Strict-improvement mismatch: {history_path}")
            if improved:
                best = validation_mse
                best_epoch = epoch
                bad_epochs = 0
            else:
                bad_epochs += 1
            if int(item["epochs_without_improvement"]) != bad_epochs or not math.isclose(
                stored_best, best, rel_tol=0.0, abs_tol=1e-15
            ):
                raise ValueError(f"Decoder patience/best history mismatch: {history_path}")
            if bad_epochs == int(
                expected_config["decoder_early_stopping_patience"]
            ) and epoch != len(history):
                raise ValueError(f"Decoder continued beyond patience: {history_path}")
        if best_epoch != int(row["best_epoch"]) or not math.isclose(
            best, float(row["best_validation_mse"]), rel_tol=0.0, abs_tol=1e-15
        ):
            raise ValueError(f"Decoder selection mismatch: {history_path}")
        if bad_epochs != int(row["terminal_bad_epochs"]):
            raise ValueError(f"Decoder terminal patience mismatch: {history_path}")
        early_stopped = bool(row["early_stopped"])
        maximum_epochs = int(expected_config["decoder_maximum_epochs"])
        patience = int(expected_config["decoder_early_stopping_patience"])
        if early_stopped and bad_epochs != patience:
            raise ValueError(f"Invalid decoder early stop: {history_path}")
        if not early_stopped and len(history) != maximum_epochs:
            raise ValueError(f"Unexplained decoder termination: {history_path}")
        if not early_stopped and bad_epochs >= patience:
            raise ValueError(f"Missing decoder patience termination: {history_path}")
        if not bool(row.get("restored_best_state_verified")) or not bool(
            row.get("checkpoint_reload_verified")
        ):
            raise ValueError(f"Decoder restoration was not verified: {result_path}")
        checkpoint_path = result_path.parent / row["decoder_checkpoint"]
        if (
            not checkpoint_path.is_file()
            or file_digest(checkpoint_path, "sha256")
            != row["decoder_checkpoint_sha256"]
        ):
            raise ValueError(f"Decoder checkpoint checksum mismatch: {checkpoint_path}")
        checkpoint = load_torch_payload(checkpoint_path)
        context = checkpoint.get("audit_context")
        if (
            int(checkpoint.get("schema_version", 0)) != RESULT_SCHEMA_VERSION
            or checkpoint.get("source_sha256") != source_sha256()
            or checkpoint.get("history_sha256") != row["decoder_history_sha256"]
            or int(checkpoint.get("best_epoch", 0)) != best_epoch
            or not math.isclose(
                float(checkpoint.get("best_validation_mse", float("nan"))),
                best,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or int(checkpoint.get("maximum_epochs", 0)) != maximum_epochs
            or int(checkpoint.get("patience", 0)) != patience
            or int(checkpoint.get("terminal_bad_epochs", -1)) != bad_epochs
            or int(checkpoint.get("seed", -1)) != int(expected_config["seed"])
            or int(checkpoint.get("representation_dim", 0))
            != int(expected_config["hidden_dims"][expected_layer - 1])
            or context
            != {"run_signature": expected_signature, "layer": expected_layer}
            or not isinstance(checkpoint.get("state_dict"), dict)
        ):
            raise ValueError(f"Decoder checkpoint metadata mismatch: {checkpoint_path}")
        decoder = Decoder(int(checkpoint["representation_dim"]))
        decoder.load_state_dict(checkpoint["state_dict"], strict=True)
    reconstruction_path = result_path.with_name("reconstruction.csv")
    if (
        not reconstruction_path.is_file()
        or file_digest(reconstruction_path, "sha256")
        != payload.get("reconstruction_csv_sha256")
    ):
        raise ValueError(f"Reconstruction CSV checksum mismatch: {reconstruction_path}")
    return payload


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    result_path = args.output_dir / "run.json"
    config = probe_configuration(args)
    backbone, backbone_config, backbone_result = load_backbone(
        args.backbone_checkpoint,
        condition=args.condition,
        seed=args.seed,
        device=device,
    )
    run_signature = probe_run_signature(config, args.backbone_checkpoint)
    if result_path.exists():
        verify_completed_run(
            result_path,
            expected_config=config,
            expected_backbone_checkpoint=args.backbone_checkpoint,
        )
        print(f"REUSE {result_path}", flush=True)
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"Incomplete output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    atomic_json(
        config_path,
        {
            "config": config,
            "run_signature": run_signature,
            "source_sha256": source_sha256(),
        },
    )
    config_sha256 = file_digest(config_path, "sha256")
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    set_seed(args.seed)
    train_dataset = ReconstructionDataset(
        args.data_dir, train=True, download=args.download
    )
    test_dataset = ReconstructionDataset(
        args.data_dir, train=False, download=args.download
    )
    train_loader, validation_loader, test_loader, split = make_split_loaders(
        train_dataset,
        test_dataset,
        seed=args.seed,
        validation_size=args.validation_size,
        batch_size=args.extraction_batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    backbone_split = backbone_result["dataset"]["split"]
    if split["validation_index_sha256"] != backbone_split["validation_index_sha256"]:
        raise RuntimeError("Probe and backbone validation splits do not match")
    encode_true_label = args.representation_input == "true-label"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_features, train_targets = extract_representations(
        backbone, train_loader, device=device, encode_true_label=encode_true_label
    )
    validation_features, validation_targets = extract_representations(
        backbone, validation_loader, device=device, encode_true_label=encode_true_label
    )
    test_features, test_targets = extract_representations(
        backbone, test_loader, device=device, encode_true_label=encode_true_label
    )
    rows: list[dict[str, Any]] = []
    for layer_index, (train_layer, validation_layer, test_layer) in enumerate(
        zip(train_features, validation_features, test_features, strict=True), start=1
    ):
        layer_dir = args.output_dir / f"layer_{layer_index}"
        checkpoint_path = layer_dir / "best_decoder.pt"
        decoder, selection, history = train_decoder(
            train_layer,
            train_targets,
            validation_layer,
            validation_targets,
            device=device,
            maximum_epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            evaluation_batch_size=args.evaluation_batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            checkpoint_path=checkpoint_path,
            audit_context={"run_signature": run_signature, "layer": layer_index},
        )
        del history
        test_evaluations = 0
        test_evaluations += 1
        test_mse = reconstruction_mse(
            decoder,
            test_layer,
            test_targets,
            device=device,
            batch_size=args.evaluation_batch_size,
        )
        if test_evaluations != 1:
            raise RuntimeError("Each decoder must evaluate the test split exactly once")
        rows.append(
            {
                "condition": args.condition,
                "seed": args.seed,
                "layer": layer_index,
                "test_mse": test_mse,
                "best_validation_mse": selection["best_validation_mse"],
                "best_epoch": selection["best_epoch"],
                "epochs_trained": selection["epochs_trained"],
                "early_stopped": selection["early_stopped"],
                "decoder_checkpoint": str(checkpoint_path.relative_to(args.output_dir)),
                "decoder_checkpoint_sha256": file_digest(checkpoint_path, "sha256"),
                "decoder_history": str(
                    checkpoint_path.with_name("history.csv").relative_to(args.output_dir)
                ),
                "decoder_history_sha256": selection["history_sha256"],
                "terminal_bad_epochs": selection["terminal_bad_epochs"],
                "restored_best_state_verified": selection[
                    "restored_best_state_verified"
                ],
                "checkpoint_reload_verified": selection[
                    "checkpoint_reload_verified"
                ],
                "test_evaluations": test_evaluations,
            }
        )
        print(
            f"{args.condition} seed={args.seed} layer={layer_index} "
            f"val_mse={float(selection['best_validation_mse']):.8f} "
            f"test_mse={test_mse:.8f}",
            flush=True,
        )
    reconstruction_path = args.output_dir / "reconstruction.csv"
    atomic_csv(
        reconstruction_path,
        rows,
        tuple(rows[0]),
    )
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete",
        "config": config,
        "config_sha256": config_sha256,
        "source_sha256": source_sha256(),
        "run_signature": run_signature,
        "comparison_scope": (
            "independently trained backbones with identical encoded-input L2 "
            "normalization and with versus without hidden-to-hidden L2 normalization; "
            "the result measures recoverability from learned representations and is "
            "not a direct causal estimate of information discarded by normalization"
        ),
        "dataset": {
            "training_samples": len(train_loader.dataset),
            "validation_samples": len(validation_loader.dataset),
            "test_samples": len(test_loader.dataset),
            "normalization_mean": list(FASHION_MEAN),
            "normalization_std": list(FASHION_STD),
            "augmentation": "none",
            "split": split,
            "source_manifest": dataset_manifest(args.data_dir, "fashionmnist"),
        },
        "backbone": {
            "checkpoint": str(args.backbone_checkpoint),
            "checkpoint_sha256": file_digest(args.backbone_checkpoint, "sha256"),
            "config": backbone_config,
            "selection": backbone_result["selection"],
        },
        "rows": rows,
        "test_evaluations_per_decoder": 1,
        "reconstruction_csv_sha256": file_digest(reconstruction_path, "sha256"),
        "elapsed_seconds": time.perf_counter() - started,
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "software": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "hardware": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "command": [sys.executable, *sys.argv],
    }
    atomic_json(result_path, result)
    verify_completed_run(
        result_path,
        expected_config=config,
        expected_backbone_checkpoint=args.backbone_checkpoint,
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--backbone-checkpoint", type=Path, required=True)
    root.add_argument("--condition", choices=tuple(EXPECTED_METHODS), required=True)
    root.add_argument("--seed", choices=(424, 425, 426), type=int, required=True)
    root.add_argument("--data-dir", type=Path, default=Path("data"))
    root.add_argument("--output-dir", type=Path, required=True)
    root.add_argument("--device", default="cuda")
    root.add_argument("--epochs", type=int, default=40)
    root.add_argument("--patience", type=int, default=10)
    root.add_argument("--batch-size", type=int, default=512)
    root.add_argument("--evaluation-batch-size", type=int, default=1024)
    root.add_argument("--extraction-batch-size", type=int, default=1024)
    root.add_argument("--learning-rate", type=float, default=1e-3)
    root.add_argument("--validation-size", type=int, default=5000)
    root.add_argument("--num-workers", type=int, default=4)
    root.add_argument(
        "--representation-input",
        choices=("true-label", "unencoded"),
        default="true-label",
        help="input used to obtain the supervised FF representation",
    )
    root.add_argument(
        "--download", action=argparse.BooleanOptionalAction, default=False
    )
    return root


def main() -> None:
    args = parser().parse_args()
    for name in (
        "epochs",
        "patience",
        "batch_size",
        "evaluation_batch_size",
        "extraction_batch_size",
        "validation_size",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.validation_size % 10 != 0:
        raise ValueError("--validation-size must be divisible by 10")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    run(args)


if __name__ == "__main__":
    main()
