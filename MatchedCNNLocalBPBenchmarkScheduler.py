#!/usr/bin/env python3
"""Run and verify the fixed 24-run CNN BP/Local-BP benchmark matrix."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
from typing import Any
import uuid

import torch


METHODS = ("bp", "local-bp")
DATASETS = ("mnist", "fashionmnist", "cifar10", "cifar100")
SEEDS = (424, 425, 426)
BACKBONE_WIDTHS = [64, 128, 256, 512]
DATASET_SIZES = {
    "mnist": (55_000, 5_000, 10_000),
    "fashionmnist": (55_000, 5_000, 10_000),
    "cifar10": (45_000, 5_000, 10_000),
    "cifar100": (45_000, 5_000, 10_000),
}
BAD_LOG_PATTERN = re.compile(
    r"traceback|floatingpointerror|non-finite|(?:^|[^a-z])(?:nan|inf)(?:[^a-z]|$)",
    re.IGNORECASE | re.MULTILINE,
)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
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


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older supported PyTorch
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid checkpoint payload in {path}")
    return payload


def parse_utc(value: Any, context: str) -> datetime:
    if not isinstance(value, str):
        raise RuntimeError(f"Missing UTC timestamp for {context}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise RuntimeError(f"Invalid UTC timestamp for {context}: {value!r}") from error
    if parsed.tzinfo is None:
        raise RuntimeError(f"Naive timestamp for {context}: {value!r}")
    return parsed.astimezone(timezone.utc)


def jobs() -> list[tuple[str, str, int]]:
    return [
        (dataset, method, seed)
        for dataset in DATASETS
        for method in METHODS
        for seed in SEEDS
    ]


def job_output_dir(run_dir: Path, dataset: str, method: str, seed: int) -> Path:
    return run_dir / "results" / dataset / method / f"seed_{seed}"


def command_for(
    python_executable: str,
    project_root: Path,
    run_dir: Path,
    dataset: str,
    method: str,
    seed: int,
    provenance_run_id: str,
) -> list[str]:
    return [
        python_executable,
        str(project_root / "LocalBPCNNBenchmark.py"),
        dataset,
        "--method",
        method,
        "--epochs",
        "200",
        "--patience",
        "15",
        "--minimum-delta",
        "0",
        "--validation-size",
        "5000",
        "--optimizer",
        "sgd",
        "--learning-rate",
        "0.1",
        "--batch-size",
        "128",
        "--eval-batch-size",
        "512",
        "--seeds",
        str(seed),
        "--num-workers",
        "4",
        "--data-dir",
        str(project_root / "data"),
        "--output-dir",
        str(job_output_dir(run_dir, dataset, method, seed)),
        "--device",
        "cuda",
        "--no-download",
        "--save-checkpoints",
        "--provenance-run-id",
        provenance_run_id,
    ]


def read_single_csv_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one row in {path}, found {len(rows)}")
    return rows[0]


def require_float(row: dict[str, str], field: str, path: Path) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"Invalid {field!r} in {path}") from error
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite {field!r} in {path}")
    return value


def verify_history(
    path: Path,
    maximum_epochs: int,
    patience: int,
    minimum_delta: float,
    dataset: str,
    method: str,
    seed: int,
    split_hash: str,
) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len(rows) > maximum_epochs:
        raise RuntimeError(f"Invalid epoch count in {path}: {len(rows)}")

    best_validation_accuracy = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    for expected_epoch, row in enumerate(rows, start=1):
        if int(row["epoch"]) != expected_epoch:
            raise RuntimeError(f"Non-contiguous epoch history in {path}")
        identity = (
            row.get("dataset"),
            row.get("method"),
            int(row.get("seed", -1)),
            row.get("split_sha256"),
        )
        if identity != (dataset, method, seed, split_hash):
            raise RuntimeError(f"History identity mismatch in {path}: {identity}")
        for field in (
            "learning_rate",
            "train_loss",
            "train_accuracy",
            "validation_accuracy",
            "best_validation_accuracy",
            "train_seconds",
        ):
            require_float(row, field, path)
        learning_rate = require_float(row, "learning_rate", path)
        expected_learning_rate = 0.05 * (
            1.0 + math.cos(math.pi * (expected_epoch - 1) / maximum_epochs)
        )
        if not math.isclose(
            learning_rate,
            expected_learning_rate,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"Learning-rate history mismatch in {path}, epoch {expected_epoch}")
        train_loss = require_float(row, "train_loss", path)
        train_accuracy = require_float(row, "train_accuracy", path)
        validation_accuracy = require_float(row, "validation_accuracy", path)
        train_seconds = require_float(row, "train_seconds", path)
        if train_loss < 0.0 or train_seconds <= 0.0:
            raise RuntimeError(f"Invalid loss/runtime history in {path}")
        if not 0.0 <= train_accuracy <= 1.0 or not 0.0 <= validation_accuracy <= 1.0:
            raise RuntimeError(f"Accuracy outside [0, 1] in {path}")

        if validation_accuracy > best_validation_accuracy + minimum_delta:
            best_validation_accuracy = validation_accuracy
            best_epoch = expected_epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        recorded_best = require_float(row, "best_validation_accuracy", path)
        if not math.isclose(recorded_best, best_validation_accuracy, abs_tol=1e-15):
            raise RuntimeError(f"Strict-best validation trace mismatch in {path}")
        if int(row["best_epoch"]) != best_epoch:
            raise RuntimeError(f"Best-epoch trace mismatch in {path}")
        if int(row["epochs_without_improvement"]) != epochs_without_improvement:
            raise RuntimeError(f"Patience-counter mismatch in {path}")
        if expected_epoch < len(rows) and epochs_without_improvement >= patience:
            raise RuntimeError(f"Training continued after patience was exhausted in {path}")

        peak_memory = row.get("epoch_peak_gpu_training_memory_bytes", "")
        if peak_memory and int(peak_memory) <= 0:
            raise RuntimeError(f"Invalid peak-memory history in {path}")
        for layer_index in range(1, 5):
            value = row.get(f"train_layer{layer_index}_loss", "")
            if value:
                if require_float(row, f"train_layer{layer_index}_loss", path) < 0.0:
                    raise RuntimeError(f"Negative local loss in {path}")

    early_stopping_triggered = epochs_without_improvement >= patience
    if len(rows) < maximum_epochs and not early_stopping_triggered:
        raise RuntimeError(f"History ended before maximum epochs without exhausting patience: {path}")
    return {
        "epochs_trained": len(rows),
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_validation_accuracy,
        "final_epochs_without_improvement": epochs_without_improvement,
        "early_stopping_triggered": early_stopping_triggered,
        "termination_reason": "patience" if early_stopping_triggered else "maximum_epochs",
    }


def expected_config(
    dataset: str,
    method: str,
    seed: int,
    provenance_run_id: str,
) -> dict[str, Any]:
    return {
        "method": method,
        "dataset": dataset,
        "epochs": 200,
        "patience": 15,
        "minimum_delta": 0.0,
        "validation_size": 5000,
        "optimizer": "sgd",
        "optimizer_parameters": "momentum=0.9,dampening=0,nesterov=false",
        "learning_rate": 0.1,
        "scheduler": "cosine-annealing",
        "batch_size": 128,
        "eval_batch_size": 512,
        "seeds": [seed],
        "device": "cuda",
        "backbone_widths": BACKBONE_WIDTHS,
        "validation_split": "seed-specific-stratified",
        "test_evaluation_policy": "once-after-best-checkpoint-restoration",
        "augmentation": "none",
        "dropout": 0.0,
        "weight_decay": 0.0,
        "local_bp_prediction": "sum-local-logits",
        "precision": "float32",
        "checkpoint_selection": "strict-validation-accuracy-improvement",
        "num_workers": 4,
        "download": False,
        "save_checkpoints": True,
        "provenance_run_id": provenance_run_id,
        "overwrite": False,
    }


def verify_config(
    config: dict[str, Any],
    dataset: str,
    method: str,
    seed: int,
    provenance_run_id: str,
) -> None:
    for key, expected in expected_config(dataset, method, seed, provenance_run_id).items():
        if config.get(key) != expected:
            raise RuntimeError(
                f"Configuration mismatch for {dataset}/{method}/seed {seed}: "
                f"{key}={config.get(key)!r}, expected {expected!r}"
            )


def load_verified_job(
    run_dir: Path,
    dataset: str,
    method: str,
    seed: int,
    provenance_run_id: str,
    expected_source_sha256: dict[str, str],
    scheduler_started_at: datetime,
    project_root: Path,
) -> dict[str, Any]:
    output_dir = job_output_dir(run_dir, dataset, method, seed)
    config_path = output_dir / "config.json"
    result_path = output_dir / "per_seed_results.csv"
    metadata_path = output_dir / "run_metadata.json"
    history_path = output_dir / "history.csv"
    checkpoint_path = output_dir / f"seed_{seed}_best.pt"
    for path in (config_path, result_path, metadata_path, history_path, checkpoint_path):
        if not path.is_file():
            raise RuntimeError(f"Missing required output: {path}")
        modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if modified_at < scheduler_started_at - timedelta(seconds=2):
            raise RuntimeError(f"Output predates this scheduler run: {path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    verify_config(config, dataset, method, seed, provenance_run_id)
    if Path(config.get("data_dir", "")).resolve() != (project_root / "data").resolve():
        raise RuntimeError(f"Data-directory mismatch in {config_path}")
    if Path(config.get("output_dir", "")).resolve() != output_dir.resolve():
        raise RuntimeError(f"Output-directory mismatch in {config_path}")
    result = read_single_csv_row(result_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    identity = (result.get("dataset"), result.get("method"), int(result.get("seed", -1)))
    if identity != (dataset, method, seed):
        raise RuntimeError(f"Result identity mismatch in {result_path}: {identity}")
    if int(result["test_evaluations"]) != 1:
        raise RuntimeError(f"Expected one test evaluation in {result_path}")
    if result.get("best_checkpoint_restored", "").lower() != "true":
        raise RuntimeError(f"Best checkpoint was not restored in {result_path}")
    if result.get("finite", "").lower() != "true":
        raise RuntimeError(f"Run was not marked finite in {result_path}")
    observed_sizes = tuple(
        int(result[field])
        for field in ("train_samples", "validation_samples", "test_samples")
    )
    if observed_sizes != DATASET_SIZES[dataset]:
        raise RuntimeError(
            f"Dataset-size mismatch in {result_path}: {observed_sizes}, "
            f"expected {DATASET_SIZES[dataset]}"
        )
    test_accuracy = require_float(result, "test_accuracy", result_path)
    validation_accuracy = require_float(result, "best_validation_accuracy", result_path)
    if not 0.0 <= test_accuracy <= 1.0 or not 0.0 <= validation_accuracy <= 1.0:
        raise RuntimeError(f"Accuracy outside [0, 1] in {result_path}")
    split_hash = result.get("split_sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", split_hash):
        raise RuntimeError(f"Invalid validation split hash in {result_path}")
    history_evidence = verify_history(
        history_path,
        maximum_epochs=200,
        patience=15,
        minimum_delta=0.0,
        dataset=dataset,
        method=method,
        seed=seed,
        split_hash=split_hash,
    )
    epochs_trained = int(history_evidence["epochs_trained"])
    if int(result["epochs_trained"]) != epochs_trained:
        raise RuntimeError(f"History length mismatch in {result_path}")
    if int(result["validation_evaluations"]) != epochs_trained:
        raise RuntimeError(f"Validation-evaluation count mismatch in {result_path}")
    if int(result["best_epoch"]) != history_evidence["best_epoch"]:
        raise RuntimeError(f"Selected epoch is not the strict history best in {result_path}")
    if not math.isclose(
        validation_accuracy,
        float(history_evidence["best_validation_accuracy"]),
        abs_tol=1e-15,
    ):
        raise RuntimeError(f"Selected validation accuracy is not the history best in {result_path}")
    if int(result["final_epochs_without_improvement"]) != history_evidence[
        "final_epochs_without_improvement"
    ]:
        raise RuntimeError(f"Final patience counter mismatch in {result_path}")
    if result.get("early_stopping_triggered", "").lower() != str(
        history_evidence["early_stopping_triggered"]
    ).lower():
        raise RuntimeError(f"Early-stopping marker mismatch in {result_path}")
    if result.get("termination_reason") != history_evidence["termination_reason"]:
        raise RuntimeError(f"Termination reason mismatch in {result_path}")

    expected_checkpoint_file = checkpoint_path.name
    if result.get("checkpoint_file") != expected_checkpoint_file:
        raise RuntimeError(f"Checkpoint filename mismatch in {result_path}")
    checkpoint_hash = file_sha256(checkpoint_path)
    if result.get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError(f"Checkpoint file hash mismatch in {result_path}")
    if int(result["checkpoint_bytes"]) != checkpoint_path.stat().st_size:
        raise RuntimeError(f"Checkpoint size mismatch in {result_path}")
    model_state_hash = result.get("model_state_sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", model_state_hash):
        raise RuntimeError(f"Invalid model-state hash in {result_path}")
    if result.get("restored_state_sha256") != model_state_hash:
        raise RuntimeError(f"Restored-state hash mismatch in {result_path}")
    if result.get("provenance_run_id") != provenance_run_id:
        raise RuntimeError(f"Provenance run ID mismatch in {result_path}")

    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint_identity = (
        checkpoint.get("dataset"),
        checkpoint.get("method"),
        int(checkpoint.get("seed", -1)),
    )
    if checkpoint_identity != (dataset, method, seed):
        raise RuntimeError(f"Checkpoint identity mismatch in {checkpoint_path}")
    if checkpoint.get("provenance_run_id") != provenance_run_id:
        raise RuntimeError(f"Checkpoint provenance mismatch in {checkpoint_path}")
    if checkpoint.get("config") != config:
        raise RuntimeError(f"Checkpoint/config mismatch in {checkpoint_path}")
    if checkpoint.get("split_sha256") != split_hash:
        raise RuntimeError(f"Checkpoint split mismatch in {checkpoint_path}")
    if int(checkpoint.get("best_epoch", -1)) != history_evidence["best_epoch"]:
        raise RuntimeError(f"Checkpoint epoch mismatch in {checkpoint_path}")
    if not math.isclose(
        float(checkpoint.get("best_validation_accuracy", math.nan)),
        validation_accuracy,
        abs_tol=1e-15,
    ):
        raise RuntimeError(f"Checkpoint validation metric mismatch in {checkpoint_path}")
    if "test_accuracy" in checkpoint or "test_evaluations" in checkpoint:
        raise RuntimeError(f"Checkpoint improperly contains test-derived fields: {checkpoint_path}")
    checkpoint_state = checkpoint.get("model_state_dict")
    if not isinstance(checkpoint_state, dict) or not checkpoint_state:
        raise RuntimeError(f"Checkpoint state is missing in {checkpoint_path}")
    if any(
        not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all().item()
        for tensor in checkpoint_state.values()
    ):
        raise RuntimeError(f"Checkpoint contains invalid/non-finite tensors: {checkpoint_path}")
    if checkpoint.get("model_state_sha256") != model_state_hash:
        raise RuntimeError(f"Checkpoint state-hash marker mismatch in {checkpoint_path}")
    if state_dict_sha256(checkpoint_state) != model_state_hash:
        raise RuntimeError(f"Checkpoint tensor-state hash mismatch in {checkpoint_path}")

    if metadata.get("status") != "complete":
        raise RuntimeError(f"Incomplete metadata in {metadata_path}")
    if int(metadata.get("test_evaluations_total", 0)) != 1:
        raise RuntimeError(f"Test-evaluation count mismatch in {metadata_path}")
    if metadata.get("nonfinite_values_detected") is not False:
        raise RuntimeError(f"Non-finite marker in {metadata_path}")
    if metadata.get("best_checkpoints_restored") is not True:
        raise RuntimeError(f"Checkpoint-restoration marker missing in {metadata_path}")
    wall_seconds = float(metadata.get("wall_seconds", 0.0))
    if not math.isfinite(wall_seconds) or wall_seconds <= 0.0:
        raise RuntimeError(f"Invalid runtime in {metadata_path}")
    if metadata.get("split_sha256_by_seed", {}).get(str(seed)) != split_hash:
        raise RuntimeError(f"Split-hash metadata mismatch in {metadata_path}")
    if metadata.get("checkpoint_sha256_by_seed", {}).get(str(seed)) != checkpoint_hash:
        raise RuntimeError(f"Checkpoint-hash metadata mismatch in {metadata_path}")
    if metadata.get("model_state_sha256_by_seed", {}).get(str(seed)) != model_state_hash:
        raise RuntimeError(f"Model-state metadata mismatch in {metadata_path}")
    if metadata.get("config") != config:
        raise RuntimeError(f"Metadata/config mismatch in {metadata_path}")
    if metadata.get("provenance_run_id") != provenance_run_id:
        raise RuntimeError(f"Metadata provenance mismatch in {metadata_path}")
    process_instance_id = metadata.get("process_instance_id", "")
    if not re.fullmatch(r"[0-9a-f]{32}", process_instance_id):
        raise RuntimeError(f"Invalid process-instance ID in {metadata_path}")
    if metadata.get("source_sha256") != expected_source_sha256:
        raise RuntimeError(f"Source-hash metadata mismatch in {metadata_path}")
    metadata_started_at = parse_utc(metadata.get("started_at_utc"), str(metadata_path))
    metadata_finished_at = parse_utc(metadata.get("finished_at_utc"), str(metadata_path))
    seed_started_at = parse_utc(result.get("seed_started_at_utc"), str(result_path))
    seed_finished_at = parse_utc(result.get("seed_finished_at_utc"), str(result_path))
    now = datetime.now(timezone.utc) + timedelta(minutes=5)
    if not (
        scheduler_started_at <= metadata_started_at <= seed_started_at
        <= seed_finished_at <= metadata_finished_at <= now
    ):
        raise RuntimeError(f"Fresh-run timestamp ordering failed for {output_dir}")
    environment = metadata.get("environment", {})
    if environment.get("device") != "cuda" or environment.get("precision") != "float32":
        raise RuntimeError(f"Device/precision metadata mismatch in {metadata_path}")
    if not environment.get("gpu", {}).get("name"):
        raise RuntimeError(f"GPU metadata missing in {metadata_path}")
    seed_wall_seconds = require_float(result, "seed_wall_seconds", result_path)
    mean_train_seconds = require_float(result, "mean_train_epoch_seconds", result_path)
    peak_memory = require_float(result, "peak_gpu_training_memory_bytes", result_path)
    if seed_wall_seconds <= 0.0 or mean_train_seconds <= 0.0 or peak_memory <= 0.0:
        raise RuntimeError(f"Invalid runtime/memory summary in {result_path}")

    return {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "config": config,
        "split_sha256": split_hash,
        "test_accuracy": test_accuracy,
        "best_validation_accuracy": validation_accuracy,
        "best_epoch": int(result["best_epoch"]),
        "epochs_trained": epochs_trained,
        "checkpoint_sha256": checkpoint_hash,
        "model_state_sha256": model_state_hash,
        "early_stopping_triggered": history_evidence["early_stopping_triggered"],
        "process_instance_id": process_instance_id,
        "test_evaluations": 1,
        "wall_seconds": wall_seconds,
        "hardware": environment,
        "finite": True,
    }


def comparable_config(config: dict[str, Any]) -> dict[str, Any]:
    comparable = dict(config)
    comparable.pop("method")
    comparable.pop("output_dir")
    return comparable


def verify_logs(run_dir: Path) -> None:
    for path in sorted((run_dir / "logs").glob("*.log")):
        text = path.read_text(encoding="utf-8", errors="replace")
        match = BAD_LOG_PATTERN.search(text)
        if match:
            raise RuntimeError(f"Error/non-finite marker {match.group(0)!r} in {path}")


def aggregate_and_verify(run_dir: Path, project_root: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise RuntimeError(f"Unsupported manifest schema in {manifest_path}")
    if Path(manifest.get("project_root", "")).resolve() != project_root.resolve():
        raise RuntimeError(f"Project-root mismatch in {manifest_path}")
    if Path(manifest.get("run_dir", "")).resolve() != run_dir.resolve():
        raise RuntimeError(f"Run-directory mismatch in {manifest_path}")
    expected_jobs = [
        {"dataset": dataset, "method": method, "seed": seed}
        for dataset, method, seed in jobs()
    ]
    if manifest.get("jobs") != expected_jobs:
        raise RuntimeError(f"Job matrix mismatch in {manifest_path}")
    provenance_run_id = manifest.get("provenance_run_id")
    if not isinstance(provenance_run_id, str) or not re.fullmatch(
        r"[0-9a-f]{32}", provenance_run_id
    ):
        raise RuntimeError(f"Missing provenance run ID in {manifest_path}")
    scheduler_started_at = parse_utc(manifest.get("started_at_utc"), str(manifest_path))
    expected_source_sha256 = manifest.get("source_sha256")
    if not isinstance(expected_source_sha256, dict):
        raise RuntimeError(f"Missing source hashes in {manifest_path}")
    if any(not re.fullmatch(r"[0-9a-f]{64}", digest) for digest in expected_source_sha256.values()):
        raise RuntimeError(f"Invalid source hash in {manifest_path}")
    current_source_sha256 = {
        name: file_sha256(project_root / name)
        for name in ("LocalBPCNNBenchmark.py", "MatchedCNNLocalBPBenchmarkScheduler.py")
    }
    if expected_source_sha256 != current_source_sha256:
        raise RuntimeError("Benchmark source changed after the scheduler manifest was created")

    verified: dict[tuple[str, str, int], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for dataset, method, seed in jobs():
        verified[(dataset, method, seed)] = load_verified_job(
            run_dir,
            dataset,
            method,
            seed,
            provenance_run_id,
            expected_source_sha256,
            scheduler_started_at,
            project_root,
        )
    process_instance_ids = [item["process_instance_id"] for item in verified.values()]
    if len(set(process_instance_ids)) != len(process_instance_ids):
        raise RuntimeError("A process-instance ID was reused across scheduled jobs")
    verify_logs(run_dir)

    for dataset in DATASETS:
        for method in METHODS:
            group = [verified[(dataset, method, seed)] for seed in SEEDS]
            test_values = [float(item["test_accuracy"]) for item in group]
            validation_values = [
                float(item["best_validation_accuracy"]) for item in group
            ]
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "seeds": ",".join(map(str, SEEDS)),
                    "test_accuracy_mean": statistics.mean(test_values),
                    "test_accuracy_sample_std": statistics.stdev(test_values),
                    "best_validation_accuracy_mean": statistics.mean(validation_values),
                    "best_epoch_mean": statistics.mean(
                        int(item["best_epoch"]) for item in group
                    ),
                    "wall_seconds_total": sum(float(item["wall_seconds"]) for item in group),
                    "per_seed_test_accuracy": json.dumps(test_values),
                }
            )

    paired_deltas = []
    for dataset in DATASETS:
        deltas = []
        for seed in SEEDS:
            bp = verified[(dataset, "bp", seed)]
            local = verified[(dataset, "local-bp", seed)]
            if comparable_config(bp["config"]) != comparable_config(local["config"]):
                raise RuntimeError(f"Unmatched training configuration for {dataset}, seed {seed}")
            if bp["split_sha256"] != local["split_sha256"]:
                raise RuntimeError(f"Unmatched validation split for {dataset}, seed {seed}")
            if bp["config"]["backbone_widths"] != local["config"]["backbone_widths"]:
                raise RuntimeError(f"Unmatched backbone for {dataset}, seed {seed}")
            deltas.append(float(local["test_accuracy"]) - float(bp["test_accuracy"]))
        paired_deltas.append(
            {
                "dataset": dataset,
                "local_minus_bp_per_seed": deltas,
                "local_minus_bp_mean": statistics.mean(deltas),
                "local_minus_bp_sample_std": statistics.stdev(deltas),
            }
        )

    csv_path = run_dir / "matched_cnn_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": 2,
        "provenance_run_id": provenance_run_id,
        "source_sha256": expected_source_sha256,
        "comparison_scope": (
            "same Conv-BN-ReLU-MaxPool backbone and training protocol; BP uses one "
            "terminal head whereas Local BP uses detached per-block heads"
        ),
        "methods": list(METHODS),
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "completed_runs": len(verified),
        "rows": rows,
        "paired_deltas": paired_deltas,
        "verification": {
            "fresh_process_per_job": True,
            "all_configs_exact": True,
            "all_split_hashes_paired": True,
            "all_test_evaluations_exactly_one": True,
            "all_best_checkpoints_restored": True,
            "all_checkpoints_persisted_and_hashed": True,
            "all_checkpoint_states_match_strict_validation_bests": True,
            "all_early_stopping_traces_reconstructed": True,
            "all_outputs_fresh_for_provenance_run": True,
            "all_process_instance_ids_unique": True,
            "all_source_hashes_match_manifest": True,
            "all_values_finite": True,
            "error_and_nonfinite_log_scan_empty": True,
            "hardware_and_runtime_metadata_present": True,
        },
    }
    atomic_json(run_dir / "matched_cnn_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    run_started_at_utc = datetime.now(timezone.utc).isoformat()
    run_start = time.perf_counter()
    provenance_run_id = uuid.uuid4().hex
    run_dir = args.run_dir.resolve()
    project_root = args.project_root.resolve()
    if (run_dir / "manifest.json").exists() or (run_dir / "results").exists():
        raise FileExistsError(f"Refusing to reuse non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)

    matrix = jobs()
    manifest = {
        "schema_version": 2,
        "provenance_run_id": provenance_run_id,
        "started_at_utc": run_started_at_utc,
        "project_root": str(project_root),
        "run_dir": str(run_dir),
        "python": args.python,
        "protocol": {
            "comparison_scope": (
                "same four-block Conv-BN-ReLU-MaxPool backbone and fixed training "
                "protocol; method-specific classification heads and credit assignment"
            ),
            "methods": list(METHODS),
            "datasets": list(DATASETS),
            "seeds": list(SEEDS),
            "backbone_widths": BACKBONE_WIDTHS,
            "maximum_epochs": 200,
            "validation_size": 5000,
            "validation_split": "seed-specific stratified official-training holdout",
            "early_stopping_patience": 15,
            "early_stopping_strict_improvement": True,
            "restore_best_checkpoint": True,
            "optimizer": "SGD",
            "learning_rate": 0.1,
            "momentum": 0.9,
            "dampening": 0.0,
            "nesterov": False,
            "batch_size": 128,
            "evaluation_batch_size": 512,
            "scheduler": "CosineAnnealingLR(T_max=200,eta_min=0)",
            "augmentation": "none",
            "dropout": 0.0,
            "weight_decay": 0.0,
            "precision": "float32",
            "test_policy": "exactly once after best validation checkpoint restoration",
            "fresh_process_per_job": True,
            "persist_and_hash_best_checkpoint": True,
        },
        "source_sha256": {
            name: file_sha256(project_root / name)
            for name in (
                "LocalBPCNNBenchmark.py",
                "MatchedCNNLocalBPBenchmarkScheduler.py",
            )
        },
        "jobs": [
            {"dataset": dataset, "method": method, "seed": seed}
            for dataset, method, seed in matrix
        ],
    }
    atomic_json(run_dir / "manifest.json", manifest)

    completed = 0
    failures: list[dict[str, Any]] = []
    for index, (dataset, method, seed) in enumerate(matrix, start=1):
        atomic_json(
            run_dir / "status.json",
            {
                "provenance_run_id": provenance_run_id,
                "total": len(matrix),
                "completed": completed,
                "failed": len(failures),
                "current_index": index,
                "current": {"dataset": dataset, "method": method, "seed": seed},
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        command = command_for(
            args.python,
            project_root,
            run_dir,
            dataset,
            method,
            seed,
            provenance_run_id,
        )
        log_path = run_dir / "logs" / f"{index:02d}_{dataset}_{method}_seed{seed}.log"
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=project_root,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if result.returncode == 0:
            completed += 1
        else:
            failures.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "seed": seed,
                    "returncode": result.returncode,
                    "log": str(log_path),
                }
            )

    summary = None
    if not failures:
        try:
            summary = aggregate_and_verify(run_dir, project_root)
        except Exception as error:  # preserve the full run for diagnosis
            verification_error = run_dir / "logs" / "verification_error.log"
            verification_error.write_text(f"{type(error).__name__}: {error}\n", encoding="utf-8")
            failures.append(
                {
                    "stage": "verification",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "log": str(verification_error),
                }
            )

    final = {
        "schema_version": 2,
        "provenance_run_id": provenance_run_id,
        "status": "complete" if not failures else "failed",
        "completed": completed,
        "expected": len(matrix),
        "failed": failures,
        "started_at_utc": run_started_at_utc,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "scheduler_wall_seconds": time.perf_counter() - run_start,
        "summary": summary,
    }
    atomic_json(run_dir / "scheduler.done", final)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
