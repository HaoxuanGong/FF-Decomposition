#!/usr/bin/env python3
"""Run the reduced BP versus Local BP MLP-Mixer reference benchmark."""

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
import statistics
import subprocess
import sys
from typing import Any

from MLPMixerBenchmarkSuite import (
    RESULT_SCHEMA_VERSION,
    ensure_tiny,
    source_sha256,
    validate_dataset_integrity_record,
    validate_reusable_row,
)


DATASETS = ("cifar10", "cifar100", "pathmnist", "tinyimagenet")
MANIFEST_SCHEMA_VERSION = 2
METHODS = ("bp", "local-bp")
SEEDS = (424, 425, 426)
PATCH_SIZES = {"cifar10": 4, "cifar100": 4, "pathmnist": 7, "tinyimagenet": 8}
PAIR_MATCH_FIELDS = (
    "dataset",
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
    "device",
    "amp_enabled",
    "torch_version",
    "torchvision_version",
    "cuda_runtime_version",
    "cudnn_version",
    "gpu_name",
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


def jobs() -> list[tuple[str, int, str]]:
    """Pair BP and Local BP adjacently for every dataset and seed."""

    return [
        (dataset, seed, method)
        for dataset in DATASETS
        for seed in SEEDS
        for method in METHODS
    ]


def job_dir(run_dir: Path, dataset: str, method: str, seed: int) -> Path:
    return run_dir / "raw" / dataset / method / f"seed_{seed}"


def command_for(
    python_executable: str,
    project_root: Path,
    run_dir: Path,
    dataset: str,
    method: str,
    seed: int,
) -> list[str]:
    output_dir = job_dir(run_dir, dataset, method, seed)
    command = [
        python_executable,
        str(project_root / "MLPMixerBenchmarkSuite.py"),
        "--datasets",
        dataset,
        "--methods",
        method,
        "--depths",
        "5",
        "--dims",
        "256",
        "--seed",
        str(seed),
        "--epochs",
        "200",
        "--early-stop-patience",
        "15",
        "--early-stop-min-delta",
        "0",
        "--validation-fraction",
        "0.1",
        "--batch-size",
        "128",
        "--eval-batch-size",
        "256",
        "--num-workers",
        "4",
        "--optimizer",
        "adamw",
        "--lr",
        "0.0003",
        "--weight-decay",
        "0.05",
        "--momentum",
        "0.9",
        "--token-ratio",
        "1",
        "--channel-ratio",
        "4",
        "--local-bp-updates-per-block",
        "1",
        "--data-dir",
        str(project_root / "data"),
        "--output-dir",
        str(output_dir),
        "--results-csv",
        "result.csv",
        "--summary-csv",
        "summary.csv",
        "--checkpoint-dir",
        str(run_dir / "checkpoints"),
        "--device",
        "cuda",
    ]
    if dataset == "tinyimagenet":
        command.append("--download-tinyimagenet")
    return command


def _single_result_row(run_dir: Path, dataset: str, method: str, seed: int) -> dict[str, str]:
    result_path = job_dir(run_dir, dataset, method, seed) / "result.csv"
    with result_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    successful = [row for row in rows if row.get("status") == "ok"]
    if len(successful) != 1:
        raise RuntimeError(
            f"Expected one successful result row in {result_path}, found {len(successful)}"
        )
    row = successful[0]
    if row.get("test_evaluations") != "1":
        raise RuntimeError(f"Incomplete result row in {result_path}")
    return row


def _as_float(row: dict[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid numeric field {field!r} in result row") from exc
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite field {field!r} in result row")
    return value


def _row_dataset_integrity(row: dict[str, str]) -> dict[str, object]:
    try:
        notes = json.loads(row["notes_json"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Invalid result notes metadata") from exc
    if not isinstance(notes, dict):
        raise RuntimeError("Result notes metadata must be an object")
    try:
        return validate_dataset_integrity_record(
            row.get("dataset", ""), notes.get("dataset_integrity")
        )
    except ValueError as exc:
        raise RuntimeError("Invalid dataset-integrity record") from exc


def _verify_dataset_integrity_against_manifest(
    run_dir: Path, row: dict[str, str]
) -> None:
    """Bind each child result to the dataset fingerprint frozen by the scheduler."""

    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("dataset_integrity", {}).get(row.get("dataset", ""), {})
    if _row_dataset_integrity(row) != expected:
        raise RuntimeError(
            f"Dataset provenance differs from the run manifest for {row.get('dataset')}"
        )


def validate_result(row: dict[str, str], dataset: str, method: str, seed: int) -> None:
    expected = {
        "dataset": dataset,
        "method": method,
        "seed": str(seed),
        "depth": "5",
        "dim": "256",
        "token_dim": "256",
        "channel_dim": "1024",
        "patch_size": str(PATCH_SIZES[dataset]),
        "epochs_target": "200",
        "early_stop_patience": "15",
        "early_stop_min_delta": "0.0",
        "validation_fraction": "0.1",
        "dropout": "0.1",
        "scheduler": "cosine_annealing",
        "scheduler_t_max": "200",
        "optimizer": "adamw",
        "lr": "0.0003",
        "weight_decay": "0.05",
        "batch_size": "128",
        "eval_batch_size": "256",
        "momentum": "0.9",
        "num_workers": "4",
        "device": "cuda",
        "amp_enabled": "1",
    }
    for field, expected_value in expected.items():
        if row.get(field) != expected_value:
            raise RuntimeError(
                f"Protocol mismatch for {dataset}/{method}/seed{seed}: "
                f"{field}={row.get(field)!r}, expected {expected_value!r}"
            )
    expected_updates = "1" if method == "local-bp" else "0"
    if row.get("local_bp_updates_per_block") != expected_updates:
        raise RuntimeError(f"Local-update mismatch for {dataset}/{method}/seed{seed}")
    for digest_field in (
        "train_index_sha256",
        "validation_index_sha256",
        "test_index_sha256",
        "checkpoint_sha256",
    ):
        digest = row.get(digest_field, "")
        if len(digest) != 64:
            raise RuntimeError(f"Missing SHA-256 field {digest_field!r}")
    checkpoint = Path(row["checkpoint_path"])
    if not checkpoint.is_file() or file_sha256(checkpoint) != row["checkpoint_sha256"]:
        raise RuntimeError(f"Checkpoint verification failed: {checkpoint}")
    for field in (
        "test_primary",
        "test_secondary",
        "best_validation_primary",
        "peak_train_mem_gb",
        "runtime_seconds",
    ):
        value = _as_float(row, field)
        if field in {"peak_train_mem_gb", "runtime_seconds"} and value <= 0:
            raise RuntimeError(f"Expected positive {field!r}")
    if not row.get("gpu_name") or not row.get("dataset_version"):
        raise RuntimeError("Hardware or dataset-version metadata is missing")
    try:
        validate_reusable_row(row)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Result provenance/audit verification failed for "
            f"{dataset}/{method}/seed{seed}"
        ) from exc
    dataset_integrity = _row_dataset_integrity(row)
    signature = json.loads(row["run_signature"])
    expected_signature = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "source_sha256": source_sha256(),
        "dataset_integrity": dataset_integrity,
        "dataset": dataset,
        "method": method,
        "depth": 5,
        "dim": 256,
        "token_dim": 256,
        "channel_dim": 1024,
        "patch_size": PATCH_SIZES[dataset],
        "seed": seed,
        "epochs": 200,
        "early_stop_patience": 15,
        "early_stop_min_delta": 0.0,
        "validation_fraction": 0.1,
        "dropout": 0.1,
        "scheduler": "cosine_annealing",
        "scheduler_t_max": 200,
        "batch_size": 128,
        "eval_batch_size": 256,
        "optimizer": "adamw",
        "lr": 0.0003,
        "weight_decay": 0.05,
        "momentum": 0.9,
        "local_bp_updates_per_block": 1 if method == "local-bp" else 0,
        "amp": True,
        "device": "cuda",
        "num_workers": 4,
    }
    if signature != expected_signature:
        raise RuntimeError(
            f"Run-signature mismatch for {dataset}/{method}/seed{seed}"
        )


def _mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def aggregate(run_dir: Path) -> dict[str, Any]:
    indexed: dict[tuple[str, str, int], dict[str, str]] = {}
    for dataset in DATASETS:
        for seed in SEEDS:
            for method in METHODS:
                row = _single_result_row(run_dir, dataset, method, seed)
                validate_result(row, dataset, method, seed)
                _verify_dataset_integrity_against_manifest(run_dir, row)
                indexed[(dataset, method, seed)] = row

    method_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for method in METHODS:
            group = [indexed[(dataset, method, seed)] for seed in SEEDS]
            summary_row: dict[str, Any] = {
                "dataset": dataset,
                "method": method,
                "seeds": ",".join(map(str, SEEDS)),
                "num_runs": len(group),
                "metric_primary_name": group[0]["metric_primary_name"],
                "metric_secondary_name": group[0]["metric_secondary_name"],
            }
            for field in (
                "test_primary",
                "test_secondary",
                "best_validation_primary",
                "peak_train_mem_gb",
                "runtime_seconds",
                "best_epoch",
            ):
                mean, sample_std = _mean_std([_as_float(row, field) for row in group])
                summary_row[f"{field}_mean"] = mean
                summary_row[f"{field}_sample_std"] = sample_std
            summary_row["per_seed_test_primary"] = json.dumps(
                [_as_float(row, "test_primary") for row in group]
            )
            method_rows.append(summary_row)

    paired_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        primary_deltas: list[float] = []
        secondary_deltas: list[float] = []
        memory_ratios: list[float] = []
        for seed in SEEDS:
            bp = indexed[(dataset, "bp", seed)]
            local = indexed[(dataset, "local-bp", seed)]
            for field in PAIR_MATCH_FIELDS:
                if bp.get(field) != local.get(field):
                    raise RuntimeError(
                        f"Unmatched pair for {dataset}, seed {seed}: {field} differs"
                    )
            if _row_dataset_integrity(bp) != _row_dataset_integrity(local):
                raise RuntimeError(
                    f"Unmatched dataset provenance for {dataset}, seed {seed}"
                )
            primary_deltas.append(
                _as_float(local, "test_primary") - _as_float(bp, "test_primary")
            )
            secondary_deltas.append(
                _as_float(local, "test_secondary") - _as_float(bp, "test_secondary")
            )
            memory_ratios.append(
                _as_float(local, "peak_train_mem_gb")
                / _as_float(bp, "peak_train_mem_gb")
            )
        primary_mean, primary_std = _mean_std(primary_deltas)
        secondary_mean, secondary_std = _mean_std(secondary_deltas)
        ratio_mean, ratio_std = _mean_std(memory_ratios)
        paired_rows.append(
            {
                "dataset": dataset,
                "seeds": ",".join(map(str, SEEDS)),
                "local_minus_bp_primary_mean": primary_mean,
                "local_minus_bp_primary_sample_std": primary_std,
                "local_minus_bp_secondary_mean": secondary_mean,
                "local_minus_bp_secondary_sample_std": secondary_std,
                "local_over_bp_memory_mean": ratio_mean,
                "local_over_bp_memory_sample_std": ratio_std,
                "per_seed_primary_deltas": json.dumps(primary_deltas),
                "per_seed_secondary_deltas": json.dumps(secondary_deltas),
                "per_seed_memory_ratios": json.dumps(memory_ratios),
            }
        )

    method_csv = run_dir / "reduced_mixer_summary.csv"
    with method_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(method_rows[0]))
        writer.writeheader()
        writer.writerows(method_rows)
    paired_csv = run_dir / "reduced_mixer_paired.csv"
    with paired_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    summary = {
        "schema_version": 1,
        "completed_runs": len(indexed),
        "datasets": list(DATASETS),
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "method_rows": method_rows,
        "paired_rows": paired_rows,
        "pair_verification": (
            "backbone, data, split, optimization, hardware, and evaluation metadata "
            "match within each seed; Local BP differs by local heads, local objectives, "
            "gradient detachment, and summed-head inference"
        ),
    }
    atomic_json(run_dir / "reduced_mixer_summary.json", summary)
    return summary


def scheduler_environment() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }


def _manifest_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return immutable fields that must match before any output is reused."""

    return {
        key: manifest[key]
        for key in (
            "schema_version",
            "project_root",
            "run_dir",
            "python",
            "scheduler_environment",
            "protocol",
            "source_sha256",
            "dataset_integrity",
            "jobs",
        )
    }


def prepare_run_directory(run_dir: Path, manifest: dict[str, Any]) -> bool:
    """Create a fresh run or prove that an existing run is exactly resumable."""

    if run_dir.exists() and any(run_dir.iterdir()):
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            entries = {path.name for path in run_dir.iterdir()}
            allowed_prelaunch_entries = {
                "source",
                "source_snapshot.sha256",
                "scheduler.log",
            }
            if (
                not entries.issubset(allowed_prelaunch_entries)
                or not (run_dir / "source").is_dir()
                or not (run_dir / "source_snapshot.sha256").is_file()
                or (
                    (run_dir / "scheduler.log").exists()
                    and not (run_dir / "scheduler.log").is_file()
                )
            ):
                raise RuntimeError(
                    f"Refusing non-empty run directory without a manifest: {run_dir}"
                )
            atomic_json(manifest_path, manifest)
            return False
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _manifest_identity(existing) != _manifest_identity(manifest):
            raise RuntimeError(
                f"Existing run source/configuration does not match: {run_dir}"
            )
        return True
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / "manifest.json", manifest)
    return False


def _verified_existing_result(
    run_dir: Path, dataset: str, method: str, seed: int
) -> bool:
    result_path = job_dir(run_dir, dataset, method, seed) / "result.csv"
    if not result_path.exists():
        return False
    with result_path.open("r", newline="", encoding="utf-8") as handle:
        successful = [
            row for row in csv.DictReader(handle) if row.get("status") == "ok"
        ]
    if not successful:
        return False
    if len(successful) != 1:
        raise RuntimeError(f"Ambiguous successful reuse rows: {result_path}")
    row = _single_result_row(run_dir, dataset, method, seed)
    validate_result(row, dataset, method, seed)
    _verify_dataset_integrity_against_manifest(run_dir, row)
    return True


def _next_log_path(base: Path) -> Path:
    if not base.exists():
        return base
    retry = 1
    while True:
        candidate = base.with_name(f"{base.stem}.retry{retry}{base.suffix}")
        if not candidate.exists():
            return candidate
        retry += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    project_root = args.project_root.resolve()
    matrix = jobs()
    _tiny_root, tiny_integrity = ensure_tiny(
        project_root / "data", allow_download=True
    )
    validate_dataset_integrity_record("tinyimagenet", tiny_integrity)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "run_dir": str(run_dir),
        "python": args.python,
        "scheduler_environment": scheduler_environment(),
        "protocol": {
            "purpose": "BP and Local BP reference benchmark; not a causal locality ablation",
            "datasets": list(DATASETS),
            "methods": list(METHODS),
            "seeds": list(SEEDS),
            "architecture": {
                "mixer_blocks": 5,
                "embedding_dimension": 256,
                "token_mlp_hidden_dimension": 256,
                "channel_mlp_hidden_dimension": 1024,
                "patch_sizes": PATCH_SIZES,
            },
            "maximum_epochs": 200,
            "validation": (
                "official MedMNIST validation; otherwise seed-specific stratified "
                "10% training holdout"
            ),
            "early_stopping_patience": 15,
            "early_stopping_min_delta": 0.0,
            "checkpoint_selection": "strict validation-primary improvement",
            "test_policy": "exactly once after best-checkpoint restoration",
            "optimizer": "AdamW",
            "learning_rate": 0.0003,
            "weight_decay": 0.05,
            "dropout": 0.1,
            "batch_size": 128,
            "eval_batch_size": 256,
            "scheduler": "CosineAnnealingLR(T_max=200)",
            "mixed_precision": True,
            "local_bp_updates_per_block": 1,
            "memory_metric": (
                "maximum torch.cuda.max_memory_allocated across training epochs; "
                "one fresh process per run"
            ),
            "final_tinyimagenet_split": "official validation split with labels",
            "dataset_integrity": (
                "owner-published structure validation plus computed archive/tree "
                "SHA-256; no source-published Tiny ImageNet checksum was found"
            ),
            "expected_runs": len(matrix),
        },
        "source_sha256": {
            name: file_sha256(project_root / name)
            for name in (
                "MLPMixerBenchmarkSuite.py",
                "medmnist_support.py",
                "ReducedMixerBenchmarkScheduler.py",
            )
        },
        "dataset_integrity": {"tinyimagenet": tiny_integrity},
        "jobs": [
            {
                "dataset": dataset,
                "method": method,
                "seed": seed,
                "command": command_for(
                    args.python, project_root, run_dir, dataset, method, seed
                ),
            }
            for dataset, seed, method in matrix
        ],
    }
    prepare_run_directory(run_dir, manifest)
    (run_dir / "logs").mkdir(exist_ok=True)

    completed = 0
    failures: list[dict[str, Any]] = []
    for index, (dataset, seed, method) in enumerate(matrix, start=1):
        if _verified_existing_result(run_dir, dataset, method, seed):
            completed += 1
            continue
        atomic_json(
            run_dir / "status.json",
            {
                "total": len(matrix),
                "completed": completed,
                "failed": len(failures),
                "current_index": index,
                "current": {"dataset": dataset, "method": method, "seed": seed},
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        command = command_for(args.python, project_root, run_dir, dataset, method, seed)
        log_path = _next_log_path(
            run_dir / "logs" / f"{index:02d}_{dataset}_{method}_seed{seed}.log"
        )
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
            try:
                row = _single_result_row(run_dir, dataset, method, seed)
                validate_result(row, dataset, method, seed)
                _verify_dataset_integrity_against_manifest(run_dir, row)
                completed += 1
            except Exception as exc:
                failures.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "seed": seed,
                        "stage": "post_run_verification",
                        "error": repr(exc),
                        "log": str(log_path),
                    }
                )
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

    summary: dict[str, Any] | None = None
    if not failures:
        try:
            summary = aggregate(run_dir)
        except Exception as exc:
            failures.append({"stage": "aggregate_and_verify", "error": repr(exc)})
    final = {
        "status": "complete" if not failures else "failed",
        "completed": completed,
        "failed": failures,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
    }
    atomic_json(run_dir / "scheduler.done", final)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
