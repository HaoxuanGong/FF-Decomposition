#!/usr/bin/env python3
"""Run a legacy seven-method exploratory MLP matrix.

This scheduler predates the eleven-variant paper protocol. Use
MLPThresholdSweepScheduler.py and MLPOptimizerSweepScheduler.py for the current
paper experiments.
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
import statistics
import subprocess
import sys
from typing import Any


CORE_METHODS = ("bp", "local-bp", "ff", "ff-ge")
FULL_COMPARISON_METHODS = ("fc-ff", "fc-ff-ge", "fc-nn-ff-ge")
METHODS = CORE_METHODS + FULL_COMPARISON_METHODS
METHOD_LABELS = {
    "bp": "BP",
    "local-bp": "Local BP",
    "ff": "Vanilla FF",
    "ff-ge": "FF + GE",
    "fc-ff": "FC-FF",
    "fc-ff-ge": "FC-FF + GE",
    "fc-nn-ff-ge": "FC-NN-FF + GE",
}
DATASETS = ("mnist", "fashionmnist", "cifar10", "cifar100")
DATASET_LABELS = {
    "mnist": "MNIST",
    "fashionmnist": "F-MNIST",
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
}
SEEDS = (424, 425, 426)
ARCHITECTURES = {
    "mnist": [1000, 1000],
    "fashionmnist": [1000, 1000],
    "cifar10": [2000, 2000, 2000],
    "cifar100": [2000, 2000, 2000],
}
SOURCE_FILES = (
    "decomposition_core.py",
    "MLPBenchmarkSuite.py",
    "BroadMLPBenchmarkScheduler.py",
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


def json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def methods_for(dataset: str) -> tuple[str, ...]:
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if dataset == "cifar100":
        return CORE_METHODS
    return METHODS


def jobs() -> list[tuple[str, str, int]]:
    return [
        (dataset, method, seed)
        for dataset in DATASETS
        for method in methods_for(dataset)
        for seed in SEEDS
    ]


def command_for(
    python_executable: str,
    project_root: Path,
    run_dir: Path,
    dataset: str,
    method: str,
    seed: int,
) -> list[str]:
    if method not in methods_for(dataset):
        raise ValueError(f"Method {method} is not scheduled for {dataset}")
    if seed not in SEEDS:
        raise ValueError(f"Unexpected seed: {seed}")
    return [
        python_executable,
        str(project_root / "MLPBenchmarkSuite.py"),
        "run",
        "--dataset",
        dataset,
        "--method",
        method,
        "--seed",
        str(seed),
        "--hidden-dims",
        *(str(width) for width in ARCHITECTURES[dataset]),
        "--epochs",
        "200",
        "--patience",
        "15",
        "--minimum-epochs",
        "0",
        "--validation-size",
        "5000",
        "--optimizer",
        "adam",
        "--learning-rate",
        "0.001",
        "--batch-size",
        "128",
        "--scheduler",
        "none",
        "--evaluation-batch-size",
        "256",
        "--candidate-chunk",
        "10",
        "--num-workers",
        "4",
        "--data-dir",
        str(project_root / "data"),
        "--output-dir",
        str(run_dir / "results"),
        "--device",
        "cuda",
        "--no-download",
    ]


def expected_protocol() -> dict[str, Any]:
    return {
        "comparison": "broad MLP benchmark",
        "methods": list(METHODS),
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "architectures": ARCHITECTURES,
        "full_comparison_cifar100": "not evaluated",
        "maximum_epochs": 200,
        "validation_size": 5000,
        "validation_split": "seed-specific stratified holdout from the training split",
        "early_stopping_patience": 15,
        "early_stopping_strict_improvement": True,
        "restore_best_checkpoint": True,
        "test_evaluations_per_run": 1,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "batch_size": 128,
        "evaluation_batch_size": 256,
        "scheduler": "none",
        "weight_decay": 0.0,
        "dropout": 0.0,
        "augmentation": "none",
        "precision": "float32",
        "goodness": "mean squared activation",
        "goodness_threshold_for_pairwise_ff": 2.0,
        "normalization_epsilon_for_ff_inputs": 1e-4,
        "test_policy": "once after best validation checkpoint restoration",
    }


def _expected_config(dataset: str, method: str, seed: int) -> dict[str, Any]:
    normalized = method in {"ff", "ff-ge", "fc-ff", "fc-ff-ge"}
    first_input_normalized = normalized or method == "fc-nn-ff-ge"
    return {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "epochs": 200,
        "hidden_dims": ARCHITECTURES[dataset],
        "activation": "relu",
        "optimizer": "adam",
        "learning_rate": 0.001,
        "adam_betas": [0.9, 0.999],
        "adam_epsilon": 1e-8,
        "sgd_momentum": None,
        "sgd_dampening": None,
        "sgd_nesterov": None,
        "batch_size": 128,
        "evaluation_batch_size": 256,
        "scheduler": "none",
        "step_size": None,
        "step_gamma": None,
        "weight_decay": 0.0,
        "dropout": 0.0,
        "precision": "float32",
        "amp": False,
        "early_stopping": True,
        "early_stopping_patience": 15,
        "early_stopping_minimum_epochs": 0,
        "early_stopping_monitor": "validation accuracy",
        "early_stopping_mode": "max",
        "early_stopping_min_delta": 0.0,
        "early_stopping_tie_policy": "ties count as non-improvement",
        "restore_best_checkpoint": True,
        "validation_size": 5000,
        "validation_split": "seeded stratified holdout from the original training split",
        "model_selection": "restore the checkpoint with the highest validation accuracy",
        "test_policy": "one evaluation after restoring the best validation checkpoint",
        "inter_layer_normalization": normalized,
        "first_layer_input_normalization": first_input_normalized,
        "normalization_epsilon": 1e-4 if first_input_normalized else None,
        "goodness_threshold": 2.0 if method in {"ff", "ff-ge"} else None,
        "matched_locality_control": False,
        "matched_loss_placement": None,
        "detach_between_layers": True if method == "local-bp" else None,
        "candidate_chunk": 10,
        "num_workers": 4,
        "download": False,
        "requested_device": "cuda",
        "train_limit": None,
        "test_limit": None,
    }


def _manifest_static_fields(
    python_executable: str,
    project_root: Path,
    run_dir: Path,
) -> dict[str, Any]:
    matrix = jobs()
    return {
        "schema_version": 1,
        "project_root": str(project_root),
        "run_dir": str(run_dir),
        "python": python_executable,
        "protocol": expected_protocol(),
        "source_sha256": {
            name: file_sha256(project_root / name) for name in SOURCE_FILES
        },
        "jobs": [
            {"dataset": dataset, "method": method, "seed": seed}
            for dataset, method, seed in matrix
        ],
    }


def _new_manifest(
    python_executable: str,
    project_root: Path,
    run_dir: Path,
) -> dict[str, Any]:
    manifest = {
        **_manifest_static_fields(python_executable, project_root, run_dir),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest["integrity_sha256"] = json_sha256(manifest)
    return manifest


def _verify_manifest(
    manifest: dict[str, Any],
    python_executable: str,
    project_root: Path,
    run_dir: Path,
) -> None:
    expected = _manifest_static_fields(python_executable, project_root, run_dir)
    allowed_keys = set(expected) | {"started_at_utc", "integrity_sha256"}
    if set(manifest) != allowed_keys:
        raise RuntimeError("Existing manifest has an unexpected schema")
    observed = {key: manifest.get(key) for key in expected}
    if observed != expected:
        raise RuntimeError("Existing manifest does not match the fixed protocol or source")
    unsigned = {key: value for key, value in manifest.items() if key != "integrity_sha256"}
    if manifest.get("integrity_sha256") != json_sha256(unsigned):
        raise RuntimeError("Existing manifest integrity check failed")


def _prepare_manifest(
    path: Path,
    python_executable: str,
    project_root: Path,
    run_dir: Path,
) -> dict[str, Any]:
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        _verify_manifest(manifest, python_executable, project_root, run_dir)
        return manifest
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"Refusing to use a non-empty run directory without a manifest: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = _new_manifest(python_executable, project_root, run_dir)
    atomic_json(path, manifest)
    return manifest


def _load_result(run_dir: Path, dataset: str, method: str, seed: int) -> dict[str, Any]:
    path = run_dir / "results" / dataset / method / f"seed_{seed}" / "run.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or payload.get("test_evaluations") != 1:
        raise RuntimeError(f"Invalid completed result: {path}")
    config = payload.get("config", {})
    expected = _expected_config(dataset, method, seed)
    mismatches = {
        key: {"expected": expected_value, "observed": config.get(key)}
        for key, expected_value in expected.items()
        if config.get(key) != expected_value
    }
    if mismatches:
        raise RuntimeError(f"Fixed-protocol mismatch in {path}: {mismatches}")
    dataset_metadata = payload.get("dataset", {})
    if dataset_metadata.get("augmentation") != "none":
        raise RuntimeError(f"Unexpected data augmentation in {path}")
    split = dataset_metadata.get("split", {})
    if (
        split.get("scheme")
        != "seeded stratified holdout from the original training split"
        or int(split.get("seed", -1)) != seed
        or int(split.get("validation_size", -1)) != 5000
    ):
        raise RuntimeError(f"Validation split mismatch in {path}")
    if not dataset_metadata.get("source_manifest"):
        raise RuntimeError(f"Dataset source manifest is missing in {path}")
    if not math.isfinite(float(payload.get("test", {}).get("accuracy", float("nan")))):
        raise RuntimeError(f"Non-finite test accuracy in {path}")
    return payload


def aggregate(run_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        for dataset in DATASETS:
            if method not in methods_for(dataset):
                continue
            group = [_load_result(run_dir, dataset, method, seed) for seed in SEEDS]
            test_values = [float(item["test"]["accuracy"]) for item in group]
            validation_values = [
                float(item["selection"]["best_validation_accuracy"]) for item in group
            ]
            best_epochs = [int(item["selection"]["best_epoch"]) for item in group]
            rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS[dataset],
                    "seeds": ",".join(map(str, SEEDS)),
                    "test_accuracy_mean": statistics.mean(test_values),
                    "test_accuracy_sample_std": statistics.stdev(test_values),
                    "per_seed_test_accuracy": json.dumps(test_values),
                    "best_validation_accuracy_mean": statistics.mean(validation_values),
                    "best_epoch_mean": statistics.mean(best_epochs),
                }
            )
    summary = {
        "schema_version": 1,
        "completed_runs": len(rows) * len(SEEDS),
        "expected_runs": len(jobs()),
        "methods": list(METHODS),
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "rows": rows,
    }
    atomic_json(run_dir / "broad_mlp_summary.json", summary)
    with (run_dir / "broad_mlp_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    project_root = args.project_root.resolve()
    suite_path = project_root / "MLPBenchmarkSuite.py"
    if not suite_path.is_file():
        raise FileNotFoundError(f"MLP benchmark runner not found: {suite_path}")
    manifest = _prepare_manifest(
        run_dir / "manifest.json",
        args.python,
        project_root,
        run_dir,
    )
    (run_dir / "logs").mkdir(exist_ok=True)

    matrix = jobs()
    completed = 0
    failures: list[dict[str, Any]] = []
    for index, (dataset, method, seed) in enumerate(matrix, start=1):
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
        log_path = run_dir / "logs" / f"{index:03d}_{dataset}_{method}_seed{seed}.log"
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

    if not failures:
        verification_command = [
            args.python,
            str(suite_path),
            "verify",
            "--results-dir",
            str(run_dir / "results"),
            "--expected-count",
            str(len(matrix)),
            "--load-models",
        ]
        verification_log = run_dir / "logs" / "verification.log"
        with verification_log.open("w", encoding="utf-8") as log:
            verification = subprocess.run(
                verification_command,
                cwd=project_root,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if verification.returncode != 0:
            failures.append(
                {
                    "stage": "verification",
                    "returncode": verification.returncode,
                    "log": str(verification_log),
                }
            )

    summary = None
    if not failures:
        try:
            _verify_manifest(manifest, args.python, project_root, run_dir)
            summary = aggregate(run_dir)
        except Exception as error:  # preserve verification failure in scheduler.done
            failures.append({"stage": "aggregation", "error": str(error)})

    final = {
        "status": "complete" if not failures else "failed",
        "completed": completed,
        "failed": failures,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_integrity_sha256": manifest["integrity_sha256"],
        "summary": summary,
    }
    atomic_json(run_dir / "scheduler.done", final)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
