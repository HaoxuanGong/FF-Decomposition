#!/usr/bin/env python3
"""Run a fixed-protocol inter-layer-normalization ablation for FC-FF+GE."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

from MatchedCELocalityControlScheduler import atomic_json, file_sha256


METHODS = ("fc-ff-ge", "fc-nn-ff-ge")
DATASETS = ("mnist", "fashionmnist", "cifar10")
SEEDS = (424, 425, 426)
ARCHITECTURES = {
    "mnist": [1000, 1000],
    "fashionmnist": [1000, 1000],
    "cifar10": [2000, 2000, 2000],
}
NORMALIZATION_EPSILON = 1e-4


def jobs() -> list[tuple[str, str, int]]:
    return [
        (dataset, method, seed)
        for dataset in DATASETS
        for seed in SEEDS
        for method in METHODS
    ]


def command_for(
    python_executable: str,
    project_root: Path,
    run_dir: Path,
    dataset: str,
    method: str,
    seed: int,
) -> list[str]:
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


def _load_result(run_dir: Path, dataset: str, method: str, seed: int) -> dict[str, Any]:
    path = run_dir / "results" / dataset / method / f"seed_{seed}" / "run.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or payload.get("test_evaluations") != 1:
        raise RuntimeError(f"Invalid completed result: {path}")
    config = payload["config"]
    if (
        config.get("dataset") != dataset
        or config.get("method") != method
        or config.get("seed") != seed
    ):
        raise RuntimeError(f"Result identity mismatch: {path}")
    expected_normalization = method == "fc-ff-ge"
    expected_scope = (
        "input to every FF layer, including the encoded input to layer 1"
        if expected_normalization
        else "encoded input to layer 1 only"
    )
    expected = {
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
        "local_update_schedule": "one end-to-end update per minibatch",
        "inter_layer_normalization": expected_normalization,
        "first_layer_input_normalization": True,
        "normalization_scope": expected_scope,
        "normalization_epsilon": NORMALIZATION_EPSILON,
        "goodness_definition": "mean squared activation",
        "goodness_threshold": None,
        "matched_locality_control": False,
        "matched_loss_placement": None,
        "detach_between_layers": None,
        "prediction": "final-layer goodness",
        "candidate_chunk": 10,
        "num_workers": 4,
        "download": False,
        "requested_device": "cuda",
        "negative_sampling": "all classes",
        "label_encoding": "replace the first C normalized input entries with a 0/1 one-hot label",
        "train_limit": None,
        "test_limit": None,
    }
    mismatches = {
        key: {"expected": expected_value, "observed": config.get(key)}
        for key, expected_value in expected.items()
        if config.get(key) != expected_value
    }
    if mismatches:
        raise RuntimeError(f"Fixed-protocol mismatch in {path}: {mismatches}")
    return payload


def _matched_config(config: dict[str, Any]) -> dict[str, Any]:
    comparable = dict(config)
    for key in ("method", "inter_layer_normalization", "normalization_scope"):
        comparable.pop(key)
    return comparable


def aggregate(run_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for dataset in DATASETS:
        for method in METHODS:
            group = [_load_result(run_dir, dataset, method, seed) for seed in SEEDS]
            values = [float(item["test"]["accuracy"]) for item in group]
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "seeds": ",".join(map(str, SEEDS)),
                    "test_accuracy_mean": statistics.mean(values),
                    "test_accuracy_sample_std": statistics.stdev(values),
                    "per_seed_test_accuracy": json.dumps(values),
                    "best_validation_accuracy_mean": statistics.mean(
                        float(item["selection"]["best_validation_accuracy"])
                        for item in group
                    ),
                }
            )
            grouped.setdefault(dataset, {})[method] = group

    paired_deltas = []
    for dataset in DATASETS:
        normalized = grouped[dataset]["fc-ff-ge"]
        no_inter_layer_normalization = grouped[dataset]["fc-nn-ff-ge"]
        deltas = []
        for seed, norm_item, no_norm_item in zip(
            SEEDS, normalized, no_inter_layer_normalization, strict=True
        ):
            if _matched_config(norm_item["config"]) != _matched_config(
                no_norm_item["config"]
            ):
                raise RuntimeError(f"Unmatched configuration for {dataset}, seed {seed}")
            if norm_item["dataset"]["split"] != no_norm_item["dataset"]["split"]:
                raise RuntimeError(f"Unmatched split for {dataset}, seed {seed}")
            if norm_item["dataset"]["source_manifest"] != no_norm_item["dataset"]["source_manifest"]:
                raise RuntimeError(f"Unmatched dataset source for {dataset}, seed {seed}")
            if norm_item["parameter_count"] != no_norm_item["parameter_count"]:
                raise RuntimeError(f"Unmatched architecture for {dataset}, seed {seed}")
            deltas.append(
                float(no_norm_item["test"]["accuracy"])
                - float(norm_item["test"]["accuracy"])
            )
        paired_deltas.append(
            {
                "dataset": dataset,
                "no_inter_layer_normalization_minus_normalization_per_seed": deltas,
                "no_inter_layer_normalization_minus_normalization_mean": statistics.mean(
                    deltas
                ),
                "no_inter_layer_normalization_minus_normalization_sample_std": (
                    statistics.stdev(deltas)
                ),
            }
        )

    with (run_dir / "normalization_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": 1,
        "methods": list(METHODS),
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "completed_runs": len(rows) * len(SEEDS),
        "rows": rows,
        "paired_deltas": paired_deltas,
        "pair_verification": (
            "architecture, initialization seed, dataset source and split, encoded-input "
            "normalization, optimizer, inference, and model-selection protocol match; "
            "normalization between hidden layers is the only intended intervention"
        ),
    }
    atomic_json(run_dir / "normalization_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    project_root = args.project_root.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    matrix = jobs()
    manifest = {
        "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "run_dir": str(run_dir),
        "python": args.python,
        "protocol": {
            "comparison": "FC-FF+GE with versus without inter-layer L2 normalization",
            "methods": list(METHODS),
            "datasets": list(DATASETS),
            "seeds": list(SEEDS),
            "architectures": ARCHITECTURES,
            "maximum_epochs": 200,
            "validation_size": 5000,
            "validation_split": "seed-specific stratified training holdout",
            "early_stopping_patience": 15,
            "early_stopping_strict_improvement": True,
            "optimizer": "adam",
            "learning_rate": 0.001,
            "batch_size": 128,
            "scheduler": "none",
            "weight_decay": 0.0,
            "dropout": 0.0,
            "first_layer_input_normalization": True,
            "normalization_epsilon": NORMALIZATION_EPSILON,
            "intervention": "disable normalization between hidden layers only",
            "test_policy": "once after best validation checkpoint restoration",
        },
        "source_sha256": {
            name: file_sha256(project_root / name)
            for name in (
                "decomposition_core.py",
                "MLPBenchmarkSuite.py",
                "MatchedCELocalityControlScheduler.py",
                "NormalizationAblationScheduler.py",
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
                "total": len(matrix),
                "completed": completed,
                "failed": len(failures),
                "current_index": index,
                "current": {"dataset": dataset, "method": method, "seed": seed},
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        command = command_for(args.python, project_root, run_dir, dataset, method, seed)
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

    if not failures:
        verification_command = [
            args.python,
            str(project_root / "MLPBenchmarkSuite.py"),
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

    summary = aggregate(run_dir) if not failures else None
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
