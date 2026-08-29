#!/usr/bin/env python3
"""Run the matched cross-entropy locality-control matrix and aggregate results."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any


METHODS = ("ce-matched-local", "ce-matched-ge")
DATASETS = ("mnist", "fashionmnist", "cifar10", "cifar100")
SEEDS = (424, 425, 426)


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


def jobs() -> list[tuple[str, str, int]]:
    return [
        (dataset, method, seed)
        for dataset in DATASETS
        for method in METHODS
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


def _matched_config(config: dict[str, Any]) -> dict[str, Any]:
    comparable = dict(config)
    comparable.pop("method")
    comparable.pop("detach_between_layers")
    return comparable


def _load_result(run_dir: Path, dataset: str, method: str, seed: int) -> dict[str, Any]:
    path = run_dir / "results" / dataset / method / f"seed_{seed}" / "run.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or payload.get("test_evaluations") != 1:
        raise RuntimeError(f"Invalid completed result: {path}")
    config = payload["config"]
    if config.get("method") != method or config.get("seed") != seed:
        raise RuntimeError(f"Result identity mismatch: {path}")
    if not config.get("matched_locality_control"):
        raise RuntimeError(f"Matched-control metadata missing: {path}")
    expected_detach = method == "ce-matched-local"
    if config.get("detach_between_layers") is not expected_detach:
        raise RuntimeError(f"Detach metadata mismatch: {path}")
    return payload


def aggregate(run_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    by_dataset: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for dataset in DATASETS:
        for method in METHODS:
            group = [_load_result(run_dir, dataset, method, seed) for seed in SEEDS]
            values = [float(item["test"]["accuracy"]) for item in group]
            validation = [
                float(item["selection"]["best_validation_accuracy"]) for item in group
            ]
            epochs = [int(item["selection"]["best_epoch"]) for item in group]
            row = {
                "dataset": dataset,
                "method": method,
                "seeds": ",".join(map(str, SEEDS)),
                "test_accuracy_mean": statistics.mean(values),
                "test_accuracy_sample_std": statistics.stdev(values),
                "best_validation_accuracy_mean": statistics.mean(validation),
                "best_epoch_mean": statistics.mean(epochs),
                "per_seed_test_accuracy": json.dumps(values),
            }
            rows.append(row)
            by_dataset.setdefault(dataset, {})[method] = group

    paired_deltas = []
    for dataset in DATASETS:
        local = by_dataset[dataset]["ce-matched-local"]
        global_ = by_dataset[dataset]["ce-matched-ge"]
        deltas = []
        for seed, local_item, global_item in zip(SEEDS, local, global_, strict=True):
            if _matched_config(local_item["config"]) != _matched_config(
                global_item["config"]
            ):
                raise RuntimeError(f"Unmatched configuration for {dataset}, seed {seed}")
            if local_item["dataset"]["split"] != global_item["dataset"]["split"]:
                raise RuntimeError(f"Unmatched split for {dataset}, seed {seed}")
            if local_item["parameter_count"] != global_item["parameter_count"]:
                raise RuntimeError(f"Unmatched architecture for {dataset}, seed {seed}")
            deltas.append(
                float(global_item["test"]["accuracy"])
                - float(local_item["test"]["accuracy"])
            )
        paired_deltas.append(
            {
                "dataset": dataset,
                "global_minus_local_per_seed": deltas,
                "global_minus_local_mean": statistics.mean(deltas),
                "global_minus_local_sample_std": statistics.stdev(deltas),
            }
        )

    with (run_dir / "matched_ce_summary.csv").open(
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
            "configuration, architecture, and split match; only method and "
            "detach_between_layers differ"
        ),
    }
    atomic_json(run_dir / "matched_ce_summary.json", summary)
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
            "comparison": (
                "identical LocalBPMLP architecture and layerwise cross-entropy "
                "losses; stop-gradient placement only"
            ),
            "loss_aggregation": "equal-weight mean over all layer losses",
            "optimizer_count": 1,
            "minibatch_passes_per_update": 1,
            "prediction": "sum all layer logits, then take argmax",
            "methods": list(METHODS),
            "datasets": list(DATASETS),
            "seeds": list(SEEDS),
            "architectures": {
                "mnist": [1000, 1000],
                "fashionmnist": [1000, 1000],
                "cifar10": [2000, 2000, 2000],
                "cifar100": [2000, 2000, 2000],
            },
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
            "test_policy": "once after best validation checkpoint restoration",
        },
        "source_sha256": {
            name: file_sha256(project_root / name)
            for name in (
                "decomposition_core.py",
                "MLPBenchmarkSuite.py",
                "MatchedCELocalityControlScheduler.py",
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
        command = command_for(
            args.python,
            project_root,
            run_dir,
            dataset,
            method,
            seed,
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
