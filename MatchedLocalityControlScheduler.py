#!/usr/bin/env python3
"""Run the matched FF locality-control matrix and aggregate its results."""

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


METHODS = ("ff-matched-local", "ff-matched-ge")
DATASETS = ("mnist", "fashionmnist", "cifar10", "cifar100")
SEEDS = (424, 425, 426)
ARCHITECTURES = {
    "mnist": [1000, 1000],
    "fashionmnist": [1000, 1000],
    "cifar10": [2000, 2000, 2000],
    "cifar100": [2000, 2000, 2000],
}
NORMALIZATION_EPSILON = 1e-4
SOURCE_FILES = (
    "decomposition_core.py",
    "MLPBenchmarkSuite.py",
    "MatchedLocalityControlScheduler.py",
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
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


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


def expected_protocol() -> dict[str, Any]:
    return {
        "comparison": "identical layerwise FF losses; stop-gradient placement only",
        "loss_aggregation": "equal-weight mean over all layer losses",
        "optimizer_count": 1,
        "minibatch_passes_per_update": 1,
        "prediction": "sum all layer goodness values, then take argmax",
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
        "evaluation_batch_size": 256,
        "scheduler": "none",
        "weight_decay": 0.0,
        "dropout": 0.0,
        "precision": "float32",
        "goodness": "mean squared activation",
        "goodness_threshold": 2.0,
        "negative_sampling": "one uniformly sampled incorrect class per example per update",
        "inter_layer_normalization": True,
        "normalization_epsilon": NORMALIZATION_EPSILON,
        "test_policy": "once after best validation checkpoint restoration",
    }


def _expected_config(dataset: str, method: str, seed: int) -> dict[str, Any]:
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
        "local_update_schedule": "one matched multi-loss update per minibatch",
        "inter_layer_normalization": True,
        "first_layer_input_normalization": True,
        "normalization_scope": "input to every FF layer, including the encoded input to layer 1",
        "normalization_epsilon": NORMALIZATION_EPSILON,
        "goodness_definition": "mean squared activation",
        "goodness_threshold": 2.0,
        "matched_locality_control": True,
        "matched_loss_placement": (
            "equal-weight mean of one thresholded FF loss at every layer"
        ),
        "detach_between_layers": method == "ff-matched-local",
        "prediction": "summed layer goodness",
        "terminal_classifier_bias": None,
        "candidate_chunk": 10,
        "num_workers": 4,
        "download": False,
        "requested_device": "cuda",
        "negative_sampling": (
            "one uniformly sampled incorrect class per example per update"
        ),
        "label_encoding": (
            "replace the first C normalized input entries with a 0/1 one-hot label"
        ),
        "dnc_policy": (
            "report every finite completed accuracy numerically; DNC only on run failure"
        ),
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
        "schema_version": 2,
        "project_root": str(project_root),
        "run_dir": str(run_dir),
        "python": python_executable,
        "protocol": expected_protocol(),
        "source_sha256": {
            name: file_sha256(project_root / name) for name in SOURCE_FILES
        },
        "jobs": [
            {
                "dataset": dataset,
                "method": method,
                "seed": seed,
                "command": command_for(
                    python_executable,
                    project_root,
                    run_dir,
                    dataset,
                    method,
                    seed,
                ),
                "expected_config_sha256": json_sha256(
                    _expected_config(dataset, method, seed)
                ),
            }
            for dataset, method, seed in matrix
        ],
    }


def _new_manifest(
    python_executable: str,
    project_root: Path,
    run_dir: Path,
) -> dict[str, Any]:
    static = _manifest_static_fields(python_executable, project_root, run_dir)
    unsigned = {
        **static,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return {**unsigned, "integrity_sha256": json_sha256(unsigned)}


def _verify_manifest(
    manifest: dict[str, Any],
    python_executable: str,
    project_root: Path,
    run_dir: Path,
) -> None:
    expected = _manifest_static_fields(python_executable, project_root, run_dir)
    allowed_keys = set(expected) | {"started_at_utc", "integrity_sha256"}
    if set(manifest) != allowed_keys:
        unexpected = sorted(set(manifest) - allowed_keys)
        missing = sorted(allowed_keys - set(manifest))
        raise RuntimeError(
            f"Manifest key mismatch; unexpected={unexpected}, missing={missing}"
        )
    observed = {key: manifest.get(key) for key in expected}
    if observed != expected:
        differing = sorted(key for key in expected if observed.get(key) != expected[key])
        raise RuntimeError(
            "Manifest does not match the fixed protocol or current source: "
            + ", ".join(differing)
        )
    started = manifest.get("started_at_utc")
    if not isinstance(started, str) or not started:
        raise RuntimeError("Manifest start timestamp is missing")
    unsigned = {key: value for key, value in manifest.items() if key != "integrity_sha256"}
    if manifest.get("integrity_sha256") != json_sha256(unsigned):
        raise RuntimeError("Manifest integrity checksum mismatch")


def _prepare_manifest(
    path: Path,
    python_executable: str,
    project_root: Path,
    run_dir: Path,
) -> dict[str, Any]:
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        _verify_manifest(
            manifest,
            python_executable,
            project_root,
            run_dir,
        )
        return manifest
    manifest = _new_manifest(python_executable, project_root, run_dir)
    atomic_json(path, manifest)
    return manifest


def _matched_config(config: dict[str, Any]) -> dict[str, Any]:
    comparable = dict(config)
    comparable.pop("method")
    comparable.pop("detach_between_layers")
    return comparable


def _finite_number(value: Any, *, field: str, path: Path) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"Invalid numeric value for {field} in {path}") from error
    if not math.isfinite(number):
        raise RuntimeError(f"Non-finite value for {field} in {path}")
    return number


def _verify_history_and_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    config = payload["config"]
    history_path = path.with_name("history.csv")
    config_path = path.with_name("config.json")
    if json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise RuntimeError(f"Config artifact mismatch: {config_path}")
    with history_path.open("r", newline="", encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    selection = payload["selection"]
    epochs_trained = int(selection["epochs_trained"])
    best_epoch = int(selection["best_epoch"])
    maximum_epochs = int(config["epochs"])
    patience = int(config["early_stopping_patience"])
    if len(history) != epochs_trained or not history:
        raise RuntimeError(f"History length mismatch: {history_path}")
    if not 1 <= best_epoch <= epochs_trained <= maximum_epochs:
        raise RuntimeError(f"Invalid best/terminal epoch metadata: {path}")
    if int(selection["maximum_epochs"]) != maximum_epochs:
        raise RuntimeError(f"Maximum-epoch metadata mismatch: {path}")
    if int(selection["patience"]) != patience:
        raise RuntimeError(f"Patience metadata mismatch: {path}")

    validation_values: list[float] = []
    running_best = float("-inf")
    bad_epochs = 0
    for expected_epoch, row in enumerate(history, start=1):
        if int(row["epoch"]) != expected_epoch:
            raise RuntimeError(f"Non-contiguous epoch history: {history_path}")
        _finite_number(row["train_loss"], field="train_loss", path=history_path)
        validation = _finite_number(
            row["validation_accuracy"],
            field="validation_accuracy",
            path=history_path,
        )
        recorded_best = _finite_number(
            row["best_validation_accuracy"],
            field="best_validation_accuracy",
            path=history_path,
        )
        _finite_number(row["learning_rate"], field="learning_rate", path=history_path)
        _finite_number(row["seconds"], field="seconds", path=history_path)
        learning_rates = json.loads(row["all_learning_rates"])
        if not learning_rates or any(
            not math.isfinite(float(rate)) for rate in learning_rates
        ):
            raise RuntimeError(f"Invalid learning-rate history: {history_path}")
        if float(row["learning_rate"]) != 0.001 or [
            float(rate) for rate in learning_rates
        ] != [0.001]:
            raise RuntimeError(f"Unexpected learning-rate schedule: {history_path}")
        improved = validation > running_best
        if improved:
            running_best = validation
            bad_epochs = 0
        else:
            bad_epochs += 1
        if row["improved"].strip().lower() != str(improved).lower():
            raise RuntimeError(f"Strict-improvement flag mismatch: {history_path}")
        if int(row["epochs_without_improvement"]) != bad_epochs:
            raise RuntimeError(f"Patience counter mismatch: {history_path}")
        if not math.isclose(recorded_best, running_best, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"Running best mismatch: {history_path}")
        if row["finite"].strip().lower() != "true":
            raise RuntimeError(f"History row is not marked finite: {history_path}")
        validation_values.append(validation)

    maximum_validation = max(validation_values)
    earliest_best_epoch = validation_values.index(maximum_validation) + 1
    stored_best = _finite_number(
        selection["best_validation_accuracy"],
        field="selection.best_validation_accuracy",
        path=path,
    )
    duplicated_best = _finite_number(
        selection["best_validation_metrics"]["accuracy"],
        field="selection.best_validation_metrics.accuracy",
        path=path,
    )
    if not math.isclose(stored_best, maximum_validation, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"Stored best validation accuracy mismatch: {path}")
    if not math.isclose(stored_best, duplicated_best, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"Duplicated best validation accuracy mismatch: {path}")
    if best_epoch != earliest_best_epoch:
        raise RuntimeError(f"Best epoch violates strict-improvement handling: {path}")
    if not isinstance(selection["early_stopped"], bool):
        raise RuntimeError(f"Invalid early-stop flag: {path}")
    early_stopped = selection["early_stopped"]
    terminal_bad_epochs = int(history[-1]["epochs_without_improvement"])
    if early_stopped and terminal_bad_epochs != patience:
        raise RuntimeError(f"Early-stop patience mismatch: {path}")
    if not early_stopped and epochs_trained != maximum_epochs:
        raise RuntimeError(f"Run ended before its budget without early stopping: {path}")

    checkpoint_path = path.with_name(payload["checkpoint"])
    if not checkpoint_path.is_file():
        raise RuntimeError(f"Best checkpoint is missing: {checkpoint_path}")
    if file_sha256(checkpoint_path) != payload.get("checkpoint_sha256"):
        raise RuntimeError(f"Checkpoint checksum mismatch: {checkpoint_path}")


def _load_result(run_dir: Path, dataset: str, method: str, seed: int) -> dict[str, Any]:
    path = run_dir / "results" / dataset / method / f"seed_{seed}" / "run.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or payload.get("test_evaluations") != 1:
        raise RuntimeError(f"Invalid completed result: {path}")
    config = payload.get("config")
    expected = _expected_config(dataset, method, seed)
    if config != expected:
        observed = config if isinstance(config, dict) else {}
        differing = sorted(
            key for key in set(expected) | set(observed) if observed.get(key) != expected.get(key)
        )
        raise RuntimeError(
            f"Fixed-protocol configuration mismatch in {path}: {', '.join(differing)}"
        )
    dataset_metadata = payload["dataset"]
    split = dataset_metadata["split"]
    if (
        int(dataset_metadata["validation_samples"]) != 5000
        or int(split["seed"]) != seed
        or int(split["validation_size"]) != 5000
        or len(str(split["validation_index_sha256"])) != 64
    ):
        raise RuntimeError(f"Validation-split metadata mismatch: {path}")
    if not dataset_metadata.get("source_manifest"):
        raise RuntimeError(f"Dataset source manifest is missing: {path}")
    test_accuracy = _finite_number(
        payload["test"]["accuracy"],
        field="test.accuracy",
        path=path,
    )
    if not 0.0 <= test_accuracy <= 100.0:
        raise RuntimeError(f"Test accuracy is outside [0, 100]: {path}")
    for metric_group, metrics in (
        ("test", payload["test"]),
        ("selection.best_validation_metrics", payload["selection"]["best_validation_metrics"]),
    ):
        for metric_name, value in metrics.items():
            _finite_number(
                value,
                field=f"{metric_group}.{metric_name}",
                path=path,
            )
    if int(payload["test"]["total"]) != int(dataset_metadata["test_samples"]):
        raise RuntimeError(f"Test sample count mismatch: {path}")
    if int(payload.get("parameter_count", 0)) <= 0:
        raise RuntimeError(f"Invalid parameter count: {path}")
    _verify_history_and_checkpoint(path, payload)
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
        local = by_dataset[dataset]["ff-matched-local"]
        global_ = by_dataset[dataset]["ff-matched-ge"]
        deltas = []
        for seed, local_item, global_item in zip(SEEDS, local, global_, strict=True):
            if _matched_config(local_item["config"]) != _matched_config(
                global_item["config"]
            ):
                raise RuntimeError(f"Unmatched configuration for {dataset}, seed {seed}")
            if local_item["dataset"]["split"] != global_item["dataset"]["split"]:
                raise RuntimeError(f"Unmatched split for {dataset}, seed {seed}")
            if (
                local_item["dataset"]["source_manifest"]
                != global_item["dataset"]["source_manifest"]
            ):
                raise RuntimeError(f"Unmatched dataset source for {dataset}, seed {seed}")
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

    with (run_dir / "matched_summary.csv").open(
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
            "configuration, architecture, dataset source and split, training budget, "
            "optimizer, inference, and model-selection protocol match; only method and "
            "detach_between_layers differ"
        ),
    }
    atomic_json(run_dir / "matched_summary.json", summary)
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
    manifest_path = run_dir / "manifest.json"
    manifest = _prepare_manifest(
        manifest_path,
        args.python,
        project_root,
        run_dir,
    )
    completed = 0
    failures: list[dict[str, Any]] = []
    for index, (dataset, method, seed) in enumerate(matrix, start=1):
        status = {
            "total": len(matrix),
            "completed": completed,
            "failed": len(failures),
            "current_index": index,
            "current": {"dataset": dataset, "method": method, "seed": seed},
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(run_dir / "status.json", status)
        command = command_for(
            args.python, project_root, run_dir, dataset, method, seed
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
                    "stage": "checkpoint_and_history_verification",
                    "returncode": verification.returncode,
                    "log": str(verification_log),
                }
            )

    summary = None
    if not failures:
        try:
            _verify_manifest(
                manifest,
                args.python,
                project_root,
                run_dir,
            )
            summary = aggregate(run_dir)
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            failures.append(
                {
                    "stage": "artifact_and_pair_verification",
                    "error": str(error),
                }
            )
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
