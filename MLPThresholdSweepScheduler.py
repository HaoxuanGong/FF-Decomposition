#!/usr/bin/env python3
"""Plan, run, and select the paper's preliminary MLP threshold sweep.

This scheduler searches only the goodness threshold.  Every candidate uses the
fixed constant-Adam training profile and defers the official test split.  The
selected threshold for each dataset/method is determined from the mean best
validation accuracy across seeds 424--426, then written as a plain JSON mapping
accepted by ``MLPOptimizerSweepScheduler.py --thresholds-json``.

Typical use on a prepared GPU host::

    python MLPThresholdSweepScheduler.py plan --run-dir /path/to/run
    nohup python MLPThresholdSweepScheduler.py launch --run-dir /path/to/run \
        > /path/to/run/launcher.log 2>&1 &
    python MLPThresholdSweepScheduler.py status --run-dir /path/to/run
    python MLPThresholdSweepScheduler.py select --run-dir /path/to/run

Planning never starts training or downloads datasets.  Candidate commands use
``--no-download --defer-test`` so this preliminary study cannot inspect test
accuracy.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import statistics
import subprocess
import sys
from typing import Any, Mapping, Sequence
import uuid

import torch

import MLPBenchmarkSuite as suite


SCHEMA_VERSION = 1
SEEDS = (424, 425, 426)
DATASETS = ("mnist", "fashionmnist", "cifar10", "cifar100")
METHODS = ("ff", "ff-matched-ge", "ff-ge", "nn-ff-ge")
METHOD_LABELS = {
    "ff": "Vanilla FF (Local, Multi-Head)",
    "ff-matched-ge": "FF (Global, Multi-Head)",
    "ff-ge": "FF (Global, Terminal)",
    "nn-ff-ge": "NN-FF (Global, Terminal)",
}
THRESHOLDS = (1.0, 2.0, 4.0)
# Exact ties prefer the smaller threshold.  The order is recorded in every
# manifest and selection artifact so reruns cannot silently change this rule.
THRESHOLD_TIE_BREAK = THRESHOLDS
ARCHITECTURES = {
    "mnist": (1000, 1000),
    "fashionmnist": (1000, 1000),
    "cifar10": (2000, 2000, 2000),
    "cifar100": (2000, 2000, 2000),
}
MAXIMUM_EPOCHS = 200
PATIENCE = 15
MINIMUM_EPOCHS = 0
VALIDATION_SIZE = 5000
BATCH_SIZE = 128
EVALUATION_BATCH_SIZE = 256
CANDIDATE_CHUNK = 10
NUM_WORKERS = 4
DEADLINE_BUFFER = timedelta(minutes=20)
HEARTBEAT_INTERVAL_SECONDS = 30.0
SOURCE_FILES = (
    "decomposition_core.py",
    "MLPBenchmarkSuite.py",
    "MLPThresholdSweepScheduler.py",
)


Job = tuple[float, str, str, int]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def signed_payload(payload: dict[str, Any]) -> dict[str, Any]:
    unsigned = {key: value for key, value in payload.items() if key != "integrity_sha256"}
    return {**unsigned, "integrity_sha256": object_sha256(unsigned)}


def verify_signature(payload: Mapping[str, Any], *, context: str) -> None:
    unsigned = {key: value for key, value in payload.items() if key != "integrity_sha256"}
    if payload.get("integrity_sha256") != object_sha256(unsigned):
        raise RuntimeError(f"Integrity checksum mismatch: {context}")


def threshold_slug(threshold: float) -> str:
    if float(threshold).is_integer():
        return f"theta_{int(threshold)}"
    return "theta_" + format(threshold, ".12g").replace(".", "p")


def normalize_datasets(datasets: Sequence[str] | None = None) -> tuple[str, ...]:
    selected = tuple(DATASETS if datasets is None else datasets)
    if not selected:
        raise ValueError("At least one dataset must be selected")
    unknown = [dataset for dataset in selected if dataset not in DATASETS]
    if unknown:
        raise ValueError(f"Unsupported datasets: {unknown}")
    if len(set(selected)) != len(selected):
        raise ValueError("Dataset selection must not contain duplicates")
    return selected


def identities(datasets: Sequence[str] | None = None) -> list[tuple[str, str]]:
    return [(dataset, method) for dataset in normalize_datasets(datasets) for method in METHODS]


def jobs(datasets: Sequence[str] | None = None) -> list[Job]:
    return [
        (threshold, dataset, method, seed)
        for dataset, method in identities(datasets)
        for seed in SEEDS
        for threshold in THRESHOLDS
    ]


def validate_runner_contract() -> None:
    observed = set(getattr(suite, "THRESHOLDED_FF_METHODS", set()))
    missing = set(METHODS) - observed
    if missing:
        raise RuntimeError(f"MLPBenchmarkSuite lacks thresholded paper methods: {sorted(missing)}")
    for dataset, expected in ARCHITECTURES.items():
        observed_architecture = tuple(getattr(suite, "ARCHITECTURES", {}).get(dataset, ()))
        if observed_architecture != expected:
            raise RuntimeError(
                f"MLPBenchmarkSuite architecture mismatch for {dataset}: "
                f"{observed_architecture!r} != {expected!r}"
            )


def source_hashes(project_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in SOURCE_FILES:
        path = project_root / name
        if not path.is_file():
            raise FileNotFoundError(f"Required source file is missing: {path}")
        hashes[name] = file_sha256(path)
    return hashes


def verify_source_hashes(project_root: Path, expected: Mapping[str, str]) -> None:
    observed = source_hashes(project_root)
    if observed != dict(expected):
        changed = sorted(
            name for name in set(observed) | set(expected) if observed.get(name) != expected.get(name)
        )
        raise RuntimeError(f"Source changed after planning the threshold sweep: {changed}")


def result_directory(
    run_dir: Path, threshold: float, dataset: str, method: str, seed: int
) -> Path:
    return run_dir / "results" / threshold_slug(threshold) / dataset / method / f"seed_{seed}"


def command_for(
    python_executable: str,
    project_root: Path,
    run_dir: Path,
    threshold: float,
    dataset: str,
    method: str,
    seed: int,
) -> list[str]:
    if threshold not in THRESHOLDS:
        raise ValueError(f"Unsupported threshold: {threshold}")
    if dataset not in DATASETS or method not in METHODS or seed not in SEEDS:
        raise ValueError(f"Job is outside the paper matrix: {(threshold, dataset, method, seed)}")
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
        str(MAXIMUM_EPOCHS),
        "--patience",
        str(PATIENCE),
        "--minimum-epochs",
        str(MINIMUM_EPOCHS),
        "--validation-size",
        str(VALIDATION_SIZE),
        "--optimizer",
        "adam",
        "--learning-rate",
        "0.001",
        "--batch-size",
        str(BATCH_SIZE),
        "--evaluation-batch-size",
        str(EVALUATION_BATCH_SIZE),
        "--scheduler",
        "none",
        "--candidate-chunk",
        str(CANDIDATE_CHUNK),
        "--num-workers",
        str(NUM_WORKERS),
        "--goodness-threshold",
        format(threshold, "g"),
        "--data-dir",
        str(project_root / "data"),
        "--output-dir",
        str(run_dir / "results" / threshold_slug(threshold)),
        "--device",
        "cuda",
        "--no-download",
        "--defer-test",
    ]


def expected_config(
    python_executable: str,
    project_root: Path,
    run_dir: Path,
    threshold: float,
    dataset: str,
    method: str,
    seed: int,
) -> dict[str, Any]:
    command = command_for(
        python_executable, project_root, run_dir, threshold, dataset, method, seed
    )
    args = suite.parser().parse_args(command[2:])
    return suite.configuration(args)


def build_manifest(
    python_executable: str,
    project_root: Path,
    run_dir: Path,
    datasets: Sequence[str] | None = None,
) -> dict[str, Any]:
    validate_runner_contract()
    selected = normalize_datasets(datasets)
    project_root = project_root.resolve()
    run_dir = run_dir.resolve()
    matrix = jobs(selected)
    records = []
    for threshold, dataset, method, seed in matrix:
        records.append(
            {
                "threshold": threshold,
                "threshold_slug": threshold_slug(threshold),
                "dataset": dataset,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "seed": seed,
                "command": command_for(
                    python_executable,
                    project_root,
                    run_dir,
                    threshold,
                    dataset,
                    method,
                    seed,
                ),
            }
        )
    spec = {
        "schema_version": SCHEMA_VERSION,
        "project_root": str(project_root),
        "run_dir": str(run_dir),
        "python": python_executable,
        "source_sha256": source_hashes(project_root),
        "protocol": {
            "stage": "preliminary goodness-threshold selection",
            "datasets": list(selected),
            "methods": list(METHODS),
            "method_labels": METHOD_LABELS,
            "thresholds": list(THRESHOLDS),
            "threshold_tie_break_order": list(THRESHOLD_TIE_BREAK),
            "seeds": list(SEEDS),
            "architectures": {dataset: list(ARCHITECTURES[dataset]) for dataset in selected},
            "optimizer": "adam",
            "learning_rate": 1e-3,
            "scheduler": "none",
            "maximum_epochs": MAXIMUM_EPOCHS,
            "patience": PATIENCE,
            "minimum_epochs": MINIMUM_EPOCHS,
            "validation_size": VALIDATION_SIZE,
            "batch_size": BATCH_SIZE,
            "evaluation_batch_size": EVALUATION_BATCH_SIZE,
            "candidate_training_test_policy": "deferred; zero test evaluations",
            "selection_metric": "mean best-validation accuracy over seeds 424--426",
            "selection_scope": "one threshold per dataset and method",
            "selected_test_evaluations": 0,
            "candidate_jobs": len(matrix),
        },
        "jobs": records,
    }
    return signed_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": utc_now(),
            "spec_sha256": object_sha256(spec),
            **spec,
        }
    )


def manifest_path(run_dir: Path) -> Path:
    return run_dir / "manifest.json"


def write_or_resume_manifest(run_dir: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    path = manifest_path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        verify_signature(existing, context=str(path))
        if existing.get("spec_sha256") != candidate.get("spec_sha256"):
            raise RuntimeError(f"Existing run manifest has a different protocol: {path}")
        return existing
    unexpected = [entry for entry in run_dir.iterdir() if entry.name != "scheduler.lock"]
    if unexpected:
        raise RuntimeError(f"Refusing non-empty run directory without a manifest: {run_dir}")
    atomic_json(path, candidate)
    return candidate


def load_manifest(run_dir: Path) -> dict[str, Any]:
    path = manifest_path(run_dir.resolve())
    if not path.is_file():
        raise FileNotFoundError(f"Run has not been planned: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    verify_signature(manifest, context=str(path))
    if Path(manifest["run_dir"]) != run_dir.resolve():
        raise RuntimeError("Manifest run directory does not match its location")
    return manifest


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older supported torch
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid checkpoint: {path}")
    return payload


def finite_accuracy(value: Any, *, context: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 100.0:
        raise RuntimeError(f"Invalid accuracy for {context}: {value!r}")
    return number


def expected_config_from_manifest(
    manifest: Mapping[str, Any], threshold: float, dataset: str, method: str, seed: int
) -> dict[str, Any]:
    return expected_config(
        str(manifest["python"]),
        Path(manifest["project_root"]),
        Path(manifest["run_dir"]),
        threshold,
        dataset,
        method,
        seed,
    )


def verify_training_run(
    manifest: Mapping[str, Any], threshold: float, dataset: str, method: str, seed: int
) -> dict[str, Any]:
    directory = result_directory(Path(manifest["run_dir"]), threshold, dataset, method, seed)
    run_path = directory / "run.json"
    config_path = directory / "config.json"
    history_path = directory / "history.csv"
    checkpoint_path = directory / "best_model.pt"
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    expected = expected_config_from_manifest(manifest, threshold, dataset, method, seed)
    if payload.get("schema_version") != 2:
        raise RuntimeError(f"Unexpected run schema: {run_path}")
    if payload.get("status") != "train_complete":
        raise RuntimeError(f"Candidate is not an untouched deferred-test run: {run_path}")
    if payload.get("test") is not None or int(payload.get("test_evaluations", -1)) != 0:
        raise RuntimeError(f"Candidate test data must be absent: {run_path}")
    if payload.get("config") != expected:
        raise RuntimeError(f"Candidate configuration mismatch: {run_path}")
    if json.loads(config_path.read_text(encoding="utf-8")) != expected:
        raise RuntimeError(f"Config artifact mismatch: {config_path}")
    if not history_path.is_file() or not checkpoint_path.is_file():
        raise RuntimeError(f"Missing candidate artifact in {directory}")
    if payload.get("checkpoint_sha256") != file_sha256(checkpoint_path):
        raise RuntimeError(f"Checkpoint checksum mismatch: {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint.get("config") != expected:
        raise RuntimeError(f"Checkpoint configuration mismatch: {checkpoint_path}")
    selection = payload.get("selection", {})
    best_validation = finite_accuracy(
        selection.get("best_validation_accuracy"), context=str(run_path)
    )
    best_epoch = int(selection.get("best_epoch", 0))
    epochs_trained = int(selection.get("epochs_trained", 0))
    if not 1 <= best_epoch <= epochs_trained <= MAXIMUM_EPOCHS:
        raise RuntimeError(f"Invalid training epoch metadata: {run_path}")
    if int(selection.get("maximum_epochs", -1)) != MAXIMUM_EPOCHS:
        raise RuntimeError(f"Maximum-epoch mismatch: {run_path}")
    if int(selection.get("patience", -1)) != PATIENCE:
        raise RuntimeError(f"Patience mismatch: {run_path}")
    split = payload.get("dataset", {}).get("split", {})
    split_hash = str(split.get("validation_index_sha256", ""))
    if int(split.get("seed", -1)) != seed or len(split_hash) != 64:
        raise RuntimeError(f"Validation-split provenance mismatch: {run_path}")
    return {
        "threshold": threshold,
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "best_validation_accuracy": best_validation,
        "best_epoch": best_epoch,
        "epochs_trained": epochs_trained,
        "validation_index_sha256": split_hash,
        "run_path": str(run_path),
        "run_sha256": file_sha256(run_path),
        "config_sha256": file_sha256(config_path),
        "history_sha256": file_sha256(history_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
    }


def verify_all_candidates(manifest: Mapping[str, Any]) -> dict[Job, dict[str, Any]]:
    selected_datasets = manifest["protocol"]["datasets"]
    records: dict[Job, dict[str, Any]] = {}
    for job in jobs(selected_datasets):
        records[job] = verify_training_run(manifest, *job)
    for dataset in selected_datasets:
        for seed in SEEDS:
            split_hashes = {
                records[(threshold, dataset, method, seed)]["validation_index_sha256"]
                for method in METHODS
                for threshold in THRESHOLDS
            }
            if len(split_hashes) != 1:
                raise RuntimeError(
                    f"Validation split differs across candidates for {dataset}/seed {seed}"
                )
    return records


def freeze_selection(
    manifest: Mapping[str, Any], records: Mapping[Job, Mapping[str, Any]]
) -> tuple[dict[str, dict[str, float]], dict[str, Any], list[dict[str, Any]]]:
    selected: dict[str, dict[str, float]] = {}
    summaries: list[dict[str, Any]] = []
    choices: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for dataset, method in identities(manifest["protocol"]["datasets"]):
        selected.setdefault(dataset, {})
        candidate_scores: list[tuple[float, int, float]] = []
        method_summaries: list[dict[str, Any]] = []
        for preference, threshold in enumerate(THRESHOLD_TIE_BREAK):
            values = [
                float(records[(threshold, dataset, method, seed)]["best_validation_accuracy"])
                for seed in SEEDS
            ]
            summary = {
                "dataset": dataset,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "threshold": threshold,
                "seeds": list(SEEDS),
                "per_seed_best_validation_accuracy": values,
                "mean_best_validation_accuracy": statistics.mean(values),
                "sample_std_best_validation_accuracy": statistics.stdev(values),
                "best_epochs": [
                    int(records[(threshold, dataset, method, seed)]["best_epoch"])
                    for seed in SEEDS
                ],
                "run_sha256": [
                    str(records[(threshold, dataset, method, seed)]["run_sha256"])
                    for seed in SEEDS
                ],
                "checkpoint_sha256": [
                    str(records[(threshold, dataset, method, seed)]["checkpoint_sha256"])
                    for seed in SEEDS
                ],
            }
            method_summaries.append(summary)
            summaries.append(summary)
            candidate_scores.append((summary["mean_best_validation_accuracy"], -preference, threshold))
        _score, _preference, selected_threshold = max(candidate_scores)
        selected[dataset][method] = float(selected_threshold)
        choices.append(
            {
                "dataset": dataset,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "selected_threshold": selected_threshold,
                "selection_metric": "mean_best_validation_accuracy",
                "test_metrics_used_for_selection": False,
                "candidate_mean_best_validation_accuracy": {
                    format(item["threshold"], "g"): item["mean_best_validation_accuracy"]
                    for item in method_summaries
                },
                "selected_runs": [
                    {
                        key: records[(selected_threshold, dataset, method, seed)][key]
                        for key in (
                            "seed",
                            "run_path",
                            "run_sha256",
                            "config_sha256",
                            "history_sha256",
                            "checkpoint_path",
                            "checkpoint_sha256",
                            "validation_index_sha256",
                        )
                    }
                    for seed in SEEDS
                ],
            }
        )
        for summary in method_summaries:
            csv_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "threshold": format(summary["threshold"], "g"),
                    "seed_424_best_validation_accuracy": summary[
                        "per_seed_best_validation_accuracy"
                    ][0],
                    "seed_425_best_validation_accuracy": summary[
                        "per_seed_best_validation_accuracy"
                    ][1],
                    "seed_426_best_validation_accuracy": summary[
                        "per_seed_best_validation_accuracy"
                    ][2],
                    "mean_best_validation_accuracy": summary[
                        "mean_best_validation_accuracy"
                    ],
                    "sample_std_best_validation_accuracy": summary[
                        "sample_std_best_validation_accuracy"
                    ],
                    "selected": summary["threshold"] == selected_threshold,
                }
            )
    evidence = signed_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": utc_now(),
            "manifest_integrity_sha256": manifest["integrity_sha256"],
            "selection_metric": "mean best-validation accuracy over seeds 424--426",
            "threshold_tie_break_order": list(THRESHOLD_TIE_BREAK),
            "tie_break_rule": "prefer the first threshold in threshold_tie_break_order",
            "test_metrics_used_for_selection": False,
            "test_evaluations": 0,
            "threshold_summaries": summaries,
            "choices": choices,
            "selected_thresholds": selected,
        }
    )
    return selected, evidence, csv_rows


def selection_paths(run_dir: Path) -> tuple[Path, Path, Path]:
    return (
        run_dir / "selected_thresholds.json",
        run_dir / "threshold_selection.json",
        run_dir / "threshold_selection.csv",
    )


def write_or_verify_selection(
    manifest: Mapping[str, Any],
    selected: dict[str, dict[str, float]],
    evidence: dict[str, Any],
    csv_rows: list[dict[str, Any]],
) -> None:
    run_dir = Path(manifest["run_dir"])
    plain_path, evidence_path, csv_path = selection_paths(run_dir)
    if evidence_path.exists():
        if not plain_path.is_file() or json.loads(plain_path.read_text(encoding="utf-8")) != selected:
            raise RuntimeError("Plain selected-threshold mapping is missing or inconsistent")
        if not csv_path.is_file():
            raise RuntimeError("Threshold selection CSV is missing")
        expected_plain_sha256 = file_sha256(plain_path)
        evidence = signed_payload(
            {**evidence, "selected_thresholds_sha256": expected_plain_sha256}
        )
        existing = json.loads(evidence_path.read_text(encoding="utf-8"))
        verify_signature(existing, context=str(evidence_path))
        ignored = {"created_at_utc", "integrity_sha256"}
        existing_substantive = {key: value for key, value in existing.items() if key not in ignored}
        candidate_substantive = {key: value for key, value in evidence.items() if key not in ignored}
        if existing_substantive != candidate_substantive:
            raise RuntimeError("Frozen threshold selection no longer matches candidate artifacts")
        return
    if plain_path.exists() or csv_path.exists():
        raise RuntimeError("Incomplete threshold selection artifacts already exist")
    atomic_json(plain_path, selected)
    evidence = signed_payload(
        {**evidence, "selected_thresholds_sha256": file_sha256(plain_path)}
    )
    atomic_json(evidence_path, evidence)
    atomic_csv(csv_path, csv_rows)


def parse_utc(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("Timestamp must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def deadline_allows_start(stop_before_utc: datetime | None, *, now: datetime | None = None) -> bool:
    if stop_before_utc is None:
        return True
    current = now or datetime.now(timezone.utc)
    return stop_before_utc - current.astimezone(timezone.utc) > DEADLINE_BUFFER


def process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def acquire_lock(run_dir: Path) -> tuple[Path, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "scheduler.lock"
    token = uuid.uuid4().hex
    payload = {
        "token": token,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "created_at_utc": utc_now(),
    }
    descriptor: int | None = None
    for _attempt in range(2):
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            break
        except FileExistsError as error:
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {}
            same_host = existing.get("hostname") == socket.gethostname()
            if same_host and not process_is_alive(int(existing.get("pid", -1))):
                path.unlink(missing_ok=True)
                continue
            raise RuntimeError(f"Another scheduler operation holds {path}") from error
    if descriptor is None:
        raise RuntimeError(f"Could not acquire scheduler lock: {path}")
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path, token


def release_lock(path: Path, token: str) -> None:
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return
    if existing.get("token") == token:
        path.unlink(missing_ok=True)


def archive_partial(directory: Path, run_dir: Path, slug: str) -> Path | None:
    if not directory.exists() or not any(directory.iterdir()):
        return None
    destination = run_dir / "partial_attempts" / slug
    retry = 1
    while destination.exists():
        destination = run_dir / "partial_attempts" / f"{slug}.retry{retry}"
        retry += 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(directory), str(destination))
    return destination


def write_runtime_state(run_dir: Path, name: str, payload: Mapping[str, Any]) -> None:
    atomic_json(run_dir / name, dict(payload))


def status_snapshot(manifest: Mapping[str, Any]) -> dict[str, Any]:
    run_dir = Path(manifest["run_dir"])
    counts = {"planned": 0, "partial": 0, "train_complete": 0, "invalid": 0}
    for threshold, dataset, method, seed in jobs(manifest["protocol"]["datasets"]):
        directory = result_directory(run_dir, threshold, dataset, method, seed)
        path = directory / "run.json"
        if not path.exists():
            counts["partial" if directory.exists() and any(directory.iterdir()) else "planned"] += 1
            continue
        try:
            verify_training_run(manifest, threshold, dataset, method, seed)
        except Exception:
            counts["invalid"] += 1
        else:
            counts["train_complete"] += 1
    plain_path, evidence_path, csv_path = selection_paths(run_dir)
    status_payload = None
    if (run_dir / "status.json").is_file():
        status_payload = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    return {
        "run_dir": str(run_dir),
        "candidate_runs": counts,
        "candidate_total": len(manifest["jobs"]),
        "selected_thresholds_ready": plain_path.is_file(),
        "selection_evidence_ready": evidence_path.is_file() and csv_path.is_file(),
        "scheduler_status": status_payload,
        "updated_at_utc": utc_now(),
    }


def plan_command(args: argparse.Namespace) -> int:
    project_root = args.project_root.resolve()
    run_dir = args.run_dir.resolve()
    candidate = build_manifest(args.python, project_root, run_dir, args.datasets)
    manifest = write_or_resume_manifest(run_dir, candidate)
    print(
        json.dumps(
            {
                "status": "planned_only",
                "training_started": False,
                "run_dir": str(run_dir),
                "expected_jobs": len(manifest["jobs"]),
                "manifest": str(manifest_path(run_dir)),
            },
            indent=2,
        )
    )
    return 0


def launch_command(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.run_dir.resolve())
    project_root = Path(manifest["project_root"])
    run_dir = Path(manifest["run_dir"])
    verify_source_hashes(project_root, manifest["source_sha256"])
    running = {
        "state": "running",
        "pid": os.getpid(),
        "started_at_utc": utc_now(),
        "expected": len(manifest["jobs"]),
    }
    write_runtime_state(run_dir, "scheduler.running.json", running)
    (run_dir / "scheduler.done.json").unlink(missing_ok=True)
    (run_dir / "scheduler.failed.json").unlink(missing_ok=True)
    completed = 0
    reused = 0
    launched = 0
    for index, record in enumerate(manifest["jobs"], start=1):
        job = (
            float(record["threshold"]),
            str(record["dataset"]),
            str(record["method"]),
            int(record["seed"]),
        )
        directory = result_directory(run_dir, *job)
        if (directory / "run.json").exists():
            verify_training_run(manifest, *job)
            completed += 1
            reused += 1
            continue
        if args.max_jobs is not None and launched >= args.max_jobs:
            write_runtime_state(
                run_dir,
                "status.json",
                {
                    "state": "paused_after_max_jobs",
                    "completed": completed,
                    "remaining": len(manifest["jobs"]) - completed,
                    "next_job": record,
                    "updated_at_utc": utc_now(),
                },
            )
            (run_dir / "scheduler.running.json").unlink(missing_ok=True)
            return 0
        if not deadline_allows_start(args.stop_before_utc):
            write_runtime_state(
                run_dir,
                "status.json",
                {
                    "state": "paused_before_deadline",
                    "completed": completed,
                    "remaining": len(manifest["jobs"]) - completed,
                    "next_job": record,
                    "updated_at_utc": utc_now(),
                },
            )
            (run_dir / "scheduler.running.json").unlink(missing_ok=True)
            return 0
        slug = (
            f"{threshold_slug(job[0])}_{job[1]}_{job[2]}_seed{job[3]}"
        )
        archived = archive_partial(directory, run_dir, slug)
        state = {
            "state": "running",
            "current_index": index,
            "current_job": record,
            "completed": completed,
            "remaining": len(manifest["jobs"]) - completed,
            "archived_partial_attempt": str(archived) if archived else None,
            "updated_at_utc": utc_now(),
        }
        write_runtime_state(run_dir, "status.json", state)
        log_path = run_dir / "logs" / f"{index:04d}_{slug}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(
                list(record["command"]),
                cwd=project_root,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            while True:
                try:
                    returncode = process.wait(timeout=HEARTBEAT_INTERVAL_SECONDS)
                    break
                except subprocess.TimeoutExpired:
                    candidate_heartbeat = directory / "heartbeat.json"
                    atomic_json(
                        run_dir / "heartbeat.json",
                        {
                            "stage": "training",
                            "scheduler_pid": os.getpid(),
                            "candidate_pid": process.pid,
                            "current_index": index,
                            "current_job": record,
                            "completed": completed,
                            "expected": len(manifest["jobs"]),
                            "candidate_heartbeat": str(candidate_heartbeat),
                            "candidate_heartbeat_mtime": (
                                candidate_heartbeat.stat().st_mtime
                                if candidate_heartbeat.exists()
                                else None
                            ),
                            "updated_at_utc": utc_now(),
                        },
                    )
        launched += 1
        if returncode != 0:
            failure = {
                "state": "failed",
                "job": record,
                "returncode": returncode,
                "log": str(log_path),
                "failed_at_utc": utc_now(),
            }
            write_runtime_state(run_dir, "status.json", failure)
            write_runtime_state(run_dir, "scheduler.failed.json", failure)
            (run_dir / "scheduler.running.json").unlink(missing_ok=True)
            return returncode or 1
        verify_source_hashes(project_root, manifest["source_sha256"])
        verify_training_run(manifest, *job)
        completed += 1
        atomic_json(
            run_dir / "heartbeat.json",
            {
                "stage": "training",
                "completed": completed,
                "expected": len(manifest["jobs"]),
                "updated_at_utc": utc_now(),
            },
        )
    final = {
        "state": "training_complete",
        "completed": completed,
        "reused": reused,
        "launched": launched,
        "remaining": 0,
        "finished_at_utc": utc_now(),
    }
    write_runtime_state(run_dir, "status.json", final)
    write_runtime_state(run_dir, "scheduler.done.json", final)
    (run_dir / "scheduler.running.json").unlink(missing_ok=True)
    return 0


def status_command(args: argparse.Namespace) -> int:
    print(json.dumps(status_snapshot(load_manifest(args.run_dir.resolve())), indent=2, sort_keys=True))
    return 0


def select_command(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.run_dir.resolve())
    verify_source_hashes(Path(manifest["project_root"]), manifest["source_sha256"])
    records = verify_all_candidates(manifest)
    selected, evidence, rows = freeze_selection(manifest, records)
    write_or_verify_selection(manifest, selected, evidence, rows)
    plain_path, evidence_path, csv_path = selection_paths(Path(manifest["run_dir"]))
    print(
        json.dumps(
            {
                "status": "selected",
                "test_evaluations": 0,
                "selected_thresholds": str(plain_path),
                "selection_evidence_json": str(evidence_path),
                "selection_evidence_csv": str(csv_path),
            },
            indent=2,
        )
    )
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="freeze the threshold candidate manifest")
    plan.add_argument("--run-dir", type=Path, required=True)
    plan.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    plan.add_argument("--python", default=sys.executable)
    plan.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    plan.set_defaults(function=plan_command, mutates=True)

    launch = commands.add_parser("launch", help="train or resume validation-only candidates")
    launch.add_argument("--run-dir", type=Path, required=True)
    launch.add_argument(
        "--stop-before-utc",
        type=parse_utc,
        default=None,
        help="optional deadline; no candidate starts in the final 20 minutes",
    )
    launch.add_argument(
        "--max-jobs",
        type=int,
        default=None,
        help="optional number of new jobs to start before pausing",
    )
    launch.set_defaults(function=launch_command, mutates=True)

    status = commands.add_parser("status", help="inspect durable sweep artifacts")
    status.add_argument("--run-dir", type=Path, required=True)
    status.set_defaults(function=status_command, mutates=False)

    select = commands.add_parser(
        "select", help="select thresholds using validation and write optimizer-sweep input"
    )
    select.add_argument("--run-dir", type=Path, required=True)
    select.set_defaults(function=select_command, mutates=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if getattr(args, "max_jobs", None) is not None and args.max_jobs <= 0:
        raise ValueError("--max-jobs must be positive")
    if not getattr(args, "mutates", False):
        return int(args.function(args))
    run_dir = args.run_dir.resolve()
    lock_path, token = acquire_lock(run_dir)
    try:
        code = int(args.function(args))
        (run_dir / "scheduler.exit_code").write_text(f"{code}\n", encoding="utf-8")
        return code
    except Exception as error:
        atomic_json(
            run_dir / "scheduler.failed.json",
            {"failed_at_utc": utc_now(), "error": repr(error)},
        )
        (run_dir / "scheduler.running.json").unlink(missing_ok=True)
        (run_dir / "scheduler.exit_code").write_text("1\n", encoding="utf-8")
        raise
    finally:
        release_lock(lock_path, token)


if __name__ == "__main__":
    raise SystemExit(main())
