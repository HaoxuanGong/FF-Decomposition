#!/usr/bin/env python3
"""Plan, run, select, finalize, and aggregate the paper MLP sweep.

Training and test evaluation are deliberately separate.  Every optimizer
candidate is trained with ``MLPBenchmarkSuite.py run --defer-test``.  A profile
is then frozen for each dataset/method using the mean best-validation accuracy
over seeds 424--426.  Only the three checkpoints belonging to that frozen
profile are evaluated on the official test split, exactly once.

Goodness thresholds are inputs to this scheduler, not optimizer-sweep
hyperparameters.  ``FROZEN_THRESHOLDS`` records the choices made by the paper's
earlier threshold study.  A complete replacement mapping can be supplied to
``plan --thresholds-json``; it is validated, hashed, and frozen in the run
manifest before any optimizer candidate starts.
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


SCHEMA_VERSION = 3
SEEDS = (424, 425, 426)
DATASETS = ("mnist", "fashionmnist", "cifar10", "cifar100")
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
METHOD_LABELS = {
    "ff": "Vanilla FF (Local, Multi-Head)",
    "ff-matched-ge": "FF (Global, Multi-Head)",
    "ff-ge": "FF (Global, Terminal)",
    "nn-ff-ge": "NN-FF (Global, Terminal)",
    "fc-ff": "FC-FF (Local, Multi-Head)",
    "fc-ff-matched-ge": "FC-FF (Global, Multi-Head)",
    "fc-ff-ge": "FC-FF (Global, Terminal)",
    "fc-nn-ff-ge": "FC-NN-FF (Global, Terminal)",
    "local-bp": "CE (Local, Multi-Head)",
    "ce-matched-ge": "CE (Global, Multi-Head)",
    "bp": "CE (Global, Terminal)",
}
FC_METHODS = {
    "fc-ff",
    "fc-ff-matched-ge",
    "fc-ff-ge",
    "fc-nn-ff-ge",
}
THRESHOLDED_METHODS = {"ff", "ff-matched-ge", "ff-ge", "nn-ff-ge"}
ARCHITECTURES = {
    "mnist": (1000, 1000),
    "fashionmnist": (1000, 1000),
    "cifar10": (2000, 2000, 2000),
    "cifar100": (2000, 2000, 2000),
}

# These paper-facing values were selected by the separate constant-Adam
# threshold study and are frozen before the optimizer sweep. The compatibility
# local controls are outside the paper matrix and are not scheduled here.
FROZEN_THRESHOLDS: dict[str, dict[str, float]] = {
    "mnist": {"ff": 4.0, "ff-matched-ge": 1.0, "ff-ge": 4.0, "nn-ff-ge": 4.0},
    "fashionmnist": {
        "ff": 4.0,
        "ff-matched-ge": 1.0,
        "ff-ge": 2.0,
        "nn-ff-ge": 2.0,
    },
    "cifar10": {"ff": 2.0, "ff-matched-ge": 1.0, "ff-ge": 1.0, "nn-ff-ge": 4.0},
    "cifar100": {"ff": 1.0, "ff-matched-ge": 1.0, "ff-ge": 1.0, "nn-ff-ge": 4.0},
}

PROFILES: dict[str, dict[str, Any]] = {
    "adam_constant": {
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "momentum": None,
        "nesterov": None,
        "scheduler": "none",
        "cosine_t_max": None,
        "cosine_eta_min": None,
        "step_size": None,
        "step_gamma": None,
    },
    "adam_cosine": {
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "momentum": None,
        "nesterov": None,
        "scheduler": "cosine",
        "cosine_t_max": 200,
        "cosine_eta_min": 0.0,
        "step_size": None,
        "step_gamma": None,
    },
    "sgd_step30": {
        "optimizer": "sgd",
        "learning_rate": 0.1,
        "momentum": 0.9,
        "nesterov": False,
        "scheduler": "step",
        "cosine_t_max": None,
        "cosine_eta_min": None,
        "step_size": 30,
        "step_gamma": 0.1,
    },
}
PROFILE_TIE_BREAK = tuple(PROFILES)
MAXIMUM_EPOCHS = 200
PATIENCE = 15
MINIMUM_EPOCHS = 0
VALIDATION_SIZE = 5000
BATCH_SIZE = 128
EVALUATION_BATCH_SIZE = 256
CANDIDATE_CHUNK = 10
NUM_WORKERS = 4
DEADLINE_BUFFER = timedelta(minutes=20)
SOURCE_FILES = (
    "decomposition_core.py",
    "MLPBenchmarkSuite.py",
    "MLPOptimizerSweepScheduler.py",
)


Job = tuple[str, str, str, int]


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


def methods_for(dataset: str) -> tuple[str, ...]:
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if dataset == "cifar100":
        return tuple(method for method in PRIMARY_METHODS if method not in FC_METHODS)
    return PRIMARY_METHODS


def identities(datasets: Sequence[str] | None = None) -> list[tuple[str, str]]:
    selected = normalize_datasets(datasets)
    return [(dataset, method) for dataset in selected for method in methods_for(dataset)]


def jobs(datasets: Sequence[str] | None = None) -> list[Job]:
    return [
        (profile, dataset, method, seed)
        for dataset, method in identities(datasets)
        for seed in SEEDS
        for profile in PROFILES
    ]


def validate_runner_contract() -> None:
    observed_primary = tuple(getattr(suite, "PRIMARY_METHODS", ()))
    if observed_primary != PRIMARY_METHODS:
        raise RuntimeError(
            "MLPBenchmarkSuite.PRIMARY_METHODS does not match the paper scheduler: "
            f"{observed_primary!r} != {PRIMARY_METHODS!r}"
        )
    observed_fc = set(getattr(suite, "FC_METHODS", set()))
    if observed_fc != FC_METHODS:
        raise RuntimeError(
            f"MLPBenchmarkSuite.FC_METHODS mismatch: {observed_fc!r} != {FC_METHODS!r}"
        )


def validate_thresholds(
    thresholds: Mapping[str, Mapping[str, Any]],
    datasets: Sequence[str] | None = None,
) -> dict[str, dict[str, float]]:
    selected = normalize_datasets(datasets)
    normalized: dict[str, dict[str, float]] = {}
    for dataset in selected:
        if dataset not in thresholds:
            raise ValueError(f"Missing frozen thresholds for {dataset}")
        expected_methods = THRESHOLDED_METHODS.intersection(methods_for(dataset))
        supplied = set(thresholds[dataset])
        if supplied != expected_methods:
            raise ValueError(
                f"Frozen-threshold methods for {dataset} must be "
                f"{sorted(expected_methods)}, got {sorted(supplied)}"
            )
        normalized[dataset] = {}
        for method in sorted(expected_methods):
            value = float(thresholds[dataset][method])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid frozen threshold for {dataset}/{method}: {value}")
            normalized[dataset][method] = value
    return normalized


def load_thresholds(
    path: Path | None,
    datasets: Sequence[str] | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    if path is None:
        return validate_thresholds(FROZEN_THRESHOLDS, datasets), {
            "kind": "embedded-paper-threshold-selection",
            "path": None,
            "sha256": None,
        }
    resolved = path.resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--thresholds-json must contain a dataset-to-method mapping")
    return validate_thresholds(payload, datasets), {
        "kind": "external-frozen-threshold-selection",
        "path": str(resolved),
        "sha256": file_sha256(resolved),
    }


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
            name
            for name in set(observed) | set(expected)
            if observed.get(name) != expected.get(name)
        )
        raise RuntimeError(f"Source changed after planning the sweep: {changed}")


def result_directory(
    run_dir: Path,
    profile: str,
    dataset: str,
    method: str,
    seed: int,
) -> Path:
    return run_dir / "results" / profile / dataset / method / f"seed_{seed}"


def test_record_path(
    run_dir: Path,
    profile: str,
    dataset: str,
    method: str,
    seed: int,
) -> Path:
    return run_dir / "final" / profile / dataset / method / f"seed_{seed}" / "test.json"


def command_for(
    python_executable: str,
    project_root: Path,
    run_dir: Path,
    profile: str,
    dataset: str,
    method: str,
    seed: int,
    thresholds: Mapping[str, Mapping[str, float]] = FROZEN_THRESHOLDS,
) -> list[str]:
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile: {profile}")
    if method not in methods_for(dataset) or seed not in SEEDS:
        raise ValueError(f"Job is outside the paper matrix: {(profile, dataset, method, seed)}")
    settings = PROFILES[profile]
    command = [
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
        str(settings["optimizer"]),
        "--learning-rate",
        str(settings["learning_rate"]),
        "--batch-size",
        str(BATCH_SIZE),
        "--evaluation-batch-size",
        str(EVALUATION_BATCH_SIZE),
        "--scheduler",
        str(settings["scheduler"]),
        "--candidate-chunk",
        str(CANDIDATE_CHUNK),
        "--num-workers",
        str(NUM_WORKERS),
        "--data-dir",
        str(project_root / "data"),
        "--output-dir",
        str(run_dir / "results" / profile),
        "--device",
        "cuda",
        "--no-download",
        "--defer-test",
    ]
    if method in THRESHOLDED_METHODS:
        command.extend(["--goodness-threshold", str(thresholds[dataset][method])])
    if profile == "sgd_step30":
        command.extend(
            [
                "--momentum",
                "0.9",
                "--step-size",
                "30",
                "--step-gamma",
                "0.1",
            ]
        )
    return command


def expected_config(
    python_executable: str,
    project_root: Path,
    run_dir: Path,
    profile: str,
    dataset: str,
    method: str,
    seed: int,
    thresholds: Mapping[str, Mapping[str, float]] = FROZEN_THRESHOLDS,
) -> dict[str, Any]:
    command = command_for(
        python_executable,
        project_root,
        run_dir,
        profile,
        dataset,
        method,
        seed,
        thresholds,
    )
    args = suite.parser().parse_args(command[2:])
    return suite.configuration(args)


def job_record(
    job: Job,
    *,
    python_executable: str,
    project_root: Path,
    run_dir: Path,
    thresholds: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    profile, dataset, method, seed = job
    return {
        "profile": profile,
        "dataset": dataset,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "seed": seed,
        "goodness_threshold": (
            thresholds[dataset][method] if method in THRESHOLDED_METHODS else None
        ),
        "command": command_for(
            python_executable,
            project_root,
            run_dir,
            profile,
            dataset,
            method,
            seed,
            thresholds,
        ),
    }


def build_manifest(
    python_executable: str,
    project_root: Path,
    run_dir: Path,
    datasets: Sequence[str] | None = None,
    *,
    thresholds: Mapping[str, Mapping[str, float]] = FROZEN_THRESHOLDS,
    threshold_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_runner_contract()
    selected = normalize_datasets(datasets)
    frozen = validate_thresholds(thresholds, selected)
    project_root = project_root.resolve()
    run_dir = run_dir.resolve()
    matrix = jobs(selected)
    spec = {
        "schema_version": SCHEMA_VERSION,
        "project_root": str(project_root),
        "run_dir": str(run_dir),
        "python": python_executable,
        "source_sha256": source_hashes(project_root),
        "protocol": {
            "datasets": list(selected),
            "methods_by_dataset": {dataset: list(methods_for(dataset)) for dataset in selected},
            "method_labels": METHOD_LABELS,
            "seeds": list(SEEDS),
            "architectures": {dataset: list(ARCHITECTURES[dataset]) for dataset in selected},
            "profiles": PROFILES,
            "profile_tie_break_order": list(PROFILE_TIE_BREAK),
            "maximum_epochs": MAXIMUM_EPOCHS,
            "patience": PATIENCE,
            "minimum_epochs": MINIMUM_EPOCHS,
            "validation_size": VALIDATION_SIZE,
            "batch_size": BATCH_SIZE,
            "evaluation_batch_size": EVALUATION_BATCH_SIZE,
            "candidate_training_test_policy": "deferred; zero test evaluations",
            "profile_selection": (
                "one profile per dataset/method, maximizing mean best-validation "
                "accuracy over seeds 424--426; test metrics are unavailable"
            ),
            "final_test_policy": (
                "evaluate each selected seed checkpoint exactly once after selection"
            ),
            "threshold_policy": {
                "stage": "separate preliminary threshold sweep",
                "frozen_before_optimizer_profile_selection": True,
                "searched_by_this_scheduler": False,
                "values": frozen,
                "source": dict(threshold_source or {"kind": "caller-supplied"}),
            },
            "candidate_jobs": len(matrix),
            "selected_test_evaluations": len(identities(selected)) * len(SEEDS),
        },
        "jobs": [
            job_record(
                job,
                python_executable=python_executable,
                project_root=project_root,
                run_dir=run_dir,
                thresholds=frozen,
            )
            for job in matrix
        ],
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


def manifest_thresholds(manifest: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    return validate_thresholds(
        manifest["protocol"]["threshold_policy"]["values"],
        manifest["protocol"]["datasets"],
    )


def expected_config_from_manifest(
    manifest: Mapping[str, Any], profile: str, dataset: str, method: str, seed: int
) -> dict[str, Any]:
    return expected_config(
        str(manifest["python"]),
        Path(manifest["project_root"]),
        Path(manifest["run_dir"]),
        profile,
        dataset,
        method,
        seed,
        manifest_thresholds(manifest),
    )


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older supported torch
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid checkpoint: {path}")
    return payload


def finite_accuracy(value: Any, *, context: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 100.0:
        raise RuntimeError(f"Invalid accuracy for {context}: {value!r}")
    return number


def verify_training_run(
    manifest: Mapping[str, Any], profile: str, dataset: str, method: str, seed: int
) -> dict[str, Any]:
    directory = result_directory(Path(manifest["run_dir"]), profile, dataset, method, seed)
    run_path = directory / "run.json"
    config_path = directory / "config.json"
    history_path = directory / "history.csv"
    checkpoint_path = directory / "best_model.pt"
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    expected = expected_config_from_manifest(manifest, profile, dataset, method, seed)
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
        "profile": profile,
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
        "payload": payload,
    }


def verify_all_candidates(manifest: Mapping[str, Any]) -> dict[Job, dict[str, Any]]:
    records: dict[Job, dict[str, Any]] = {}
    for job in jobs(manifest["protocol"]["datasets"]):
        records[job] = verify_training_run(manifest, *job)
    for dataset, method in identities(manifest["protocol"]["datasets"]):
        for seed in SEEDS:
            hashes = {
                records[(profile, dataset, method, seed)]["validation_index_sha256"]
                for profile in PROFILES
            }
            if len(hashes) != 1:
                raise RuntimeError(
                    f"Validation split differs across profiles for {dataset}/{method}/seed {seed}"
                )
    return records


def freeze_selection(
    manifest: Mapping[str, Any], records: Mapping[Job, Mapping[str, Any]]
) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    choices: list[dict[str, Any]] = []
    for dataset, method in identities(manifest["protocol"]["datasets"]):
        candidates: list[tuple[float, int, str]] = []
        for preference, profile in enumerate(PROFILE_TIE_BREAK):
            values = [
                float(records[(profile, dataset, method, seed)]["best_validation_accuracy"])
                for seed in SEEDS
            ]
            mean = statistics.mean(values)
            summaries.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "profile": profile,
                    "seeds": list(SEEDS),
                    "per_seed_best_validation_accuracy": values,
                    "mean_best_validation_accuracy": mean,
                    "sample_std_best_validation_accuracy": statistics.stdev(values),
                }
            )
            candidates.append((mean, -preference, profile))
        _mean, _preference, selected_profile = max(candidates)
        choices.append(
            {
                "dataset": dataset,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "selected_profile": selected_profile,
                "selection_metric": "mean_best_validation_accuracy",
                "test_metrics_used_for_selection": False,
                "candidate_mean_best_validation_accuracy": {
                    summary["profile"]: summary["mean_best_validation_accuracy"]
                    for summary in summaries
                    if summary["dataset"] == dataset and summary["method"] == method
                },
                "selected_runs": [
                    {
                        key: records[(selected_profile, dataset, method, seed)][key]
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
    return signed_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": utc_now(),
            "manifest_integrity_sha256": manifest["integrity_sha256"],
            "profile_tie_break_order": list(PROFILE_TIE_BREAK),
            "selection_metric": "mean best-validation accuracy over seeds 424--426",
            "test_metrics_used_for_selection": False,
            "profile_summaries": summaries,
            "choices": choices,
        }
    )


def selection_path(run_dir: Path) -> Path:
    return run_dir / "selection.json"


def write_or_verify_selection(run_dir: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    def substantive(item: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in item.items()
            if key not in {"created_at_utc", "integrity_sha256"}
        }

    path = selection_path(run_dir)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        verify_signature(existing, context=str(path))
        if substantive(existing) != substantive(candidate):
            raise RuntimeError("Frozen selection no longer matches candidate artifacts")
        return existing
    atomic_json(path, candidate)
    return candidate


def verify_frozen_selection(manifest: Mapping[str, Any], selection: Mapping[str, Any]) -> None:
    verify_signature(selection, context="selection.json")
    if selection.get("manifest_integrity_sha256") != manifest.get("integrity_sha256"):
        raise RuntimeError("Selection was created from a different manifest")
    for choice in selection["choices"]:
        for record in choice["selected_runs"]:
            for path_key, digest_key in (
                ("run_path", "run_sha256"),
                ("checkpoint_path", "checkpoint_sha256"),
            ):
                path = Path(record[path_key])
                if not path.is_file() or file_sha256(path) != record[digest_key]:
                    raise RuntimeError(f"Frozen artifact changed: {path}")
            directory = Path(record["run_path"]).parent
            for name, digest_key in (
                ("config.json", "config_sha256"),
                ("history.csv", "history_sha256"),
            ):
                path = directory / name
                if file_sha256(path) != record[digest_key]:
                    raise RuntimeError(f"Frozen artifact changed: {path}")


def evaluate_checkpoint(
    manifest: Mapping[str, Any],
    choice: Mapping[str, Any],
    selected_run: Mapping[str, Any],
    *,
    data_dir: Path,
    device: torch.device,
) -> tuple[dict[str, Any], str]:
    dataset = str(choice["dataset"])
    method = str(choice["method"])
    seed = int(selected_run["seed"])
    source = json.loads(Path(selected_run["run_path"]).read_text(encoding="utf-8"))
    config = source["config"]
    suite.set_seed(seed)
    training_data, test_data = suite.load_dataset(dataset, data_dir, False)
    num_classes = int(suite.DATASET_SPECS[dataset]["classes"])
    _train_loader, _validation_loader, test_loader, split = suite.make_loaders(
        training_data,
        test_data,
        seed=seed,
        num_classes=num_classes,
        validation_size=int(config["validation_size"]),
        device=device,
        evaluation_batch_size=int(config["evaluation_batch_size"]),
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        train_limit=config.get("train_limit"),
        test_limit=config.get("test_limit"),
    )
    if split["validation_index_sha256"] != selected_run["validation_index_sha256"]:
        raise RuntimeError(f"Evaluation split mismatch for {dataset}/{method}/seed {seed}")
    current_manifest = suite.dataset_manifest(data_dir, dataset)
    if current_manifest != source["dataset"]["source_manifest"]:
        raise RuntimeError(f"Evaluation dataset differs from training data for {dataset}")
    model = suite.build_model(
        method,
        dataset,
        hidden_dims=tuple(int(width) for width in config["hidden_dims"]),
    ).to(device)
    checkpoint = load_checkpoint(Path(selected_run["checkpoint_path"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    metrics = suite.evaluate(
        method,
        model,
        test_loader,
        device=device,
        num_classes=num_classes,
        candidate_chunk=int(config["candidate_chunk"]),
    )
    return dict(metrics), object_sha256(current_manifest)


def verify_test_record(
    path: Path,
    selection: Mapping[str, Any],
    choice: Mapping[str, Any],
    selected_run: Mapping[str, Any],
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    verify_signature(payload, context=str(path))
    expected = {
        "profile": choice["selected_profile"],
        "dataset": choice["dataset"],
        "method": choice["method"],
        "seed": selected_run["seed"],
        "checkpoint_sha256": selected_run["checkpoint_sha256"],
        "source_run_sha256": selected_run["run_sha256"],
        "selection_integrity_sha256": selection["integrity_sha256"],
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Test provenance mismatch: {path}")
    if payload.get("status") != "complete" or int(payload.get("test_evaluations", -1)) != 1:
        raise RuntimeError(f"Invalid final test record: {path}")
    finite_accuracy(payload.get("test", {}).get("accuracy"), context=str(path))
    return payload


def finalize_selected(
    manifest: Mapping[str, Any],
    selection: Mapping[str, Any],
    *,
    data_dir: Path,
    device_name: str,
) -> list[dict[str, Any]]:
    verify_frozen_selection(manifest, selection)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; refusing CPU fallback")
    completed: list[dict[str, Any]] = []
    run_dir = Path(manifest["run_dir"])
    for choice in selection["choices"]:
        profile = str(choice["selected_profile"])
        dataset = str(choice["dataset"])
        method = str(choice["method"])
        for selected_run in choice["selected_runs"]:
            seed = int(selected_run["seed"])
            output = test_record_path(run_dir, profile, dataset, method, seed)
            started = output.with_name("test.started.json")
            if output.exists():
                completed.append(verify_test_record(output, selection, choice, selected_run))
                continue
            if started.exists():
                raise RuntimeError(
                    f"Prior test evaluation did not produce a result; refusing rerun: {started}"
                )
            atomic_json(
                started,
                {
                    "started_at_utc": utc_now(),
                    "selection_integrity_sha256": selection["integrity_sha256"],
                    "checkpoint_sha256": selected_run["checkpoint_sha256"],
                    "pid": os.getpid(),
                },
            )
            metrics, dataset_manifest_sha256 = evaluate_checkpoint(
                manifest,
                choice,
                selected_run,
                data_dir=data_dir,
                device=device,
            )
            payload = signed_payload(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "complete",
                    "profile": profile,
                    "dataset": dataset,
                    "method": method,
                    "seed": seed,
                    "selection_integrity_sha256": selection["integrity_sha256"],
                    "test_metrics_used_for_selection": False,
                    "checkpoint_sha256": selected_run["checkpoint_sha256"],
                    "source_run_sha256": selected_run["run_sha256"],
                    "dataset_manifest_sha256": dataset_manifest_sha256,
                    "test": metrics,
                    "test_evaluations": 1,
                    "evaluated_at_utc": utc_now(),
                    "device": str(device),
                    "gpu": (torch.cuda.get_device_name(device) if device.type == "cuda" else None),
                }
            )
            atomic_json(output, payload)
            completed.append(verify_test_record(output, selection, choice, selected_run))
            atomic_json(
                run_dir / "heartbeat.json",
                {
                    "stage": "finalize",
                    "updated_at_utc": utc_now(),
                    "completed_test_evaluations": len(completed),
                    "expected_test_evaluations": len(identities(manifest["protocol"]["datasets"]))
                    * len(SEEDS),
                },
            )
    return completed


def aggregate_results(manifest: Mapping[str, Any], selection: Mapping[str, Any]) -> dict[str, Any]:
    verify_frozen_selection(manifest, selection)
    run_dir = Path(manifest["run_dir"])
    rows: list[dict[str, Any]] = []
    for choice in selection["choices"]:
        profile = str(choice["selected_profile"])
        dataset = str(choice["dataset"])
        method = str(choice["method"])
        tests = [
            verify_test_record(
                test_record_path(run_dir, profile, dataset, method, int(record["seed"])),
                selection,
                choice,
                record,
            )
            for record in choice["selected_runs"]
        ]
        values = [float(test["test"]["accuracy"]) for test in tests]
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "selected_profile": profile,
                "goodness_threshold": (
                    manifest_thresholds(manifest)[dataset][method]
                    if method in THRESHOLDED_METHODS
                    else None
                ),
                "seeds": list(SEEDS),
                "per_seed_test_accuracy": values,
                "test_accuracy_mean": statistics.mean(values),
                "test_accuracy_sample_std": statistics.stdev(values),
                "test_evaluations": len(values),
            }
        )
    output = signed_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": utc_now(),
            "manifest_integrity_sha256": manifest["integrity_sha256"],
            "selection_integrity_sha256": selection["integrity_sha256"],
            "selection_policy": selection["selection_metric"],
            "test_metrics_used_for_selection": False,
            "rows": rows,
        }
    )
    atomic_json(run_dir / "summary.json", output)
    csv_rows = [
        {
            **row,
            "seeds": json.dumps(row["seeds"]),
            "per_seed_test_accuracy": json.dumps(row["per_seed_test_accuracy"]),
        }
        for row in rows
    ]
    atomic_csv(run_dir / "summary.csv", csv_rows)
    return output


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
    # ``os.kill(pid, 0)`` is a harmless existence probe on POSIX, where the
    # long sweep runs.  On Windows it can terminate a process, so retain the
    # lock conservatively.
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


def status_snapshot(manifest: Mapping[str, Any]) -> dict[str, Any]:
    run_dir = Path(manifest["run_dir"])
    counts = {"planned": 0, "partial": 0, "train_complete": 0, "invalid": 0}
    for profile, dataset, method, seed in jobs(manifest["protocol"]["datasets"]):
        directory = result_directory(run_dir, profile, dataset, method, seed)
        path = directory / "run.json"
        if not path.exists():
            counts["partial" if directory.exists() and any(directory.iterdir()) else "planned"] += 1
            continue
        try:
            verify_training_run(manifest, profile, dataset, method, seed)
        except Exception:
            counts["invalid"] += 1
        else:
            counts["train_complete"] += 1
    tests = len(list((run_dir / "final").glob("*/*/*/seed_*/test.json")))
    selection = selection_path(run_dir).is_file()
    return {
        "run_dir": str(run_dir),
        "candidate_runs": counts,
        "candidate_total": len(manifest["jobs"]),
        "selection_frozen": selection,
        "final_test_records": tests,
        "expected_final_test_records": manifest["protocol"]["selected_test_evaluations"],
        "summary_ready": (run_dir / "summary.json").is_file(),
        "updated_at_utc": utc_now(),
    }


def plan_command(args: argparse.Namespace) -> int:
    project_root = args.project_root.resolve()
    run_dir = args.run_dir.resolve()
    thresholds, source = load_thresholds(args.thresholds_json, args.datasets)
    candidate = build_manifest(
        args.python,
        project_root,
        run_dir,
        args.datasets,
        thresholds=thresholds,
        threshold_source=source,
    )
    manifest = write_or_resume_manifest(run_dir, candidate)
    print(
        json.dumps(
            {
                "status": "planned",
                "run_dir": str(run_dir),
                "candidate_jobs": len(manifest["jobs"]),
                "selected_test_evaluations": manifest["protocol"]["selected_test_evaluations"],
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
    completed = 0
    reused = 0
    for index, record in enumerate(manifest["jobs"], start=1):
        job = (
            str(record["profile"]),
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
        if not deadline_allows_start(args.stop_before_utc):
            atomic_json(
                run_dir / "status.json",
                {
                    "state": "paused_before_deadline",
                    "completed": completed,
                    "remaining": len(manifest["jobs"]) - completed,
                    "next_job": record,
                    "updated_at_utc": utc_now(),
                },
            )
            return 0
        slug = f"{job[0]}_{job[1]}_{job[2]}_seed{job[3]}"
        archived = archive_partial(directory, run_dir, slug)
        atomic_json(
            run_dir / "status.json",
            {
                "state": "running",
                "current_index": index,
                "current_job": record,
                "completed": completed,
                "remaining": len(manifest["jobs"]) - completed,
                "archived_partial_attempt": str(archived) if archived else None,
                "updated_at_utc": utc_now(),
            },
        )
        log = run_dir / "logs" / f"{index:04d}_{slug}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as handle:
            process = subprocess.run(
                list(record["command"]),
                cwd=project_root,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if process.returncode != 0:
            atomic_json(
                run_dir / "status.json",
                {
                    "state": "failed",
                    "job": record,
                    "returncode": process.returncode,
                    "log": str(log),
                    "updated_at_utc": utc_now(),
                },
            )
            return process.returncode or 1
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
    atomic_json(
        run_dir / "status.json",
        {
            "state": "training_complete",
            "completed": completed,
            "reused": reused,
            "remaining": 0,
            "updated_at_utc": utc_now(),
        },
    )
    return 0


def status_command(args: argparse.Namespace) -> int:
    snapshot = status_snapshot(load_manifest(args.run_dir.resolve()))
    print(json.dumps(snapshot, indent=2, sort_keys=True))
    return 0


def finalize_command(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.run_dir.resolve())
    verify_source_hashes(Path(manifest["project_root"]), manifest["source_sha256"])
    records = verify_all_candidates(manifest)
    candidate = freeze_selection(manifest, records)
    selection = write_or_verify_selection(Path(manifest["run_dir"]), candidate)
    if args.selection_only:
        print(json.dumps(selection, indent=2, sort_keys=True))
        return 0
    tests = finalize_selected(
        manifest,
        selection,
        data_dir=args.data_dir.resolve(),
        device_name=args.device,
    )
    print(
        json.dumps(
            {
                "status": "finalized",
                "selection": str(selection_path(Path(manifest["run_dir"]))),
                "verified_test_evaluations": len(tests),
            },
            indent=2,
        )
    )
    return 0


def aggregate_command(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.run_dir.resolve())
    selection = json.loads(selection_path(Path(manifest["run_dir"])).read_text(encoding="utf-8"))
    summary = aggregate_results(manifest, selection)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="freeze the complete candidate manifest")
    plan.add_argument("--run-dir", type=Path, required=True)
    plan.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    plan.add_argument("--python", default=sys.executable)
    plan.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    plan.add_argument(
        "--thresholds-json",
        type=Path,
        default=None,
        help="complete frozen threshold mapping from a separate threshold study",
    )
    plan.set_defaults(function=plan_command, mutates=True)

    launch = commands.add_parser("launch", help="train or resume deferred-test candidates")
    launch.add_argument("--run-dir", type=Path, required=True)
    launch.add_argument(
        "--stop-before-utc",
        type=parse_utc,
        default=None,
        help="optional deadline; no candidate starts in the final 20 minutes",
    )
    launch.set_defaults(function=launch_command, mutates=True)

    status = commands.add_parser("status", help="inspect durable run artifacts")
    status.add_argument("--run-dir", type=Path, required=True)
    status.set_defaults(function=status_command, mutates=False)

    finalize = commands.add_parser(
        "finalize", help="select by validation and test only selected checkpoints"
    )
    finalize.add_argument("--run-dir", type=Path, required=True)
    finalize.add_argument("--data-dir", type=Path, required=True)
    finalize.add_argument("--device", default="cuda")
    finalize.add_argument("--selection-only", action="store_true")
    finalize.set_defaults(function=finalize_command, mutates=True)

    aggregate = commands.add_parser("aggregate", help="verify and summarize final tests")
    aggregate.add_argument("--run-dir", type=Path, required=True)
    aggregate.set_defaults(function=aggregate_command, mutates=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
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
        (run_dir / "scheduler.exit_code").write_text("1\n", encoding="utf-8")
        raise
    finally:
        release_lock(lock_path, token)


if __name__ == "__main__":
    raise SystemExit(main())
