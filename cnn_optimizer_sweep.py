#!/usr/bin/env python3
"""Run the paper's validation-selected CNN optimizer sweep.

Optimizer candidates train in fresh processes with test evaluation deferred.
The scheduler freezes one profile per dataset and method using mean validation
accuracy over seeds 424--426, then tests only the selected checkpoints once.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import fields
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess
import sys
from typing import Any, Callable, Sequence
import uuid

import torch

METHODS = ("bp", "local-bp", "ce-matched-ge")
METHOD_LABELS = {
    "bp": "CE (Global, Terminal)",
    "local-bp": "CE (Local, Multi-Head)",
    "ce-matched-ge": "CE (Global, Multi-Head)",
}
DATASETS = ("mnist", "fashionmnist", "cifar10", "cifar100")
SEEDS = (424, 425, 426)
BACKBONE_WIDTHS = [64, 128, 256, 512]
MAXIMUM_EPOCHS = 200
PATIENCE = 15
MINIMUM_EPOCHS = 0
VALIDATION_SIZE = 5000
NO_NEW_JOB_BUFFER = timedelta(minutes=20)
SOURCE_FILES = ("cnn_experiment.py", "cnn_optimizer_sweep.py")
DATASET_SIZES = {
    "mnist": (55_000, 5_000, 10_000),
    "fashionmnist": (55_000, 5_000, 10_000),
    "cifar10": (45_000, 5_000, 10_000),
    "cifar100": (45_000, 5_000, 10_000),
}
PROFILES: dict[str, dict[str, Any]] = {
    "adam_cosine": {
        "optimizer": "adam",
        "optimizer_parameters": "betas=(0.9,0.999),eps=1e-8",
        "learning_rate": 0.001,
        "runner_scheduler": "cosine",
        "scheduler": "cosine-annealing",
        "scheduler_parameters": "T_max=200,eta_min=0",
        "step_size": 30,
        "step_gamma": 0.1,
    },
    "sgd_step30": {
        "optimizer": "sgd",
        "optimizer_parameters": "momentum=0.9,dampening=0,nesterov=false",
        "learning_rate": 0.1,
        "runner_scheduler": "step",
        "scheduler": "step",
        "scheduler_parameters": "step_size=30,gamma=0.1",
        "step_size": 30,
        "step_gamma": 0.1,
    },
}
EXACT_TIE_PREFERENCE = "adam_cosine"
BAD_LOG_PATTERN = re.compile(
    r"traceback|floatingpointerror|non-finite|(?:^|[^a-z])(?:nan|inf)(?:[^a-z]|$)",
    re.IGNORECASE | re.MULTILINE,
)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid checkpoint payload: {path}")
    return payload


def normalize_datasets(datasets: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(datasets)
    if not selected:
        raise ValueError("At least one dataset must be selected")
    unknown = sorted(set(selected) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unsupported datasets: {unknown}")
    if len(set(selected)) != len(selected):
        raise ValueError("Selected datasets must be unique")
    return selected


def jobs(datasets: Sequence[str] = DATASETS) -> list[tuple[str, str, str, int]]:
    return [
        (profile, dataset, method, seed)
        for profile in PROFILES
        for dataset in normalize_datasets(datasets)
        for method in METHODS
        for seed in SEEDS
    ]


def job_slug(profile: str, dataset: str, method: str, seed: int) -> str:
    return f"{profile}_{dataset}_{method}_seed{seed}"


def job_output_dir(
    run_dir: Path, profile: str, dataset: str, method: str, seed: int
) -> Path:
    return run_dir / "candidates" / profile / dataset / method / f"seed_{seed}"


def heartbeat_path(
    run_dir: Path, profile: str, dataset: str, method: str, seed: int
) -> Path:
    return job_output_dir(run_dir, profile, dataset, method, seed) / "heartbeat.json"


def command_for(
    python_executable: str,
    project_root: Path,
    run_dir: Path,
    profile: str,
    dataset: str,
    method: str,
    seed: int,
    provenance_run_id: str,
) -> list[str]:
    if (
        profile not in PROFILES
        or dataset not in DATASETS
        or method not in METHODS
        or seed not in SEEDS
    ):
        raise ValueError(
            f"Job is outside the paper matrix: {(profile, dataset, method, seed)}"
        )
    settings = PROFILES[profile]
    return [
        python_executable,
        str(project_root / "cnn_experiment.py"),
        dataset,
        "--method",
        method,
        "--epochs",
        str(MAXIMUM_EPOCHS),
        "--patience",
        str(PATIENCE),
        "--minimum-delta",
        "0",
        "--validation-size",
        str(VALIDATION_SIZE),
        "--optimizer",
        str(settings["optimizer"]),
        "--learning-rate",
        str(settings["learning_rate"]),
        "--scheduler",
        str(settings["runner_scheduler"]),
        "--step-size",
        str(settings["step_size"]),
        "--step-gamma",
        str(settings["step_gamma"]),
        "--batch-size",
        "128",
        "--eval-batch-size",
        "512",
        "--seed",
        str(seed),
        "--num-workers",
        "4",
        "--data-dir",
        str(project_root / "data"),
        "--output-dir",
        str(job_output_dir(run_dir, profile, dataset, method, seed)),
        "--device",
        "cuda",
        "--no-download",
        "--save-checkpoints",
        "--defer-test",
        "--heartbeat-file",
        str(heartbeat_path(run_dir, profile, dataset, method, seed)),
        "--provenance-run-id",
        provenance_run_id,
        "--overwrite",
    ]


def expected_config(
    project_root: Path,
    run_dir: Path,
    profile: str,
    dataset: str,
    method: str,
    seed: int,
    provenance_run_id: str,
) -> dict[str, Any]:
    settings = PROFILES[profile]
    return {
        "method": method,
        "dataset": dataset,
        "epochs": 200,
        "patience": 15,
        "minimum_delta": 0.0,
        "validation_size": 5000,
        "optimizer": settings["optimizer"],
        "optimizer_parameters": settings["optimizer_parameters"],
        "learning_rate": settings["learning_rate"],
        "scheduler": settings["scheduler"],
        "scheduler_parameters": settings["scheduler_parameters"],
        "step_size": 30,
        "step_gamma": 0.1,
        "batch_size": 128,
        "eval_batch_size": 512,
        "seed": seed,
        "num_workers": 4,
        "data_dir": str(project_root / "data"),
        "output_dir": str(job_output_dir(run_dir, profile, dataset, method, seed)),
        "device": "cuda",
        "download": False,
        "save_checkpoints": True,
        "provenance_run_id": provenance_run_id,
        "overwrite": True,
        "defer_test": True,
        "heartbeat_file": str(heartbeat_path(run_dir, profile, dataset, method, seed)),
        "terminal_classifier_bias": False,
        "backbone_widths": BACKBONE_WIDTHS,
        "validation_split": "seed-specific-stratified",
        "test_evaluation_policy": "deferred-until-profile-selection",
        "augmentation": "none",
        "dropout": 0.0,
        "weight_decay": 0.0,
        "local_bp_prediction": "sum-local-logits",
        "precision": "float32",
        "checkpoint_selection": "strict-validation-accuracy-improvement",
    }


def expected_learning_rate(profile: str, epoch: int) -> float:
    if epoch <= 0:
        raise ValueError("epoch must be positive")
    initial = float(PROFILES[profile]["learning_rate"])
    if PROFILES[profile]["scheduler"] == "cosine-annealing":
        return initial * 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / 200))
    return initial * 0.1 ** ((epoch - 1) // 30)


def _single_csv_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one row in {path}, found {len(rows)}")
    return rows[0]


def _finite_float(row: dict[str, str], field: str, path: Path) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"Invalid {field!r} in {path}") from error
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite {field!r} in {path}")
    return value


def verify_history(
    path: Path, profile: str, dataset: str, method: str, seed: int, split_hash: str
) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not 1 <= len(rows) <= 200:
        raise RuntimeError(f"Invalid history length in {path}")
    best, best_epoch, bad_epochs = -1.0, 0, 0
    for expected_epoch, row in enumerate(rows, 1):
        identity = (
            row.get("dataset"),
            row.get("method"),
            int(row.get("seed", -1)),
            row.get("split_sha256"),
        )
        if identity != (dataset, method, seed, split_hash):
            raise RuntimeError(f"History identity mismatch in {path}")
        if int(row.get("epoch", -1)) != expected_epoch:
            raise RuntimeError(f"Non-contiguous history in {path}")
        if not math.isclose(
            _finite_float(row, "learning_rate", path),
            expected_learning_rate(profile, expected_epoch),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"Learning-rate trace mismatch in {path}")
        validation = _finite_float(row, "validation_accuracy", path)
        if validation > best:
            best, best_epoch, bad_epochs = validation, expected_epoch, 0
        else:
            bad_epochs += 1
        if not math.isclose(
            _finite_float(row, "best_validation_accuracy", path), best, abs_tol=1e-15
        ):
            raise RuntimeError(f"Best-validation trace mismatch in {path}")
        if int(row.get("best_epoch", -1)) != best_epoch:
            raise RuntimeError(f"Best-epoch trace mismatch in {path}")
        if int(row.get("epochs_without_improvement", -1)) != bad_epochs:
            raise RuntimeError(f"Patience trace mismatch in {path}")
        if expected_epoch < len(rows) and bad_epochs >= 15:
            raise RuntimeError(f"Training continued after patience in {path}")
    if len(rows) < 200 and bad_epochs < 15:
        raise RuntimeError(f"History ended before its stopping condition in {path}")
    return {
        "epochs_trained": len(rows),
        "best_epoch": best_epoch,
        "best_validation_accuracy": best,
        "bad_epochs": bad_epochs,
    }


def verify_success_log(path: Path) -> None:
    match = BAD_LOG_PATTERN.search(path.read_text(encoding="utf-8", errors="replace"))
    if match:
        raise RuntimeError(f"Failure marker {match.group(0)!r} in {path}")


def load_verified_candidate(
    run_dir: Path,
    project_root: Path,
    profile: str,
    dataset: str,
    method: str,
    seed: int,
    provenance_run_id: str,
    source_hashes: dict[str, str],
    log_path: Path | None = None,
) -> dict[str, Any]:
    output = job_output_dir(run_dir, profile, dataset, method, seed)
    paths = {
        "config": output / "config.json",
        "history": output / "history.csv",
        "result": output / "per_seed_results.csv",
        "metadata": output / "run_metadata.json",
        "checkpoint": output / f"seed_{seed}_best.pt",
        "heartbeat": heartbeat_path(run_dir, profile, dataset, method, seed),
    }
    for path in paths.values():
        if not path.is_file():
            raise RuntimeError(f"Missing candidate artifact: {path}")
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    expected = expected_config(
        project_root, run_dir, profile, dataset, method, seed, provenance_run_id
    )
    if config != expected:
        different = sorted(
            key
            for key in set(config) | set(expected)
            if config.get(key) != expected.get(key)
        )
        raise RuntimeError(f"Candidate configuration mismatch: {different}")
    result = _single_csv_row(paths["result"])
    if (result.get("dataset"), result.get("method"), int(result.get("seed", -1))) != (
        dataset,
        method,
        seed,
    ):
        raise RuntimeError("Result identity mismatch")
    if result.get("test_evaluations") != "0" or result.get("test_accuracy") not in (
        "",
        None,
    ):
        raise RuntimeError("Optimizer candidates must not evaluate the test split")
    if result.get("best_checkpoint_restored", "").lower() != "true":
        raise RuntimeError("Validation-selected checkpoint was not restored")
    if result.get("finite", "").lower() != "true":
        raise RuntimeError("Candidate was not marked finite")
    sizes = tuple(
        int(result[field])
        for field in ("train_samples", "validation_samples", "test_samples")
    )
    if sizes != DATASET_SIZES[dataset]:
        raise RuntimeError("Dataset sizes do not match the protocol")
    split_hash = result.get("split_sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", split_hash):
        raise RuntimeError("Invalid split hash")
    evidence = verify_history(
        paths["history"], profile, dataset, method, seed, split_hash
    )
    if int(result.get("epochs_trained", -1)) != evidence["epochs_trained"]:
        raise RuntimeError("History/result epoch mismatch")
    if int(result.get("best_epoch", -1)) != evidence["best_epoch"]:
        raise RuntimeError("History/result best-epoch mismatch")
    validation = _finite_float(result, "best_validation_accuracy", paths["result"])
    if not math.isclose(
        validation, evidence["best_validation_accuracy"], abs_tol=1e-15
    ):
        raise RuntimeError("History/result validation mismatch")
    checkpoint_hash = file_sha256(paths["checkpoint"])
    if result.get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("Checkpoint file hash mismatch")
    checkpoint = load_checkpoint(paths["checkpoint"])
    if checkpoint.get("config") != config:
        raise RuntimeError("Checkpoint configuration mismatch")
    checkpoint_identity = (
        checkpoint.get("dataset"),
        checkpoint.get("method"),
        int(checkpoint.get("seed", -1)),
    )
    if checkpoint_identity != (dataset, method, seed):
        raise RuntimeError("Checkpoint identity mismatch")
    if checkpoint.get("provenance_run_id") != provenance_run_id:
        raise RuntimeError("Checkpoint provenance mismatch")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise RuntimeError("Checkpoint state is missing")
    if any(
        not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all().item()
        for tensor in state.values()
    ):
        raise RuntimeError("Checkpoint state is non-finite")
    state_hash = state_dict_sha256(state)
    if checkpoint.get("model_state_sha256") != state_hash:
        raise RuntimeError("Checkpoint tensor-state hash mismatch")
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    if metadata.get("status") != "train_complete" or metadata.get("config") != config:
        raise RuntimeError("Candidate metadata is incomplete")
    if metadata.get("provenance_run_id") != provenance_run_id:
        raise RuntimeError("Candidate metadata provenance mismatch")
    if int(metadata.get("test_evaluations_total", -1)) != 0:
        raise RuntimeError("Candidate metadata records a test evaluation")
    metadata_evidence = {
        "split_sha256": split_hash,
        "checkpoint_sha256": checkpoint_hash,
        "model_state_sha256": state_hash,
        "best_checkpoint_restored": True,
    }
    if any(metadata.get(name) != value for name, value in metadata_evidence.items()):
        raise RuntimeError("Candidate metadata evidence mismatch")
    if (
        metadata.get("source_sha256", {}).get(SOURCE_FILES[0])
        != source_hashes[SOURCE_FILES[0]]
    ):
        raise RuntimeError("CNN runner changed after manifest creation")
    process_id = metadata.get("process_instance_id", "")
    if not re.fullmatch(r"[0-9a-f]{32}", str(process_id)):
        raise RuntimeError("Invalid candidate process identifier")
    heartbeat = json.loads(paths["heartbeat"].read_text(encoding="utf-8"))
    if (
        heartbeat.get("dataset") != dataset
        or heartbeat.get("method") != method
        or int(heartbeat.get("seed", -1)) != seed
        or int(heartbeat.get("epoch", -1)) != evidence["epochs_trained"]
    ):
        raise RuntimeError("Candidate heartbeat does not match its final epoch")
    if log_path is not None:
        verify_success_log(log_path)
    return {
        "profile": profile,
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "config": config,
        "split_sha256": split_hash,
        "best_validation_accuracy": validation,
        "best_epoch": int(result["best_epoch"]),
        "checkpoint_path": str(paths["checkpoint"]),
        "checkpoint_sha256": checkpoint_hash,
        "model_state_sha256": state_hash,
        "process_instance_id": process_id,
        "wall_seconds": float(metadata["wall_seconds"]),
    }


def _comparable_config(config: dict[str, Any]) -> dict[str, Any]:
    comparable = dict(config)
    for name in ("method", "output_dir", "heartbeat_file"):
        comparable.pop(name)
    return comparable


def select_profiles(
    run_dir: Path,
    verified: dict[tuple[str, str, str, int], dict[str, Any]],
    manifest: dict[str, Any],
    datasets: Sequence[str],
) -> dict[str, Any]:
    selected_datasets = normalize_datasets(datasets)
    if set(verified) != set(jobs(selected_datasets)):
        raise RuntimeError("Cannot select profiles from an incomplete candidate matrix")
    process_ids = [str(item["process_instance_id"]) for item in verified.values()]
    if len(process_ids) != len(set(process_ids)):
        raise RuntimeError("A process identifier was reused across candidates")
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for profile in PROFILES:
        for dataset in selected_datasets:
            for method in METHODS:
                group = [verified[(profile, dataset, method, seed)] for seed in SEEDS]
                scores = [
                    100.0 * float(item["best_validation_accuracy"]) for item in group
                ]
                row = {
                    "profile": profile,
                    "dataset": dataset,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "optimizer": PROFILES[profile]["optimizer"],
                    "learning_rate": PROFILES[profile]["learning_rate"],
                    "scheduler": PROFILES[profile]["scheduler"],
                    "scheduler_parameters": PROFILES[profile]["scheduler_parameters"],
                    "seeds": ",".join(map(str, SEEDS)),
                    "best_validation_accuracy_mean_percent": statistics.mean(scores),
                    "best_validation_accuracy_sample_std_percent": statistics.stdev(
                        scores
                    ),
                    "per_seed_best_validation_accuracy_percent": json.dumps(scores),
                }
                rows.append(row)
                grouped[(profile, dataset, method)] = row
    for dataset in selected_datasets:
        for seed in SEEDS:
            split_hashes = {
                verified[(profile, dataset, method, seed)]["split_sha256"]
                for profile in PROFILES
                for method in METHODS
            }
            if len(split_hashes) != 1:
                raise RuntimeError(
                    f"Candidates use different splits for {dataset}/seed {seed}"
                )
            for profile in PROFILES:
                reference = verified[(profile, dataset, METHODS[0], seed)]["config"]
                for method in METHODS[1:]:
                    other = verified[(profile, dataset, method, seed)]["config"]
                    if _comparable_config(reference) != _comparable_config(other):
                        raise RuntimeError("Unmatched method configurations")
    winners: list[dict[str, Any]] = []
    for dataset in selected_datasets:
        for method in METHODS:
            selected_profile = EXACT_TIE_PREFERENCE
            selected_row = grouped[(selected_profile, dataset, method)]
            for profile in PROFILES:
                candidate = grouped[(profile, dataset, method)]
                if (
                    candidate["best_validation_accuracy_mean_percent"]
                    > selected_row["best_validation_accuracy_mean_percent"]
                ):
                    selected_profile, selected_row = profile, candidate
            winners.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "selected_profile": selected_profile,
                    "optimizer": selected_row["optimizer"],
                    "learning_rate": selected_row["learning_rate"],
                    "scheduler": selected_row["scheduler"],
                    "scheduler_parameters": selected_row["scheduler_parameters"],
                    "selection_metric": "mean best-validation accuracy over three seeds",
                    "selected_best_validation_accuracy_mean_percent": selected_row[
                        "best_validation_accuracy_mean_percent"
                    ],
                    "candidate_validation_accuracy_mean_percent": json.dumps(
                        {
                            profile: grouped[(profile, dataset, method)][
                                "best_validation_accuracy_mean_percent"
                            ]
                            for profile in PROFILES
                        },
                        sort_keys=True,
                    ),
                    "selected_checkpoint_sha256_by_seed": {
                        str(seed): verified[(selected_profile, dataset, method, seed)][
                            "checkpoint_sha256"
                        ]
                        for seed in SEEDS
                    },
                }
            )
    payload = {
        "schema_version": 1,
        "provenance_run_id": manifest["provenance_run_id"],
        "source_sha256": manifest["source_sha256"],
        "method_labels": METHOD_LABELS,
        "selection_policy": {
            "scope": "one profile per dataset and method",
            "metric": "mean best-validation accuracy across seeds 424--426",
            "exact_tie_preference": EXACT_TIE_PREFERENCE,
            "test_evaluations_before_selection": 0,
            "test_metrics_used_for_selection": False,
        },
        "candidate_rows": rows,
        "winners": winners,
    }
    selection_path = run_dir / "selection.json"
    if selection_path.exists():
        if json.loads(selection_path.read_text(encoding="utf-8")) != payload:
            raise RuntimeError("Frozen profile selection no longer matches candidates")
    else:
        atomic_json(selection_path, payload)
    atomic_csv(run_dir / "candidate_validation_summary.csv", rows)
    return payload


def final_test_path(run_dir: Path, dataset: str, method: str, seed: int) -> Path:
    return run_dir / "final_tests" / dataset / method / f"seed_{seed}.json"


def _production_evaluator(item: dict[str, Any]) -> float:
    from cnn_experiment import (
        RunConfig,
        build_loaders,
        build_model,
        evaluate,
        set_seed,
    )

    allowed = {field.name for field in fields(RunConfig)}
    config = RunConfig(
        **{key: value for key, value in item["config"].items() if key in allowed}
    )
    seed = int(item["seed"])
    set_seed(seed)
    _train, _validation, test_loader, split_hash = build_loaders(config, seed)
    if split_hash != item["split_sha256"]:
        raise RuntimeError("Finalizer reconstructed a different split")
    checkpoint = load_checkpoint(Path(item["checkpoint_path"]))
    state = checkpoint.get("model_state_dict")
    if (
        not isinstance(state, dict)
        or state_dict_sha256(state) != item["model_state_sha256"]
    ):
        raise RuntimeError("Selected checkpoint changed before final evaluation")
    device = torch.device(config.device)
    model = build_model(config).to(device)
    model.load_state_dict(state)
    accuracy = float(evaluate(model, test_loader, device))
    if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
        raise RuntimeError("Final test evaluation returned an invalid accuracy")
    return accuracy


def finalize_selected_tests(
    run_dir: Path,
    selection: dict[str, Any],
    verified: dict[tuple[str, str, str, int], dict[str, Any]],
    evaluator: Callable[[dict[str, Any]], float] = _production_evaluator,
) -> dict[tuple[str, str, int], dict[str, Any]]:
    finals: dict[tuple[str, str, int], dict[str, Any]] = {}
    selection_hash = canonical_sha256(selection)
    for winner in selection["winners"]:
        dataset, method = winner["dataset"], winner["method"]
        profile = winner["selected_profile"]
        for seed in SEEDS:
            key = (dataset, method, seed)
            candidate = verified[(profile, dataset, method, seed)]
            result_path = final_test_path(run_dir, dataset, method, seed)
            started_path = result_path.with_suffix(".started.json")
            identity = {
                "schema_version": 1,
                "provenance_run_id": selection["provenance_run_id"],
                "selection_sha256": selection_hash,
                "dataset": dataset,
                "method": method,
                "seed": seed,
                "selected_profile": profile,
                "checkpoint_sha256": candidate["checkpoint_sha256"],
                "model_state_sha256": candidate["model_state_sha256"],
            }
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if any(result.get(name) != value for name, value in identity.items()):
                    raise RuntimeError(f"Final-test artifact mismatch: {result_path}")
                if result.get("test_evaluations") != 1:
                    raise RuntimeError(f"Invalid final-test count: {result_path}")
                accuracy = float(result.get("test_accuracy", float("nan")))
                if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
                    raise RuntimeError(f"Invalid final-test accuracy: {result_path}")
                finals[key] = result
                continue
            if started_path.exists():
                raise RuntimeError(
                    "A selected test evaluation started without a durable result; "
                    f"refusing to evaluate it again: {started_path}"
                )
            atomic_json(
                started_path,
                {**identity, "started_at_utc": datetime.now(timezone.utc).isoformat()},
            )
            accuracy = float(evaluator(candidate))
            if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
                raise RuntimeError("Evaluator returned an invalid test accuracy")
            result = {
                **identity,
                "test_accuracy": accuracy,
                "test_evaluations": 1,
                "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            atomic_json(result_path, result)
            finals[key] = result
    return finals


def write_summary(
    run_dir: Path,
    selection: dict[str, Any],
    finals: dict[tuple[str, str, int], dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for winner in selection["winners"]:
        dataset, method = winner["dataset"], winner["method"]
        values = [
            100.0 * finals[(dataset, method, seed)]["test_accuracy"] for seed in SEEDS
        ]
        rows.append(
            {
                **{
                    key: winner[key]
                    for key in (
                        "dataset",
                        "method",
                        "method_label",
                        "selected_profile",
                        "optimizer",
                        "learning_rate",
                        "scheduler",
                        "scheduler_parameters",
                    )
                },
                "seeds": ",".join(map(str, SEEDS)),
                "best_validation_accuracy_mean_percent": winner[
                    "selected_best_validation_accuracy_mean_percent"
                ],
                "test_accuracy_mean_percent": statistics.mean(values),
                "test_accuracy_sample_std_percent": statistics.stdev(values),
                "per_seed_test_accuracy_percent": json.dumps(values),
                "test_evaluations_per_seed": 1,
            }
        )
    atomic_csv(run_dir / "summary.csv", rows)
    summary = {
        "schema_version": 1,
        "provenance_run_id": selection["provenance_run_id"],
        "methods": list(METHODS),
        "method_labels": METHOD_LABELS,
        "seeds": list(SEEDS),
        "selection_policy": selection["selection_policy"],
        "rows": rows,
        "verification": {
            "profiles_selected_without_test_metrics": True,
            "only_selected_checkpoints_tested": True,
            "test_evaluations_per_selected_seed": 1,
        },
    }
    atomic_json(run_dir / "summary.json", summary)
    return summary


def current_source_hashes(project_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in SOURCE_FILES:
        path = project_root / name
        if not path.is_file():
            raise FileNotFoundError(f"Required source file not found: {path}")
        hashes[name] = file_sha256(path)
    return hashes


def manifest_payload(
    run_dir: Path,
    project_root: Path,
    python_executable: str,
    datasets: Sequence[str] = DATASETS,
) -> dict[str, Any]:
    selected = normalize_datasets(datasets)
    matrix = jobs(selected)
    return {
        "schema_version": 2,
        "provenance_run_id": uuid.uuid4().hex,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "run_dir": str(run_dir),
        "python": python_executable,
        "source_sha256": current_source_hashes(project_root),
        "profiles": PROFILES,
        "datasets": list(selected),
        "protocol": {
            "methods": list(METHODS),
            "method_labels": METHOD_LABELS,
            "seeds": list(SEEDS),
            "backbone_widths": BACKBONE_WIDTHS,
            "maximum_epochs": 200,
            "patience": 15,
            "minimum_epochs": 0,
            "validation_size": 5000,
            "training_batch_size": 128,
            "evaluation_batch_size": 512,
            "profile_selection": "mean best-validation accuracy over all three seeds",
            "candidate_test_evaluations": 0,
            "selected_checkpoint_test_evaluations": 1,
            "expected_candidate_jobs": len(matrix),
        },
        "jobs": [
            {
                "profile": p,
                "dataset": d,
                "method": m,
                "method_label": METHOD_LABELS[m],
                "seed": s,
            }
            for p, d, m, s in matrix
        ],
    }


def prepare_manifest(
    run_dir: Path,
    project_root: Path,
    python_executable: str,
    datasets: Sequence[str] = DATASETS,
) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    if not path.exists():
        allowed_prelaunch = {"scheduler.lock", "scheduler.log", "scheduler.pid"}
        unexpected = [
            entry for entry in run_dir.iterdir() if entry.name not in allowed_prelaunch
        ]
        if unexpected:
            raise RuntimeError(
                f"Refusing non-empty run directory without manifest: {run_dir}"
            )
        payload = manifest_payload(run_dir, project_root, python_executable, datasets)
        atomic_json(path, payload)
        return payload
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected_jobs = [
        {
            "profile": p,
            "dataset": d,
            "method": m,
            "method_label": METHOD_LABELS[m],
            "seed": s,
        }
        for p, d, m, s in jobs(datasets)
    ]
    if (
        manifest.get("schema_version") != 2
        or Path(manifest.get("project_root", "")).resolve() != project_root
        or Path(manifest.get("run_dir", "")).resolve() != run_dir
        or manifest.get("python") != python_executable
        or manifest.get("jobs") != expected_jobs
        or manifest.get("profiles") != PROFILES
        or manifest.get("datasets") != list(normalize_datasets(datasets))
        or manifest.get("protocol", {}).get("method_labels") != METHOD_LABELS
        or manifest.get("source_sha256") != current_source_hashes(project_root)
    ):
        raise RuntimeError(
            "CNN sweep source or protocol changed after manifest creation"
        )
    return manifest


def parse_utc(value: str) -> datetime:
    rendered = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(rendered)
    if parsed.tzinfo is None:
        raise ValueError("--stop-before-utc must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def may_start_new_job(now: datetime, stop_before: datetime | None) -> bool:
    return (
        stop_before is None
        or now.astimezone(timezone.utc) < stop_before - NO_NEW_JOB_BUFFER
    )


def acquire_lock(run_dir: Path) -> tuple[Path, str]:
    path, token = run_dir / "scheduler.lock", uuid.uuid4().hex
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as error:
        raise RuntimeError(f"Another scheduler invocation may hold {path}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "token": token,
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            handle,
            indent=2,
        )
    return path, token


def release_lock(path: Path, token: str) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return
    if payload.get("token") == token:
        path.unlink()


def _next_log_path(
    run_dir: Path, index: int, profile: str, dataset: str, method: str, seed: int
) -> Path:
    stem = f"{index:03d}_{job_slug(profile, dataset, method, seed)}"
    attempt = 1
    while True:
        path = run_dir / "logs" / f"{stem}_attempt{attempt:02d}.log"
        if not path.exists():
            return path
        attempt += 1


def run_scheduler(args: argparse.Namespace) -> int:
    run_dir, project_root = args.run_dir.resolve(), args.project_root.resolve()
    python_executable = os.path.abspath(os.path.expanduser(args.python))
    datasets = normalize_datasets(args.datasets)
    deadline = parse_utc(args.stop_before_utc) if args.stop_before_utc else None
    run_dir.mkdir(parents=True, exist_ok=True)
    lock, token = acquire_lock(run_dir)
    try:
        manifest = prepare_manifest(run_dir, project_root, python_executable, datasets)
        matrix = jobs(datasets)
        (run_dir / "logs").mkdir(exist_ok=True)
        verified: dict[tuple[str, str, str, int], dict[str, Any]] = {}
        for index, (profile, dataset, method, seed) in enumerate(matrix, 1):
            key = (profile, dataset, method, seed)
            try:
                verified[key] = load_verified_candidate(
                    run_dir,
                    project_root,
                    profile,
                    dataset,
                    method,
                    seed,
                    manifest["provenance_run_id"],
                    manifest["source_sha256"],
                )
                continue
            except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError):
                pass
            if not may_start_new_job(datetime.now(timezone.utc), deadline):
                atomic_json(
                    run_dir / "status.json",
                    {
                        "status": "paused_before_deadline",
                        "completed_candidates": len(verified),
                        "expected_candidates": len(matrix),
                        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
                return 0
            log_path = _next_log_path(run_dir, index, profile, dataset, method, seed)
            atomic_json(
                run_dir / "status.json",
                {
                    "status": "training_candidate",
                    "completed_candidates": len(verified),
                    "expected_candidates": len(matrix),
                    "current": {
                        "profile": profile,
                        "dataset": dataset,
                        "method": method,
                        "seed": seed,
                    },
                    "heartbeat": str(
                        heartbeat_path(run_dir, profile, dataset, method, seed)
                    ),
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            command = command_for(
                python_executable,
                project_root,
                run_dir,
                profile,
                dataset,
                method,
                seed,
                manifest["provenance_run_id"],
            )
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.run(
                    command,
                    cwd=project_root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            if process.returncode != 0:
                raise RuntimeError(
                    f"Candidate failed with exit {process.returncode}: {log_path}"
                )
            verified[key] = load_verified_candidate(
                run_dir,
                project_root,
                profile,
                dataset,
                method,
                seed,
                manifest["provenance_run_id"],
                manifest["source_sha256"],
                log_path,
            )
        selection = select_profiles(run_dir, verified, manifest, datasets)
        atomic_json(
            run_dir / "status.json",
            {
                "status": "final_test_evaluation",
                "completed_candidates": len(verified),
                "expected_candidates": len(matrix),
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        finals = finalize_selected_tests(run_dir, selection, verified)
        summary = write_summary(run_dir, selection, finals)
        final = {
            "schema_version": 1,
            "status": "complete",
            "completed_candidates": len(verified),
            "expected_candidates": len(matrix),
            "completed_final_tests": len(finals),
            "expected_final_tests": len(datasets) * len(METHODS) * len(SEEDS),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
        }
        atomic_json(run_dir / "scheduler.done", final)
        atomic_json(run_dir / "status.json", final)
        return 0
    except Exception as error:
        atomic_json(
            run_dir / "scheduler.failed",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise
    finally:
        release_lock(lock, token)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--run-dir", type=Path, required=True)
    root.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parent
    )
    root.add_argument("--python", default=sys.executable)
    root.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    root.add_argument(
        "--stop-before-utc",
        help="Optional booked-server deadline as an offset-aware ISO-8601 timestamp.",
    )
    return root


def main(argv: list[str] | None = None) -> int:
    return run_scheduler(parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
