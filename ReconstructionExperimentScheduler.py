#!/usr/bin/env python3
"""Run and verify the three-seed FashionMNIST reconstruction experiment."""

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
from typing import Any, Sequence

import matplotlib.pyplot as plt
import torch

from ReconstructionProbe import probe_configuration, verify_completed_run


CONDITIONS = {
    "normalized": "fc-ff-ge",
    "no-inter-layer-normalization": "fc-nn-ff-ge",
}
CONDITION_LABELS = {
    "normalized": r"Inter-layer $L_2$ normalization",
    "no-inter-layer-normalization": r"No inter-layer $L_2$ normalization",
}
SEEDS = (424, 425, 426)
LAYERS = (1, 2, 3, 4, 5)
HIDDEN_DIMS = (500, 500, 500, 500, 500)
MANIFEST_SCHEMA_VERSION = 3


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jobs() -> list[tuple[str, str, int]]:
    return [
        (condition, method, seed)
        for condition, method in CONDITIONS.items()
        for seed in SEEDS
    ]


def backbone_command(
    python_executable: str,
    project_root: Path,
    run_dir: Path,
    condition: str,
    method: str,
    seed: int,
) -> list[str]:
    del condition
    return [
        python_executable,
        str(project_root / "MLPBenchmarkSuite.py"),
        "run",
        "--dataset",
        "fashionmnist",
        "--method",
        method,
        "--seed",
        str(seed),
        "--hidden-dims",
        *[str(width) for width in HIDDEN_DIMS],
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
        str(run_dir / "backbones"),
        "--device",
        "cuda",
        "--no-download",
    ]


def probe_command(
    python_executable: str,
    project_root: Path,
    run_dir: Path,
    condition: str,
    method: str,
    seed: int,
) -> list[str]:
    checkpoint = (
        run_dir
        / "backbones"
        / "fashionmnist"
        / method
        / f"seed_{seed}"
        / "best_model.pt"
    )
    return [
        python_executable,
        str(project_root / "ReconstructionProbe.py"),
        "--backbone-checkpoint",
        str(checkpoint),
        "--condition",
        condition,
        "--seed",
        str(seed),
        "--data-dir",
        str(project_root / "data"),
        "--output-dir",
        str(run_dir / "probes" / condition / f"seed_{seed}"),
        "--device",
        "cuda",
        "--epochs",
        "40",
        "--patience",
        "10",
        "--batch-size",
        "512",
        "--evaluation-batch-size",
        "1024",
        "--extraction-batch-size",
        "1024",
        "--learning-rate",
        "0.001",
        "--validation-size",
        "5000",
        "--num-workers",
        "4",
        "--representation-input",
        "true-label",
        "--no-download",
    ]


def _load_torch_payload(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid checkpoint payload: {path}")
    return payload


def backbone_paths(
    run_dir: Path, method: str, seed: int
) -> tuple[Path, Path, Path, Path]:
    directory = run_dir / "backbones" / "fashionmnist" / method / f"seed_{seed}"
    return (
        directory / "run.json",
        directory / "history.csv",
        directory / "best_model.pt",
        directory / "reconstruction_provenance.json",
    )


def verify_backbone(
    run_dir: Path,
    *,
    condition: str,
    method: str,
    seed: int,
    project_root: Path,
    require_provenance: bool,
) -> dict[str, Any]:
    result_path, history_path, checkpoint_path, provenance_path = backbone_paths(
        run_dir, method, seed
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    config = result.get("config", {})
    expected = {
        "dataset": "fashionmnist",
        "method": method,
        "seed": seed,
        "hidden_dims": list(HIDDEN_DIMS),
        "epochs": 200,
        "early_stopping_patience": 15,
        "early_stopping_minimum_epochs": 0,
        "validation_size": 5000,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "batch_size": 128,
        "evaluation_batch_size": 256,
        "scheduler": "none",
        "weight_decay": 0.0,
        "dropout": 0.0,
        "first_layer_input_normalization": True,
        "inter_layer_normalization": condition == "normalized",
    }
    mismatches = [key for key, value in expected.items() if config.get(key) != value]
    if result.get("status") != "complete" or mismatches:
        raise RuntimeError(
            f"Backbone status/configuration mismatch ({mismatches}): {result_path}"
        )
    if int(result.get("test_evaluations", 0)) != 1:
        raise RuntimeError(f"Backbone test count mismatch: {result_path}")
    dataset = result.get("dataset", {})
    split = dataset.get("split", {})
    if (
        int(dataset.get("training_samples", -1)) != 55000
        or int(dataset.get("validation_samples", -1)) != 5000
        or int(dataset.get("test_samples", -1)) != 10000
        or int(result.get("test", {}).get("total", -1)) != 10000
        or int(split.get("seed", -1)) != seed
        or len(split.get("validation_index_sha256", "")) != 64
        or not dataset.get("source_manifest")
    ):
        raise RuntimeError(f"Backbone dataset/split metadata mismatch: {result_path}")
    with history_path.open("r", newline="", encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    selection = result["selection"]
    epochs_trained = int(selection["epochs_trained"])
    best_epoch = int(selection["best_epoch"])
    if len(history) != epochs_trained or not 1 <= best_epoch <= epochs_trained <= 200:
        raise RuntimeError(f"Backbone history bounds mismatch: {history_path}")
    best = -float("inf")
    replay_best_epoch = 0
    bad_epochs = 0
    for epoch, row in enumerate(history, start=1):
        if int(row["epoch"]) != epoch:
            raise RuntimeError(f"Non-contiguous backbone history: {history_path}")
        values = [float(row[field]) for field in ("train_loss", "validation_accuracy")]
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"Non-finite backbone history: {history_path}")
        improved = values[1] > best
        if (row["improved"].lower() == "true") != improved:
            raise RuntimeError(f"Backbone strict-selection mismatch: {history_path}")
        if improved:
            best = values[1]
            replay_best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
        if int(row["epochs_without_improvement"]) != bad_epochs:
            raise RuntimeError(f"Backbone patience history mismatch: {history_path}")
        if bad_epochs == 15 and epoch != len(history):
            raise RuntimeError(f"Backbone continued beyond patience: {history_path}")
    if replay_best_epoch != best_epoch or not math.isclose(
        best,
        float(selection["best_validation_accuracy"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(f"Backbone best selection mismatch: {result_path}")
    if bool(selection["early_stopped"]) and bad_epochs != 15:
        raise RuntimeError(f"Backbone patience termination mismatch: {result_path}")
    if not bool(selection["early_stopped"]) and epochs_trained != 200:
        raise RuntimeError(f"Unexplained backbone termination: {result_path}")
    if not bool(selection["early_stopped"]) and bad_epochs >= 15:
        raise RuntimeError(f"Missing backbone patience termination: {result_path}")
    finite_result_values = (
        float(result["test"]["accuracy"]),
        float(selection["best_validation_accuracy"]),
    )
    if not all(math.isfinite(value) for value in finite_result_values):
        raise RuntimeError(f"Non-finite backbone result: {result_path}")
    if file_sha256(checkpoint_path) != result.get("checkpoint_sha256"):
        raise RuntimeError(f"Backbone checkpoint checksum mismatch: {checkpoint_path}")
    checkpoint = _load_torch_payload(checkpoint_path)
    if (
        checkpoint.get("config") != config
        or int(checkpoint.get("best_epoch", 0)) != best_epoch
        or not isinstance(checkpoint.get("state_dict"), dict)
    ):
        raise RuntimeError(f"Backbone checkpoint metadata mismatch: {checkpoint_path}")
    source_hashes = {
        name: file_sha256(project_root / name)
        for name in ("MLPBenchmarkSuite.py", "decomposition_core.py")
    }
    provenance = {
        "schema_version": 1,
        "source_sha256": source_hashes,
        "run_json_sha256": file_sha256(result_path),
        "history_sha256": file_sha256(history_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "condition": condition,
        "method": method,
        "seed": seed,
    }
    if require_provenance:
        if not provenance_path.is_file() or json.loads(
            provenance_path.read_text(encoding="utf-8")
        ) != provenance:
            raise RuntimeError(f"Backbone provenance mismatch: {provenance_path}")
    return {"result": result, "provenance": provenance, "path": provenance_path}


def verify_probe(path: Path, *, condition: str, seed: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise RuntimeError(f"Incomplete probe result: {path}")
    config = payload["config"]
    run_dir = path.parents[3]
    scheduler_manifest = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    expected_full_config = probe_configuration(
        argparse.Namespace(
            condition=condition,
            seed=seed,
            representation_input="true-label",
            learning_rate=0.001,
            batch_size=512,
            evaluation_batch_size=1024,
            extraction_batch_size=1024,
            epochs=40,
            patience=10,
            validation_size=5000,
            num_workers=4,
            device="cuda",
            data_dir=Path(scheduler_manifest["project_root"]) / "data",
            download=False,
        )
    )
    if config != expected_full_config:
        raise RuntimeError(f"Probe full configuration mismatch: {path}")
    expected = {
        "condition": condition,
        "seed": seed,
        "dataset": "fashionmnist",
        "hidden_dims": list(HIDDEN_DIMS),
        "representation_input": "true-label",
        "decoder_maximum_epochs": 40,
        "decoder_early_stopping_patience": 10,
        "validation_size": 5000,
        "decoder_weight_decay": 0.0,
        "decoder_dropout": 0.0,
        "first_layer_input_normalization": True,
        "inter_layer_normalization": condition == "normalized",
    }
    mismatches = [name for name, value in expected.items() if config.get(name) != value]
    if mismatches:
        raise RuntimeError(f"Probe configuration mismatch {mismatches}: {path}")
    if int(payload.get("test_evaluations_per_decoder", 0)) != 1:
        raise RuntimeError(f"Unexpected test-evaluation policy: {path}")
    if len(payload.get("rows", [])) != len(LAYERS):
        raise RuntimeError(f"Unexpected layer count: {path}")
    for row in payload["rows"]:
        if int(row["test_evaluations"]) != 1:
            raise RuntimeError(f"Unexpected layer test count: {path}")
        if not math.isfinite(float(row["test_mse"])):
            raise RuntimeError(f"Non-finite test MSE: {path}")
        checkpoint_path = path.parent / row["decoder_checkpoint"]
        if file_sha256(checkpoint_path) != row["decoder_checkpoint_sha256"]:
            raise RuntimeError(f"Decoder checkpoint checksum mismatch: {checkpoint_path}")
        history_path = checkpoint_path.with_name("history.csv")
        with history_path.open("r", newline="", encoding="utf-8") as handle:
            history = list(csv.DictReader(handle))
        validation = [float(item["validation_mse"]) for item in history]
        if not validation or not all(math.isfinite(value) for value in validation):
            raise RuntimeError(f"Invalid decoder validation history: {history_path}")
        earliest_best = validation.index(min(validation)) + 1
        if earliest_best != int(row["best_epoch"]):
            raise RuntimeError(f"Decoder best epoch is not the earliest strict minimum: {path}")
        if not math.isclose(
            min(validation),
            float(row["best_validation_mse"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"Decoder best validation MSE mismatch: {path}")
    backbone_checkpoint = backbone_paths(
        run_dir, CONDITIONS[condition], seed
    )[2]
    try:
        return verify_completed_run(
            path,
            expected_config=config,
            expected_backbone_checkpoint=backbone_checkpoint,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Probe provenance/audit verification failed: {path}") from exc


def aggregate(run_dir: Path) -> dict[str, Any]:
    runs: dict[tuple[str, int], dict[str, Any]] = {}
    for condition, _method, seed in jobs():
        path = run_dir / "probes" / condition / f"seed_{seed}" / "run.json"
        runs[(condition, seed)] = verify_probe(path, condition=condition, seed=seed)
    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for layer in LAYERS:
            values = [
                float(
                    next(
                        row
                        for row in runs[(condition, seed)]["rows"]
                        if int(row["layer"]) == layer
                    )["test_mse"]
                )
                for seed in SEEDS
            ]
            rows.append(
                {
                    "condition": condition,
                    "layer": layer,
                    "seeds": ",".join(map(str, SEEDS)),
                    "test_mse_mean": statistics.mean(values),
                    "test_mse_sample_std": statistics.stdev(values),
                    "per_seed_test_mse": json.dumps(values),
                }
            )
    paired_deltas = []
    for layer in LAYERS:
        deltas = []
        for seed in SEEDS:
            normalized = next(
                row
                for row in runs[("normalized", seed)]["rows"]
                if int(row["layer"]) == layer
            )
            no_inter_layer_normalization = next(
                row
                for row in runs[("no-inter-layer-normalization", seed)]["rows"]
                if int(row["layer"]) == layer
            )
            deltas.append(
                float(normalized["test_mse"])
                - float(no_inter_layer_normalization["test_mse"])
            )
        paired_deltas.append(
            {
                "layer": layer,
                "normalized_minus_no_inter_layer_normalization_mean": statistics.mean(
                    deltas
                ),
                "normalized_minus_no_inter_layer_normalization_sample_std": (
                    statistics.stdev(deltas)
                ),
                "per_seed_delta": deltas,
            }
        )
    atomic_csv(
        run_dir / "reconstruction_summary.csv",
        rows,
        tuple(rows[0]),
    )
    figure, axis = plt.subplots(figsize=(5.2, 3.4))
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        axis.errorbar(
            [int(row["layer"]) for row in selected],
            [1000.0 * float(row["test_mse_mean"]) for row in selected],
            yerr=[1000.0 * float(row["test_mse_sample_std"]) for row in selected],
            marker="o",
            capsize=2.5,
            label=CONDITION_LABELS[condition],
        )
    axis.set_xlabel("Hidden layer")
    axis.set_ylabel(r"Test reconstruction MSE ($\times 10^{-3}$)")
    axis.set_xticks(list(LAYERS))
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(run_dir / "reconstruction_summary.pdf", bbox_inches="tight")
    figure.savefig(run_dir / "reconstruction_summary.png", dpi=300, bbox_inches="tight")
    plt.close(figure)
    summary = {
        "schema_version": 2,
        "status": "complete",
        "completed_backbones": len(jobs()),
        "completed_decoders": len(jobs()) * len(LAYERS),
        "conditions": CONDITIONS,
        "seeds": list(SEEDS),
        "rows": rows,
        "paired_deltas": paired_deltas,
        "interpretation_scope": (
            "reconstruction recoverability from independently learned representations "
            "that share encoded-input normalization and differ in hidden-to-hidden "
            "normalization; not a direct mutual-information estimate"
        ),
    }
    atomic_json(run_dir / "reconstruction_summary.json", summary)
    return summary


def _manifest_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        key: manifest[key]
        for key in (
            "schema_version",
            "project_root",
            "run_dir",
            "python",
            "protocol",
            "source_sha256",
            "jobs",
        )
    }


def prepare_run_directory(run_dir: Path, manifest: dict[str, Any]) -> bool:
    if run_dir.exists() and any(run_dir.iterdir()):
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            entries = {path.name for path in run_dir.iterdir()}
            allowed_prelaunch_entries = {"source", "scheduler.log"}
            if (
                not entries.issubset(allowed_prelaunch_entries)
                or not (run_dir / "source").is_dir()
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
    source_names = (
        "decomposition_core.py",
        "MLPBenchmarkSuite.py",
        "ReconstructionProbe.py",
        "ReconstructionExperimentScheduler.py",
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "run_dir": str(run_dir),
        "python": args.python,
        "protocol": {
            "dataset": "FashionMNIST",
            "comparison": (
                "FC-FF+GE with versus without hidden-to-hidden L2 normalization; "
                "both conditions normalize the label-encoded input before layer 1"
            ),
            "conditions": CONDITIONS,
            "condition_labels": CONDITION_LABELS,
            "seeds": list(SEEDS),
            "hidden_dims": list(HIDDEN_DIMS),
            "backbone_maximum_epochs": 200,
            "backbone_validation_size": 5000,
            "backbone_patience": 15,
            "backbone_optimizer": "Adam",
            "backbone_learning_rate": 0.001,
            "backbone_batch_size": 128,
            "backbone_scheduler": "none",
            "decoder": "500-unit ReLU hidden layer with sigmoid pixel output",
            "decoder_maximum_epochs": 40,
            "decoder_patience": 10,
            "decoder_optimizer": "Adam",
            "decoder_learning_rate": 0.001,
            "decoder_batch_size": 512,
            "decoder_selection": "strict minimum validation per-pixel MSE",
            "representation_input": "true-label positive encoding",
            "first_layer_input_normalization": True,
            "inter_layer_normalization": {
                "normalized": True,
                "no-inter-layer-normalization": False,
            },
            "target": "raw input pixels in [0, 1]",
            "test_policy": "once per decoder after checkpoint restoration",
            "weight_decay": 0.0,
            "dropout": 0.0,
        },
        "source_sha256": {
            name: file_sha256(project_root / name) for name in source_names
        },
        "jobs": [
            {"condition": condition, "method": method, "seed": seed}
            for condition, method, seed in matrix
        ],
    }
    prepare_run_directory(run_dir, manifest)
    (run_dir / "logs").mkdir(exist_ok=True)
    failures: list[dict[str, Any]] = []
    completed = 0
    for index, (condition, method, seed) in enumerate(matrix, start=1):
        atomic_json(
            run_dir / "status.json",
            {
                "total": len(matrix),
                "completed": completed,
                "failed": len(failures),
                "current_index": index,
                "current": {"condition": condition, "method": method, "seed": seed},
                "phase": "backbone",
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        probe_result_path = run_dir / "probes" / condition / f"seed_{seed}" / "run.json"
        try:
            if probe_result_path.exists():
                verify_backbone(
                    run_dir,
                    condition=condition,
                    method=method,
                    seed=seed,
                    project_root=project_root,
                    require_provenance=True,
                )
                verify_probe(probe_result_path, condition=condition, seed=seed)
                completed += 1
                continue
            probe_directory = probe_result_path.parent
            if probe_directory.exists() and any(probe_directory.iterdir()):
                raise RuntimeError(
                    f"Refusing incomplete/stale probe directory: {probe_directory}"
                )

            backbone_result_path = backbone_paths(run_dir, method, seed)[0]
            if backbone_result_path.exists():
                verify_backbone(
                    run_dir,
                    condition=condition,
                    method=method,
                    seed=seed,
                    project_root=project_root,
                    require_provenance=True,
                )
            else:
                backbone_directory = backbone_result_path.parent
                if backbone_directory.exists() and any(backbone_directory.iterdir()):
                    raise RuntimeError(
                        f"Refusing incomplete/stale backbone directory: {backbone_directory}"
                    )
                command = backbone_command(
                    args.python, project_root, run_dir, condition, method, seed
                )
                log_path = _next_log_path(
                    run_dir
                    / "logs"
                    / f"{index:02d}_{condition}_seed{seed}_backbone.log"
                )
                with log_path.open("w", encoding="utf-8") as log:
                    execution = subprocess.run(
                        command,
                        cwd=project_root,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=False,
                    )
                if execution.returncode != 0:
                    raise RuntimeError(
                        f"Backbone command failed with {execution.returncode}; log={log_path}"
                    )
                audit = verify_backbone(
                    run_dir,
                    condition=condition,
                    method=method,
                    seed=seed,
                    project_root=project_root,
                    require_provenance=False,
                )
                atomic_json(audit["path"], audit["provenance"])
                verify_backbone(
                    run_dir,
                    condition=condition,
                    method=method,
                    seed=seed,
                    project_root=project_root,
                    require_provenance=True,
                )

            command = probe_command(
                args.python, project_root, run_dir, condition, method, seed
            )
            log_path = _next_log_path(
                run_dir / "logs" / f"{index:02d}_{condition}_seed{seed}_probe.log"
            )
            with log_path.open("w", encoding="utf-8") as log:
                execution = subprocess.run(
                    command,
                    cwd=project_root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            if execution.returncode != 0:
                raise RuntimeError(
                    f"Probe command failed with {execution.returncode}; log={log_path}"
                )
            verify_probe(probe_result_path, condition=condition, seed=seed)
            completed += 1
        except Exception as exc:
            failures.append(
                {
                    "condition": condition,
                    "method": method,
                    "seed": seed,
                    "error": repr(exc),
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
