from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from MatchedLocalityControlScheduler import (
    METHODS,
    SOURCE_FILES,
    _expected_config,
    _matched_config,
    _new_manifest,
    _verify_history_and_checkpoint,
    _verify_manifest,
    command_for,
    file_sha256,
    jobs,
)
from MLPBenchmarkSuite import configuration, parser


def test_matched_ff_scheduler_defines_exactly_24_unique_jobs() -> None:
    matrix = jobs()
    assert len(matrix) == 24
    assert len(set(matrix)) == 24
    assert {method for _dataset, method, _seed in matrix} == set(METHODS)


def test_matched_ff_pair_commands_differ_only_by_method() -> None:
    project_root = Path("/project")
    run_dir = Path("/run")
    local = command_for(
        "python", project_root, run_dir, "cifar10", "ff-matched-local", 424
    )
    global_ = command_for(
        "python", project_root, run_dir, "cifar10", "ff-matched-ge", 424
    )
    method_index = local.index("--method") + 1
    local[method_index] = "METHOD"
    global_[method_index] = "METHOD"
    assert local == global_


def test_expected_config_matches_the_benchmark_runner() -> None:
    benchmark_parser = parser()
    project_root = Path("/project")
    run_dir = Path("/run")
    for dataset, method, seed in jobs():
        command = command_for(
            "python", project_root, run_dir, dataset, method, seed
        )
        arguments = benchmark_parser.parse_args(command[2:])
        assert configuration(arguments) == _expected_config(dataset, method, seed)


def test_pair_configs_differ_only_by_method_and_detach_policy() -> None:
    local = _expected_config("mnist", "ff-matched-local", 424)
    global_ = _expected_config("mnist", "ff-matched-ge", 424)
    assert local["detach_between_layers"] is True
    assert global_["detach_between_layers"] is False
    assert _matched_config(local) == _matched_config(global_)


def test_manifest_verification_detects_source_drift(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    for name in SOURCE_FILES:
        (project_root / name).write_text(f"source:{name}\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    manifest = _new_manifest("python", project_root, run_dir)
    _verify_manifest(manifest, "python", project_root, run_dir)

    changed_timestamp = dict(manifest)
    changed_timestamp["started_at_utc"] = "2026-01-01T00:00:00+00:00"
    with pytest.raises(RuntimeError, match="integrity checksum"):
        _verify_manifest(changed_timestamp, "python", project_root, run_dir)

    (project_root / SOURCE_FILES[0]).write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="current source"):
        _verify_manifest(manifest, "python", project_root, run_dir)


def test_history_verification_checks_strict_early_stopping_and_finiteness(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "run.json"
    config = _expected_config("mnist", "ff-matched-local", 424)
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    checkpoint_path = tmp_path / "best_model.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    history = []
    for epoch in range(1, 18):
        validation = 40.0 if epoch == 1 else 50.0 if epoch == 2 else 49.0
        history.append(
            {
                "epoch": epoch,
                "train_loss": 1.0,
                "validation_accuracy": validation,
                "best_validation_accuracy": 40.0 if epoch == 1 else 50.0,
                "improved": epoch <= 2,
                "epochs_without_improvement": 0 if epoch <= 2 else epoch - 2,
                "learning_rate": 0.001,
                "all_learning_rates": "[0.001]",
                "finite": True,
                "seconds": 1.0,
            }
        )
    with (tmp_path / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    payload = {
        "config": config,
        "selection": {
            "epochs_trained": 17,
            "best_epoch": 2,
            "maximum_epochs": 200,
            "patience": 15,
            "early_stopped": True,
            "best_validation_accuracy": 50.0,
            "best_validation_metrics": {
                "accuracy": 50.0,
                "correct": 2500,
                "total": 5000,
            },
        },
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": file_sha256(checkpoint_path),
    }
    _verify_history_and_checkpoint(result_path, payload)

    history[-1]["train_loss"] = "nan"
    with (tmp_path / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    with pytest.raises(RuntimeError, match="Non-finite"):
        _verify_history_and_checkpoint(result_path, payload)
