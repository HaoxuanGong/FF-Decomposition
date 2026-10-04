from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import socket

import pytest

import MLPBenchmarkSuite as suite
from MLPOptimizerSweepScheduler import validate_thresholds
from MLPThresholdSweepScheduler import (
    ARCHITECTURES,
    DEADLINE_BUFFER,
    MAXIMUM_EPOCHS,
    METHODS,
    MINIMUM_EPOCHS,
    PATIENCE,
    SEEDS,
    SOURCE_FILES,
    THRESHOLDS,
    THRESHOLD_TIE_BREAK,
    acquire_lock,
    build_manifest,
    command_for,
    deadline_allows_start,
    freeze_selection,
    jobs,
    parser,
    release_lock,
    selection_paths,
    write_or_resume_manifest,
    write_or_verify_selection,
)


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    for name in SOURCE_FILES:
        (project / name).write_text(f"fixture:{name}\n", encoding="utf-8")
    return project


def test_matrix_is_the_paper_threshold_sweep() -> None:
    assert METHODS == ("ff", "ff-matched-ge", "ff-ge", "nn-ff-ge")
    assert THRESHOLDS == THRESHOLD_TIE_BREAK == (1.0, 2.0, 4.0)
    assert SEEDS == (424, 425, 426)
    assert len(jobs()) == len(set(jobs())) == 144
    assert Counter(threshold for threshold, *_rest in jobs()) == {
        1.0: 48,
        2.0: 48,
        4.0: 48,
    }


def test_candidate_commands_use_constant_adam_and_never_test_or_download() -> None:
    project = Path("/paper")
    run_dir = Path("/runs")
    runner_parser = suite.parser()
    for threshold, dataset, method, seed in jobs():
        command = command_for(
            "python", project, run_dir, threshold, dataset, method, seed
        )
        args = runner_parser.parse_args(command[2:])
        assert args.defer_test is True
        assert args.download is False
        assert args.epochs == MAXIMUM_EPOCHS == 200
        assert args.patience == PATIENCE == 15
        assert args.minimum_epochs == MINIMUM_EPOCHS == 0
        assert args.validation_size == 5000
        assert args.batch_size == 128
        assert args.evaluation_batch_size == 256
        assert tuple(args.hidden_dims) == ARCHITECTURES[dataset]
        assert args.optimizer == "adam"
        assert args.learning_rate == 1e-3
        assert args.scheduler == "none"
        assert args.goodness_threshold == threshold


def test_plan_is_resumable_and_records_zero_test_evaluations(tmp_path: Path) -> None:
    project = _project(tmp_path)
    run_dir = tmp_path / "run"
    candidate = build_manifest("python", project, run_dir, ("mnist",))
    stored = write_or_resume_manifest(run_dir, candidate)
    resumed = build_manifest("python", project, run_dir, ("mnist",))
    assert write_or_resume_manifest(run_dir, resumed) == stored
    assert len(stored["jobs"]) == 36
    protocol = stored["protocol"]
    assert protocol["candidate_training_test_policy"] == "deferred; zero test evaluations"
    assert protocol["selected_test_evaluations"] == 0
    assert protocol["selection_scope"] == "one threshold per dataset and method"
    assert protocol["threshold_tie_break_order"] == [1.0, 2.0, 4.0]

    changed = json.loads(json.dumps(stored))
    changed["protocol"]["patience"] = 99
    changed["spec_sha256"] = "different"
    with pytest.raises(RuntimeError, match="different protocol"):
        write_or_resume_manifest(run_dir, changed)


def _synthetic_records(dataset: str = "mnist"):
    records = {}
    for threshold, ds, method, seed in jobs((dataset,)):
        if method == "ff":
            value = 80.0  # exact tie; threshold 1 must win
        elif method == "ff-matched-ge":
            value = {1.0: 81.0, 2.0: 82.0, 4.0: 83.0}[threshold]
        else:
            value = {1.0: 82.0, 2.0: 84.0, 4.0: 83.0}[threshold]
        records[(threshold, ds, method, seed)] = {
            "seed": seed,
            "best_validation_accuracy": value + (seed - 425) * 0.01,
            "best_epoch": 10,
            "run_path": f"/{threshold}/{ds}/{method}/{seed}/run.json",
            "run_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "history_sha256": "c" * 64,
            "checkpoint_path": f"/{threshold}/{ds}/{method}/{seed}/best_model.pt",
            "checkpoint_sha256": "d" * 64,
            "validation_index_sha256": f"{seed:064x}",
        }
    return records


def test_selection_uses_mean_validation_with_stable_ties_and_writes_optimizer_input(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = {
        "run_dir": str(run_dir),
        "integrity_sha256": "manifest-hash",
        "protocol": {"datasets": ["mnist"]},
    }
    selected, evidence, rows = freeze_selection(manifest, _synthetic_records())
    assert selected["mnist"] == {
        "ff": 1.0,
        "ff-matched-ge": 4.0,
        "ff-ge": 2.0,
        "nn-ff-ge": 2.0,
    }
    assert evidence["test_metrics_used_for_selection"] is False
    assert evidence["test_evaluations"] == 0
    assert len(rows) == len(METHODS) * len(THRESHOLDS)

    write_or_verify_selection(manifest, selected, evidence, rows)
    # Repeating selection is an integrity check and must not change the files.
    write_or_verify_selection(manifest, selected, evidence, rows)
    plain_path, evidence_path, csv_path = selection_paths(run_dir)
    plain = json.loads(plain_path.read_text(encoding="utf-8"))
    assert plain == selected
    assert validate_thresholds(plain, ("mnist",)) == selected
    assert evidence_path.is_file()
    assert csv_path.is_file()
    assert "integrity_sha256" not in plain


def test_deadline_lock_and_subcommands(tmp_path: Path) -> None:
    now = datetime(2030, 1, 1, tzinfo=timezone.utc)
    assert DEADLINE_BUFFER == timedelta(minutes=20)
    assert not deadline_allows_start(now + DEADLINE_BUFFER, now=now)
    assert deadline_allows_start(now + DEADLINE_BUFFER + timedelta(seconds=1), now=now)
    assert deadline_allows_start(None, now=now)

    root = parser()
    for command in ("plan", "launch", "status", "select"):
        parsed = root.parse_args([command, "--run-dir", "/run"])
        assert parsed.command == command

    lock, token = acquire_lock(tmp_path / "lock-run")
    try:
        with pytest.raises(RuntimeError, match="Another scheduler operation"):
            acquire_lock(tmp_path / "lock-run")
    finally:
        release_lock(lock, token)

    if os.name != "nt":
        lock.write_text(
            json.dumps({"hostname": socket.gethostname(), "pid": -1, "token": "stale"}),
            encoding="utf-8",
        )
        recovered, recovered_token = acquire_lock(tmp_path / "lock-run")
        release_lock(recovered, recovered_token)
        assert not recovered.exists()
