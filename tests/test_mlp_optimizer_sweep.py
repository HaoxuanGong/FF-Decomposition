from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import socket

import pytest

import MLPBenchmarkSuite as suite
from MLPOptimizerSweepScheduler import (
    DEADLINE_BUFFER,
    FC_METHODS,
    FROZEN_THRESHOLDS,
    MAXIMUM_EPOCHS,
    MINIMUM_EPOCHS,
    PATIENCE,
    PRIMARY_METHODS,
    PROFILES,
    PROFILE_TIE_BREAK,
    SEEDS,
    SOURCE_FILES,
    THRESHOLDED_METHODS,
    acquire_lock,
    aggregate_results,
    build_manifest,
    command_for,
    deadline_allows_start,
    file_sha256,
    finalize_selected,
    freeze_selection,
    identities,
    jobs,
    methods_for,
    parser,
    release_lock,
    signed_payload,
    validate_thresholds,
    write_or_resume_manifest,
)


EXPECTED_PRIMARY_METHODS = (
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


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    for name in SOURCE_FILES:
        (project / name).write_text(f"fixture:{name}\n", encoding="utf-8")
    return project


def test_matrix_is_the_canonical_paper_matrix() -> None:
    assert PRIMARY_METHODS == EXPECTED_PRIMARY_METHODS
    assert tuple(suite.PRIMARY_METHODS) == PRIMARY_METHODS
    assert len(identities()) == 40
    assert len(jobs()) == len(set(jobs())) == 360
    assert Counter(profile for profile, *_rest in jobs()) == {
        "adam_constant": 120,
        "adam_cosine": 120,
        "sgd_step30": 120,
    }
    assert methods_for("cifar100") == tuple(
        method for method in PRIMARY_METHODS if method not in FC_METHODS
    )
    assert not FC_METHODS.intersection(methods_for("cifar100"))
    assert "ff-matched-local" not in PRIMARY_METHODS
    assert "ce-matched-local" not in PRIMARY_METHODS


def test_candidate_commands_match_the_three_profiles_and_defer_test() -> None:
    project = Path("/paper")
    run_dir = Path("/runs")
    parser_ = suite.parser()
    for profile, dataset, method, seed in jobs():
        command = command_for("python", project, run_dir, profile, dataset, method, seed)
        args = parser_.parse_args(command[2:])
        assert args.defer_test is True
        assert args.epochs == MAXIMUM_EPOCHS == 200
        assert args.patience == PATIENCE == 15
        assert args.minimum_epochs == MINIMUM_EPOCHS == 0
        assert args.validation_size == 5000
        assert args.batch_size == 128
        assert args.evaluation_batch_size == 256
        assert args.hidden_dims == list(
            (1000, 1000) if dataset in {"mnist", "fashionmnist"} else (2000,) * 3
        )
        settings = PROFILES[profile]
        assert args.optimizer == settings["optimizer"]
        assert args.learning_rate == settings["learning_rate"]
        assert args.scheduler == settings["scheduler"]
        if profile == "sgd_step30":
            assert args.momentum == 0.9
            assert args.step_size == 30
            assert args.step_gamma == 0.1
        if method in THRESHOLDED_METHODS:
            assert args.goodness_threshold == FROZEN_THRESHOLDS[dataset][method]


def test_thresholds_are_frozen_inputs_not_an_optimizer_profile() -> None:
    assert not any("threshold" in profile for profile in PROFILES)
    assert validate_thresholds(FROZEN_THRESHOLDS) == FROZEN_THRESHOLDS
    broken = json.loads(json.dumps(FROZEN_THRESHOLDS))
    del broken["mnist"]["ff"]
    with pytest.raises(ValueError, match="must be"):
        validate_thresholds(broken)
    broken = json.loads(json.dumps(FROZEN_THRESHOLDS))
    broken["mnist"]["ff"] = 0
    with pytest.raises(ValueError, match="Invalid frozen threshold"):
        validate_thresholds(broken)


def test_manifest_is_resumable_and_records_validation_only_selection(tmp_path: Path) -> None:
    project = _project(tmp_path)
    run_dir = tmp_path / "run"
    first = build_manifest(
        "python", project, run_dir, ("fashionmnist",), threshold_source={"kind": "test"}
    )
    stored = write_or_resume_manifest(run_dir, first)
    resumed = build_manifest(
        "python", project, run_dir, ("fashionmnist",), threshold_source={"kind": "test"}
    )
    assert write_or_resume_manifest(run_dir, resumed) == stored
    assert len(stored["jobs"]) == 99
    protocol = stored["protocol"]
    assert protocol["candidate_training_test_policy"] == "deferred; zero test evaluations"
    assert protocol["selected_test_evaluations"] == 33
    assert protocol["threshold_policy"]["searched_by_this_scheduler"] is False
    assert protocol["threshold_policy"]["frozen_before_optimizer_profile_selection"] is True
    assert "test metrics are unavailable" in protocol["profile_selection"]

    changed = json.loads(json.dumps(stored))
    changed["protocol"]["patience"] = 99
    changed["spec_sha256"] = "different"
    changed = signed_payload(changed)
    with pytest.raises(RuntimeError, match="different protocol"):
        write_or_resume_manifest(run_dir, changed)


def _selection_records(dataset: str = "fashionmnist"):
    records = {}
    for profile, ds, method, seed in jobs((dataset,)):
        # BP is an exact tie and must follow the declared profile order.  Every
        # other method selects SGD.  Deliberately contradictory test numbers
        # demonstrate that selection never reads them.
        validation = 80.0 if method == "bp" else (90.0 if profile == "sgd_step30" else 70.0)
        records[(profile, ds, method, seed)] = {
            "seed": seed,
            "best_validation_accuracy": validation + (seed - 425) * 0.01,
            "test_accuracy": 100.0 if profile == "adam_cosine" else 0.0,
            "run_path": f"/{profile}/{ds}/{method}/{seed}/run.json",
            "run_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "history_sha256": "c" * 64,
            "checkpoint_path": f"/{profile}/{ds}/{method}/{seed}/best_model.pt",
            "checkpoint_sha256": "d" * 64,
            "validation_index_sha256": f"{seed:064x}",
        }
    return records


def test_profile_selection_uses_mean_validation_only_and_has_stable_ties() -> None:
    manifest = {
        "integrity_sha256": "manifest",
        "protocol": {"datasets": ["fashionmnist"]},
    }
    selection = freeze_selection(manifest, _selection_records())
    assert selection["test_metrics_used_for_selection"] is False
    assert tuple(selection["profile_tie_break_order"]) == PROFILE_TIE_BREAK
    choices = {choice["method"]: choice for choice in selection["choices"]}
    assert choices["bp"]["selected_profile"] == "adam_constant"
    assert choices["local-bp"]["selected_profile"] == "sgd_step30"
    assert all(choice["test_metrics_used_for_selection"] is False for choice in choices.values())


def _frozen_artifact(directory: Path, name: str, text: str) -> tuple[str, str]:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path), file_sha256(path)


def test_finalize_evaluates_only_selected_checkpoints_once(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    manifest = {
        "run_dir": str(run_dir),
        "integrity_sha256": "manifest-hash",
        "protocol": {"datasets": ["mnist"]},
    }
    selected_runs = []
    for seed in SEEDS:
        directory = run_dir / "training" / f"seed_{seed}"
        run_path, run_hash = _frozen_artifact(directory, "run.json", f"run {seed}")
        config_path, config_hash = _frozen_artifact(directory, "config.json", f"config {seed}")
        history_path, history_hash = _frozen_artifact(directory, "history.csv", f"history {seed}")
        checkpoint_path, checkpoint_hash = _frozen_artifact(
            directory, "best_model.pt", f"checkpoint {seed}"
        )
        selected_runs.append(
            {
                "seed": seed,
                "run_path": run_path,
                "run_sha256": run_hash,
                "config_sha256": config_hash,
                "history_sha256": history_hash,
                "checkpoint_path": checkpoint_path,
                "checkpoint_sha256": checkpoint_hash,
                "validation_index_sha256": f"{seed:064x}",
            }
        )
    selection = signed_payload(
        {
            "schema_version": 3,
            "created_at_utc": "2030-01-01T00:00:00+00:00",
            "manifest_integrity_sha256": "manifest-hash",
            "selection_metric": "mean best-validation accuracy over seeds 424--426",
            "choices": [
                {
                    "dataset": "mnist",
                    "method": "bp",
                    "selected_profile": "adam_cosine",
                    "selected_runs": selected_runs,
                }
            ],
        }
    )
    calls = []

    def fake_evaluate(_manifest, choice, selected_run, **_kwargs):
        calls.append((choice["selected_profile"], selected_run["seed"]))
        return {"accuracy": 90.0 + selected_run["seed"] - 424, "total": 10_000}, "e" * 64

    monkeypatch.setattr("MLPOptimizerSweepScheduler.evaluate_checkpoint", fake_evaluate)
    first = finalize_selected(manifest, selection, data_dir=tmp_path / "data", device_name="cpu")
    second = finalize_selected(manifest, selection, data_dir=tmp_path / "data", device_name="cpu")
    assert len(first) == len(second) == 3
    assert calls == [("adam_cosine", seed) for seed in SEEDS]
    assert all(record["test_evaluations"] == 1 for record in first)

    summary = aggregate_results(manifest, selection)
    assert summary["test_metrics_used_for_selection"] is False
    assert summary["rows"][0]["per_seed_test_accuracy"] == [90.0, 91.0, 92.0]
    assert summary["rows"][0]["test_accuracy_mean"] == 91.0


def test_deadline_guard_and_subcommands() -> None:
    now = datetime(2030, 1, 1, tzinfo=timezone.utc)
    assert DEADLINE_BUFFER == timedelta(minutes=20)
    assert not deadline_allows_start(now + DEADLINE_BUFFER, now=now)
    assert deadline_allows_start(now + DEADLINE_BUFFER + timedelta(seconds=1), now=now)
    assert deadline_allows_start(None, now=now)

    root = parser()
    for command in ("plan", "launch", "status", "finalize", "aggregate"):
        argv = [command, "--run-dir", "/run"]
        if command == "finalize":
            argv += ["--data-dir", "/data"]
        parsed = root.parse_args(argv)
        assert parsed.command == command


def test_lock_blocks_live_owner_and_recovers_stale_local_owner(tmp_path: Path) -> None:
    lock, token = acquire_lock(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="Another scheduler operation"):
            acquire_lock(tmp_path)
    finally:
        release_lock(lock, token)

    if os.name != "nt":
        lock.write_text(
            json.dumps({"hostname": socket.gethostname(), "pid": -1, "token": "stale"}),
            encoding="utf-8",
        )
        recovered, recovered_token = acquire_lock(tmp_path)
        release_lock(recovered, recovered_token)
        assert not recovered.exists()
