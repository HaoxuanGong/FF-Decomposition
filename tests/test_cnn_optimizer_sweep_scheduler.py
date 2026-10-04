from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

import pytest

from CNNOptimizerSweepScheduler import (
    DATASETS,
    METHODS,
    PROFILES,
    SEEDS,
    SOURCE_FILES,
    command_for,
    expected_config,
    expected_learning_rate,
    finalize_selected_tests,
    jobs,
    prepare_manifest,
    select_profiles,
    write_summary,
)
from LocalBPCNNBenchmark import config_from_args, parse_args


def test_cnn_paper_sweep_defines_72_deferred_candidate_jobs() -> None:
    matrix = jobs()
    assert len(matrix) == 72
    assert len(set(matrix)) == 72
    assert {profile for profile, _dataset, _method, _seed in matrix} == set(PROFILES)
    assert {dataset for _profile, dataset, _method, _seed in matrix} == set(DATASETS)
    assert {method for _profile, _dataset, method, _seed in matrix} == set(METHODS)
    assert {seed for _profile, _dataset, _method, seed in matrix} == set(SEEDS)


def test_cnn_fresh_launch_accepts_precreated_log_and_pid(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    for name in SOURCE_FILES:
        (project_root / name).write_text(f"source:{name}\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "scheduler.log").touch()
    (run_dir / "scheduler.pid").write_text("123\n", encoding="utf-8")

    manifest = prepare_manifest(
        run_dir, project_root, sys.executable, ("mnist",)
    )
    assert manifest["datasets"] == ["mnist"]
    assert (run_dir / "manifest.json").is_file()


@pytest.mark.parametrize("profile", tuple(PROFILES))
@pytest.mark.parametrize("method", METHODS)
def test_candidate_command_matches_the_exact_core_config(
    tmp_path: Path, profile: str, method: str
) -> None:
    project_root = (tmp_path / "project").resolve()
    run_dir = (tmp_path / "run").resolve()
    command = command_for(
        sys.executable, project_root, run_dir, profile, "cifar100", method,
        425, "a" * 32,
    )
    observed = json.loads(json.dumps(asdict(config_from_args(parse_args(command[2:])))))
    assert observed == expected_config(
        project_root, run_dir, profile, "cifar100", method, 425, "a" * 32
    )
    assert "--defer-test" in command
    assert "--save-checkpoints" in command
    assert command[command.index("--step-size") + 1] == "30"
    assert command[command.index("--step-gamma") + 1] == "0.1"


def test_learning_rate_traces_match_the_paper_profiles() -> None:
    assert expected_learning_rate("adam_cosine", 1) == pytest.approx(0.001)
    assert expected_learning_rate("adam_cosine", 200) > 0.0
    assert expected_learning_rate("sgd_step30", 30) == pytest.approx(0.1)
    assert expected_learning_rate("sgd_step30", 31) == pytest.approx(0.01)
    assert expected_learning_rate("sgd_step30", 61) == pytest.approx(0.001)


def _verified_matrix() -> dict[tuple[str, str, str, int], dict[str, object]]:
    verified: dict[tuple[str, str, str, int], dict[str, object]] = {}
    process = 0
    for profile, dataset, method, seed in jobs(("fashionmnist",)):
        process += 1
        score = 0.82 if profile == "sgd_step30" and method == "local-bp" else 0.80
        verified[(profile, dataset, method, seed)] = {
            "profile": profile,
            "dataset": dataset,
            "method": method,
            "seed": seed,
            "config": {
                "method": method,
                "output_dir": f"/{profile}/{method}/{seed}",
                "heartbeat_file": f"/{profile}/{method}/{seed}/heartbeat.json",
                "shared": "paper-protocol",
            },
            "split_sha256": f"{seed:064x}",
            "best_validation_accuracy": score,
            "best_epoch": 10,
            "checkpoint_path": f"/{profile}/{method}/{seed}.pt",
            "checkpoint_sha256": f"{process:064x}",
            "model_state_sha256": f"{process + 100:064x}",
            "process_instance_id": f"{process:032x}",
            "wall_seconds": 1.0,
        }
    return verified


def test_selection_is_validation_only_and_final_tests_are_exactly_once(
    tmp_path: Path,
) -> None:
    verified = _verified_matrix()
    manifest = {
        "provenance_run_id": "a" * 32,
        "source_sha256": {"runner": "b" * 64},
    }
    selection = select_profiles(tmp_path, verified, manifest, ("fashionmnist",))
    winners = {row["method"]: row for row in selection["winners"]}
    assert winners["local-bp"]["selected_profile"] == "sgd_step30"
    assert winners["bp"]["selected_profile"] == "adam_cosine"
    assert selection["selection_policy"]["test_evaluations_before_selection"] == 0
    assert selection["selection_policy"]["test_metrics_used_for_selection"] is False

    calls: list[tuple[str, str, int]] = []

    def evaluator(item: dict[str, object]) -> float:
        calls.append((str(item["dataset"]), str(item["method"]), int(item["seed"])))
        return 0.75

    first = finalize_selected_tests(tmp_path, selection, verified, evaluator)
    second = finalize_selected_tests(tmp_path, selection, verified, evaluator)
    assert len(first) == len(METHODS) * len(SEEDS)
    assert second == first
    assert len(calls) == len(METHODS) * len(SEEDS)
    assert all(row["test_evaluations"] == 1 for row in first.values())

    summary = write_summary(tmp_path, selection, first)
    assert len(summary["rows"]) == len(METHODS)
    assert summary["verification"]["only_selected_checkpoints_tested"] is True
    assert (tmp_path / "cnn_paper_results.csv").is_file()


def test_interrupted_final_test_is_not_silently_repeated(tmp_path: Path) -> None:
    verified = _verified_matrix()
    manifest = {
        "provenance_run_id": "a" * 32,
        "source_sha256": {"runner": "b" * 64},
    }
    selection = select_profiles(tmp_path, verified, manifest, ("fashionmnist",))
    first = selection["winners"][0]
    started = (
        tmp_path / "final_tests" / first["dataset"] / first["method"]
        / f"seed_{SEEDS[0]}.started.json"
    )
    started.parent.mkdir(parents=True)
    started.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="refusing to evaluate it again"):
        finalize_selected_tests(tmp_path, selection, verified, lambda _item: 0.5)
