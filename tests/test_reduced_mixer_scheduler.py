from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from MLPMixerBenchmarkSuite import (
    RESULT_SCHEMA_VERSION,
    TINY_REFERENCE_URL,
    TINY_URL,
    source_sha256,
)

from ReducedMixerBenchmarkScheduler import (
    DATASETS,
    METHODS,
    PATCH_SIZES,
    SEEDS,
    aggregate,
    command_for,
    file_sha256,
    job_dir,
    jobs,
    prepare_run_directory,
)


def test_reduced_mixer_scheduler_defines_24_fresh_process_jobs() -> None:
    matrix = jobs()
    assert len(matrix) == 24
    assert len(set(matrix)) == 24
    assert {dataset for dataset, _seed, _method in matrix} == set(DATASETS)
    assert {seed for _dataset, seed, _method in matrix} == set(SEEDS)
    assert {method for _dataset, _seed, method in matrix} == set(METHODS)


def test_reduced_mixer_command_uses_the_fixed_protocol(tmp_path: Path) -> None:
    command = command_for(
        "python",
        Path("project"),
        tmp_path,
        "cifar10",
        "local-bp",
        424,
    )
    joined = " ".join(map(str, command))
    assert "--depths 5" in joined
    assert "--dims 256" in joined
    assert "--epochs 200" in joined
    assert "--early-stop-patience 15" in joined
    assert "--early-stop-min-delta 0" in joined
    assert "--optimizer adamw" in joined
    assert "--lr 0.0003" in joined
    assert "--weight-decay 0.05" in joined
    assert "--batch-size 128" in joined
    assert "--local-bp-updates-per-block 1" in joined
    assert "--checkpoint-dir" in joined
    assert "--download-tinyimagenet" not in command

    tiny_command = command_for(
        "python",
        Path("project"),
        tmp_path,
        "tinyimagenet",
        "bp",
        424,
    )
    assert "--download-tinyimagenet" in tiny_command


def test_reduced_mixer_resume_requires_exact_manifest_identity(tmp_path: Path) -> None:
    manifest = {
        "schema_version": 2,
        "project_root": "project",
        "run_dir": str(tmp_path),
        "python": "python",
        "scheduler_environment": {},
        "protocol": {"epochs": 200},
        "source_sha256": {"suite": "a" * 64},
        "dataset_integrity": {"tinyimagenet": {"tree_sha256": "c" * 64}},
        "jobs": [{"dataset": "cifar10"}],
        "started_at_utc": "first",
    }
    assert prepare_run_directory(tmp_path, manifest) is False
    resumed = dict(manifest, started_at_utc="second")
    assert prepare_run_directory(tmp_path, resumed) is True
    changed = dict(resumed, source_sha256={"suite": "b" * 64})
    with pytest.raises(RuntimeError, match="source/configuration"):
        prepare_run_directory(tmp_path, changed)


def test_reduced_mixer_accepts_only_launcher_precreated_entries(
    tmp_path: Path,
) -> None:
    manifest = {
        "schema_version": 2,
        "project_root": "project",
        "run_dir": str(tmp_path),
        "python": "python",
        "scheduler_environment": {},
        "protocol": {"epochs": 200},
        "source_sha256": {"suite": "a" * 64},
        "dataset_integrity": {"tinyimagenet": {"tree_sha256": "c" * 64}},
        "jobs": [{"dataset": "cifar10"}],
        "started_at_utc": "first",
    }
    (tmp_path / "source").mkdir()
    (tmp_path / "source_snapshot.sha256").write_text(
        "snapshot", encoding="utf-8"
    )
    (tmp_path / "scheduler.log").touch()
    assert prepare_run_directory(tmp_path, manifest) is False
    assert (tmp_path / "manifest.json").is_file()

    unexpected = tmp_path / "unexpected"
    unexpected.mkdir()
    (unexpected / "source").mkdir()
    (unexpected / "source_snapshot.sha256").write_text(
        "snapshot", encoding="utf-8"
    )
    (unexpected / "partial-result.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="without a manifest"):
        prepare_run_directory(unexpected, dict(manifest, run_dir=str(unexpected)))


def _fake_dataset_integrity(dataset: str) -> dict[str, object]:
    if dataset != "tinyimagenet":
        return {}
    return {
        "dataset": "tinyimagenet",
        "status": "expected_structure_validated",
        "source_url": TINY_URL,
        "reference_url": TINY_REFERENCE_URL,
        "source_published_checksum_available": False,
        "fingerprint_scope": "test fixture",
        "expected_structure": {},
        "archive": {"available": False, "sha256": "", "size_bytes": 0},
        "tree_sha256": "d" * 64,
        "tree_file_count": 110002,
        "tree_total_bytes": 1,
    }


def _write_fake_result(
    run_dir: Path,
    dataset: str,
    method: str,
    seed: int,
) -> None:
    dataset_integrity = _fake_dataset_integrity(dataset)
    checkpoint = run_dir / "checkpoints" / dataset / method / f"seed_{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    signature_payload = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "source_sha256": source_sha256(),
        "dataset_integrity": dataset_integrity,
        "dataset": dataset,
        "method": method,
        "depth": 5,
        "dim": 256,
        "token_dim": 256,
        "channel_dim": 1024,
        "patch_size": PATCH_SIZES[dataset],
        "seed": seed,
        "epochs": 200,
        "early_stop_patience": 15,
        "early_stop_min_delta": 0.0,
        "validation_fraction": 0.1,
        "dropout": 0.1,
        "scheduler": "cosine_annealing",
        "scheduler_t_max": 200,
        "batch_size": 128,
        "eval_batch_size": 256,
        "optimizer": "adamw",
        "lr": 0.0003,
        "weight_decay": 0.05,
        "momentum": 0.9,
        "local_bp_updates_per_block": 1 if method == "local-bp" else 0,
        "amp": True,
        "device": "cuda",
        "num_workers": 4,
    }
    signature = json.dumps(signature_payload, sort_keys=True, separators=(",", ":"))
    history_path = run_dir / "history" / dataset / method / f"seed_{seed}.json"
    history = []
    best = float("-inf")
    best_epoch = 0
    bad_epochs = 0
    for epoch in range(1, 36):
        validation = 0.5 + 0.01 * min(epoch, 20)
        improved = validation > best
        if improved:
            best = validation
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "learning_rate": 0.0003,
                "train_loss": 1.0 / epoch,
                "train_primary": validation,
                "train_secondary": validation,
                "validation_loss": 1.0 / epoch,
                "validation_primary": validation,
                "validation_secondary": validation,
                "improved": improved,
                "best_validation_primary": best,
                "best_epoch": best_epoch,
                "epochs_without_improvement": bad_epochs,
            }
        )
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        json.dumps(
            {
                "schema_version": RESULT_SCHEMA_VERSION,
                "run_signature": signature,
                "source_sha256": source_sha256(),
                "dataset_integrity": dataset_integrity,
                "epochs": history,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    history_digest = file_sha256(history_path)
    torch.save(
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "state_dict": {"weight": torch.tensor([1.0])},
            "run_signature": signature,
            "best_epoch": 20,
            "best_validation_primary": 0.70,
            "history_sha256": history_digest,
            "source_sha256": source_sha256(),
            "dataset_integrity": dataset_integrity,
        },
        checkpoint,
    )
    is_local = method == "local-bp"
    row = {
        "result_schema_version": str(RESULT_SCHEMA_VERSION),
        "suite_source_sha256": source_sha256()["MLPMixerBenchmarkSuite.py"],
        "support_source_sha256": source_sha256()["medmnist_support.py"],
        "status": "ok",
        "run_signature": signature,
        "test_evaluations": "1",
        "dataset": dataset,
        "method": method,
        "seed": str(seed),
        "depth": "5",
        "dim": "256",
        "token_dim": "256",
        "channel_dim": "1024",
        "patch_size": str(PATCH_SIZES[dataset]),
        "num_classes": "10",
        "in_channels": "3",
        "image_size": "32",
        "task": "multi-class",
        "metric_primary_name": "top1",
        "metric_secondary_name": "top5",
        "dataset_source": dataset,
        "dataset_version": "test-version",
        "training_augmentation": "test-augmentation",
        "final_evaluation_split": "official_test",
        "epochs_target": "200",
        "epochs_ran": "35",
        "early_stopped": "1",
        "early_stop_patience": "15",
        "early_stop_min_delta": "0.0",
        "validation_fraction": "0.1",
        "validation_is_official": "0",
        "dropout": "0.1",
        "scheduler": "cosine_annealing",
        "scheduler_t_max": "200",
        "local_bp_updates_per_block": "1" if is_local else "0",
        "optimizer": "adamw",
        "lr": "0.0003",
        "weight_decay": "0.05",
        "momentum": "0.9",
        "batch_size": "128",
        "eval_batch_size": "256",
        "num_workers": "4",
        "train_samples": "45000",
        "validation_samples": "5000",
        "test_samples": "10000",
        "split_seed": str(seed),
        "split_protocol": "seed_specific_stratified_training_holdout",
        "train_index_sha256": f"{seed:064x}",
        "validation_index_sha256": f"{seed + 1:064x}",
        "test_index_sha256": f"{seed + 2:064x}",
        "device": "cuda",
        "amp_enabled": "1",
        "best_validation_primary": "0.70",
        "best_validation_secondary": "0.70",
        "best_validation_loss": "0.05",
        "best_epoch": "20",
        "test_primary": "0.71" if is_local else "0.70",
        "test_secondary": "0.91" if is_local else "0.90",
        "test_loss": "0.04",
        "last_train_loss": "0.03",
        "last_train_primary": "0.70",
        "last_train_secondary": "0.70",
        "peak_train_mem_gb": "0.4" if is_local else "1.0",
        "runtime_seconds": "10.0",
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "history_path": str(history_path),
        "history_sha256": history_digest,
        "terminal_bad_epochs": "15",
        "selection_rule": "validation_primary_strict_improvement",
        "restored_best_state_verified": "1",
        "checkpoint_reload_verified": "1",
        "torch_version": "test-torch",
        "torchvision_version": "test-torchvision",
        "cuda_runtime_version": "test-cuda",
        "cudnn_version": "test-cudnn",
        "gpu_name": "test-gpu",
        "notes_json": json.dumps(
            {"dataset_integrity": dataset_integrity}, sort_keys=True
        ),
    }
    output = job_dir(run_dir, dataset, method, seed)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "result.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def test_reduced_mixer_aggregate_computes_paired_deltas_and_memory_ratios(
    tmp_path: Path,
) -> None:
    for dataset in DATASETS:
        for seed in SEEDS:
            for method in METHODS:
                _write_fake_result(tmp_path, dataset, method, seed)

    summary = aggregate(tmp_path)

    assert summary["completed_runs"] == 24
    assert len(summary["method_rows"]) == 8
    assert len(summary["paired_rows"]) == 4
    for row in summary["paired_rows"]:
        assert row["local_minus_bp_primary_mean"] == pytest.approx(0.01)
        assert row["local_over_bp_memory_mean"] == pytest.approx(0.4)
    assert (tmp_path / "reduced_mixer_summary.json").is_file()
    assert (tmp_path / "reduced_mixer_summary.csv").is_file()
    assert (tmp_path / "reduced_mixer_paired.csv").is_file()
