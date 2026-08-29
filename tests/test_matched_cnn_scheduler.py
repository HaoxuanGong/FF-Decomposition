from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from MatchedCNNLocalBPBenchmarkScheduler import (
    DATASETS,
    METHODS,
    SEEDS,
    command_for,
    jobs,
    verify_history,
)


def test_matched_cnn_scheduler_defines_exactly_24_unique_jobs() -> None:
    matrix = jobs()
    assert len(matrix) == 24
    assert len(set(matrix)) == 24
    assert {dataset for dataset, _method, _seed in matrix} == set(DATASETS)
    assert {method for _dataset, method, _seed in matrix} == set(METHODS)
    assert {seed for _dataset, _method, seed in matrix} == set(SEEDS)


def test_matched_cnn_scheduler_uses_the_fixed_protocol_and_one_seed_per_process() -> None:
    command = command_for(
        "/remote/python",
        Path("/remote/project"),
        Path("/remote/run"),
        "cifar100",
        "local-bp",
        425,
        "fresh-run-123",
    )
    joined = " ".join(command)
    assert "--epochs 200" in joined
    assert "--patience 15" in joined
    assert "--minimum-delta 0" in joined
    assert "--validation-size 5000" in joined
    assert "--optimizer sgd" in joined
    assert "--learning-rate 0.1" in joined
    assert "--batch-size 128" in joined
    assert "--eval-batch-size 512" in joined
    assert "--seeds 425" in joined
    assert "--device cuda" in joined
    assert "--no-download" in joined
    assert "--save-checkpoints" in joined
    assert "--provenance-run-id fresh-run-123" in joined
    assert command.count("425") == 1


def _write_history(path: Path, recorded_best_at_epoch_2: float = 0.60) -> None:
    fieldnames = [
        "dataset",
        "method",
        "seed",
        "split_sha256",
        "epoch",
        "learning_rate",
        "train_loss",
        "train_accuracy",
        "validation_accuracy",
        "best_validation_accuracy",
        "best_epoch",
        "epochs_without_improvement",
        "train_seconds",
        "epoch_peak_gpu_training_memory_bytes",
        *(f"train_layer{index}_loss" for index in range(1, 5)),
    ]
    validation = (0.60, 0.60, 0.59)
    rows = []
    for epoch, accuracy in enumerate(validation, start=1):
        rows.append(
            {
                "dataset": "cifar10",
                "method": "bp",
                "seed": 424,
                "split_sha256": "a" * 64,
                "epoch": epoch,
                "learning_rate": 0.05 * (1 + math.cos(math.pi * (epoch - 1) / 10)),
                "train_loss": 1.0,
                "train_accuracy": 0.5,
                "validation_accuracy": accuracy,
                "best_validation_accuracy": (
                    recorded_best_at_epoch_2 if epoch == 2 else 0.60
                ),
                "best_epoch": 1,
                "epochs_without_improvement": epoch - 1,
                "train_seconds": 1.0,
                "epoch_peak_gpu_training_memory_bytes": 100,
                "train_layer1_loss": 1.0,
                "train_layer2_loss": "",
                "train_layer3_loss": "",
                "train_layer4_loss": "",
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_history_verifier_reconstructs_strict_best_and_patience(tmp_path: Path) -> None:
    path = tmp_path / "history.csv"
    _write_history(path)
    evidence = verify_history(
        path,
        maximum_epochs=10,
        patience=2,
        minimum_delta=0.0,
        dataset="cifar10",
        method="bp",
        seed=424,
        split_hash="a" * 64,
    )
    assert evidence == {
        "epochs_trained": 3,
        "best_epoch": 1,
        "best_validation_accuracy": 0.60,
        "final_epochs_without_improvement": 2,
        "early_stopping_triggered": True,
        "termination_reason": "patience",
    }


def test_history_verifier_rejects_a_tie_marked_as_an_improvement(tmp_path: Path) -> None:
    path = tmp_path / "history.csv"
    _write_history(path, recorded_best_at_epoch_2=0.61)
    with pytest.raises(RuntimeError, match="Strict-best validation trace mismatch"):
        verify_history(
            path,
            maximum_epochs=10,
            patience=2,
            minimum_delta=0.0,
            dataset="cifar10",
            method="bp",
            seed=424,
            split_hash="a" * 64,
        )
