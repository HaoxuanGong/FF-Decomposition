from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from BroadMLPBenchmarkScheduler import (
    ARCHITECTURES,
    CORE_METHODS,
    DATASETS,
    FULL_COMPARISON_METHODS,
    METHODS,
    SEEDS,
    _expected_config,
    command_for,
    expected_protocol,
    jobs,
    methods_for,
)
from MLPBenchmarkSuite import configuration, parser


def test_broad_matrix_is_the_exact_75_run_paper_matrix() -> None:
    matrix = jobs()
    assert len(matrix) == 75
    assert len(set(matrix)) == 75
    assert {dataset for dataset, _method, _seed in matrix} == set(DATASETS)
    assert {method for _dataset, method, _seed in matrix} == set(METHODS)
    assert {seed for _dataset, _method, seed in matrix} == set(SEEDS)

    counts = Counter(dataset for dataset, _method, _seed in matrix)
    assert counts == {
        "mnist": 21,
        "fashionmnist": 21,
        "cifar10": 21,
        "cifar100": 12,
    }
    assert methods_for("cifar100") == CORE_METHODS
    assert all(
        method not in FULL_COMPARISON_METHODS
        for dataset, method, _seed in matrix
        if dataset == "cifar100"
    )
    for dataset in ("mnist", "fashionmnist", "cifar10"):
        assert methods_for(dataset) == METHODS


def test_every_command_constructs_the_exact_runner_configuration() -> None:
    project_root = Path("/project")
    run_dir = Path("/run")
    suite_parser = parser()

    for dataset, method, seed in jobs():
        command = command_for(
            "python",
            project_root,
            run_dir,
            dataset,
            method,
            seed,
        )
        assert command[:3] == [
            "python",
            str(project_root / "MLPBenchmarkSuite.py"),
            "run",
        ]
        args = suite_parser.parse_args(command[2:])
        observed = configuration(args)
        expected = _expected_config(dataset, method, seed)
        assert {
            key: observed.get(key) for key in expected
        } == expected
        assert list(args.hidden_dims) == ARCHITECTURES[dataset]
        assert args.epochs == 200
        assert args.patience == 15
        assert args.minimum_epochs == 0
        assert args.validation_size == 5000
        assert args.optimizer == "adam"
        assert args.learning_rate == 0.001
        assert args.batch_size == 128
        assert args.evaluation_batch_size == 256
        assert args.scheduler == "none"
        assert args.num_workers == 4
        assert args.candidate_chunk == 10
        assert args.device == "cuda"
        assert args.download is False
        assert args.train_limit is None
        assert args.test_limit is None


def test_protocol_records_all_fixed_non_command_controls() -> None:
    protocol = expected_protocol()
    assert protocol["architectures"] == ARCHITECTURES
    assert protocol["seeds"] == list(SEEDS)
    assert protocol["maximum_epochs"] == 200
    assert protocol["validation_size"] == 5000
    assert protocol["early_stopping_patience"] == 15
    assert protocol["early_stopping_strict_improvement"] is True
    assert protocol["restore_best_checkpoint"] is True
    assert protocol["test_evaluations_per_run"] == 1
    assert protocol["scheduler"] == "none"
    assert protocol["weight_decay"] == 0.0
    assert protocol["dropout"] == 0.0
    assert protocol["augmentation"] == "none"
    assert protocol["goodness_threshold_for_pairwise_ff"] == 2.0
    assert protocol["normalization_epsilon_for_ff_inputs"] == 1e-4


def test_command_rejects_jobs_outside_the_fixed_matrix() -> None:
    with pytest.raises(ValueError, match="not scheduled"):
        command_for(
            "python",
            Path("/project"),
            Path("/run"),
            "cifar100",
            "fc-ff-ge",
            424,
        )
    with pytest.raises(ValueError, match="Unexpected seed"):
        command_for(
            "python",
            Path("/project"),
            Path("/run"),
            "mnist",
            "bp",
            999,
        )
