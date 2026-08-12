"""Focused checks for the fixed main-text decomposition runner."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import FFDecompositionBenchmark as benchmark


def test_public_protocol_defaults_are_fixed():
    args = benchmark.parse_args([])

    assert args.datasets == ["mnist", "fashionmnist", "cifar10"]
    assert args.methods == ["ff", "ff-ge", "fc-ff-ge", "fc-nn-ff-ge", "bp"]
    assert args.seeds == [424, 425, 426]
    assert args.epochs == 200
    assert benchmark.HIDDEN_DIMS == (1000, 1000)
    assert benchmark.BATCH_SIZE == 128
    assert benchmark.LEARNING_RATE == pytest.approx(0.001)


def test_dataset_registry_contains_only_main_text_datasets():
    assert tuple(benchmark.DATASETS) == ("mnist", "fashionmnist", "cifar10")
    assert benchmark.DATASETS["mnist"].input_dim == 784
    assert benchmark.DATASETS["fashionmnist"].input_dim == 784
    assert benchmark.DATASETS["cifar10"].input_dim == 3072


@pytest.mark.parametrize(
    ("method", "expected_normalization"),
    [
        ("ff", True),
        ("ff-ge", True),
        ("fc-ff-ge", True),
        ("fc-nn-ff-ge", False),
    ],
)
def test_goodness_methods_use_the_prescribed_normalization(method, expected_normalization):
    model = benchmark.build_model(method, input_dim=784, num_classes=10)

    assert isinstance(model, benchmark.GoodnessMLP)
    assert model.normalize is expected_normalization
    assert [(layer.in_features, layer.out_features) for layer in model.layers] == [
        (784, 1000),
        (1000, 1000),
    ]


def test_bp_has_two_matched_hidden_layers_and_a_clean_classifier():
    model = benchmark.build_model("bp", input_dim=784, num_classes=10)

    assert isinstance(model, benchmark.BackpropMLP)
    linear_layers = [layer for layer in model.features if isinstance(layer, torch.nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in linear_layers] == [
        (784, 1000),
        (1000, 1000),
    ]
    assert (model.classifier.in_features, model.classifier.out_features) == (1000, 10)


def test_wrong_label_sampling_never_returns_the_true_class():
    torch.manual_seed(7)
    labels = torch.arange(10).repeat(100)

    wrong = benchmark.sample_wrong_labels(labels, num_classes=10)

    assert torch.all(wrong != labels)
    assert torch.all((0 <= wrong) & (wrong < 10))


def test_candidate_goodness_scores_all_classes_and_propagates_global_error():
    model = benchmark.GoodnessMLP(6, hidden_dims=(5, 4), normalize=True)
    inputs = torch.randn(3, 6)

    scores = benchmark.candidate_goodness(
        model,
        inputs,
        num_classes=3,
        aggregation="sum",
        chunk_size=2,
    )
    scores.sum().backward()

    assert scores.shape == (3, 3)
    assert all(layer.weight.grad is not None for layer in model.layers)


def test_vanilla_ff_uses_one_optimizer_per_layer():
    model = benchmark.GoodnessMLP(8, hidden_dims=(4, 3), normalize=True)

    optimizers = benchmark._make_optimizers("ff", model)

    assert len(optimizers) == 2
    first_parameters = set(optimizers[0].param_groups[0]["params"])
    second_parameters = set(optimizers[1].param_groups[0]["params"])
    assert first_parameters == set(model.layers[0].parameters())
    assert second_parameters == set(model.layers[1].parameters())
    assert first_parameters.isdisjoint(second_parameters)


@pytest.mark.parametrize("method", benchmark.METHOD_NAMES)
def test_every_method_completes_a_tiny_training_epoch(method):
    inputs = torch.randn(12, 12)
    labels = torch.arange(12) % 3
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=4, shuffle=False)
    if method == "bp":
        model = benchmark.BackpropMLP(12, 3, hidden_dims=(7, 5))
    else:
        model = benchmark.GoodnessMLP(
            12,
            hidden_dims=(7, 5),
            normalize=method != "fc-nn-ff-ge",
        )
    optimizers = benchmark._make_optimizers(method, model)

    loss = benchmark.train_one_epoch(
        method,
        model,
        loader,
        optimizers,
        device=torch.device("cpu"),
        num_classes=3,
        candidate_chunk=2,
    )

    assert math.isfinite(loss)


def test_run_seed_reads_test_loader_only_after_all_epochs(monkeypatch, tmp_path: Path):
    events = []
    training_data = TensorDataset(torch.randn(4, 2), torch.tensor([0, 1, 0, 1]))
    bundle = benchmark.DatasetBundle(training_data, training_data, input_dim=2, num_classes=2)
    loaders = benchmark.Loaders(
        train="training-loader",
        train_evaluation="training-evaluation-loader",
        test="test-loader",
    )
    monkeypatch.setattr(benchmark, "make_loaders", lambda *_args, **_kwargs: loaders)
    monkeypatch.setattr(
        benchmark,
        "build_model",
        lambda *_args, **_kwargs: torch.nn.Linear(2, 2),
    )

    def fake_train(*_args, **_kwargs):
        events.append("train")
        return 0.25

    def fake_accuracy(_method, _model, loader, **_kwargs):
        events.append(loader)
        return 75.0 if loader == "training-evaluation-loader" else 70.0

    monkeypatch.setattr(benchmark, "train_one_epoch", fake_train)
    monkeypatch.setattr(benchmark, "classification_accuracy", fake_accuracy)

    result = benchmark.run_seed(
        dataset_name="mnist",
        method="bp",
        seed=424,
        epochs=3,
        bundle=bundle,
        output_dir=tmp_path,
        device=torch.device("cpu"),
        evaluation_batch_size=4,
        candidate_chunk=2,
        num_workers=0,
        resume=False,
    )

    assert events == [
        "train",
        "train",
        "train",
        "training-evaluation-loader",
        "test-loader",
    ]
    assert result["final_test_accuracy"] == 70.0
    with (tmp_path / "mnist" / "bp" / "seed_424" / "history.csv").open(encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 3


def test_aggregate_results_reports_sample_standard_deviation():
    results = [
        {
            "config": {"dataset": "mnist", "method": "bp", "seed": seed, "epochs": 200},
            "final_train_accuracy": value + 1,
            "final_test_accuracy": value,
        }
        for seed, value in zip((424, 425, 426), (90.0, 92.0, 94.0), strict=True)
    ]

    summary = benchmark.aggregate_results(results)

    assert len(summary) == 1
    assert summary[0]["epochs"] == 200
    assert summary[0]["final_test_accuracy_mean"] == pytest.approx(92.0)
    assert summary[0]["final_test_accuracy_std"] == pytest.approx(2.0)


def test_aggregate_results_keeps_different_epoch_protocols_separate():
    results = [
        {
            "config": {
                "dataset": "mnist",
                "method": "bp",
                "seed": seed,
                "epochs": epochs,
            },
            "final_train_accuracy": accuracy + 1,
            "final_test_accuracy": accuracy,
        }
        for seed, epochs, accuracy in ((424, 1, 50.0), (425, 200, 90.0))
    ]

    summary = benchmark.aggregate_results(results)

    assert [(row["epochs"], row["runs"]) for row in summary] == [(1, 1), (200, 1)]
    assert [row["final_test_accuracy_mean"] for row in summary] == [50.0, 90.0]


def test_write_summaries_keeps_different_epoch_protocols_separate(tmp_path: Path):
    results = [
        {
            "schema_version": 1,
            "status": "complete",
            "config": benchmark.protocol_config("mnist", "bp", seed, epochs),
            "device": "cpu",
            "parameter_count": 10,
            "final_train_accuracy": accuracy + 1,
            "final_test_accuracy": accuracy,
        }
        for seed, epochs, accuracy in ((424, 1, 50.0), (425, 200, 90.0))
    ]

    benchmark.write_summaries(tmp_path, results)

    with (tmp_path / "runs.csv").open(newline="", encoding="utf-8") as handle:
        runs = list(csv.DictReader(handle))
    with (tmp_path / "summary.csv").open(newline="", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    assert [int(row["epochs"]) for row in runs] == [1, 200]
    assert [(int(row["epochs"]), int(row["runs"])) for row in summary] == [
        (1, 1),
        (200, 1),
    ]


def test_write_summaries_preserves_prior_subset_runs_and_deduplicates(tmp_path: Path):
    def result(dataset: str, method: str, seed: int, accuracy: float):
        return {
            "schema_version": 1,
            "status": "complete",
            "config": benchmark.protocol_config(dataset, method, seed, epochs=1),
            "device": "cpu",
            "parameter_count": 10,
            "final_train_accuracy": accuracy + 1,
            "final_test_accuracy": accuracy,
        }

    prior = result("mnist", "bp", 424, 90.0)
    current = result("fashionmnist", "ff", 425, 80.0)
    prior_path = tmp_path / "mnist" / "bp" / "seed_424" / "run.json"
    current_path = tmp_path / "fashionmnist" / "ff" / "seed_425" / "run.json"
    benchmark._atomic_write_json(prior_path, prior)
    benchmark._atomic_write_json(current_path, current)

    benchmark.write_summaries(tmp_path, [current, current])

    with (tmp_path / "runs.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["dataset"], row["method"], int(row["seed"])) for row in rows] == [
        ("fashionmnist", "ff", 425),
        ("mnist", "bp", 424),
    ]
    assert {int(row["epochs"]) for row in rows} == {1}
    with (tmp_path / "summary.csv").open(newline="", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    assert len(summary) == 2


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--epochs", "0"], "epochs"),
        (["--num-workers", "-1"], "num-workers"),
        (["--seeds", "424", "424"], "duplicates"),
    ],
)
def test_invalid_runtime_options_are_rejected(arguments, message, tmp_path):
    args = benchmark.parse_args(
        [*arguments, "--data-dir", str(tmp_path / "data"), "--output-dir", str(tmp_path / "out")]
    )

    with pytest.raises(ValueError, match=message):
        benchmark.validate_args(args)
