from __future__ import annotations

from collections import Counter

import torch
from torch.utils.data import TensorDataset

from decomposition_core import LocalBPMLP
from MLPBenchmarkSuite import (
    FC_METHODS,
    PRIMARY_METHODS,
    build_model,
    configuration,
    expected_matrix,
    make_optimizers,
    parser,
    stratified_split_indices,
)


def test_stratified_split_is_deterministic_balanced_and_disjoint() -> None:
    targets = torch.arange(4).repeat_interleave(20)
    dataset = TensorDataset(torch.zeros(len(targets), 2), targets)
    dataset.targets = targets  # type: ignore[attr-defined]
    train_a, validation_a = stratified_split_indices(
        dataset, seed=17, num_classes=4, validation_size=20
    )
    train_b, validation_b = stratified_split_indices(
        dataset, seed=17, num_classes=4, validation_size=20
    )
    assert train_a == train_b
    assert validation_a == validation_b
    assert not set(train_a).intersection(validation_a)
    assert Counter(targets[validation_a].tolist()) == Counter({0: 5, 1: 5, 2: 5, 3: 5})


def test_no_inter_layer_normalization_variant_retains_first_layer_normalization() -> None:
    for method in ("nn-ff-ge", "fc-nn-ff-ge"):
        model = build_model(method, "mnist")
        assert getattr(model, "normalize") is False
        assert getattr(model, "normalize_first_layer_input") is True
        assert model.normalizes_layer_input(0) is True
        assert model.normalizes_layer_input(1) is False


def test_hidden_dimension_override_is_recorded_and_built() -> None:
    args = parser().parse_args(
        [
            "run",
            "--dataset",
            "fashionmnist",
            "--method",
            "fc-ff-ge",
            "--seed",
            "424",
            "--hidden-dims",
            "500",
            "500",
            "500",
            "500",
            "500",
            "--output-dir",
            "results",
        ]
    )
    config = configuration(args)
    assert config["hidden_dims"] == [500] * 5
    model = build_model(
        "fc-ff-ge", "fashionmnist", hidden_dims=tuple(config["hidden_dims"])
    )
    assert [layer.out_features for layer in model.layers] == [500] * 5


def test_direct_mlp_step_scheduler_default_matches_the_paper_profile() -> None:
    args = parser().parse_args(
        [
            "run",
            "--dataset",
            "mnist",
            "--method",
            "bp",
            "--seed",
            "424",
            "--scheduler",
            "step",
            "--output-dir",
            "results",
        ]
    )
    assert args.step_size == 30
    assert args.step_gamma == 0.1


def test_cross_entropy_metadata_does_not_claim_a_goodness_objective() -> None:
    for method, expected_detach in (("bp", None), ("local-bp", True)):
        args = parser().parse_args(
            [
                "run",
                "--dataset",
                "mnist",
                "--method",
                method,
                "--seed",
                "424",
                "--output-dir",
                "results",
            ]
        )
        config = configuration(args)
        assert config["goodness_definition"] is None
        assert config["detach_between_layers"] is expected_detach


def test_matched_ce_methods_differ_only_in_gradient_locality() -> None:
    local_args = parser().parse_args(
        [
            "run",
            "--dataset",
            "mnist",
            "--method",
            "ce-matched-local",
            "--seed",
            "424",
            "--output-dir",
            "results",
        ]
    )
    global_args = parser().parse_args(
        [
            "run",
            "--dataset",
            "mnist",
            "--method",
            "ce-matched-ge",
            "--seed",
            "424",
            "--output-dir",
            "results",
        ]
    )
    local_config = configuration(local_args)
    global_config = configuration(global_args)
    differences = {
        key for key in local_config if local_config[key] != global_config[key]
    }
    assert differences == {"method", "detach_between_layers"}

    for method in ("ce-matched-local", "ce-matched-ge"):
        model = build_model(method, "mnist")
        assert isinstance(model, LocalBPMLP)
        optimizers = make_optimizers(
            method,
            model,
            optimizer_name="adam",
            learning_rate=1e-3,
            momentum=0.9,
        )
        assert len(optimizers) == 1


def test_full_paper_matrix_has_expected_omissions() -> None:
    matrix = expected_matrix()
    assert len(PRIMARY_METHODS) == 11
    assert len(matrix) == 120
    assert {method for _dataset, method, _seed in matrix} == set(PRIMARY_METHODS)
    assert not any(
        dataset == "cifar100" and method in FC_METHODS
        for dataset, method, _seed in matrix
    )
    assert not any(
        method in {"ff-matched-local", "ce-matched-local"}
        for _dataset, method, _seed in matrix
    )
