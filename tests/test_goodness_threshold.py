from __future__ import annotations

import copy
import json

import pytest
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

import decomposition_core as core
import MLPBenchmarkSuite as suite


THRESHOLD_METHODS = (
    "ff",
    "ff-ge",
    "nn-ff-ge",
    "ff-matched-local",
    "ff-matched-ge",
)


def run_args(*extra, method="ff"):
    return suite.parser().parse_args([
        "run", "--dataset", "fashionmnist", "--method", method,
        "--seed", "424", "--output-dir", "results", *extra,
    ])


@pytest.mark.parametrize("threshold", [1.0, 2.0, 4.0])
@pytest.mark.parametrize("method", THRESHOLD_METHODS)
def test_threshold_parser_and_configuration_record_actual_value(threshold, method):
    args = run_args("--goodness-threshold", str(threshold), method=method)
    assert args.goodness_threshold == threshold
    assert suite.configuration(args)["goodness_threshold"] == threshold


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf", "abc"])
def test_threshold_parser_rejects_nonpositive_nonfinite_and_nonnumeric_values(value):
    with pytest.raises(SystemExit):
        run_args(f"--goodness-threshold={value}")


@pytest.mark.parametrize("method", sorted(set(suite.METHODS) - set(THRESHOLD_METHODS)))
def test_non_ff_methods_reject_inapplicable_thresholds_before_loading_data(method, monkeypatch):
    def unexpected_load(*args):
        raise AssertionError("Invalid threshold must be rejected before dataset access")

    monkeypatch.setattr(suite, "load_dataset", unexpected_load)
    assert suite.configuration(run_args(method=method))["goodness_threshold"] is None
    with pytest.raises(ValueError, match="does not apply"):
        suite.run(run_args("--goodness-threshold", "4", method=method))


def tiny_problem():
    core.set_seed(424)
    model = core.GoodnessMLP(16, hidden_dims=(8, 8), normalize=True)
    # Positive biases keep each layer active, so the threshold can affect updates.
    with torch.no_grad():
        for layer in model.layers:
            layer.bias.fill_(0.3)
    data = TensorDataset(torch.randn(8, 16), torch.arange(8) % 4)
    return model, DataLoader(data, batch_size=4, shuffle=False)


def optimizers_for(method, model, optimizer="adam"):
    return suite.make_optimizers(
        method, model, optimizer_name=optimizer, learning_rate=0.001, momentum=0.0,
    )


def untouched_core_epoch(method, model, loader, optimizers):
    kwargs = {"device": torch.device("cpu"), "num_classes": 4}
    if method == "ff":
        return core.train_vanilla_ff_epoch(model, loader, optimizers, **kwargs)
    if method == "ff-ge":
        return core.train_pairwise_global_epoch(model, loader, optimizers[0], **kwargs)
    if method == "nn-ff-ge":
        return core.train_pairwise_global_epoch(model, loader, optimizers[0], **kwargs)
    return core.train_matched_ff_epoch(
        model, loader, optimizers[0],
        detach_between_layers=method == "ff-matched-local", **kwargs,
    )


@pytest.mark.parametrize("method", THRESHOLD_METHODS)
def test_default_and_explicit_two_preserve_original_core_updates_exactly(method):
    initial, loader = tiny_problem()
    expected = copy.deepcopy(initial)
    expected_optimizers = optimizers_for(method, expected)
    core.set_seed(842)
    expected_loss = untouched_core_epoch(method, expected, loader, expected_optimizers)
    for threshold_kwargs in ({}, {"goodness_threshold": 2.0}):
        actual = copy.deepcopy(initial)
        actual_optimizers = optimizers_for(method, actual)
        core.set_seed(842)
        actual_loss = suite.train_epoch(
            method, actual, loader, actual_optimizers,
            device=torch.device("cpu"), num_classes=4, candidate_chunk=4,
            **threshold_kwargs,
        )
        assert actual_loss == expected_loss
        for name, expected_parameter in expected.state_dict().items():
            assert torch.equal(actual.state_dict()[name], expected_parameter), name


@pytest.mark.parametrize("method", THRESHOLD_METHODS)
def test_explicit_threshold_reaches_actual_loss_and_changes_learning(method, monkeypatch):
    initial, loader = tiny_problem()
    original_loss = core.softplus_goodness_loss
    seen_thresholds = []

    def observed_loss(positive, negative, *, threshold=core.GOODNESS_THRESHOLD):
        seen_thresholds.append(threshold)
        return original_loss(positive, negative, threshold=threshold)

    monkeypatch.setattr(core, "softplus_goodness_loss", observed_loss)
    outcomes = []
    for threshold in (1.0, 2.0, 4.0):
        seen_thresholds.clear()
        model = copy.deepcopy(initial)
        optimizers = optimizers_for(method, model, optimizer="sgd")
        core.set_seed(842)
        loss = suite.train_epoch(
            method, model, loader, optimizers, device=torch.device("cpu"),
            num_classes=4, candidate_chunk=4, goodness_threshold=threshold,
        )
        assert seen_thresholds and set(seen_thresholds) == {threshold}
        outcomes.append((loss, model.state_dict()))
    for left, right in zip(outcomes, outcomes[1:]):
        assert left[0] != right[0]
        assert any(not torch.equal(left[1][name], right[1][name]) for name in left[1])


class UnreadTestDataset(Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, index):
        raise AssertionError("Deferred training must not access test samples")


def test_run_forwards_threshold_and_saves_it_in_result_and_checkpoint(tmp_path, monkeypatch):
    targets = torch.arange(10).repeat_interleave(2)
    train_data = TensorDataset(torch.ones(20, 784), targets)
    train_data.targets = targets
    monkeypatch.setattr(suite, "load_dataset", lambda *args: (train_data, UnreadTestDataset()))
    monkeypatch.setattr(suite, "dataset_manifest", lambda *args: {"synthetic": True})
    observed_thresholds = []
    original_loss = core.softplus_goodness_loss

    def observed_loss(positive, negative, *, threshold=core.GOODNESS_THRESHOLD):
        observed_thresholds.append(threshold)
        return original_loss(positive, negative, threshold=threshold)

    monkeypatch.setattr(core, "softplus_goodness_loss", observed_loss)
    args = run_args(
        "--output-dir", str(tmp_path), "--goodness-threshold", "4",
        "--hidden-dims", "8", "8", "--epochs", "1", "--validation-size", "10",
        "--device", "cpu", "--defer-test",
    )
    suite.run(args)
    assert observed_thresholds and set(observed_thresholds) == {4.0}
    directory = tmp_path / "fashionmnist" / "ff" / "seed_424"
    payload = json.loads((directory / "run.json").read_text())
    config = json.loads((directory / "config.json").read_text())
    checkpoint = torch.load(directory / "best_model.pt", weights_only=True)
    assert payload["config"]["goodness_threshold"] == 4.0
    assert config["goodness_threshold"] == 4.0
    assert checkpoint["config"]["goodness_threshold"] == 4.0
    assert payload["status"] == "train_complete"
    assert payload["test_evaluations"] == 0
