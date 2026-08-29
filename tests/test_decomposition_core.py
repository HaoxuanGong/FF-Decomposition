from __future__ import annotations

import torch
from torch.nn import functional as F

from decomposition_core import (
    GoodnessMLP,
    LocalBPMLP,
    candidate_goodness,
    mark_inputs,
    matched_ce_layer_losses,
    matched_ff_layer_losses,
    sample_wrong_labels,
)


def test_goodness_is_mean_squared_activation() -> None:
    activations = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert torch.equal(GoodnessMLP.activation_goodness(activations), torch.tensor([7.5]))


def test_normalization_is_applied_before_first_layer() -> None:
    model = GoodnessMLP(2, hidden_dims=(2,), normalize=True)
    with torch.no_grad():
        model.layers[0].weight.copy_(torch.eye(2))
        model.layers[0].bias.zero_()
    output = model.forward_layer(torch.tensor([[3.0, 4.0]]), 0)
    expected = torch.tensor([[3.0, 4.0]]) / (5.0 + model.normalization_epsilon)
    assert torch.allclose(output, expected)


def test_first_layer_normalization_can_be_retained_without_inter_layer_normalization() -> None:
    model = GoodnessMLP(
        2,
        hidden_dims=(2, 2),
        normalize=False,
        normalize_first_layer_input=True,
    )
    assert model.normalizes_layer_input(0) is True
    assert model.normalizes_layer_input(1) is False
    with torch.no_grad():
        for layer in model.layers:
            layer.weight.copy_(torch.eye(2))
            layer.bias.zero_()
    first = model.forward_layer(torch.tensor([[3.0, 4.0]]), 0)
    expected_first = torch.tensor([[3.0, 4.0]]) / (5.0 + model.normalization_epsilon)
    assert torch.allclose(first, expected_first)
    assert torch.allclose(model.forward_layer(first, 1), first)


def test_wrong_label_sampler_never_returns_true_label() -> None:
    labels = torch.arange(1000) % 10
    sampled = sample_wrong_labels(labels, 10)
    assert torch.all(sampled != labels)
    assert sampled.min() >= 0
    assert sampled.max() < 10


def test_candidate_scores_have_batch_by_class_shape() -> None:
    model = GoodnessMLP(16, hidden_dims=(8, 8), normalize=True)
    scores = candidate_goodness(
        model,
        torch.randn(7, 16),
        4,
        aggregation="final",
        chunk_size=3,
    )
    assert scores.shape == (7, 4)


def test_terminal_loss_reaches_every_global_layer() -> None:
    model = GoodnessMLP(16, hidden_dims=(8, 8), normalize=True)
    inputs = torch.randn(6, 16)
    labels = torch.arange(6) % 4
    scores = candidate_goodness(
        model, inputs, 4, aggregation="final", chunk_size=4
    )
    F.cross_entropy(scores, labels).backward()
    assert all(layer.weight.grad is not None for layer in model.layers)


def test_matched_ff_pair_differs_only_in_gradient_connectivity() -> None:
    inputs = torch.randn(6, 16)
    labels = torch.arange(6) % 4
    positive = mark_inputs(inputs, labels, 4)
    negative = mark_inputs(inputs, (labels + 1) % 4, 4)

    local_model = GoodnessMLP(16, hidden_dims=(8, 8), normalize=True)
    local_losses = matched_ff_layer_losses(
        local_model,
        positive,
        negative,
        detach_between_layers=True,
    )
    local_losses[-1].backward()
    assert local_model.layers[0].weight.grad is None
    assert local_model.layers[1].weight.grad is not None

    global_model = GoodnessMLP(16, hidden_dims=(8, 8), normalize=True)
    global_model.load_state_dict(local_model.state_dict())
    global_losses = matched_ff_layer_losses(
        global_model,
        positive,
        negative,
        detach_between_layers=False,
    )
    assert all(
        torch.allclose(local_loss.detach(), global_loss.detach())
        for local_loss, global_loss in zip(local_losses, global_losses, strict=True)
    )
    global_losses[-1].backward()
    assert all(layer.weight.grad is not None for layer in global_model.layers)


def test_matched_ce_pair_differs_only_in_gradient_connectivity() -> None:
    torch.manual_seed(23)
    inputs = torch.randn(6, 16)
    labels = torch.arange(6) % 4

    local_model = LocalBPMLP(16, 4, hidden_dims=(8, 8, 8))
    global_model = LocalBPMLP(16, 4, hidden_dims=(8, 8, 8))
    global_model.load_state_dict(local_model.state_dict())

    local_losses = matched_ce_layer_losses(
        local_model,
        inputs,
        labels,
        detach_between_layers=True,
    )
    global_losses = matched_ce_layer_losses(
        global_model,
        inputs,
        labels,
        detach_between_layers=False,
    )
    assert all(
        torch.equal(local_loss.detach(), global_loss.detach())
        for local_loss, global_loss in zip(local_losses, global_losses, strict=True)
    )

    local_losses[-1].backward()
    assert all(layer.linear.weight.grad is None for layer in local_model.layers[:-1])
    assert local_model.layers[-1].linear.weight.grad is not None
    assert local_model.layers[-1].classifier.weight.grad is not None

    global_losses[-1].backward()
    assert all(layer.linear.weight.grad is not None for layer in global_model.layers)
    assert global_model.layers[-1].classifier.weight.grad is not None
    assert all(layer.classifier.weight.grad is None for layer in global_model.layers[:-1])


def test_local_bp_detach_blocks_earlier_layer_gradient() -> None:
    model = LocalBPMLP(16, 4, hidden_dims=(8, 8))
    inputs = torch.randn(6, 16)
    labels = torch.arange(6) % 4
    first = model.layers[0](inputs).detach()
    second = model.layers[1](first)
    F.cross_entropy(model.layers[1].classifier(second), labels).backward()
    assert all(parameter.grad is None for parameter in model.layers[0].parameters())
    assert any(parameter.grad is not None for parameter in model.layers[1].parameters())


def test_label_encoding_replaces_only_the_class_region() -> None:
    inputs = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    labels = torch.tensor([0, 1, 2])
    marked = mark_inputs(inputs, labels, 3)
    assert torch.equal(marked[:, :3], torch.eye(3))
    assert torch.equal(marked[:, 3:], inputs[:, 3:])
