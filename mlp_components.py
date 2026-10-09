"""Models and objectives shared by the MLP decomposition experiments.

The module deliberately records the implementation used by the experiments:
goodness is the mean squared activation, and normalized FF variants normalize
the input to every FF layer (including the encoded input to the first layer).
The normalization ablation retains this first-layer input normalization while
removing normalization between hidden layers. These choices must be stated
identically in the manuscript.
"""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


GOODNESS_THRESHOLD = 2.0
NORMALIZATION_EPSILON = 1e-4


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch deterministically."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class GoodnessMLP(nn.Module):
    """ReLU MLP whose mean-squared activations define layer goodness."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dims: Sequence[int],
        normalize: bool,
        normalize_first_layer_input: bool | None = None,
        normalization_epsilon: float = NORMALIZATION_EPSILON,
    ) -> None:
        super().__init__()
        if not hidden_dims or any(width <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")
        if normalization_epsilon <= 0:
            raise ValueError("normalization_epsilon must be positive")
        dims = (input_dim, *hidden_dims)
        self.layers = nn.ModuleList(
            nn.Linear(dims[index], dims[index + 1]) for index in range(len(dims) - 1)
        )
        # ``normalize`` controls only normalization between hidden layers.
        # By default the first-layer policy follows it, preserving the API used
        # by the original FF variants.  Normalization ablations can retain the
        # common encoded-input normalization while disabling only inter-layer
        # normalization.
        self.normalize = normalize
        self.normalize_first_layer_input = (
            normalize
            if normalize_first_layer_input is None
            else normalize_first_layer_input
        )
        self.normalization_epsilon = normalization_epsilon

    def normalizes_layer_input(self, layer_index: int) -> bool:
        if not 0 <= layer_index < len(self.layers):
            raise IndexError(f"layer_index out of range: {layer_index}")
        return self.normalize_first_layer_input if layer_index == 0 else self.normalize

    def forward_layer(self, inputs: torch.Tensor, layer_index: int) -> torch.Tensor:
        if self.normalizes_layer_input(layer_index):
            denominator = inputs.norm(p=2, dim=1, keepdim=True)
            inputs = inputs / (denominator + self.normalization_epsilon)
        return F.relu(self.layers[layer_index](inputs))

    @staticmethod
    def activation_goodness(activations: torch.Tensor) -> torch.Tensor:
        return activations.square().mean(dim=1)

    def layer_goodness(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        scores: list[torch.Tensor] = []
        for layer_index in range(len(self.layers)):
            inputs = self.forward_layer(inputs, layer_index)
            scores.append(self.activation_goodness(inputs))
        return scores

    def goodness(self, inputs: torch.Tensor, aggregation: str) -> torch.Tensor:
        scores = self.layer_goodness(inputs)
        if aggregation == "final":
            return scores[-1]
        if aggregation == "sum":
            return torch.stack(scores, dim=1).sum(dim=1)
        raise ValueError(f"Unknown goodness aggregation: {aggregation}")


class BackpropMLP(nn.Module):
    """End-to-end cross-entropy baseline."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        hidden_dims: Sequence[int],
    ) -> None:
        super().__init__()
        dims = (input_dim, *hidden_dims)
        modules: list[nn.Module] = []
        for index in range(len(dims) - 1):
            modules.extend((nn.Linear(dims[index], dims[index + 1]), nn.ReLU()))
        self.features = nn.Sequential(*modules)
        # No-bias BP control: omit only the terminal classifier's additive bias.
        self.classifier = nn.Linear(dims[-1], num_classes, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


class LocalBPLayer(nn.Module):
    """One backbone layer and its bias-free local linear classifier."""

    def __init__(self, in_features: int, out_features: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.classifier = nn.Linear(out_features, num_classes, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.relu(self.linear(inputs))


class LocalBPMLP(nn.Module):
    """MLP with one linear classification head attached to every hidden layer."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        hidden_dims: Sequence[int],
    ) -> None:
        super().__init__()
        dims = (input_dim, *hidden_dims)
        self.layers = nn.ModuleList(
            LocalBPLayer(dims[index], dims[index + 1], num_classes)
            for index in range(len(dims) - 1)
        )

    def class_logits(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        logits: list[torch.Tensor] = []
        for layer in self.layers:
            inputs = layer(inputs)
            logits.append(layer.classifier(inputs))
        return logits


def matched_ce_layer_losses(
    model: LocalBPMLP,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    detach_between_layers: bool,
) -> list[torch.Tensor]:
    """Return one cross-entropy loss per layer under a chosen gradient graph.

    Detaching a representation changes only the route by which later losses
    assign credit.  It does not change the forward activations, local logits,
    or numerical loss values.
    """
    losses: list[torch.Tensor] = []
    for layer_index, layer in enumerate(model.layers):
        if layer_index > 0 and detach_between_layers:
            inputs = inputs.detach()
        inputs = layer(inputs)
        losses.append(F.cross_entropy(layer.classifier(inputs), labels))
    return losses


def mark_inputs(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Replace the first class-sized input region with a one-hot class code."""
    if inputs.ndim != 2 or labels.ndim != 1 or inputs.size(0) != labels.size(0):
        raise ValueError("inputs must be [batch, features] and labels must be [batch]")
    if inputs.size(1) < num_classes:
        raise ValueError("input width must be at least the number of classes")
    marked = inputs.clone()
    marked[:, :num_classes] = 0.0
    marked.scatter_(1, labels.unsqueeze(1), 1.0)
    return marked


def sample_wrong_labels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Sample one incorrect class uniformly for every observation."""
    if num_classes < 2:
        raise ValueError("At least two classes are required")
    offsets = torch.randint(1, num_classes, labels.shape, device=labels.device)
    return (labels + offsets) % num_classes


def softplus_goodness_loss(
    positive_goodness: torch.Tensor,
    negative_goodness: torch.Tensor,
    *,
    threshold: float = GOODNESS_THRESHOLD,
) -> torch.Tensor:
    terms = torch.cat(
        (threshold - positive_goodness, negative_goodness - threshold),
        dim=0,
    )
    return F.softplus(terms).mean()


def candidate_goodness(
    model: GoodnessMLP,
    inputs: torch.Tensor,
    num_classes: int,
    *,
    aggregation: str,
    chunk_size: int,
) -> torch.Tensor:
    """Return class-conditioned goodness with shape ``[batch, classes]``."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    batch_size, input_dim = inputs.shape
    chunks: list[torch.Tensor] = []
    for start in range(0, num_classes, chunk_size):
        stop = min(start + chunk_size, num_classes)
        candidates = torch.arange(start, stop, device=inputs.device)
        candidates = candidates.unsqueeze(0).expand(batch_size, -1)
        expanded = inputs.unsqueeze(1).expand(-1, stop - start, -1)
        expanded = expanded.reshape(-1, input_dim)
        marked = mark_inputs(expanded, candidates.reshape(-1), num_classes)
        scores = model.goodness(marked, aggregation=aggregation)
        chunks.append(scores.view(batch_size, stop - start))
    return torch.cat(chunks, dim=1)


def _move_batch(
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs, labels = batch
    non_blocking = device.type == "cuda"
    return (
        inputs.to(device, non_blocking=non_blocking),
        labels.to(device, non_blocking=non_blocking),
    )


def train_vanilla_ff_epoch(
    model: GoodnessMLP,
    loader: DataLoader,
    optimizers: Sequence[torch.optim.Optimizer],
    *,
    device: torch.device,
    num_classes: int,
    threshold: float = GOODNESS_THRESHOLD,
) -> float:
    """Train vanilla FF with one full local pass through each layer."""
    model.train()
    total_loss = 0.0
    observations = 0
    for layer_index, optimizer in enumerate(optimizers):
        for batch in loader:
            inputs, labels = _move_batch(batch, device)
            positive = mark_inputs(inputs, labels, num_classes)
            negative = mark_inputs(
                inputs,
                sample_wrong_labels(labels, num_classes),
                num_classes,
            )
            with torch.no_grad():
                for previous_index in range(layer_index):
                    positive = model.forward_layer(positive, previous_index)
                    negative = model.forward_layer(negative, previous_index)
            positive_output = model.forward_layer(positive, layer_index)
            negative_output = model.forward_layer(negative, layer_index)
            loss = softplus_goodness_loss(
                model.activation_goodness(positive_output),
                model.activation_goodness(negative_output),
                threshold=threshold,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            observations += labels.size(0)
    return total_loss / observations


def train_pairwise_global_epoch(
    model: GoodnessMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    num_classes: int,
    threshold: float = GOODNESS_THRESHOLD,
) -> float:
    """Train FF+GE using terminal goodness and an end-to-end gradient."""
    model.train()
    total_loss = 0.0
    observations = 0
    for batch in loader:
        inputs, labels = _move_batch(batch, device)
        positive = mark_inputs(inputs, labels, num_classes)
        negative = mark_inputs(
            inputs, sample_wrong_labels(labels, num_classes), num_classes
        )
        loss = softplus_goodness_loss(
            model.goodness(positive, aggregation="final"),
            model.goodness(negative, aggregation="final"),
            threshold=threshold,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        observations += labels.size(0)
    return total_loss / observations


def matched_ff_layer_losses(
    model: GoodnessMLP,
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    detach_between_layers: bool,
    threshold: float = GOODNESS_THRESHOLD,
) -> list[torch.Tensor]:
    """Return the same FF loss at every layer under one chosen gradient graph.

    The local and global matched controls call this function with identical
    inputs and loss placement.  Their only difference is whether the incoming
    representation is detached before each layer after the first.
    """
    losses: list[torch.Tensor] = []
    for layer_index in range(len(model.layers)):
        if layer_index > 0 and detach_between_layers:
            positive = positive.detach()
            negative = negative.detach()
        positive = model.forward_layer(positive, layer_index)
        negative = model.forward_layer(negative, layer_index)
        losses.append(
            softplus_goodness_loss(
                model.activation_goodness(positive),
                model.activation_goodness(negative),
                threshold=threshold,
            )
        )
    return losses


def train_matched_ff_epoch(
    model: GoodnessMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    num_classes: int,
    detach_between_layers: bool,
    threshold: float = GOODNESS_THRESHOLD,
) -> float:
    """Train one member of the matched FF locality-control pair."""
    model.train()
    total_loss = 0.0
    observations = 0
    for batch in loader:
        inputs, labels = _move_batch(batch, device)
        positive = mark_inputs(inputs, labels, num_classes)
        negative = mark_inputs(
            inputs, sample_wrong_labels(labels, num_classes), num_classes
        )
        layer_losses = matched_ff_layer_losses(
            model,
            positive,
            negative,
            detach_between_layers=detach_between_layers,
            threshold=threshold,
        )
        loss = torch.stack(layer_losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        observations += labels.size(0)
    return total_loss / observations


def train_backprop_epoch(
    model: BackpropMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    observations = 0
    for batch in loader:
        inputs, labels = _move_batch(batch, device)
        loss = F.cross_entropy(model(inputs), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        observations += labels.size(0)
    return total_loss / observations


def train_local_bp_epoch(
    model: LocalBPMLP,
    loader: DataLoader,
    optimizers: Sequence[torch.optim.Optimizer],
    *,
    device: torch.device,
) -> float:
    """Update every local classifier once per minibatch with detached features."""
    model.train()
    total_loss = 0.0
    observations = 0
    for batch in loader:
        inputs, labels = _move_batch(batch, device)
        batch_loss = 0.0
        for layer, optimizer in zip(model.layers, optimizers, strict=True):
            activations = layer(inputs)
            loss = F.cross_entropy(layer.classifier(activations), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            inputs = activations.detach()
            batch_loss += loss.item()
        total_loss += (batch_loss / len(model.layers)) * labels.size(0)
        observations += labels.size(0)
    return total_loss / observations


def train_matched_ce_epoch(
    model: LocalBPMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    detach_between_layers: bool,
) -> float:
    """Train one member of the matched cross-entropy locality-control pair."""
    model.train()
    total_loss = 0.0
    observations = 0
    for batch in loader:
        inputs, labels = _move_batch(batch, device)
        layer_losses = matched_ce_layer_losses(
            model,
            inputs,
            labels,
            detach_between_layers=detach_between_layers,
        )
        loss = torch.stack(layer_losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        observations += labels.size(0)
    return total_loss / observations
