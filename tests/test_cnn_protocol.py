from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

import LocalBPCNNBenchmark as cnn

from LocalBPCNNBenchmark import (
    BPCNN,
    LocalBPCNN,
    build_optimizer,
    parse_args,
    stratified_split_indices,
    validation_improved,
)


def test_cnn_split_is_exact_stratified_and_deterministic() -> None:
    targets = np.repeat(np.arange(4), 20)
    train_a, validation_a = stratified_split_indices(targets, 20, 9)
    train_b, validation_b = stratified_split_indices(targets, 20, 9)
    assert train_a == train_b
    assert validation_a == validation_b
    assert len(validation_a) == 20
    assert not set(train_a).intersection(validation_a)
    assert Counter(targets[validation_a].tolist()) == Counter({0: 5, 1: 5, 2: 5, 3: 5})


def test_cnn_local_loss_cannot_update_an_earlier_block() -> None:
    model = LocalBPCNN(3, 10)
    images = torch.randn(2, 3, 32, 32)
    labels = torch.tensor([0, 1])
    first = model.blocks[0](images).detach()
    second = model.blocks[1](first)
    F.cross_entropy(model.heads[1](second), labels).backward()
    assert all(parameter.grad is None for parameter in model.blocks[0].parameters())
    assert any(parameter.grad is not None for parameter in model.blocks[1].parameters())


def test_cnn_last_local_loss_is_isolated_from_all_earlier_blocks() -> None:
    model = LocalBPCNN(3, 10)
    images = torch.randn(2, 3, 32, 32)
    labels = torch.tensor([0, 1])
    logits = model.forward_local(images)
    F.cross_entropy(logits[-1], labels).backward()
    for block in model.blocks[:-1]:
        assert all(parameter.grad is None for parameter in block.parameters())
    assert any(parameter.grad is not None for parameter in model.blocks[-1].parameters())
    assert all(parameter.grad is None for head in model.heads[:-1] for parameter in head.parameters())
    assert any(parameter.grad is not None for parameter in model.heads[-1].parameters())


def test_cnn_bp_and_local_bp_start_with_the_same_backbone_for_a_seed() -> None:
    torch.manual_seed(424)
    bp = BPCNN(3, 10)
    torch.manual_seed(424)
    local = LocalBPCNN(3, 10)
    for bp_parameter, local_parameter in zip(
        bp.blocks.parameters(), local.blocks.parameters(), strict=True
    ):
        assert torch.equal(bp_parameter, local_parameter)


def test_cnn_fixed_defaults_match_the_paper_protocol() -> None:
    args = parse_args(["cifar10", "--method", "bp", "--device", "cpu"])
    assert args.epochs == 200
    assert args.patience == 15
    assert args.minimum_delta == 0.0
    assert args.validation_size == 5000
    assert args.optimizer == "sgd"
    assert args.learning_rate == 0.1
    assert args.batch_size == 128
    assert args.eval_batch_size == 512
    assert args.seeds == [424, 425, 426]


def test_cnn_sgd_uses_fixed_momentum_without_regularization() -> None:
    layer = torch.nn.Linear(3, 2)
    optimizer = build_optimizer("sgd", layer.parameters(), 0.1)
    group = optimizer.param_groups[0]
    assert group["momentum"] == 0.9
    assert group["dampening"] == 0.0
    assert group["nesterov"] is False
    assert group["weight_decay"] == 0.0


def test_cnn_early_stopping_requires_strict_validation_improvement() -> None:
    assert validation_improved(0.81, 0.80, 0.0)
    assert not validation_improved(0.80, 0.80, 0.0)
    assert not validation_improved(0.805, 0.80, 0.01)


def test_cnn_run_persists_and_restores_hashed_best_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(cnn, "BACKBONE_WIDTHS", (2, 2, 2, 2))
    images = torch.randn(4, 3, 16, 16)
    labels = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2, shuffle=False)
    config = cnn.RunConfig(
        method="bp",
        dataset="cifar10",
        epochs=2,
        patience=1,
        minimum_delta=0.0,
        validation_size=2,
        optimizer="sgd",
        optimizer_parameters="momentum=0.9,dampening=0,nesterov=false",
        learning_rate=0.1,
        scheduler="cosine-annealing",
        batch_size=2,
        eval_batch_size=2,
        seeds=[7],
        num_workers=0,
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path),
        device="cpu",
        download=False,
        save_checkpoints=True,
        provenance_run_id="unit-test",
        overwrite=False,
        backbone_widths=(2, 2, 2, 2),
    )
    _history, result = cnn.run_seed(
        7,
        "a" * 64,
        config,
        loader,
        loader,
        loader,
    )
    checkpoint_path = tmp_path / "seed_7_best.pt"
    checkpoint = cnn.load_checkpoint(checkpoint_path)
    assert result["checkpoint_sha256"] == cnn.file_sha256(checkpoint_path)
    assert result["model_state_sha256"] == cnn.state_dict_sha256(
        checkpoint["model_state_dict"]
    )
    assert result["restored_state_sha256"] == result["model_state_sha256"]
    assert result["best_checkpoint_restored"] is True
    assert result["test_evaluations"] == 1
    assert "test_accuracy" not in checkpoint
