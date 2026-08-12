"""Fast checks for the public experiment interfaces."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
import csv
import zipfile

import pytest
import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader, TensorDataset

import MFCNNBenchmark as cnn
import MLPMixerBenchmarkSuite as mixer
import VanillaMonoForward as mlp


@pytest.mark.parametrize(
    ("dataset", "input_dim", "num_classes"),
    [
        ("mnist", 784, 10),
        ("fashionmnist", 784, 10),
        ("cifar10", 3072, 10),
        ("cifar100", 3072, 100),
    ],
)
def test_mlp_infers_dataset_dimensions(dataset: str, input_dim: int, num_classes: int):
    args = mlp.parse_args(["--dataset", dataset, "--hidden-dims", "16", "8"])
    config = mlp.Config(**vars(args))
    model = mlp.MonoForward(config, num_classes)

    assert model.layers[0].linear.in_features == input_dim
    assert [layer.linear.out_features for layer in model.layers] == [16, 8]
    assert model.layers[-1].mask_weights.shape == (num_classes, 8)


def test_legacy_architecture_argument_remains_compatible():
    args = mlp.parse_args(["--dataset", "cifar10", "--architecture", "3072", "32", "16"])

    assert args.hidden_dims == [32, 16]


def test_legacy_architecture_rejects_wrong_input_width():
    with pytest.raises(SystemExit):
        mlp.parse_args(["--dataset", "mnist", "--architecture", "3072", "32", "16"])


def test_mlp_test_mode_requires_checkpoint():
    with pytest.raises(ValueError, match="checkpoint-path"):
        mlp.main(["--mode", "test"])


@pytest.mark.parametrize(
    ("dataset", "hidden_dims"),
    [
        ("mnist", [1000, 1000]),
        ("fashionmnist", [1000, 1000]),
        ("cifar10", [2000, 2000, 2000]),
        ("cifar100", [2000, 2000, 2000]),
    ],
)
def test_mlp_defaults_match_the_main_table_protocol(dataset: str, hidden_dims: list[int]):
    args = mlp.parse_args(["--dataset", dataset])

    assert args.hidden_dims == hidden_dims
    assert args.epochs == 200
    assert args.batch_size == 128
    assert args.optimizer == "adam"
    assert args.lr == pytest.approx(0.001)
    assert args.step_size == 20
    assert args.l2_lambda == 0
    assert args.seed is None
    assert mlp.training_seeds(args.seed) == [424, 425, 426]
    assert mlp.DATASET_SPECS["cifar10"][4] == pytest.approx((0.2471, 0.2435, 0.2616))


def test_mlp_summarizes_cumulative_and_final_metrics_independently():
    history = [
        {
            "epoch": 1,
            "train_cumulative_accuracy": 0.80,
            "test_cumulative_accuracy": 0.92,
            "test_final_layer_accuracy": 0.81,
        },
        {
            "epoch": 2,
            "train_cumulative_accuracy": 0.85,
            "test_cumulative_accuracy": 0.89,
            "test_final_layer_accuracy": 0.87,
        },
    ]

    summary = mlp.summarize_history(history)

    assert summary["best_test_cumulative_accuracy"] == pytest.approx(0.92)
    assert summary["best_test_cumulative_epoch"] == 1
    assert summary["best_test_final_layer_accuracy"] == pytest.approx(0.87)
    assert summary["best_test_final_layer_epoch"] == 2


def test_mlp_aggregate_uses_sample_standard_deviation():
    common = {
        "dataset": "mnist",
        "optimizer": "adam",
        "learning_rate": 0.001,
        "batch_size": 128,
        "epochs": 200,
        "architecture": "784-1000-1000",
    }
    rows = [
        {
            **common,
            "best_test_cumulative_accuracy": 0.80,
            "best_test_cumulative_epoch": 10,
            "best_test_final_layer_accuracy": 0.85,
            "best_test_final_layer_epoch": 12,
        },
        {
            **common,
            "best_test_cumulative_accuracy": 0.90,
            "best_test_cumulative_epoch": 14,
            "best_test_final_layer_accuracy": 0.87,
            "best_test_final_layer_epoch": 16,
        },
    ]

    aggregate = mlp.aggregate_results(rows)

    assert aggregate["best_test_cumulative_accuracy_mean"] == pytest.approx(0.85)
    assert aggregate["best_test_cumulative_accuracy_std"] == pytest.approx(2**0.5 / 20)
    assert aggregate["best_test_cumulative_epoch_mean"] == pytest.approx(12)
    assert aggregate["best_test_final_layer_accuracy_mean"] == pytest.approx(0.86)
    assert aggregate["best_test_final_layer_epoch_mean"] == pytest.approx(14)


def test_mlp_default_command_runs_three_seeds_without_overwriting(tmp_path: Path, monkeypatch):
    calls: list[tuple[int, str, str | None]] = []

    def fake_run_seed(args, seed, run_history_path, run_checkpoint_path):
        calls.append(
            (
                seed,
                run_history_path.name,
                run_checkpoint_path.name if run_checkpoint_path else None,
            )
        )
        return {
            "dataset": args.dataset,
            "seed": seed,
            "optimizer": args.optimizer,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "architecture": "784-1000-1000",
            "best_test_cumulative_accuracy": 0.90,
            "best_test_cumulative_epoch": 20,
            "best_test_final_layer_accuracy": 0.89,
            "best_test_final_layer_epoch": 18,
        }

    monkeypatch.setattr(mlp, "run_seed", fake_run_seed)
    save_path = tmp_path / "history.csv"
    checkpoint = tmp_path / "model.pt"

    mlp.main(
        [
            "--dataset",
            "mnist",
            "--save-path",
            str(save_path),
            "--checkpoint-path",
            str(checkpoint),
        ]
    )

    assert calls == [
        (424, "history_seed424.csv", "model_seed424.pt"),
        (425, "history_seed425.csv", "model_seed425.pt"),
        (426, "history_seed426.csv", "model_seed426.pt"),
    ]
    per_seed, aggregate = mlp.summary_paths(str(save_path), "mnist")
    with per_seed.open(newline="", encoding="utf-8") as handle:
        assert [int(row["seed"]) for row in csv.DictReader(handle)] == [424, 425, 426]
    with aggregate.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert int(row["num_seeds"]) == 3
    assert float(row["best_test_cumulative_accuracy_mean"]) == pytest.approx(0.90)


def test_mlp_rejects_late_multiseed_conflict_before_any_run(tmp_path: Path, monkeypatch):
    save_path = tmp_path / "history.csv"
    conflict = tmp_path / "history_seed426.csv"
    conflict.write_text("existing result", encoding="utf-8")
    calls: list[int] = []

    def record_run(_args, seed, _run_history_path, _run_checkpoint_path):
        calls.append(seed)
        raise AssertionError("output preflight must run before training")

    monkeypatch.setattr(mlp, "run_seed", record_run)

    with pytest.raises(FileExistsError, match="--overwrite"):
        mlp.main(["--dataset", "mnist", "--save-path", str(save_path)])

    assert calls == []
    assert conflict.read_text(encoding="utf-8") == "existing result"
    assert not (tmp_path / "history_seed424.csv").exists()
    per_seed, aggregate = mlp.summary_paths(str(save_path), "mnist")
    assert not per_seed.exists()
    assert not aggregate.exists()


def test_mlp_overwrite_replaces_all_planned_outputs(tmp_path: Path, monkeypatch):
    save_path = tmp_path / "history.csv"
    checkpoint_path = tmp_path / "model.pt"
    args = mlp.parse_args(
        [
            "--dataset",
            "mnist",
            "--save-path",
            str(save_path),
            "--checkpoint-path",
            str(checkpoint_path),
        ]
    )
    seeds = mlp.training_seeds(args.seed)
    histories, checkpoints, per_seed, aggregate = mlp.plan_training_outputs(args, seeds)
    planned = [
        *histories.values(),
        *(path for path in checkpoints.values() if path is not None),
        per_seed,
        aggregate,
    ]
    for path in planned:
        path.write_text("old", encoding="utf-8")

    calls: list[int] = []

    def fake_run_seed(run_args, seed, run_history_path, run_checkpoint_path):
        calls.append(seed)
        mlp.write_rows(run_history_path, [{"epoch": 1, "seed": seed}])
        assert run_checkpoint_path is not None
        run_checkpoint_path.write_text(f"checkpoint {seed}", encoding="utf-8")
        return {
            "dataset": run_args.dataset,
            "seed": seed,
            "optimizer": run_args.optimizer,
            "learning_rate": run_args.lr,
            "batch_size": run_args.batch_size,
            "epochs": run_args.epochs,
            "architecture": "784-1000-1000",
            "best_test_cumulative_accuracy": 0.90,
            "best_test_cumulative_epoch": 20,
            "best_test_final_layer_accuracy": 0.89,
            "best_test_final_layer_epoch": 18,
        }

    monkeypatch.setattr(mlp, "run_seed", fake_run_seed)

    mlp.main(
        [
            "--dataset",
            "mnist",
            "--save-path",
            str(save_path),
            "--checkpoint-path",
            str(checkpoint_path),
            "--overwrite",
        ]
    )

    assert calls == [424, 425, 426]
    assert all(path.read_text(encoding="utf-8") != "old" for path in planned)
    assert all(
        checkpoints[seed].read_text(encoding="utf-8") == f"checkpoint {seed}" for seed in seeds
    )


def test_mlp_rejects_colliding_planned_outputs(tmp_path: Path, monkeypatch):
    output = tmp_path / "same.csv"
    calls: list[int] = []
    monkeypatch.setattr(mlp, "run_seed", lambda *_args: calls.append(1))

    with pytest.raises(ValueError, match="collide"):
        mlp.main(
            [
                "--dataset",
                "mnist",
                "--seed",
                "424",
                "--save-path",
                str(output),
                "--checkpoint-path",
                str(output),
            ]
        )

    assert calls == []


def test_mlp_checkpoint_test_mode_restores_dataset_and_architecture(
    tmp_path: Path, monkeypatch, capsys
):
    train_args = mlp.parse_args(
        ["--dataset", "mnist", "--hidden-dims", "4", "--seed", "424", "--device", "cpu"]
    )
    config = mlp.Config(**vars(train_args))
    model = mlp.MonoForward(config, num_classes=10)
    checkpoint = tmp_path / "model.pt"
    mlp.save_checkpoint(checkpoint, model, config, 10, {})
    loader = DataLoader(
        TensorDataset(torch.randn(4, 784), torch.tensor([0, 1, 2, 3])),
        batch_size=2,
    )

    def fake_get_loaders(restored_config):
        assert restored_config.dataset == "mnist"
        assert restored_config.hidden_dims == [4]
        assert restored_config.seed == 424
        return loader, loader

    monkeypatch.setattr(mlp, "get_loaders", fake_get_loaders)

    mlp.main(["--mode", "test", "--checkpoint-path", str(checkpoint), "--device", "cpu"])

    assert "Cumulative:" in capsys.readouterr().out


def test_mlp_rejects_negative_seed_before_loading_data():
    with pytest.raises(ValueError, match="seed"):
        mlp.main(["--seed", "-1"])


@pytest.mark.parametrize(
    ("dataset", "input_channels", "num_classes"),
    [
        ("mnist", 1, 10),
        ("fashionmnist", 1, 10),
        ("cifar10", 3, 10),
        ("cifar100", 3, 100),
    ],
)
def test_unified_cnn_matches_dataset_shape(dataset, input_channels, num_classes):
    spec = cnn.DATASETS[dataset]
    model = cnn.MFCNN(spec.input_channels, spec.num_classes)

    first_conv = model.blocks[0].layers[0]
    assert first_conv.in_channels == input_channels
    assert [block.layers[0].out_channels for block in model.blocks] == [64, 128, 256, 512]
    assert [head.classifier.out_features for head in model.heads] == [num_classes] * 4
    assert not any(isinstance(module, torch.nn.Dropout) for module in model.modules())


def test_unified_cnn_defaults_are_paper_faithful():
    args = cnn.parse_args(["fashionmnist"])
    train_transform, test_transform = cnn.build_transforms("fashionmnist")

    assert args.epochs == 300
    assert args.seeds == [41, 42, 43]
    assert args.optimizer == "sgd"
    assert args.batch_size == 128
    assert train_transform is test_transform
    assert [type(transform).__name__ for transform in train_transform.transforms] == [
        "ToTensor",
        "Normalize",
    ]
    normalization = train_transform.transforms[-1]
    assert tuple(normalization.mean) == pytest.approx((0.2860,))
    assert tuple(normalization.std) == pytest.approx((0.3530,))
    assert cnn.DATASETS["cifar100"].mean == pytest.approx((0.5071, 0.4865, 0.4409))
    assert cnn.DATASETS["cifar100"].std == pytest.approx((0.2673, 0.2564, 0.2762))


def test_unified_cnn_seed_setup_keeps_cuda_pooling_backward_available(monkeypatch):
    strict_calls: list[bool] = []
    monkeypatch.setattr(
        torch,
        "use_deterministic_algorithms",
        lambda enabled, **_kwargs: strict_calls.append(enabled),
    )

    cnn.set_seed(41)

    assert strict_calls == []


def test_unified_cnn_uses_disjoint_plain_local_optimizers():
    config = cnn.config_from_args(cnn.parse_args(["cifar10"]))
    model = cnn.MFCNN(input_channels=3, num_classes=10)

    optimizers = cnn.build_local_optimizers(model, config)
    parameter_sets = [
        {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        for optimizer in optimizers
    ]

    assert len(optimizers) == 4
    assert all(optimizer.defaults["weight_decay"] == 0 for optimizer in optimizers)
    assert all(optimizer.defaults["momentum"] == 0 for optimizer in optimizers)
    assert all(
        first.isdisjoint(second)
        for index, first in enumerate(parameter_sets)
        for second in parameter_sets[index + 1 :]
    )


def test_unified_cnn_reports_cumulative_and_final_accuracy():
    model = cnn.MFCNN(input_channels=1, num_classes=10)
    loader = DataLoader(
        TensorDataset(torch.randn(4, 1, 28, 28), torch.tensor([0, 1, 2, 3])),
        batch_size=2,
    )

    metrics = cnn.evaluate(model, loader, torch.device("cpu"))

    assert len(metrics["local_accuracies"]) == 4
    assert len(metrics["cumulative_accuracies"]) == 4
    assert metrics["cumulative_accuracy"] == metrics["cumulative_accuracies"][-1]
    assert metrics["final_accuracy"] == metrics["local_accuracies"][-1]


def test_unified_cnn_rejects_invalid_configuration_before_loading_data():
    with pytest.raises(ValueError, match="epochs"):
        cnn.main(["mnist", "--epochs", "0"])


def test_unified_cnn_rejects_negative_seeds_before_loading_data():
    with pytest.raises(ValueError, match="non-negative"):
        cnn.main(["mnist", "--seeds", "-1"])


@pytest.mark.parametrize(
    "existing_name",
    [
        "config.json",
        "history.csv",
        "per_seed_results.csv",
        "aggregate_results.csv",
        "seed_41_best.pt",
    ],
)
def test_unified_cnn_refuses_existing_outputs_before_loading_data(
    existing_name: str, tmp_path: Path, monkeypatch
):
    (tmp_path / existing_name).touch()
    loader_called = False

    def fail_if_called(*_args, **_kwargs):
        nonlocal loader_called
        loader_called = True
        raise AssertionError("data loading must not start before output preflight")

    monkeypatch.setattr(cnn, "build_loaders", fail_if_called)
    arguments = ["mnist", "--output-dir", str(tmp_path)]
    if existing_name.endswith(".pt"):
        arguments.append("--save-checkpoints")

    with pytest.raises(FileExistsError, match="--overwrite"):
        cnn.main(arguments)

    assert loader_called is False


def test_unified_cnn_overwrite_explicitly_allows_existing_outputs(tmp_path: Path):
    existing = tmp_path / "config.json"
    existing.touch()
    config = cnn.config_from_args(
        cnn.parse_args(["mnist", "--output-dir", str(tmp_path), "--overwrite"])
    )

    cnn.preflight_outputs(config)

    assert not existing.exists()


def test_unified_cnn_aggregates_independent_best_metrics():
    seed_rows = [
        {
            "dataset": "mnist",
            "optimizer": "sgd",
            "learning_rate": 0.1,
            "batch_size": 128,
            "best_test_cumulative_accuracy": 0.91,
            "best_test_cumulative_epoch": 12,
            "best_test_final_accuracy": 0.86,
            "best_test_final_epoch": 9,
            "final_epoch_cumulative_accuracy": 0.90,
            "final_epoch_final_accuracy": 0.85,
            "mean_train_epoch_seconds": 1.0,
        },
        {
            "dataset": "mnist",
            "optimizer": "sgd",
            "learning_rate": 0.1,
            "batch_size": 128,
            "best_test_cumulative_accuracy": 0.93,
            "best_test_cumulative_epoch": 14,
            "best_test_final_accuracy": 0.90,
            "best_test_final_epoch": 11,
            "final_epoch_cumulative_accuracy": 0.92,
            "final_epoch_final_accuracy": 0.89,
            "mean_train_epoch_seconds": 1.2,
        },
    ]

    aggregate = cnn.aggregate_results(seed_rows)

    assert aggregate["best_test_cumulative_accuracy_mean"] == pytest.approx(0.92)
    assert aggregate["best_test_cumulative_epoch_mean"] == pytest.approx(13.0)
    assert aggregate["best_test_final_accuracy_mean"] == pytest.approx(0.88)
    assert aggregate["best_test_final_epoch_mean"] == pytest.approx(10.0)
    assert "selection_metric" not in aggregate


def test_unified_cnn_tracks_independent_best_epochs(monkeypatch, tmp_path: Path):
    evaluations = iter(
        [
            (0.70, 0.60),
            (0.65, 0.90),
            (0.80, 0.85),
        ]
    )

    def fake_train(*_args, **_kwargs):
        return {
            "layer_losses": [1.0, 0.9, 0.8, 0.7],
            "cumulative_accuracy": 0.5,
            "final_accuracy": 0.4,
        }

    def fake_evaluate(*_args, **_kwargs):
        cumulative, final = next(evaluations)
        return {
            "local_accuracies": [0.4, 0.5, 0.6, final],
            "cumulative_accuracies": [0.4, 0.5, 0.6, cumulative],
            "cumulative_accuracy": cumulative,
            "final_accuracy": final,
        }

    class SchedulerStub:
        def __init__(self, *_args, **_kwargs):
            pass

        def step(self):
            pass

    monkeypatch.setattr(cnn, "train_one_epoch", fake_train)
    monkeypatch.setattr(cnn, "evaluate", fake_evaluate)
    monkeypatch.setattr(cnn, "CosineAnnealingLR", SchedulerStub)
    config = cnn.config_from_args(
        cnn.parse_args(
            [
                "mnist",
                "--epochs",
                "3",
                "--selection-metric",
                "final",
                "--output-dir",
                str(tmp_path),
            ]
        )
    )

    history, summary = cnn.run_seed(41, config, None, None)

    assert history[-1]["best_test_cumulative_accuracy"] == pytest.approx(0.80)
    assert history[-1]["best_test_cumulative_epoch"] == 3
    assert history[-1]["best_test_final_accuracy"] == pytest.approx(0.90)
    assert history[-1]["best_test_final_epoch"] == 2
    assert summary["best_test_cumulative_epoch"] == 3
    assert summary["best_test_final_epoch"] == 2


def test_mixer_mf_updates_multiple_blocks_on_cpu():
    model = mixer.MonoForwardMixer(
        in_channels=3,
        num_classes=10,
        image_size=8,
        patch_size=4,
        model_dim=8,
        depth=2,
        token_hidden_dim=4,
        channel_hidden_dim=16,
        dropout=0.0,
    )
    options = SimpleNamespace(
        optimizer="adam",
        lr=1e-3,
        weight_decay=0.0,
        momentum=0.9,
        epochs=1,
    )
    optimizers, _ = mixer.build_mf_optimizers(model, options)
    loader = DataLoader(
        TensorDataset(torch.randn(4, 3, 8, 8), torch.tensor([0, 1, 2, 3])),
        batch_size=2,
    )
    metadata = {"task": "multi-class", "is_medmnist": False}

    loss, top1, top5 = mixer.train_mf(
        model,
        loader,
        optimizers,
        GradScaler("cuda", enabled=False),
        torch.device("cpu"),
        local_iters=1,
        amp_enabled=False,
        metadata=metadata,
    )

    assert loss > 0
    assert 0 <= top1 <= 1
    assert 0 <= top5 <= 1


def test_mixer_defaults_match_the_main_tables():
    args = mixer.parse_args(["--datasets", "cifar10"])

    assert mixer.MAIN_DATASETS == (
        "cifar10",
        "cifar100",
        "tinyimagenet",
        "coil100",
        "pathmnist",
        "chestmnist",
        "dermamnist",
        "octmnist",
        "pneumoniamnist",
        "retinamnist",
    )
    assert args.methods == ["bp", "mf"]
    assert args.depths == [5, 8, 12]
    assert args.dims == [256, 512]
    assert args.epochs == 480
    assert mixer.resolve_seeds(args) == [41, 42, 43]
    assert args.optimizer == "adamw"
    assert args.lr == pytest.approx(3e-4)
    assert args.weight_decay == pytest.approx(5e-2)
    assert args.batch_size == 128
    assert args.mf_local_iters == 3
    assert args.early_stop_patience == 10
    assert args.early_stop_min_delta == pytest.approx(1e-4)


@pytest.mark.parametrize(
    ("dataset", "dataset_class", "mean", "std"),
    [
        ("cifar10", "CIFAR10", (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ("cifar100", "CIFAR100", (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ],
)
def test_mixer_cifar_normalization_matches_manuscript(
    dataset: str,
    dataset_class: str,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    monkeypatch,
    tmp_path: Path,
):
    def fake_dataset(_root, *, train, download, transform):
        assert download is True
        return SimpleNamespace(train=train, transform=transform)

    monkeypatch.setattr(mixer, dataset_class, fake_dataset)
    train_set, test_set, _metadata = mixer.build_dataset(dataset, tmp_path, Namespace())

    for dataset_split in (train_set, test_set):
        normalization = dataset_split.transform.transforms[-1]
        assert tuple(normalization.mean) == pytest.approx(mean)
        assert tuple(normalization.std) == pytest.approx(std)


def test_mixer_run_signature_covers_optimization_and_device_settings():
    base = Namespace(
        epochs=10,
        early_stop_patience=3,
        early_stop_min_delta=1e-4,
        batch_size=16,
        eval_batch_size=32,
        optimizer="adamw",
        lr=1e-3,
        weight_decay=1e-2,
        momentum=0.9,
        mf_local_iters=1,
        device="cpu",
        disable_amp=False,
        num_workers=0,
    )
    first = mixer.run_signature("cifar10", "mf", 2, 32, 16, 64, 4, 42, base)
    base.lr = 2e-3
    second = mixer.run_signature("cifar10", "mf", 2, 32, 16, 64, 4, 42, base)

    assert first != second


@pytest.mark.parametrize("seeds", [[7, 7], [-1]])
def test_mixer_rejects_invalid_seed_lists(seeds: list[int]):
    with pytest.raises(ValueError, match="seeds"):
        mixer.resolve_seeds(Namespace(seed=None, seeds=seeds))


def test_mixer_summary_aggregates_successful_seeds_by_runtime_setting(tmp_path: Path):
    results_path = tmp_path / "results.csv"
    summary_path = tmp_path / "summary.csv"
    setting = {
        "dataset": "cifar10",
        "method": "mf",
        "depth": 5,
        "dim": 256,
        "token_dim": 256,
        "channel_dim": 1024,
        "patch_size": 4,
        "num_classes": 10,
        "in_channels": 3,
        "image_size": 32,
        "task": "multi-class",
        "metric_primary_name": "top1",
        "metric_secondary_name": "top5",
        "epochs_target": 100,
        "early_stop_patience": 10,
        "early_stop_min_delta": 1e-4,
        "mf_local_iters": 3,
        "optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 5e-2,
        "momentum": 0.9,
        "batch_size": 128,
        "eval_batch_size": 256,
        "num_workers": 4,
        "device": "cuda:0",
        "amp_enabled": 0,
    }
    for seed, primary, secondary, memory, epoch in (
        (41, 0.8, 0.95, 4.0, 4),
        (42, 0.9, 0.97, 6.0, 6),
    ):
        mixer.append_row(
            results_path,
            {
                **setting,
                "status": "ok",
                "run_signature": f"seed-{seed}",
                "seed": seed,
                "best_test_top1": primary,
                "best_test_top5": secondary,
                "peak_train_mem_gb": memory,
                "best_epoch": epoch,
            },
        )
    mixer.append_row(
        results_path,
        {
            **setting,
            "status": "ok",
            "run_signature": "seed-42",
            "seed": 42,
            "best_test_top1": 0.7,
            "best_test_top5": 0.93,
            "peak_train_mem_gb": 8.0,
            "best_epoch": 8,
        },
    )
    mixer.append_row(
        results_path,
        {
            **setting,
            "status": "failed",
            "run_signature": "failed-seed",
            "seed": 43,
        },
    )
    mixer.append_row(
        results_path,
        {
            **setting,
            "status": "ok",
            "run_signature": "amp-seed",
            "seed": 44,
            "amp_enabled": 1,
            "best_test_top1": 0.92,
            "best_test_top5": 0.98,
            "peak_train_mem_gb": 3.0,
            "best_epoch": 5,
        },
    )

    mixer.write_summary(results_path, summary_path)

    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    no_amp = next(row for row in rows if row["amp_enabled"] == "0")
    amp = next(row for row in rows if row["amp_enabled"] == "1")
    assert no_amp["metric_primary_name"] == "top1"
    assert no_amp["metric_secondary_name"] == "top5"
    assert int(no_amp["num_runs"]) == 2
    assert float(no_amp["best_primary_mean"]) == pytest.approx(0.75)
    assert float(no_amp["best_primary_std"]) == pytest.approx(0.070710678)
    assert float(no_amp["best_secondary_mean"]) == pytest.approx(0.94)
    assert float(no_amp["peak_train_mem_gb_mean"]) == pytest.approx(6.0)
    assert float(no_amp["best_epoch_mean"]) == pytest.approx(6.0)
    assert float(no_amp["best_epoch_std"]) == pytest.approx(2 * 2**0.5)
    assert amp["num_runs"] == "1"
    assert amp["best_primary_std"] == ""


def test_mixer_writes_summary_before_returning_failure(tmp_path: Path, monkeypatch):
    train_set = TensorDataset(torch.randn(2, 3, 8, 8), torch.tensor([0, 1]))
    metadata = {
        "num_classes": 2,
        "in_channels": 3,
        "image_size": 8,
        "patch_default": 4,
        "task": "multi-class",
        "metric_primary_name": "top1",
        "metric_secondary_name": "top5",
        "is_medmnist": False,
    }
    monkeypatch.setattr(
        mixer,
        "build_dataset",
        lambda _name, _root, _args: (train_set, train_set, metadata),
    )

    def fail_run(*_args, **_kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(mixer, "run_one", fail_run)
    status = mixer.main(
        [
            "--datasets",
            "cifar10",
            "--methods",
            "mf",
            "--depths",
            "1",
            "--dims",
            "8",
            "--epochs",
            "1",
            "--seed",
            "7",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert status == 1
    with (tmp_path / "summary.csv").open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == []


def test_mixer_records_dataset_failure_and_writes_summary(tmp_path: Path, monkeypatch):
    def fail_dataset(*_args, **_kwargs):
        raise FileNotFoundError("synthetic missing dataset")

    monkeypatch.setattr(mixer, "build_dataset", fail_dataset)
    status = mixer.main(
        [
            "--datasets",
            "tinyimagenet",
            "--depths",
            "1",
            "--dims",
            "8",
            "--epochs",
            "1",
            "--seed",
            "7",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert status == 1
    with (tmp_path / "results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert rows[0]["dataset"] == "tinyimagenet"
    with (tmp_path / "summary.csv").open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == []


def test_mixer_refuses_legacy_csv_schema(tmp_path: Path):
    csv_path = tmp_path / "legacy.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "method"])
        writer.writerow(["cifar10", "mf"])

    with pytest.raises(ValueError, match="schema mismatch"):
        mixer.load_done(csv_path)


def test_zip_extraction_rejects_parent_traversal(tmp_path: Path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr("../outside.txt", "unsafe")

    with pytest.raises(ValueError, match="Unsafe path"):
        mixer.extract_zip_safely(archive, tmp_path / "destination")


def test_mixer_recovers_from_an_empty_results_file(tmp_path: Path):
    csv_path = tmp_path / "results.csv"
    csv_path.touch()

    assert mixer.load_done(csv_path) == set()
    mixer.append_row(
        csv_path,
        {"status": "ok", "run_signature": "completed-setting"},
    )

    assert mixer.load_done(csv_path) == {"completed-setting"}
