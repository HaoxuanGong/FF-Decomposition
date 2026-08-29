import json
from pathlib import Path

import pytest
import torch

from MLPBenchmarkSuite import build_model, configuration, parser
from NormalizationAblationScheduler import (
    DATASETS,
    METHODS,
    SEEDS,
    _load_result,
    command_for,
    jobs,
)


def test_normalization_matrix_has_all_expected_pairs() -> None:
    matrix = jobs()
    assert len(matrix) == len(DATASETS) * len(METHODS) * len(SEEDS) == 18
    assert {dataset for dataset, _method, _seed in matrix} == set(DATASETS)
    assert {method for _dataset, method, _seed in matrix} == set(METHODS)
    assert {seed for _dataset, _method, seed in matrix} == set(SEEDS)
    for index in range(0, len(matrix), len(METHODS)):
        pair = matrix[index : index + len(METHODS)]
        assert len({dataset for dataset, _method, _seed in pair}) == 1
        assert len({seed for _dataset, _method, seed in pair}) == 1
        assert [method for _dataset, method, _seed in pair] == list(METHODS)


def test_normalization_pair_commands_differ_only_by_method() -> None:
    project_root = Path("/project")
    run_dir = Path("/run")
    normalized = command_for(
        "python", project_root, run_dir, "cifar10", "fc-ff-ge", 424
    )
    unnormalized = command_for(
        "python", project_root, run_dir, "cifar10", "fc-nn-ff-ge", 424
    )
    method_index = normalized.index("--method") + 1
    assert normalized[method_index] == "fc-ff-ge"
    assert unnormalized[method_index] == "fc-nn-ff-ge"
    normalized[method_index] = "METHOD"
    unnormalized[method_index] = "METHOD"
    assert normalized == unnormalized
    for flag, expected in (
        ("--epochs", "200"),
        ("--patience", "15"),
        ("--validation-size", "5000"),
        ("--learning-rate", "0.001"),
        ("--batch-size", "128"),
        ("--scheduler", "none"),
    ):
        assert normalized[normalized.index(flag) + 1] == expected


def _run_args(method: str, *extra: str):
    return parser().parse_args(
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
            *extra,
        ]
    )


def test_normalization_pair_configs_differ_only_in_inter_layer_policy() -> None:
    normalized = configuration(_run_args("fc-ff-ge"))
    no_inter_layer = configuration(_run_args("fc-nn-ff-ge"))
    differences = {
        key for key in normalized if normalized[key] != no_inter_layer[key]
    }
    assert differences == {
        "method",
        "inter_layer_normalization",
        "normalization_scope",
    }
    assert normalized["first_layer_input_normalization"] is True
    assert no_inter_layer["first_layer_input_normalization"] is True
    assert normalized["normalization_epsilon"] == no_inter_layer["normalization_epsilon"]


def test_normalization_pair_has_identical_first_layer_outputs_before_update() -> None:
    torch.manual_seed(29)
    normalized = build_model(
        "fc-ff-ge", "mnist", hidden_dims=(16, 16)
    )
    torch.manual_seed(29)
    no_inter_layer = build_model(
        "fc-nn-ff-ge", "mnist", hidden_dims=(16, 16)
    )
    assert all(
        torch.equal(normalized_value, no_inter_layer_value)
        for normalized_value, no_inter_layer_value in zip(
            normalized.state_dict().values(),
            no_inter_layer.state_dict().values(),
            strict=True,
        )
    )

    inputs = torch.randn(5, 28 * 28)
    normalized_first = normalized.forward_layer(inputs, 0)
    no_inter_layer_first = no_inter_layer.forward_layer(inputs, 0)
    assert torch.equal(normalized_first, no_inter_layer_first)
    assert normalized.normalizes_layer_input(0) is True
    assert no_inter_layer.normalizes_layer_input(0) is True

    assert normalized.normalizes_layer_input(1) is True
    assert no_inter_layer.normalizes_layer_input(1) is False
    normalized_second = normalized.forward_layer(normalized_first, 1)
    no_inter_layer_second = no_inter_layer.forward_layer(no_inter_layer_first, 1)
    assert not torch.allclose(normalized_second, no_inter_layer_second)


def test_result_loader_enforces_fixed_protocol_and_common_input_normalization(
    tmp_path: Path,
) -> None:
    args = _run_args(
        "fc-nn-ff-ge",
        "--num-workers",
        "4",
        "--device",
        "cuda",
    )
    config = configuration(args)
    path = (
        tmp_path
        / "results"
        / "mnist"
        / "fc-nn-ff-ge"
        / "seed_424"
        / "run.json"
    )
    path.parent.mkdir(parents=True)
    payload = {"status": "complete", "test_evaluations": 1, "config": config}
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert _load_result(tmp_path, "mnist", "fc-nn-ff-ge", 424) == payload

    config["first_layer_input_normalization"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Fixed-protocol mismatch"):
        _load_result(tmp_path, "mnist", "fc-nn-ff-ge", 424)
