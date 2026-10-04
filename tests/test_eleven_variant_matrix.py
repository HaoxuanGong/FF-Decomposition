from __future__ import annotations

import torch

import MLPBenchmarkSuite as suite


EXPECTED_PRIMARY_METHODS = (
    "ff",
    "ff-matched-ge",
    "ff-ge",
    "nn-ff-ge",
    "fc-ff",
    "fc-ff-matched-ge",
    "fc-ff-ge",
    "fc-nn-ff-ge",
    "local-bp",
    "ce-matched-ge",
    "bp",
)

EXPECTED_PRIMARY_LABELS = (
    "Vanilla FF (Local, Multi-Head)",
    "FF (Global, Multi-Head)",
    "FF (Global, Terminal)",
    "NN-FF (Global, Terminal)",
    "FC-FF (Local, Multi-Head)",
    "FC-FF (Global, Multi-Head)",
    "FC-FF (Global, Terminal)",
    "FC-NN-FF (Global, Terminal)",
    "CE (Local, Multi-Head)",
    "CE (Global, Multi-Head)",
    "CE (Global, Terminal)",
)


def run_args(method: str, *extra: str):
    return suite.parser().parse_args(
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


def test_primary_taxonomy_contains_exactly_the_eleven_paper_variants() -> None:
    assert suite.PRIMARY_METHODS == EXPECTED_PRIMARY_METHODS
    assert len(suite.PRIMARY_METHODS) == 11
    assert tuple(suite.METHOD_LABELS[method] for method in suite.PRIMARY_METHODS) == (
        EXPECTED_PRIMARY_LABELS
    )
    assert set(suite.COMPATIBILITY_METHODS).isdisjoint(suite.PRIMARY_METHODS)


def test_new_variants_record_the_intended_ablation_dimensions() -> None:
    no_norm = suite.configuration(run_args("nn-ff-ge", "--goodness-threshold", "4"))
    assert no_norm["inter_layer_normalization"] is False
    assert no_norm["first_layer_input_normalization"] is True
    assert no_norm["goodness_threshold"] == 4.0
    assert no_norm["matched_loss_placement"] is None
    assert no_norm["detach_between_layers"] is None
    assert no_norm["prediction"] == "final-layer goodness"

    fc_global_multihead = suite.configuration(run_args("fc-ff-matched-ge"))
    assert fc_global_multihead["inter_layer_normalization"] is True
    assert fc_global_multihead["goodness_threshold"] is None
    assert fc_global_multihead["matched_loss_placement"] == (
        "equal-weight mean of one all-class goodness loss at every layer"
    )
    assert fc_global_multihead["detach_between_layers"] is False
    assert fc_global_multihead["prediction"] == "summed layer goodness"


def test_nn_ff_terminal_loss_propagates_through_all_layers_without_inter_layer_normalization() -> None:
    model = suite.build_model("nn-ff-ge", "mnist", hidden_dims=(8, 8, 8))
    inputs = torch.randn(5, 784)
    labels = torch.arange(5) % 10
    scores = suite.candidate_goodness(
        model,
        inputs,
        10,
        aggregation="final",
        chunk_size=5,
    )
    torch.nn.functional.cross_entropy(scores, labels).backward()
    assert all(layer.weight.grad is not None for layer in model.layers)
