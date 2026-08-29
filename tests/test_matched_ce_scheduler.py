from __future__ import annotations

from pathlib import Path

from MatchedCELocalityControlScheduler import METHODS, command_for, jobs


def test_matched_ce_scheduler_defines_exactly_24_unique_jobs() -> None:
    matrix = jobs()
    assert len(matrix) == 24
    assert len(set(matrix)) == 24
    assert {method for _dataset, method, _seed in matrix} == set(METHODS)


def test_matched_ce_scheduler_uses_the_fixed_protocol() -> None:
    command = command_for(
        "python",
        Path("project"),
        Path("run"),
        "cifar10",
        "ce-matched-local",
        424,
    )
    joined = " ".join(map(str, command))
    assert "--epochs 200" in joined
    assert "--patience 15" in joined
    assert "--validation-size 5000" in joined
    assert "--optimizer adam" in joined
    assert "--learning-rate 0.001" in joined
    assert "--batch-size 128" in joined
    assert "--scheduler none" in joined
    assert "--no-download" in joined
