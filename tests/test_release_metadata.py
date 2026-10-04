from __future__ import annotations

import csv
from collections import defaultdict
import json
from pathlib import Path
import statistics

from MLPOptimizerSweepScheduler import FROZEN_THRESHOLDS


ROOT = Path(__file__).resolve().parents[1]


def test_paper_protocol_and_reported_results_cover_the_same_matrix() -> None:
    protocol = json.loads((ROOT / "configs" / "paper_protocol.json").read_text())
    with (ROOT / "paper_results" / "reported_accuracy.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))

    identities = [
        (row["architecture"], row["dataset"], row["method_id"]) for row in rows
    ]
    assert len(identities) == len(set(identities)) == 68

    expected: set[tuple[str, str, str]] = set()
    for architecture in ("mlp", "cnn", "mlp_mixer"):
        section = protocol[architecture]
        expected.update(
            (architecture, dataset, method)
            for dataset in section["datasets"]
            for method in section["methods"]
        )
    assert set(identities) == expected


def test_reported_result_rows_distinguish_intentional_omissions() -> None:
    with (ROOT / "paper_results" / "reported_accuracy.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))

    omissions = {
        (row["architecture"], row["dataset"], row["method_id"])
        for row in rows
        if row["status"] == "not_evaluated"
    }
    assert omissions == {
        ("mlp", "cifar100", "fc-ff"),
        ("mlp", "cifar100", "fc-ff-matched-ge"),
        ("mlp", "cifar100", "fc-ff-ge"),
        ("mlp", "cifar100", "fc-nn-ff-ge"),
    }

    for row in rows:
        assert row["seeds"] == "424;425;426"
        if row["status"] == "reported":
            assert float(row["mean_accuracy_percent"]) > 0
            assert float(row["sample_std_percent"]) >= 0
        else:
            assert row["mean_accuracy_percent"] == ""
            assert row["sample_std_percent"] == ""


def test_frozen_thresholds_match_protocol_and_artifact_backed_rows() -> None:
    protocol = json.loads((ROOT / "configs" / "paper_protocol.json").read_text())
    assert protocol["mlp"]["threshold_selection"]["frozen_values"] == FROZEN_THRESHOLDS

    with (ROOT / "paper_results" / "per_seed_provenance.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["architecture"] != "mlp" or not row["threshold"]:
            continue
        expected = FROZEN_THRESHOLDS[row["dataset"]][row["method_id"]]
        assert float(row["threshold"]) == expected


def test_sanitized_provenance_exactly_partitions_reported_rows() -> None:
    with (ROOT / "paper_results" / "reported_accuracy.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        reported_rows = {
            (row["architecture"], row["dataset"], row["method_id"]): row
            for row in csv.DictReader(handle)
            if row["status"] == "reported"
        }
    with (ROOT / "paper_results" / "per_seed_provenance.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        provenance_rows = list(csv.DictReader(handle))
    with (ROOT / "paper_results" / "unsupported_reported_rows.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        unsupported_rows = list(csv.DictReader(handle))

    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in provenance_rows:
        key = (row["architecture"], row["dataset"], row["method_id"])
        groups[key].append(row)
        assert row["validation_only_selection"] == "true"
        joined = " ".join(row.values()).lower()
        assert "c:\\" not in joined
        assert "/home/" not in joined

    unsupported = {
        (row["architecture"], row["dataset"], row["method_id"])
        for row in unsupported_rows
    }
    assert len(groups) == 50
    assert len(provenance_rows) == 150
    assert len(unsupported) == 14
    assert set(groups).isdisjoint(unsupported)
    assert set(groups) | unsupported == set(reported_rows)

    for key, rows in groups.items():
        assert {int(row["seed"]) for row in rows} == {424, 425, 426}
        values = [float(row["test_accuracy_percent"]) for row in rows]
        expected = reported_rows[key]
        assert round(statistics.mean(values), 2) == float(
            expected["mean_accuracy_percent"]
        )
        assert round(statistics.stdev(values), 2) == float(
            expected["sample_std_percent"]
        )
