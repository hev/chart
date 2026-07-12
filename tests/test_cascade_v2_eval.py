from __future__ import annotations

import json
from pathlib import Path

from eval.cascade_v2 import FAMILIES, score_labels


ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def test_gold_set_is_id_keyed_and_sized() -> None:
    selection = load("eval/gold/cascade-v2-selection.json")
    gold = load("eval/gold/cascade-v2-gold.json")

    assert len(selection["notes"]) == 500
    assert len(gold["labels"]) == 500
    assert selection["metadata"]["dual_labeled_count"] >= 100
    assert selection["metadata"]["plan_overlap_guard"]["selected_overlap"] == 0
    assert {item["stratum"] for item in selection["notes"]} == {
        "random",
        "stop_adverse_enriched",
        "procedure_response_enriched",
    }
    for item in selection["notes"]:
        assert sorted(item) == ["dual_labeled", "id", "stratum"]
    for labels in gold["labels"].values():
        assert set(labels) == set(FAMILIES)


def test_agreement_reports_kappa_per_family() -> None:
    agreement = load("eval/out/cascade-v2-agreement.json")

    assert agreement["dual_labeled"] >= 100
    assert set(agreement["per_family"]) == set(FAMILIES)
    for stats in agreement["per_family"].values():
        assert stats["n"] == agreement["dual_labeled"]
        assert stats["cohen_kappa"] is None or -1.0 <= stats["cohen_kappa"] <= 1.0


def test_v1_baseline_report_matches_harness() -> None:
    gold = load("eval/gold/cascade-v2-gold.json")
    pred = load("eval/out/cascade-v2-v1-labels.json")
    report = load("eval/out/cascade-v2-v1-baseline.json")

    recomputed = score_labels(gold, pred)
    assert report["macro"] == recomputed["macro"]
    assert "medication_stopped_false_positive_buckets" in report
    assert set(report["per_label"]) == set(FAMILIES)


def test_query_probes_are_qrels_style() -> None:
    probe_dir = ROOT / "eval/out/cascade-v2-probes"
    queries = [json.loads(line) for line in (probe_dir / "queries.jsonl").read_text().splitlines()]
    qrels = (probe_dir / "qrels.tsv").read_text().splitlines()
    metadata = json.loads((probe_dir / "metadata.json").read_text())

    assert len(queries) == 10
    assert qrels[0] == "query-id\tcorpus-id\tscore"
    assert len(qrels) > 10
    assert metadata["shape"].startswith("BEIR/TREC")
    assert {query["facet"] for query in queries} <= set(FAMILIES)
