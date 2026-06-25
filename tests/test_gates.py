import pytest

from smoke import gates
from smoke.gates import summarize_gates


def accepted_cost_baselines():
    return {
        "embed": {
            "report": "eval/out/embed-budget.json",
            "accepted": True,
            "estimate": {"full_notes": 167000, "estimated_full_hours": 0.28},
        },
        "classifier": {
            "report": "eval/out/classify-events-budget.json",
            "accepted": True,
            "estimate": {"full_notes": 167000, "estimated_full_hours": 0.56},
        },
    }


def accepted_layer_cost_baselines():
    snapshot = {
        "as_of_ms": 1782320205904,
        "window_seconds": 86400,
        "totals": {"total_usd": 12.34},
        "lines": [{"provider": "aws", "service": "compute", "basis": "invoice", "amount_usd": 12.34}],
    }
    return {
        "embed": {
            "report": "eval/out/embed-budget.json",
            "accepted": True,
            "source": "layer",
            "layer_cost_snapshot": snapshot,
        },
        "classifier": {
            "report": "eval/out/classify-events-budget.json",
            "accepted": True,
            "source": "layer",
            "layer_cost_snapshot": snapshot,
        },
    }


def test_gate_report_marks_phase6_incomplete_when_resources_missing() -> None:
    report = summarize_gates(
        {
            "namespace": "chart-notes",
            "pipeline": {"pipeline_id": "chart-notes", "error": {"status_code": 404}},
            "udf": {"udf_id": "chart-classify-events", "error": {"status_code": 404}},
            "facets": {
                "age_band": {"values": 6},
                "gender": {"values": 2},
                "events": {"values": 0},
            },
        }
    )

    assert report["gates"] == {
        "pipeline_installed": False,
        "udf_installed": False,
        "full_index_complete": False,
        "full_classify_complete": False,
        "base_facets_visible": True,
        "event_facets_visible": False,
        "full_facets_complete": False,
        "cost_baselines_accepted": False,
        "phase6_complete": False,
    }
    assert report["targets"] == {"namespace": "chart-notes", "full_corpus_notes": 167000}
    assert report["failures"][:2] == [
        {"gate": "pipeline_installed", "reason": "HTTP 404"},
        {"gate": "udf_installed", "reason": "HTTP 404"},
    ]
    assert any(failure["gate"] == "full_facets_complete" for failure in report["failures"])


def test_gate_report_preserves_status_targets_when_present() -> None:
    report = summarize_gates(
        {
            "namespace": "chart-notes",
            "targets": {
                "namespace": "chart-notes",
                "pipeline_id": "chart-notes",
                "udf_id": "chart-classify-events",
                "full_corpus_notes": 167000,
            },
            "pipeline": {"pipeline_id": "chart-notes", "error": {"status_code": 404}},
            "udf": {"udf_id": "chart-classify-events", "error": {"status_code": 404}},
            "facets": {},
        }
    )

    assert report["targets"] == {
        "namespace": "chart-notes",
        "pipeline_id": "chart-notes",
        "udf_id": "chart-classify-events",
        "full_corpus_notes": 167000,
    }


def test_gate_report_uses_status_target_for_counts_and_failure_reasons() -> None:
    report = summarize_gates(
        {
            "namespace": "chart-notes",
            "targets": {
                "namespace": "chart-notes",
                "pipeline_id": "chart-notes",
                "udf_id": "chart-classify-events",
                "full_corpus_notes": 10,
            },
            "pipeline": {
                "pipeline_id": "chart-notes",
                "counts": {"indexed": 10},
                "pending_count": 0,
                "processing_count": 0,
            },
            "udf": {
                "udf_id": "chart-classify-events",
                "counts": {"completed": 9},
                "pending_count": 0,
                "processing_count": 0,
            },
            "facets": {
                "specialty": {"values": 4, "row_count": 10, "sha": "target-snapshot"},
                "age_band": {"values": 6, "row_count": 10, "sha": "target-snapshot"},
                "diagnosis_category": {"values": 5, "row_count": 10, "sha": "target-snapshot"},
                "gender": {"values": 2, "row_count": 10, "sha": "target-snapshot"},
                "events": {"values": 4, "row_count": 10, "sha": "target-snapshot"},
            },
        }
    )

    assert report["gates"]["full_index_complete"] is True
    assert report["gates"]["full_facets_complete"] is True
    assert report["gates"]["full_classify_complete"] is False
    assert {"gate": "full_classify_complete", "reason": "9/10 ready"} in report["failures"]


def test_gate_report_reports_malformed_numeric_status_fields() -> None:
    report = summarize_gates(
        {
            "namespace": "chart-notes",
            "pipeline": {
                "pipeline_id": "chart-notes",
                "counts": {"indexed": "not-a-number"},
                "pending_count": 0,
                "processing_count": 0,
            },
            "udf": {
                "udf_id": "chart-classify-events",
                "counts": {"completed": 167000},
                "pending_count": "pending",
                "processing_count": 0,
            },
            "facets": {
                "specialty": {"values": 4, "row_count": 167000, "sha": "full-snapshot"},
                "age_band": {"values": 6, "row_count": "many", "sha": "full-snapshot"},
                "diagnosis_category": {"values": 5, "row_count": 167000, "sha": "full-snapshot"},
                "gender": {"values": 2, "row_count": 167000, "sha": "full-snapshot"},
                "events": {"values": 4, "row_count": 167000, "sha": "full-snapshot"},
            },
        }
    )

    assert report["gates"]["full_index_complete"] is False
    assert report["gates"]["full_classify_complete"] is False
    assert report["gates"]["full_facets_complete"] is False
    assert {"gate": "full_index_complete", "reason": "invalid numeric field(s): counts.indexed"} in report["failures"]
    assert {
        "gate": "full_classify_complete",
        "reason": "invalid numeric field(s): pending_count",
    } in report["failures"]
    facet_failure = next(failure for failure in report["failures"] if failure["gate"] == "full_facets_complete")
    assert {
        "field": "age_band",
        "reasons": ["invalid numeric field(s): row_count", "row_count=0/167000"],
    } in facet_failure["facets"]


def test_gate_report_rejects_full_counts_from_wrong_pipeline_target() -> None:
    report = summarize_gates(
        {
            "namespace": "chart-notes",
            "targets": {
                "namespace": "chart-notes",
                "pipeline_id": "chart-notes",
                "udf_id": "chart-classify-events",
                "full_corpus_notes": 167000,
            },
            "pipeline": {
                "pipeline_id": "other-pipeline",
                "counts": {"indexed": 167000},
                "pending_count": 0,
                "processing_count": 0,
            },
            "udf": {
                "udf_id": "chart-classify-events",
                "counts": {"completed": 167000},
                "pending_count": 0,
                "processing_count": 0,
            },
            "facets": {
                "specialty": {"values": 4, "row_count": 167000, "sha": "full-snapshot"},
                "age_band": {"values": 6, "row_count": 167000, "sha": "full-snapshot"},
                "diagnosis_category": {"values": 5, "row_count": 167000, "sha": "full-snapshot"},
                "gender": {"values": 2, "row_count": 167000, "sha": "full-snapshot"},
                "events": {"values": 4, "row_count": 167000, "sha": "full-snapshot"},
            },
        }
    )

    assert report["gates"]["pipeline_installed"] is True
    assert report["gates"]["full_index_complete"] is False
    assert report["gates"]["phase6_complete"] is False
    assert {
        "gate": "full_index_complete",
        "reason": "pipeline_id='other-pipeline', expected 'chart-notes'",
    } in report["failures"]


def test_gate_report_attaches_kubernetes_context_to_pending_full_index() -> None:
    report = summarize_gates(
        {
            "namespace": "chart-notes",
            "targets": {
                "namespace": "chart-notes",
                "pipeline_id": "chart-notes",
                "udf_id": "chart-classify-events",
                "full_corpus_notes": 10,
            },
            "pipeline": {
                "pipeline_id": "chart-notes",
                "counts": {"indexed": 3},
                "pending_count": 7,
                "processing_count": 0,
            },
            "udf": {},
            "facets": {},
            "kubernetes": {
                "embed_pods": [{"name": "chart-embed", "phase": "Pending"}],
                "gpu_pods": [{"namespace": "hev-shop", "name": "hev-shop-embed"}],
            },
        }
    )

    failure = next(failure for failure in report["failures"] if failure["gate"] == "full_index_complete")
    assert failure["kubernetes"]["embed_pods"][0]["phase"] == "Pending"
    assert failure["kubernetes"]["gpu_pods"][0]["namespace"] == "hev-shop"


def test_gate_report_rejects_full_counts_from_wrong_udf_target() -> None:
    report = summarize_gates(
        {
            "namespace": "chart-notes",
            "targets": {
                "namespace": "chart-notes",
                "pipeline_id": "chart-notes",
                "udf_id": "chart-classify-events",
                "full_corpus_notes": 167000,
            },
            "pipeline": {
                "pipeline_id": "chart-notes",
                "counts": {"indexed": 167000},
                "pending_count": 0,
                "processing_count": 0,
            },
            "udf": {
                "udf_id": "other-udf",
                "counts": {"completed": 167000},
                "pending_count": 0,
                "processing_count": 0,
            },
            "facets": {
                "specialty": {"values": 4, "row_count": 167000, "sha": "full-snapshot"},
                "age_band": {"values": 6, "row_count": 167000, "sha": "full-snapshot"},
                "diagnosis_category": {"values": 5, "row_count": 167000, "sha": "full-snapshot"},
                "gender": {"values": 2, "row_count": 167000, "sha": "full-snapshot"},
                "events": {"values": 4, "row_count": 167000, "sha": "full-snapshot"},
            },
        }
    )

    assert report["gates"]["udf_installed"] is True
    assert report["gates"]["full_classify_complete"] is False
    assert report["gates"]["phase6_complete"] is False
    assert {
        "gate": "full_classify_complete",
        "reason": "udf_id='other-udf', expected 'chart-classify-events'",
    } in report["failures"]


def test_gate_report_marks_phase6_incomplete_when_status_bodies_are_empty() -> None:
    report = summarize_gates(
        {
            "namespace": "chart-notes",
            "pipeline": {},
            "udf": {},
            "facets": {
                "age_band": {"values": 6},
                "gender": {"values": 2},
                "events": {"values": 0},
            },
        }
    )

    assert report["gates"]["pipeline_installed"] is False
    assert report["gates"]["udf_installed"] is False
    assert report["gates"]["full_index_complete"] is False
    assert report["gates"]["full_classify_complete"] is False
    assert report["gates"]["phase6_complete"] is False
    assert report["failures"][0] == {"gate": "pipeline_installed", "reason": "status body is empty"}


def test_gate_report_does_not_mark_phase6_complete_for_slice_facets() -> None:
    report = summarize_gates(
        {
            "namespace": "chart-notes",
            "pipeline": {
                "pipeline_id": "chart-notes",
                "counts": {"indexed": 167000},
                "pending_count": 0,
                "processing_count": 0,
            },
            "udf": {
                "udf_id": "chart-classify-events",
                "counts": {"completed": 167000},
                "pending_count": 0,
                "processing_count": 0,
            },
            "facets": {
                "age_band": {"values": 6, "row_count": 2000, "sha": "full-snapshot"},
                "gender": {"values": 2, "row_count": 2000, "sha": "full-snapshot"},
                "specialty": {"values": 4, "row_count": 2000, "sha": "full-snapshot"},
                "diagnosis_category": {"values": 5, "row_count": 2000, "sha": "full-snapshot"},
                "events": {"values": 4, "row_count": 2000, "sha": "full-snapshot"},
            },
        }
    )

    assert report["gates"]["full_facets_complete"] is False
    assert report["gates"]["phase6_complete"] is False
    facet_failure = next(failure for failure in report["failures"] if failure["gate"] == "full_facets_complete")
    assert facet_failure["facets"][0] == {"field": "specialty", "reasons": ["row_count=2000/167000"]}


def test_gate_report_marks_phase6_complete_only_after_full_counts_and_full_event_facets() -> None:
    report = summarize_gates(
        {
            "namespace": "chart-notes",
            "cost_baselines": accepted_cost_baselines(),
            "pipeline": {
                "pipeline_id": "chart-notes",
                "counts": {"indexed": 167000},
                "pending_count": 0,
                "processing_count": 0,
            },
            "udf": {
                "udf_id": "chart-classify-events",
                "counts": {"completed": 167000},
                "pending_count": 0,
                "processing_count": 0,
            },
            "facets": {
                "specialty": {"values": 4, "row_count": 167000, "sha": "full-snapshot"},
                "age_band": {"values": 6, "row_count": 167000, "sha": "full-snapshot"},
                "diagnosis_category": {"values": 5, "row_count": 167000, "sha": "full-snapshot"},
                "gender": {"values": 2, "row_count": 167000, "sha": "full-snapshot"},
                "events": {"values": 4, "row_count": 167000, "sha": "full-snapshot"},
            },
        }
    )

    assert report["gates"]["full_facets_complete"] is True
    assert report["gates"]["cost_baselines_accepted"] is True
    assert report["gates"]["phase6_complete"] is True


def test_gate_report_requires_accepted_cost_baselines_for_phase6_completion() -> None:
    report = summarize_gates(
        {
            "namespace": "chart-notes",
            "cost_baselines": {
                "embed": {
                    "report": "eval/out/embed-budget.json",
                    "accepted": False,
                    "estimate": {"full_notes": 167000, "estimated_full_hours": 0.28},
                },
                "classifier": {
                    "accepted": True,
                    "estimate": {"full_notes": 167000, "estimated_full_hours": 0.56},
                },
            },
            "pipeline": {
                "pipeline_id": "chart-notes",
                "counts": {"indexed": 167000},
                "pending_count": 0,
                "processing_count": 0,
            },
            "udf": {
                "udf_id": "chart-classify-events",
                "counts": {"completed": 167000},
                "pending_count": 0,
                "processing_count": 0,
            },
            "facets": {
                "specialty": {"values": 4, "row_count": 167000, "sha": "full-snapshot"},
                "age_band": {"values": 6, "row_count": 167000, "sha": "full-snapshot"},
                "diagnosis_category": {"values": 5, "row_count": 167000, "sha": "full-snapshot"},
                "gender": {"values": 2, "row_count": 167000, "sha": "full-snapshot"},
                "events": {"values": 4, "row_count": 167000, "sha": "full-snapshot"},
            },
        }
    )

    assert report["gates"]["full_index_complete"] is True
    assert report["gates"]["full_classify_complete"] is True
    assert report["gates"]["full_facets_complete"] is True
    assert report["gates"]["cost_baselines_accepted"] is False
    assert report["gates"]["phase6_complete"] is False
    assert {
        "gate": "cost_baselines_accepted",
        "reason": "incomplete cost baselines",
        "baselines": [
            {"baseline": "embed", "reasons": ["not accepted"]},
            {"baseline": "classifier", "reasons": ["missing report path"]},
        ],
    } in report["failures"]


def test_gate_report_accepts_layer_cost_baseline_snapshots() -> None:
    report = summarize_gates(
        {
            "namespace": "chart-notes",
            "cost_baselines": accepted_layer_cost_baselines(),
            "pipeline": {
                "pipeline_id": "chart-notes",
                "counts": {"indexed": 167000},
                "pending_count": 0,
                "processing_count": 0,
            },
            "udf": {
                "udf_id": "chart-classify-events",
                "counts": {"completed": 167000},
                "pending_count": 0,
                "processing_count": 0,
            },
            "facets": {
                "specialty": {"values": 4, "row_count": 167000, "sha": "full-snapshot"},
                "age_band": {"values": 6, "row_count": 167000, "sha": "full-snapshot"},
                "diagnosis_category": {"values": 5, "row_count": 167000, "sha": "full-snapshot"},
                "gender": {"values": 2, "row_count": 167000, "sha": "full-snapshot"},
                "events": {"values": 4, "row_count": 167000, "sha": "full-snapshot"},
            },
        }
    )

    assert report["gates"]["cost_baselines_accepted"] is True
    assert not any(failure["gate"] == "cost_baselines_accepted" for failure in report["failures"])


def test_gate_report_requires_full_facet_snapshots_from_same_snapshot() -> None:
    report = summarize_gates(
        {
            "namespace": "chart-notes",
            "pipeline": {
                "pipeline_id": "chart-notes",
                "counts": {"indexed": 167000},
                "pending_count": 0,
                "processing_count": 0,
            },
            "udf": {
                "udf_id": "chart-classify-events",
                "counts": {"completed": 167000},
                "pending_count": 0,
                "processing_count": 0,
            },
            "facets": {
                "specialty": {"values": 4, "row_count": 167000, "sha": "snapshot-a"},
                "age_band": {"values": 6, "row_count": 167000, "sha": "snapshot-a"},
                "diagnosis_category": {"values": 5, "row_count": 167000, "sha": "snapshot-b"},
                "gender": {"values": 2, "row_count": 167000, "sha": "snapshot-a"},
                "events": {"values": 4, "row_count": 167000, "sha": "snapshot-a"},
            },
        }
    )

    assert report["gates"]["full_facets_complete"] is False
    assert report["gates"]["phase6_complete"] is False
    facet_failure = next(failure for failure in report["failures"] if failure["gate"] == "full_facets_complete")
    assert {
        "field": "diagnosis_category",
        "reasons": ["snapshot sha differs from other full facets"],
    } in facet_failure["facets"]


def test_gate_report_requires_all_configured_facets_for_phase6_completion() -> None:
    report = summarize_gates(
        {
            "namespace": "chart-notes",
            "pipeline": {
                "pipeline_id": "chart-notes",
                "counts": {"indexed": 167000},
                "pending_count": 0,
                "processing_count": 0,
            },
            "udf": {
                "udf_id": "chart-classify-events",
                "counts": {"completed": 167000},
                "pending_count": 0,
                "processing_count": 0,
            },
            "facets": {
                "age_band": {"values": 6, "row_count": 167000, "sha": "full-snapshot"},
                "gender": {"values": 2, "row_count": 167000, "sha": "full-snapshot"},
                "events": {"values": 4, "row_count": 167000, "sha": "full-snapshot"},
            },
        }
    )

    assert report["gates"]["full_facets_complete"] is False
    assert report["gates"]["phase6_complete"] is False


def test_gate_report_requires_zero_pipeline_failures() -> None:
    report = summarize_gates(
        {
            "namespace": "chart-notes",
            "pipeline": {
                "pipeline_id": "chart-notes",
                "counts": {"indexed": 167000},
                "pending_count": 0,
                "processing_count": 0,
                "failed_count": 1,
            },
            "udf": {
                "udf_id": "chart-classify-events",
                "counts": {"completed": 167000},
                "pending_count": 0,
                "processing_count": 0,
            },
            "facets": {
                "specialty": {"values": 4, "row_count": 167000, "sha": "full-snapshot"},
                "age_band": {"values": 6, "row_count": 167000, "sha": "full-snapshot"},
                "diagnosis_category": {"values": 5, "row_count": 167000, "sha": "full-snapshot"},
                "gender": {"values": 2, "row_count": 167000, "sha": "full-snapshot"},
                "events": {"values": 4, "row_count": 167000, "sha": "full-snapshot"},
            },
        }
    )

    assert report["gates"]["full_index_complete"] is False
    assert report["gates"]["phase6_complete"] is False
    assert {"gate": "full_index_complete", "reason": "1 failed"} in report["failures"]


def test_gate_report_requires_zero_udf_failures_from_counts() -> None:
    report = summarize_gates(
        {
            "namespace": "chart-notes",
            "pipeline": {
                "pipeline_id": "chart-notes",
                "counts": {"indexed": 167000},
                "pending_count": 0,
                "processing_count": 0,
            },
            "udf": {
                "udf_id": "chart-classify-events",
                "counts": {"completed": 167000, "failed": 2},
                "pending_count": 0,
                "processing_count": 0,
            },
            "facets": {
                "specialty": {"values": 4, "row_count": 167000, "sha": "full-snapshot"},
                "age_band": {"values": 6, "row_count": 167000, "sha": "full-snapshot"},
                "diagnosis_category": {"values": 5, "row_count": 167000, "sha": "full-snapshot"},
                "gender": {"values": 2, "row_count": 167000, "sha": "full-snapshot"},
                "events": {"values": 4, "row_count": 167000, "sha": "full-snapshot"},
            },
        }
    )

    assert report["gates"]["full_classify_complete"] is False
    assert report["gates"]["phase6_complete"] is False
    assert {"gate": "full_classify_complete", "reason": "2 failed"} in report["failures"]


def test_gate_report_requires_full_facet_snapshot_provenance() -> None:
    report = summarize_gates(
        {
            "namespace": "chart-notes",
            "pipeline": {
                "pipeline_id": "chart-notes",
                "counts": {"indexed": 167000},
                "pending_count": 0,
                "processing_count": 0,
            },
            "udf": {
                "udf_id": "chart-classify-events",
                "counts": {"completed": 167000},
                "pending_count": 0,
                "processing_count": 0,
            },
            "facets": {
                "specialty": {"values": 4, "row_count": 167000, "sha": "full-snapshot"},
                "age_band": {"values": 6, "row_count": 167000, "sha": "full-snapshot"},
                "diagnosis_category": {"values": 5, "row_count": 167000, "sha": "full-snapshot"},
                "gender": {"values": 2, "row_count": 167000, "sha": "full-snapshot"},
                "events": {"values": 4, "row_count": 167000, "sha": None},
            },
        }
    )

    assert report["gates"]["event_facets_visible"] is True
    assert report["gates"]["full_facets_complete"] is False
    assert report["gates"]["phase6_complete"] is False
    facet_failure = next(failure for failure in report["failures"] if failure["gate"] == "full_facets_complete")
    assert {"field": "events", "reasons": ["missing sha"]} in facet_failure["facets"]


def test_gate_report_surfaces_facet_gateway_errors() -> None:
    report = summarize_gates(
        {
            "namespace": "chart-notes",
            "pipeline": {
                "pipeline_id": "chart-notes",
                "counts": {"indexed": 167000},
                "pending_count": 0,
                "processing_count": 0,
            },
            "udf": {
                "udf_id": "chart-classify-events",
                "counts": {"completed": 167000},
                "pending_count": 0,
                "processing_count": 0,
            },
            "facets": {
                "specialty": {"values": 4, "row_count": 167000, "sha": "full-snapshot"},
                "age_band": {
                    "values": 0,
                    "row_count": None,
                    "sha": None,
                    "error": {"status_code": 502, "message": "bad gateway"},
                },
                "diagnosis_category": {"values": 5, "row_count": 167000, "sha": "full-snapshot"},
                "gender": {"values": 2, "row_count": 167000, "sha": "full-snapshot"},
                "events": {"values": 4, "row_count": 167000, "sha": "full-snapshot"},
            },
        }
    )

    assert report["gates"]["full_facets_complete"] is False
    assert report["gates"]["phase6_complete"] is False
    facet_failure = next(failure for failure in report["failures"] if failure["gate"] == "full_facets_complete")
    assert {"field": "age_band", "reasons": ["HTTP 502: bad gateway"]} in facet_failure["facets"]


def test_require_complete_cli_exits_nonzero_when_phase6_incomplete(monkeypatch, capsys) -> None:
    async def fake_collect_gate_report(*, pipeline_id=None, udf_id="chart-classify-events"):
        return {
            "namespace": "chart-notes",
            "gates": {"phase6_complete": False},
            "status": {},
        }

    monkeypatch.setattr(gates, "collect_gate_report", fake_collect_gate_report)
    monkeypatch.setattr("sys.argv", ["gates", "--require-complete"])

    with pytest.raises(SystemExit) as exc:
        gates.main()

    assert exc.value.code == 1
    assert '"phase6_complete": false' in capsys.readouterr().out


def test_require_complete_cli_exits_zero_when_phase6_complete(monkeypatch, capsys) -> None:
    async def fake_collect_gate_report(*, pipeline_id=None, udf_id="chart-classify-events"):
        return {
            "namespace": "chart-notes",
            "gates": {"phase6_complete": True},
            "status": {},
        }

    monkeypatch.setattr(gates, "collect_gate_report", fake_collect_gate_report)
    monkeypatch.setattr("sys.argv", ["gates", "--require-complete"])

    gates.main()

    assert '"phase6_complete": true' in capsys.readouterr().out


def test_gate_report_cli_writes_phase6_audit_report(monkeypatch, tmp_path, capsys) -> None:
    async def fake_collect_gate_report(*, pipeline_id=None, udf_id="chart-classify-events"):
        return {
            "namespace": "chart-notes",
            "gates": {"phase6_complete": True},
            "status": {},
        }

    out = tmp_path / "reports" / "phase6-gate-report.json"
    monkeypatch.setattr(gates, "collect_gate_report", fake_collect_gate_report)
    monkeypatch.setattr("sys.argv", ["gates", "--out", str(out)])

    gates.main()

    assert '"phase6_complete": true' in capsys.readouterr().out
    assert '"phase6_complete": true' in out.read_text()
