from types import SimpleNamespace

import pytest

from eval import holdout
from eval.holdout import edge, feature_edges, holdout_gate, overlap_report, qrel_edges


def test_holdout_edges_are_undirected_and_skip_self_edges() -> None:
    records = [
        SimpleNamespace(id="patient-1", similar_patient_ids=["patient-2", "patient-1"]),
        SimpleNamespace(id="patient-2", similar_patient_ids=["patient-1"]),
    ]

    assert edge("patient-2", "patient-1") == ("patient-1", "patient-2")
    assert feature_edges(records) == {("patient-1", "patient-2")}


def test_qrel_edges_and_overlap_report_count_leakage_examples() -> None:
    qrels = {
        "patient-1": {"patient-2": 2, "patient-3": 1},
        "patient-4": {"patient-4": 1},
    }
    report = overlap_report(
        feature={("patient-1", "patient-2"), ("patient-9", "patient-10")},
        qrels=qrel_edges(qrels),
        examples=1,
    )

    assert qrel_edges(qrels) == {("patient-1", "patient-2"), ("patient-1", "patient-3")}
    assert report == {
        "feature_edges": 2,
        "qrel_edges": 2,
        "overlap_edges": 1,
        "overlap_fraction_of_qrels": 0.5,
        "examples": [{"patient_a": "patient-1", "patient_b": "patient-2"}],
    }


def test_holdout_gate_accepts_or_rejects_overlap_threshold() -> None:
    report = {"feature_edges": 3, "qrel_edges": 2, "overlap_edges": 1}

    assert holdout_gate(report, max_overlap_edges=1) == {
        "gate": "recds_holdout_overlap",
        "checks": {
            "feature_edges_present": {"actual": 3, "ok": True},
            "qrel_edges_present": {"actual": 2, "ok": True},
            "max_overlap_edges": {"limit": 1, "actual": 1, "ok": True},
        },
        "accepted": True,
    }
    assert holdout_gate(report, max_overlap_edges=0) == {
        "gate": "recds_holdout_overlap",
        "checks": {
            "feature_edges_present": {"actual": 3, "ok": True},
            "qrel_edges_present": {"actual": 2, "ok": True},
            "max_overlap_edges": {"limit": 0, "actual": 1, "ok": False},
        },
        "accepted": False,
    }


def test_holdout_gate_rejects_empty_edge_sets_as_uninformative() -> None:
    report = {"feature_edges": 0, "qrel_edges": 0, "overlap_edges": 0}

    assert holdout_gate(report, max_overlap_edges=0) == {
        "gate": "recds_holdout_overlap",
        "checks": {
            "feature_edges_present": {"actual": 0, "ok": False},
            "qrel_edges_present": {"actual": 0, "ok": False},
            "max_overlap_edges": {"limit": 0, "actual": 0, "ok": True},
        },
        "accepted": False,
    }


def test_holdout_cli_exits_nonzero_when_overlap_exceeds_threshold(monkeypatch, capsys) -> None:
    monkeypatch.setattr(holdout, "Settings", lambda: SimpleNamespace())
    monkeypatch.setattr(
        holdout,
        "load_recds",
        lambda task, settings, split: ([], {"patient-1": {"patient-2": 2}}),
    )
    monkeypatch.setattr(
        holdout,
        "load_notes",
        lambda settings, limit=None: iter(
            [SimpleNamespace(id="patient-1", similar_patient_ids=["patient-2"])]
        ),
    )
    monkeypatch.setattr("sys.argv", ["holdout", "--max-overlap-edges", "0"])

    with pytest.raises(SystemExit) as exc:
        holdout.main()

    assert exc.value.code == 1
    assert '"accepted": false' in capsys.readouterr().out


def test_holdout_cli_rejects_negative_threshold_before_loading_data(monkeypatch) -> None:
    monkeypatch.setattr(holdout, "Settings", lambda: pytest.fail("Settings should not load for invalid args"))
    monkeypatch.setattr("sys.argv", ["holdout", "--max-overlap-edges", "-1"])

    with pytest.raises(SystemExit) as exc:
        holdout.main()

    assert exc.value.code == 2


def test_holdout_cli_exits_zero_when_overlap_is_within_threshold(monkeypatch, capsys) -> None:
    monkeypatch.setattr(holdout, "Settings", lambda: SimpleNamespace())
    monkeypatch.setattr(
        holdout,
        "load_recds",
        lambda task, settings, split: ([], {"patient-1": {"patient-2": 2}}),
    )
    monkeypatch.setattr(
        holdout,
        "load_notes",
        lambda settings, limit=None: iter(
            [SimpleNamespace(id="patient-9", similar_patient_ids=["patient-10"])]
        ),
    )
    monkeypatch.setattr("sys.argv", ["holdout", "--max-overlap-edges", "0"])

    holdout.main()

    assert '"accepted": true' in capsys.readouterr().out


def test_holdout_cli_writes_phase5_audit_report(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(holdout, "Settings", lambda: SimpleNamespace())
    monkeypatch.setattr(
        holdout,
        "load_recds",
        lambda task, settings, split: ([], {"patient-1": {"patient-2": 2}}),
    )
    monkeypatch.setattr(
        holdout,
        "load_notes",
        lambda settings, limit=None: iter(
            [SimpleNamespace(id="patient-9", similar_patient_ids=["patient-10"])]
        ),
    )
    out = tmp_path / "reports" / "holdout-report.json"
    monkeypatch.setattr("sys.argv", ["holdout", "--max-overlap-edges", "0", "--out", str(out)])

    holdout.main()

    assert '"accepted": true' in capsys.readouterr().out
    text = out.read_text()
    assert '"gate"' in text
    assert '"recds_holdout_overlap"' in text
    assert '"max_overlap_edges"' in text


def test_holdout_cli_exits_nonzero_when_audit_has_no_feature_edges(monkeypatch, capsys) -> None:
    monkeypatch.setattr(holdout, "Settings", lambda: SimpleNamespace())
    monkeypatch.setattr(
        holdout,
        "load_recds",
        lambda task, settings, split: ([], {"patient-1": {"patient-2": 2}}),
    )
    monkeypatch.setattr(
        holdout,
        "load_notes",
        lambda settings, limit=None: iter([SimpleNamespace(id="patient-9", similar_patient_ids=[])]),
    )
    monkeypatch.setattr("sys.argv", ["holdout", "--max-overlap-edges", "0"])

    with pytest.raises(SystemExit) as exc:
        holdout.main()

    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert '"feature_edges_present"' in out
    assert '"accepted": false' in out
