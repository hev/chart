from types import SimpleNamespace

import pytest

from functions import measure_classify_events
from functions.measure_classify_events import budget_report, estimate_full_run, signal_report, writeback_report


def test_estimate_full_run_extrapolates_per_note_latency() -> None:
    assert estimate_full_run(notes=10, elapsed_seconds=25.0, full_notes=100) == {
        "notes": 10,
        "elapsed_seconds": 25.0,
        "per_note_seconds": 2.5,
        "full_notes": 100,
        "estimated_full_seconds": 250.0,
        "estimated_full_hours": 0.07,
    }


def test_estimate_full_run_rejects_empty_sample() -> None:
    with pytest.raises(ValueError, match="notes must be positive"):
        estimate_full_run(notes=0, elapsed_seconds=0.0)


def test_estimate_full_run_rejects_non_positive_elapsed_time() -> None:
    with pytest.raises(ValueError, match="elapsed_seconds must be positive"):
        estimate_full_run(notes=1, elapsed_seconds=0.0)


def test_estimate_full_run_extrapolates_gpu_cost_when_rate_is_supplied() -> None:
    assert estimate_full_run(notes=10, elapsed_seconds=25.0, full_notes=100, gpu_hourly_usd=2.50) == {
        "notes": 10,
        "elapsed_seconds": 25.0,
        "per_note_seconds": 2.5,
        "full_notes": 100,
        "estimated_full_seconds": 250.0,
        "estimated_full_hours": 0.07,
        "gpu_hourly_usd": 2.5,
        "estimated_full_usd": 0.18,
    }


def test_budget_report_accepts_estimate_under_limits() -> None:
    estimate = estimate_full_run(notes=10, elapsed_seconds=25.0, full_notes=100, gpu_hourly_usd=2.50)

    assert budget_report(estimate, max_full_hours=0.08, max_full_usd=0.20) == {
        "checks": {
            "max_full_hours": {"limit": 0.08, "actual": 0.07, "ok": True},
            "max_full_usd": {"limit": 0.20, "actual": 0.18, "ok": True},
        },
        "accepted": True,
    }


def test_budget_report_rejects_estimate_over_limits_or_missing_cost() -> None:
    estimate = estimate_full_run(notes=10, elapsed_seconds=25.0, full_notes=100)

    assert budget_report(estimate, max_full_hours=0.05, max_full_usd=0.20) == {
        "checks": {
            "max_full_hours": {"limit": 0.05, "actual": 0.07, "ok": False},
            "max_full_usd": {"limit": 0.20, "actual": None, "ok": False},
        },
        "accepted": False,
    }


def test_signal_report_accepts_required_discontinuation_count() -> None:
    assert signal_report(
        {"med_discontinuation": 2},
        min_med_discontinuations=1,
        examples=[{"note_preview": "a"}, {"note_preview": "b"}],
        min_review_examples=2,
    ) == {
        "checks": {
            "min_med_discontinuations": {"limit": 1, "actual": 2, "ok": True},
            "min_review_examples": {"limit": 2, "actual": 2, "ok": True},
        },
        "accepted": True,
    }


def test_signal_report_rejects_missing_discontinuation_signal() -> None:
    assert signal_report(
        {"med_discontinuation": 0},
        min_med_discontinuations=1,
        examples=[],
        min_review_examples=1,
    ) == {
        "checks": {
            "min_med_discontinuations": {"limit": 1, "actual": 0, "ok": False},
            "min_review_examples": {"limit": 1, "actual": 0, "ok": False},
        },
        "accepted": False,
    }


def test_writeback_report_documents_one_pass_multi_label_patch() -> None:
    assert writeback_report() == {
        "mode": "tpuf.patch_columns",
        "primary_output": "events",
        "model_passes_per_note": 1,
        "patched_fields": [
            "events",
            "has_med_discontinuation",
            "has_adverse_event",
            "diagnosis_category",
            "specialty",
            "discontinuation_reason",
        ],
        "settles_multi_write": True,
    }


def test_run_counts_digest_labels_without_loading_gpu(monkeypatch) -> None:
    load_notes_kwargs = []

    def fake_load_notes(settings, **kwargs):
        load_notes_kwargs.append(kwargs)
        return iter(
            [
                SimpleNamespace(text="routine follow up"),
                SimpleNamespace(text="metformin was discontinued due to adverse effect"),
            ]
        )

    monkeypatch.setattr(
        measure_classify_events,
        "load_notes",
        fake_load_notes,
    )
    monkeypatch.setattr(measure_classify_events.time, "perf_counter", iter([10.0, 15.0]).__next__)

    result = measure_classify_events.run(
        limit=1,
        discontinuation_only=True,
        max_chars=None,
        examples=1,
        gpu_hourly_usd=2.5,
        accelerator="gpu",
        digest_fn=lambda note: {
            "events": [
                {"type": "medication_discontinued", "reason": "adverse_effect"},
                {"type": "adverse_drug_reaction"},
            ],
            "diagnosis_category": "endocrine",
            "specialty": "endocrinology",
        },
    )

    assert result["runtime"] == {"accelerator": "gpu"}
    assert result["sample"] == {
        "notes": 1,
        "med_discontinuation": 1,
        "adverse_event": 1,
        "events": {"adverse_drug_reaction": 1, "medication_discontinued": 1},
    }
    assert load_notes_kwargs == [{"include_similar_patient_ids": False}]
    assert result["estimate"]["per_note_seconds"] == 5.0
    assert result["estimate"]["estimated_full_usd"] == 579.85
    assert result["writeback"]["mode"] == "tpuf.patch_columns"
    assert result["writeback"]["model_passes_per_note"] == 1
    assert result["examples"] == [
        {
            "note_preview": "metformin was discontinued due to adverse effect",
            "events": [
                {"type": "medication_discontinued", "reason": "adverse_effect"},
                {"type": "adverse_drug_reaction"},
            ],
            "labels": {
                "events": ["adverse_drug_reaction", "medication_discontinued"],
                "has_med_discontinuation": True,
                "has_adverse_event": True,
                "diagnosis_category": "endocrine",
                "specialty": "endocrinology",
            },
            "discontinuation_reason": "adverse_effect",
        }
    ]


def test_cli_exits_nonzero_when_classifier_budget_is_rejected(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        measure_classify_events,
        "run",
        lambda **kwargs: {
            "sample": {"notes": 1},
            "estimate": {"estimated_full_hours": 3.0, "estimated_full_usd": 9.0},
            "examples": [],
        },
    )
    monkeypatch.setattr("sys.argv", ["measure", "--max-full-hours", "2.0", "--max-full-usd", "5.0"])

    with pytest.raises(SystemExit) as exc:
        measure_classify_events.main()

    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert '"accepted": false' in out
    assert '"max_full_hours"' in out


def test_cli_writes_rejected_classifier_report_before_exiting(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        measure_classify_events,
        "run",
        lambda **kwargs: {
            "sample": {"notes": 1},
            "estimate": {"estimated_full_hours": 3.0, "estimated_full_usd": 9.0},
            "examples": [],
        },
    )
    out = tmp_path / "reports" / "classify-events-budget.json"
    monkeypatch.setattr(
        "sys.argv",
        ["measure", "--max-full-hours", "2.0", "--max-full-usd", "5.0", "--out", str(out)],
    )

    with pytest.raises(SystemExit) as exc:
        measure_classify_events.main()

    assert exc.value.code == 1
    assert '"accepted": false' in out.read_text()


def test_cli_exits_zero_when_classifier_budget_is_accepted(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        measure_classify_events,
        "run",
        lambda **kwargs: {
            "sample": {"notes": 1},
            "estimate": {"estimated_full_hours": 1.0, "estimated_full_usd": 2.0},
            "examples": [],
        },
    )
    monkeypatch.setattr("sys.argv", ["measure", "--max-full-hours", "2.0", "--max-full-usd", "5.0"])

    measure_classify_events.main()

    assert '"accepted": true' in capsys.readouterr().out


def test_cli_writes_classifier_report_for_phase4_audit(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setenv("CHART_GPU_DEVICE_NAME", "NVIDIA L4")
    monkeypatch.setattr(
        measure_classify_events,
        "run",
        lambda **kwargs: {
            "sample": {"notes": 1, "med_discontinuation": 1},
            "estimate": {"estimated_full_hours": 1.0, "estimated_full_usd": 2.0},
            "examples": [{"note_preview": "metformin stopped"}],
        },
    )
    out = tmp_path / "reports" / "classify-events-budget.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "measure",
            "--accelerator",
            "gpu",
            "--max-full-hours",
            "2.0",
            "--max-full-usd",
            "5.0",
            "--min-med-discontinuations",
            "1",
            "--min-review-examples",
            "1",
            "--out",
            str(out),
        ],
    )

    measure_classify_events.main()

    assert '"accepted": true' in capsys.readouterr().out
    text = out.read_text()
    assert '"budget"' in text
    assert '"signal"' in text
    assert '"accelerator": "gpu"' in text
    assert '"gpu_device": "NVIDIA L4"' in text
    assert '"max_full_hours"' in text
    assert '"max_full_usd"' in text
    assert '"min_med_discontinuations"' in text
    assert '"min_review_examples"' in text
    assert '"writeback"' in text
    assert '"tpuf.patch_columns"' in text


def test_cli_exits_nonzero_when_required_signal_is_missing(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        measure_classify_events,
        "run",
        lambda **kwargs: {
            "sample": {"notes": 1, "med_discontinuation": 0},
            "estimate": {"estimated_full_hours": 1.0},
            "examples": [],
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        ["measure", "--min-med-discontinuations", "1", "--min-review-examples", "1"],
    )

    with pytest.raises(SystemExit) as exc:
        measure_classify_events.main()

    assert exc.value.code == 1
    assert '"signal"' in capsys.readouterr().out


def test_cli_exits_cleanly_when_vllm_runtime_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        measure_classify_events,
        "run",
        lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("vLLM is required for the Gemma classifier. Run this command on a GPU.")
        ),
    )
    monkeypatch.setattr("sys.argv", ["measure", "--limit", "1"])

    with pytest.raises(SystemExit, match="vLLM is required for the Gemma classifier"):
        measure_classify_events.main()


def test_cli_writes_failed_classifier_report_when_vllm_runtime_is_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CHART_GPU_DEVICE_NAME", "NVIDIA L4")
    monkeypatch.setattr(
        measure_classify_events,
        "run",
        lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("vLLM is required for the Gemma classifier. Run this command on a GPU.")
        ),
    )
    out = tmp_path / "reports" / "classify-events-budget.json"
    monkeypatch.setattr("sys.argv", ["measure", "--accelerator", "gpu", "--limit", "1", "--out", str(out)])

    with pytest.raises(SystemExit, match="vLLM is required for the Gemma classifier"):
        measure_classify_events.main()

    text = out.read_text()
    assert '"status": "failed"' in text
    assert '"vLLM is required for the Gemma classifier' in text
    assert '"accelerator": "gpu"' in text
    assert '"gpu_device": "NVIDIA L4"' in text
    assert '"writeback"' in text


def test_cli_fails_before_classifying_when_gpu_device_is_unverified(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.delenv("CHART_GPU_DEVICE_NAME", raising=False)
    monkeypatch.setattr(
        "chart_common.runtime.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError()),
    )
    monkeypatch.setattr(
        measure_classify_events,
        "run",
        lambda **kwargs: pytest.fail("run should not start without GPU proof"),
    )
    out = tmp_path / "reports" / "classify-events-budget.json"
    monkeypatch.setattr("sys.argv", ["measure", "--accelerator", "gpu", "--out", str(out)])

    with pytest.raises(SystemExit) as exc:
        measure_classify_events.main()

    assert exc.value.code == 1
    stdout = capsys.readouterr().out
    assert "GPU accelerator was requested but no GPU device could be verified" in stdout
    text = out.read_text()
    assert '"status": "failed"' in text
    assert '"gpu_probe": "missing"' in text
    assert '"writeback"' in text


def test_cli_exits_zero_when_required_signal_and_examples_are_present(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        measure_classify_events,
        "run",
        lambda **kwargs: {
            "sample": {"notes": 2, "med_discontinuation": 1},
            "estimate": {"estimated_full_hours": 1.0},
            "examples": [{"note_preview": "metformin stopped"}],
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        ["measure", "--min-med-discontinuations", "1", "--min-review-examples", "1"],
    )

    measure_classify_events.main()

    out = capsys.readouterr().out
    assert '"accepted": true' in out
    assert '"min_review_examples"' in out


def test_cli_rejects_negative_budget_inputs_before_running(monkeypatch) -> None:
    monkeypatch.setattr(
        measure_classify_events,
        "run",
        lambda **kwargs: pytest.fail("run should not be called for invalid args"),
    )
    monkeypatch.setattr("sys.argv", ["measure", "--max-full-usd", "-1"])

    with pytest.raises(SystemExit) as exc:
        measure_classify_events.main()

    assert exc.value.code == 2
