import asyncio
import sys
from types import ModuleType

import pytest

from functions import classify_events
from functions.classify_events import derive_labels


def test_derive_labels_extracts_closed_event_flags() -> None:
    labels = derive_labels(
        {
            "events": [
                {"type": "medication_discontinued", "drug": "atorvastatin"},
                {"type": "adverse_drug_reaction"},
                {"type": "medication_discontinued", "drug": "statin"},
                {"ignored": "missing type"},
            ],
            "diagnosis_category": "cardiology",
            "specialty": "internal_medicine",
        }
    )

    assert labels == {
        "events": ["adverse_drug_reaction", "medication_discontinued"],
        "has_med_discontinuation": True,
        "has_adverse_event": True,
        "diagnosis_category": "cardiology",
        "specialty": "internal_medicine",
    }


def test_derive_labels_defaults_missing_facets() -> None:
    labels = derive_labels({"events": []})

    assert labels["events"] == []
    assert labels["has_med_discontinuation"] is False
    assert labels["has_adverse_event"] is False
    assert labels["diagnosis_category"] == "other"
    assert labels["specialty"] == "other"


def test_derive_labels_discards_events_outside_closed_taxonomy() -> None:
    labels = derive_labels(
        {
            "events": [
                {"type": "medication_started"},
                {"type": "invented_event"},
            ]
        }
    )

    assert labels["events"] == ["medication_started"]
    assert labels["has_med_discontinuation"] is False
    assert labels["has_adverse_event"] is False


def test_derive_labels_ignores_malformed_event_payloads() -> None:
    labels = derive_labels({"events": ["bad", {"type": "adverse_drug_reaction"}, None]})

    assert labels["events"] == ["adverse_drug_reaction"]
    assert labels["has_adverse_event"] is True

    missing = derive_labels({"events": "not-a-list"})
    assert missing["events"] == []
    assert missing["has_med_discontinuation"] is False


def test_refine_discontinuation_returns_digest_reason(monkeypatch) -> None:
    monkeypatch.setattr(
        classify_events,
        "digest",
        lambda note: {
            "events": [
                {
                    "type": "medication_discontinued",
                    "drug": "metformin",
                    "reason": "adverse_effect",
                }
            ]
        },
    )

    assert classify_events.refine_discontinuation("note") == {
        "discontinuation_reason": "adverse_effect"
    }


def test_refine_discontinuation_defaults_invalid_reason(monkeypatch) -> None:
    monkeypatch.setattr(
        classify_events,
        "digest",
        lambda note: {
            "events": [
                {
                    "type": "medication_discontinued",
                    "drug": "metformin",
                    "reason": "not_in_taxonomy",
                }
            ]
        },
    )

    assert classify_events.refine_discontinuation("note") == {
        "discontinuation_reason": "unspecified"
    }


def test_discontinuation_reason_ignores_malformed_event_payloads() -> None:
    assert classify_events.discontinuation_reason({"events": ["bad", None]}) is None
    assert classify_events.discontinuation_reason({"events": "not-a-list"}) is None


def test_parse_digest_supplies_required_defaults() -> None:
    assert classify_events._parse_digest('{"events": []}') == {
        "events": [],
        "summary": "",
        "diagnosis_category": "other",
        "specialty": "other",
    }


def test_parse_digest_normalizes_model_output_to_closed_schema() -> None:
    parsed = classify_events._parse_digest(
        """
        {
          "summary": "  stopped statin  ",
          "events": [
            {"type": "medication_discontinued", "drug": " atorvastatin ", "reason": "adverse_effect"},
            {"type": "invented_event", "drug": "ignored", "reason": "not_in_taxonomy"},
            {"type": "adverse_drug_reaction", "reason": "not_in_taxonomy"},
            "bad"
          ],
          "diagnosis_category": " cardiovascular ",
          "specialty": " cardiology "
        }
        """
    )

    assert parsed == {
        "summary": "stopped statin",
        "events": [
            {
                "type": "medication_discontinued",
                "drug": "atorvastatin",
                "reason": "adverse_effect",
            },
            {"type": "adverse_drug_reaction"},
        ],
        "diagnosis_category": "cardiovascular",
        "specialty": "cardiology",
    }


def test_parse_digest_rejects_non_object_json() -> None:
    try:
        classify_events._parse_digest("[]")
    except ValueError as exc:
        assert "digest must be a JSON object" in str(exc)
    else:
        raise AssertionError("expected non-object digest to fail")


def test_sampling_params_uses_structured_outputs_when_supported(monkeypatch) -> None:
    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class StructuredOutputsParams:
        def __init__(self, *, json):
            self.json = json

    vllm = ModuleType("vllm")
    vllm.SamplingParams = SamplingParams
    sampling_params = ModuleType("vllm.sampling_params")
    sampling_params.StructuredOutputsParams = StructuredOutputsParams
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling_params)

    params = classify_events._sampling_params()

    assert params.kwargs["temperature"] == 0.0
    assert params.kwargs["max_tokens"] == 512
    assert params.kwargs["structured_outputs"].json == classify_events.DIGEST_SCHEMA


def test_engine_reports_clear_error_when_vllm_is_unavailable(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "vllm", raising=False)
    classify_events._engine.cache_clear()

    with pytest.raises(RuntimeError, match="vLLM is required for the Gemma classifier"):
        classify_events._engine()

    classify_events._engine.cache_clear()


def test_sampling_params_falls_back_to_guided_decoding(monkeypatch) -> None:
    class SamplingParams:
        def __init__(self, **kwargs):
            if "structured_outputs" in kwargs:
                raise TypeError("old vLLM")
            self.kwargs = kwargs

    class StructuredOutputsParams:
        def __init__(self, *, json):
            self.json = json

    class GuidedDecodingParams:
        def __init__(self, *, json):
            self.json = json

    vllm = ModuleType("vllm")
    vllm.SamplingParams = SamplingParams
    sampling_params = ModuleType("vllm.sampling_params")
    sampling_params.StructuredOutputsParams = StructuredOutputsParams
    sampling_params.GuidedDecodingParams = GuidedDecodingParams
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling_params)

    params = classify_events._sampling_params()

    assert "structured_outputs" not in params.kwargs
    assert params.kwargs["guided_decoding"].json == classify_events.DIGEST_SCHEMA


class FakeTpuf:
    def __init__(self) -> None:
        self.calls = []

    async def patch_columns(self, namespace, ids, attrs):
        self.calls.append((namespace, ids, attrs))


def test_classify_events_udf_patches_derived_labels(monkeypatch) -> None:
    monkeypatch.setattr(
        classify_events,
        "digest",
        lambda note: {
            "events": [
                {
                    "type": "medication_discontinued",
                    "drug": "atorvastatin",
                    "reason": "adverse_effect",
                }
            ],
            "diagnosis_category": "cardiovascular",
            "specialty": "cardiology",
        },
    )
    tpuf = FakeTpuf()

    events = asyncio.run(
        classify_events.classify_events_udf(id="patient-1", text="note", tpuf=tpuf)
    )

    assert events == ["medication_discontinued"]
    assert tpuf.calls == [
        (
            "chart-notes",
            ["patient-1"],
            {
                "events": [["medication_discontinued"]],
                "has_med_discontinuation": [True],
                "has_adverse_event": [False],
                "diagnosis_category": ["cardiovascular"],
                "specialty": ["cardiology"],
                "discontinuation_reason": ["adverse_effect"],
            },
        )
    ]


def test_classify_events_udf_patches_events_without_discontinuation_reason(monkeypatch) -> None:
    monkeypatch.setattr(
        classify_events,
        "digest",
        lambda note: {
            "events": [{"type": "adverse_drug_reaction", "reason": "adverse_effect"}],
            "diagnosis_category": "allergy",
            "specialty": "immunology",
        },
    )
    tpuf = FakeTpuf()

    events = asyncio.run(
        classify_events.classify_events_udf(id="patient-2", text="note", tpuf=tpuf)
    )

    assert events == ["adverse_drug_reaction"]
    assert tpuf.calls == [
        (
            "chart-notes",
            ["patient-2"],
            {
                "events": [["adverse_drug_reaction"]],
                "has_med_discontinuation": [False],
                "has_adverse_event": [True],
                "diagnosis_category": ["allergy"],
                "specialty": ["immunology"],
                "discontinuation_reason": [None],
            },
        )
    ]


def test_classify_events_main_supports_help_without_starting_worker(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        classify_events,
        "run_udf_worker",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("worker should not start")),
    )
    monkeypatch.setattr("sys.argv", ["classify_events", "--help"])

    try:
        classify_events.main()
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("--help should exit")

    assert "Run the chart clinical-event classifier UDF worker" in capsys.readouterr().out


def test_classify_events_main_passes_once_from_cli(monkeypatch) -> None:
    captured = {}

    async def fake_run_udf_worker(udf_fn, *, udf_id, once):
        captured["udf_fn"] = udf_fn
        captured["udf_id"] = udf_id
        captured["once"] = once

    monkeypatch.setattr(classify_events, "Settings", lambda: type("Settings", (), {"api_key": "key"})())
    monkeypatch.setattr(classify_events, "run_udf_worker", fake_run_udf_worker)
    monkeypatch.setattr("sys.argv", ["classify_events", "--once"])

    classify_events.main()

    assert captured == {
        "udf_fn": classify_events.classify_events_udf,
        "udf_id": "chart-classify-events",
        "once": True,
    }


def test_classify_events_main_exits_before_worker_without_gateway_key(monkeypatch) -> None:
    monkeypatch.setattr(classify_events, "Settings", lambda: type("Settings", (), {"api_key": None})())
    monkeypatch.setattr(
        classify_events,
        "run_udf_worker",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("worker should not start")),
    )
    monkeypatch.setattr("sys.argv", ["classify_events", "--once"])

    with pytest.raises(SystemExit, match="No gateway key"):
        classify_events.main()
