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
    # The default loop is now the batched claim/complete worker; --per-item
    # selects the client's per-item run_udf_worker path this test exercises.
    monkeypatch.setattr("sys.argv", ["classify_events", "--once", "--per-item"])

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


def test_digest_batch_one_generate_call_and_parse_fallback(monkeypatch) -> None:
    """The whole batch goes through ONE engine.generate() (vLLM continuous
    batching), empty notes are skipped without a GPU pass, and an unparseable
    output degrades to an empty digest instead of failing the batch."""
    calls = []

    class FakeOut:
        def __init__(self, text):
            self.outputs = [type("O", (), {"text": text})()]

    class FakeEngine:
        def generate(self, prompts, params):
            calls.append(list(prompts))
            texts = [
                '{"events": [{"type": "medication_discontinued", "drug": "metformin", "reason": "adverse_effect"}], "summary": "s", "diagnosis_category": "endocrine", "specialty": "endocrinology"}',
                "not json",
            ]
            return [FakeOut(t) for t in texts[: len(prompts)]]

    monkeypatch.setattr(classify_events, "_engine", lambda: FakeEngine())
    monkeypatch.setattr(classify_events, "_sampling_params", lambda: None)

    digests = classify_events.digest_batch(["note one", "", "note three"])

    assert len(calls) == 1, "expected exactly one batched generate() call"
    assert len(calls[0]) == 2, "empty note must not consume a GPU slot"
    assert digests[0]["events"][0]["type"] == "medication_discontinued"
    assert digests[1] == {"events": [], "summary": "", "diagnosis_category": "other", "specialty": "other"}
    assert digests[2]["events"] == [], "parse failure degrades to empty digest"


def test_batch_writeback_columns_aligned() -> None:
    """Tier-2 labels for a claim come out as aligned column arrays — the
    patch_columns multi-row shape (one write for N rows)."""
    digests = [
        {"events": [{"type": "medication_discontinued", "reason": "cost"}], "diagnosis_category": "cardiology", "specialty": "cardiology"},
        {"events": [], "diagnosis_category": "other", "specialty": "other"},
    ]
    columns = classify_events._batch_writeback_columns(digests)
    assert set(columns) == set(classify_events.WRITEBACK_FIELDS)
    assert all(len(v) == 2 for v in columns.values())
    assert columns["events"][0] == ["medication_discontinued"]
    assert columns["has_med_discontinuation"] == [True, False]
    assert columns["discontinuation_reason"] == ["cost", None]


def _v2_event(event_type: str, **extra):
    event = {
        "type": event_type,
        "polarity": "affirmed",
        "span_quote": f"{event_type} quote",
        "confidence": 0.9,
    }
    event.update(extra)
    return event


def test_derive_labels_v2_facets_all_balanced_families() -> None:
    events = [
        _v2_event(event_type, drug_or_treatment="metformin" if event_type == "medication_stopped" else "")
        for event_type in classify_events.EVENT_TYPES_V2
    ]

    labels = classify_events.derive_labels_v2({"events_v2": events})

    assert labels["events_v2"] == sorted(classify_events.EVENT_TYPES_V2)
    assert labels["event_groups_v2"] == sorted(set(classify_events.EVENT_GROUPS_V2.values()))
    assert labels["event_confidence_v2"]["medication_stopped"] == 0.9
    assert labels["event_spans_v2"]["diagnostic_workup"] == ["diagnostic_workup quote"]


def test_derive_labels_v2_negative_med_stop_cases_do_not_facet() -> None:
    labels = classify_events.derive_labels_v2(
        {
            "events_v2": [
                _v2_event("medication_stopped", span_quote="vomiting stopped"),
                _v2_event(
                    "medication_stopped",
                    polarity="negated",
                    span_quote="no medications discontinued",
                    drug_or_treatment="medications",
                ),
                _v2_event("medication_stopped", span_quote="interrupted sutures"),
            ]
        }
    )

    assert labels["events_v2"] == []
    assert labels["has_treatment_change_v2"] is False


def test_derive_labels_v2_requires_affirmed_polarity_and_span_quote() -> None:
    labels = classify_events.derive_labels_v2(
        {
            "events_v2": [
                _v2_event("care_transition", polarity="negated", span_quote="not admitted"),
                _v2_event("diagnostic_workup", polarity="historical_or_routine", span_quote="prior CT showed"),
                _v2_event("adverse_drug_reaction", span_quote=""),
                _v2_event("medication_stopped", drug_or_treatment="statin", span_quote="statin was stopped"),
            ]
        }
    )

    assert labels["events_v2"] == ["medication_stopped"]
    assert labels["has_treatment_change_v2"] is True
    assert labels["has_care_transition_v2"] is False
    assert labels["has_complication_v2"] is False


def test_derive_labels_v2_grouped_booleans() -> None:
    labels = classify_events.derive_labels_v2(
        {
            "events_v2": [
                _v2_event("medication_started"),
                _v2_event("treatment_failure_or_recurrence"),
                _v2_event("procedure_complication"),
                _v2_event("care_transition"),
            ]
        }
    )

    assert labels["has_treatment_change_v2"] is True
    assert labels["has_treatment_response_v2"] is True
    assert labels["has_complication_v2"] is True
    assert labels["has_care_transition_v2"] is True


def test_parse_digest_normalizes_v2_events() -> None:
    parsed = classify_events._parse_digest(
        """
        {
          "events": [],
          "events_v2": [
            {
              "type": "medication_stopped",
              "polarity": "affirmed",
              "span_quote": " statin was stopped ",
              "drug_or_treatment": " statin ",
              "confidence": 0.82
            },
            {"type": "invented", "polarity": "affirmed", "span_quote": "bad", "confidence": 1.0}
          ]
        }
        """
    )

    assert parsed["events_v2"] == [
        {
            "type": "medication_stopped",
            "polarity": "affirmed",
            "span_quote": "statin was stopped",
            "confidence": 0.82,
            "drug_or_treatment": "statin",
        }
    ]


def test_prepass_guards_mark_candidate_spans_and_polarity_hints() -> None:
    guards = classify_events.prepass_guards(
        "No medications discontinued. The patient was admitted and CT showed pneumonia. "
        "Vomiting stopped after fluids; interrupted sutures were placed."
    )

    assert any(g["family_hint"] == "medication_stop" and g["polarity_hint"] == "negated" for g in guards)
    assert any(g["family_hint"] == "care_transition" for g in guards)
    assert any(g["family_hint"] == "diagnostic_workup" for g in guards)
    assert any("interrupted sutures" in g["span"] for g in guards)


def test_digest_prompt_includes_balanced_v2_contract_and_negative_examples() -> None:
    prompt = classify_events._digest_prompt("vomiting stopped; no medications discontinued; interrupted sutures")

    for event_type in classify_events.EVENT_TYPES_V2:
        assert event_type in prompt
    assert "medication_stopped" in prompt
    assert "medication_discontinued as the headline" not in prompt
    assert '"vomiting stopped"' in prompt
    assert '"no medications discontinued"' in prompt
    assert '"interrupted sutures"' in prompt
    assert "`span_quote` must be a short exact quote" in prompt


def test_classify_events_udf_patches_v2_fields_when_digest_contains_v2(monkeypatch) -> None:
    monkeypatch.setattr(
        classify_events,
        "digest",
        lambda note: {
            "events": [],
            "events_v2": [
                _v2_event("medication_stopped", drug_or_treatment="statin", span_quote="statin was stopped"),
                _v2_event("care_transition", span_quote="patient was admitted"),
            ],
            "diagnosis_category": "cardiovascular",
            "specialty": "cardiology",
        },
    )
    tpuf = FakeTpuf()

    asyncio.run(classify_events.classify_events_udf(id="patient-v2", text="note", tpuf=tpuf))

    attrs = tpuf.calls[0][2]
    assert attrs["events"] == [[]]
    assert attrs["events_v2"] == [["care_transition", "medication_stopped"]]
    assert attrs["event_groups_v2"] == [["care_transition", "treatment_change"]]
    assert attrs["has_treatment_change_v2"] == [True]
    assert attrs["has_care_transition_v2"] == [True]
