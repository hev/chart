"""classify-events — a Gemma clinical-event cascade over note text (the GPU showcase).

THE NEW THING this demo shows beyond routing: an open-weight LLM cascade on the
GPU pool that reads each note once and pulls out **clinical events** — with
**medication discontinuation** as the headline event (a real pharmacovigilance /
cohort signal a clinical customer wants). Because we are paying for a GPU pass, we
classify many things in that one pass, not one — the RFC 0072 cascade discipline,
specialized to clinical notes and built here for the first time (moment designed
the cascade; its deployed enrichment is still rule-based — see ../moment).

Shape (RFC 0072): open weights only, vLLM **guided decoding** to a fixed schema,
A10G-fit, KEDA scale-to-zero. Three tiers, one expensive model pass:

  Tier 1 — DIGEST (the GPU pass). One Gemma call per note → a structured digest:
           a short summary + a list of typed events (type, drug, reason).
  Tier 2 — DERIVED LABELS (no extra GPU). Read straight from the digest into
           denormalized, filterable attributes: `events` (the type list),
           `has_med_discontinuation`, `has_adverse_event`, and the facet labels
           `diagnosis_category` / `specialty` — so this one pass SUBSUMES
           tag_specialty + extract_clinical_fields (one GPU pass, many labels).
  Tier 3 — GATED REFINE. Only when Tier 1 flagged a discontinuation: a second
           targeted pass for the structured reason (adverse-effect / ineffective /
           resolved / patient-choice / cost). Expensive, so gated to where it pays.

The digest prose is authored once per note; the derived labels are denormalized
onto the row (RFC 0072) so the namespace stays single-granularity and the events
become facets/filters that compose with the routing headline ("medication
discontinued due to an adverse reaction" = an `events` filter over routed search).
"""

from __future__ import annotations

import asyncio
import argparse
import json
import os
from functools import lru_cache
from typing import Any

from chart_common.config import Settings
from hevlayer.udf import run_udf_worker, udf

# The clinical-event taxonomy the guided decoder is constrained to. Closed set so
# `events` is a stable facet and the digest can't invent labels. Medication
# discontinuation is first because it is the headline.
EVENT_TYPES = [
    "medication_discontinued",      # the headline event
    "medication_started",
    "dose_changed",
    "adverse_drug_reaction",
    "allergy_noted",
    "procedure_performed",
    "diagnosis_made",
    "disease_progression",
    "remission_or_improvement",
    "hospital_admission",
    "discharge",
    "death",
]

WRITEBACK_FIELDS = [
    "events",
    "has_med_discontinuation",
    "has_adverse_event",
    "diagnosis_category",
    "specialty",
    "discontinuation_reason",
]

DISCONTINUATION_REASONS = [
    "adverse_effect", "ineffective", "condition_resolved", "patient_choice",
    "cost", "interaction", "unspecified",
]

# Open-weight model on the GPU pool (RFC 0072). Gemma, A10G-fit, vLLM-served.
MODEL = os.environ.get("CHART_EVENTS_MODEL", "google/gemma-2-9b-it")

# JSON schema the digest is guided-decoded to (vLLM guided_json / outlines). The
# closed enums are what make the output a stable facet rather than free text.
DIGEST_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": EVENT_TYPES},
                    "drug": {"type": "string"},
                    "reason": {"type": "string", "enum": DISCONTINUATION_REASONS},
                },
                "required": ["type"],
            },
        },
        "diagnosis_category": {"type": "string"},
        "specialty": {"type": "string"},
    },
    "required": ["events"],
}

_DIGEST_PROMPT = (
    "You are a clinical information extractor. Read the patient note and return the "
    "structured digest. List every clinical event; for medication_discontinued, set "
    "`drug` and, if stated, `reason`. Do not infer events that are not described.\n\nNote:\n{note}"
)


@lru_cache(maxsize=1)
def _engine():
    """Lazy vLLM engine — loaded once per worker and held resident across the
    batch (RFC 0068 setup/lifecycle, so a model-loading pod doesn't count as
    serving capacity). Stubbed import so the module is inspectable without a GPU."""
    try:
        from vllm import LLM  # noqa: F401  (GPU-only; in the `classifier` extra)
    except ImportError as exc:
        raise RuntimeError(
            "vLLM is required for the Gemma classifier. Run this command in the "
            "Linux GPU classifier image or another GPU environment with the "
            "`classifier` extra installed."
        ) from exc

    return LLM(model=MODEL)


def digest(note_text: str) -> dict:
    """Tier 1 — the one GPU pass. Guided-decode the note to DIGEST_SCHEMA.

    vLLM changed the Python structured-output API across releases: newer versions
    use `structured_outputs`, older versions use `guided_decoding`. Support both
    so the Function image can move independently of this repo.
    """
    if not note_text:
        return {"events": [], "summary": ""}
    prompt = _DIGEST_PROMPT.format(note=note_text)
    out = _engine().generate([prompt], _sampling_params())
    return _parse_digest(out[0].outputs[0].text)


def derive_labels(d: dict) -> dict:
    """Tier 2 — pure-python reads from the digest. No extra GPU. The denormalized,
    filterable label set that goes on the row."""
    events = sorted({e["type"] for e in _event_items(d) if e.get("type") in EVENT_TYPES})
    return {
        "events": events,
        "has_med_discontinuation": "medication_discontinued" in events,
        "has_adverse_event": "adverse_drug_reaction" in events,
        # One pass, many labels: these subsume tag_specialty / extract_clinical_fields.
        "diagnosis_category": d.get("diagnosis_category") or "other",
        "specialty": d.get("specialty") or "other",
    }


def discontinuation_reason(d: dict) -> str | None:
    return next(
        (
            e.get("reason")
            for e in _event_items(d)
            if e.get("type") == "medication_discontinued" and e.get("reason") in DISCONTINUATION_REASONS
        ),
        None,
    )


def _event_items(d: dict) -> list[dict]:
    events = d.get("events", [])
    if not isinstance(events, list):
        return []
    return [event for event in events if isinstance(event, dict)]


def _clean_label(value: Any, *, default: str = "other") -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def refine_discontinuation(note_text: str) -> dict:
    """Tier 3 — gated. Only call when Tier 1 found medication_discontinued. A
    targeted pass for the structured reason."""
    d = digest(note_text)
    return {"discontinuation_reason": discontinuation_reason(d) or "unspecified"}


def _sampling_params():
    from vllm import SamplingParams

    kwargs = {"temperature": 0.0, "max_tokens": 512}
    try:
        from vllm.sampling_params import StructuredOutputsParams

        return SamplingParams(
            **kwargs,
            structured_outputs=StructuredOutputsParams(json=DIGEST_SCHEMA),
        )
    except (ImportError, TypeError):
        from vllm.sampling_params import GuidedDecodingParams

        return SamplingParams(
            **kwargs,
            guided_decoding=GuidedDecodingParams(json=DIGEST_SCHEMA),
        )


def _parse_digest(text: str) -> dict:
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("digest must be a JSON object")
    events = []
    for event in _event_items(parsed):
        event_type = event.get("type")
        if event_type not in EVENT_TYPES:
            continue
        normalized = {"type": event_type}
        drug = event.get("drug")
        if isinstance(drug, str) and drug.strip():
            normalized["drug"] = drug.strip()
        reason = event.get("reason")
        if reason in DISCONTINUATION_REASONS:
            normalized["reason"] = reason
        events.append(normalized)
    return {
        "summary": _clean_label(parsed.get("summary"), default=""),
        "events": events,
        "diagnosis_category": _clean_label(parsed.get("diagnosis_category")),
        "specialty": _clean_label(parsed.get("specialty")),
    }


# --- The UDF entrypoint (moment's @udf + run_udf_worker wiring). -------------
# Primary output is `events` ([]string) on the documented single-output sugar.
# The full Tier-2 label set is patched through the injected `tpuf` client, settling
# RFC 0076's multi-write question as one cascade Function: one model pass, many
# denormalized labels.
@udf(id="chart-classify-events", inputs=["id", "text"], output="events", batch_size=16)
async def classify_events_udf(
    id: object = "", text: object = "", tpuf: Any | None = None
) -> list[str]:
    note = "" if text is None else str(text)
    d = digest(note)
    labels = derive_labels(d)
    if tpuf is not None and id:
        attrs = {
            "events": [labels["events"]],
            "has_med_discontinuation": [labels["has_med_discontinuation"]],
            "has_adverse_event": [labels["has_adverse_event"]],
            "diagnosis_category": [labels["diagnosis_category"]],
            "specialty": [labels["specialty"]],
            "discontinuation_reason": [
                discontinuation_reason(d) if labels["has_med_discontinuation"] else None
            ],
        }
        await tpuf.patch_columns(Settings().namespace, [str(id)], attrs)
    return labels["events"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the chart clinical-event classifier UDF worker")
    parser.add_argument("--once", action="store_true", help="claim one batch and exit")
    args = parser.parse_args()
    if not Settings().api_key:
        raise SystemExit(
            "No gateway key. Set LAYER_GATEWAY_API_KEY in .env — it's the upstream "
            "Turbopuffer key (1Password: layer-turbopuffer / mesh-staging)."
        )
    once = args.once or os.environ.get("CHART_UDF_ONCE", "").strip().lower() in {"1", "true", "yes"}
    asyncio.run(run_udf_worker(classify_events_udf, udf_id="chart-classify-events", once=once))


if __name__ == "__main__":
    main()
