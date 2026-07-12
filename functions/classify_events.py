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

import argparse
import asyncio
import json
import os
import re
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

EVENT_TYPES_V2 = [
    "presentation_or_symptom_change",
    "diagnostic_workup",
    "diagnosis_established",
    "diagnosis_revised_or_ruled_out",
    "procedure_or_intervention",
    "procedure_complication",
    "medication_started",
    "medication_dose_changed",
    "medication_stopped",
    "treatment_response",
    "treatment_failure_or_recurrence",
    "adverse_drug_reaction",
    "non_drug_complication",
    "care_transition",
    "incidental_finding",
    "severe_outcome_or_death",
]

EVENT_POLARITIES_V2 = ["affirmed", "negated", "historical_or_routine"]

EVENT_GROUPS_V2 = {
    "presentation_or_symptom_change": "clinical_presentation",
    "diagnostic_workup": "diagnosis",
    "diagnosis_established": "diagnosis",
    "diagnosis_revised_or_ruled_out": "diagnosis",
    "procedure_or_intervention": "procedure",
    "procedure_complication": "complication",
    "medication_started": "treatment_change",
    "medication_dose_changed": "treatment_change",
    "medication_stopped": "treatment_change",
    "treatment_response": "treatment_response",
    "treatment_failure_or_recurrence": "treatment_response",
    "adverse_drug_reaction": "complication",
    "non_drug_complication": "complication",
    "care_transition": "care_transition",
    "incidental_finding": "diagnosis",
    "severe_outcome_or_death": "outcome",
}

MIN_EVENT_CONFIDENCE_V2 = {
    "medication_stopped": 0.75,
}
DEFAULT_MIN_EVENT_CONFIDENCE_V2 = 0.5

LEGACY_WRITEBACK_FIELDS = [
    "events",
    "has_med_discontinuation",
    "has_adverse_event",
    "diagnosis_category",
    "specialty",
    "discontinuation_reason",
]

V2_WRITEBACK_FIELDS = [
    "events_v2",
    "event_groups_v2",
    "event_confidence_v2",
    "event_spans_v2",
    "has_treatment_change_v2",
    "has_treatment_response_v2",
    "has_complication_v2",
    "has_care_transition_v2",
]

WRITEBACK_FIELDS = LEGACY_WRITEBACK_FIELDS + V2_WRITEBACK_FIELDS

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
        "events_v2": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": EVENT_TYPES_V2},
                    "polarity": {"type": "string", "enum": EVENT_POLARITIES_V2},
                    "span_quote": {"type": "string"},
                    "drug_or_treatment": {"type": "string"},
                    "procedure": {"type": "string"},
                    "diagnosis": {"type": "string"},
                    "temporal_context": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["type", "polarity", "span_quote", "confidence"],
            },
        },
        "diagnosis_category": {"type": "string"},
        "specialty": {"type": "string"},
    },
    # diagnosis_category / specialty are required so guided decoding always
    # emits them — optional fields get silently omitted and the parser's
    # "other" default made two facets useless.
    "required": ["events", "diagnosis_category", "specialty"],
}

_DIGEST_PROMPT = (
    "You are a clinical information extractor. Read the patient note and return the "
    "structured digest. Preserve the legacy `events` field for compatibility. Also "
    "emit `events_v2` using the balanced family taxonomy below. Do not infer events "
    "that are not described. "
    "Set `specialty` to the single medical specialty that would own this case "
    "(e.g. cardiology, oncology, neurology, infectious_disease, endocrinology, "
    "pediatrics, psychiatry, nephrology, gastroenterology, pulmonology) and "
    "`diagnosis_category` to the primary diagnosis area in the same lowercase_snake "
    "style. Use `other` only when genuinely unclassifiable.\n\n"
    "For every v2 event, set `type`, `polarity`, `span_quote`, and `confidence`. "
    "`span_quote` must be a short exact quote from the note. Use `affirmed` only for "
    "current case events, `negated` for denied or absent events, and "
    "`historical_or_routine` for past history, routine course completion, or "
    "background care. Families are symmetric siblings: presentation_or_symptom_change; "
    "diagnostic_workup; diagnosis_established; diagnosis_revised_or_ruled_out; "
    "procedure_or_intervention; procedure_complication; medication_started; "
    "medication_dose_changed; medication_stopped; treatment_response; "
    "treatment_failure_or_recurrence; adverse_drug_reaction; non_drug_complication; "
    "care_transition; incidental_finding; severe_outcome_or_death. "
    "For medication_stopped, require explicit drug or treatment evidence. Do not mark "
    "medication_stopped for negative or non-treatment phrases such as \"vomiting stopped\", "
    "\"no medications discontinued\", or \"interrupted sutures\".\n\n"
    "Deterministic guard hints, for evidence windows only:\n{guards}\n\nNote:\n{note}"
)

_GUARD_PATTERNS = {
    "medication_stop": r"\b(discontinu(?:e|ed|ation|ing)|stopp(?:ed|ing)?|withdraw(?:n|al)|cessation|suspend(?:ed|ing)?)\b",
    "care_transition": r"\b(admitted|admission|hospitali[sz]ed|discharged|follow-?up|transferred)\b",
    "procedure": r"\b(biopsy|surgery|operation|resection|procedure|intervention|catheter|intubation|stent|graft|suture)\b",
    "diagnostic_workup": r"\b(CT|MRI|ultrasound|x-?ray|scan|imaging|biopsy|laboratory|revealed|showed|demonstrated)\b",
    "response_failure": r"\b(improv(?:ed|ement)|resolved|remission|responded|failed|failure|worsen(?:ed|ing)|recurr(?:ed|ence)|relapse[ad]?)\b",
    "adverse_reaction": r"\b(adverse|reaction|toxicity|side effects?|drug-induced|allerg(?:y|ic)|complication)\b",
}

_NEGATION_WINDOW = re.compile(r"\b(no|not|without|denied|denies|never|negative for|free of)\b", re.I)
_ROUTINE_WINDOW = re.compile(r"\b(history of|previous|previously|prior|routine|completed|after completion)\b", re.I)


# Weights source of record: our S3 mirror (in-region, no HF dependency, no
# rate limits, licensing stays ours) — see docs/vllm-udf-runbook.md and RFC
# 0094's weights-distribution section. The image bake remains the fast path;
# this is the fallback when the cache is empty, replacing the old HF fallback.
WEIGHTS_S3 = os.environ.get(
    "CHART_WEIGHTS_S3",
    "s3://hevlayer-models-186219257916-us-east-1/google/gemma-2-9b-it",
)


def _ensure_weights() -> None:
    """If the HF cache has no snapshot of MODEL, restore it from the S3 mirror
    into the standard hub layout so vLLM/huggingface_hub load fully offline.
    No-op when the image bake did its job or MODEL isn't the mirrored model."""
    if not WEIGHTS_S3 or MODEL != "google/gemma-2-9b-it":
        return
    hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    model_dir = os.path.join(hub, "models--google--gemma-2-9b-it")
    snapshots = os.path.join(model_dir, "snapshots")
    if os.path.isdir(snapshots) and any(
        os.path.exists(os.path.join(snapshots, rev, "config.json")) for rev in os.listdir(snapshots)
    ):
        return
    import boto3  # classifier extra; imported lazily so CPU-side tests don't need it

    bucket, _, prefix = WEIGHTS_S3.removeprefix("s3://").partition("/")
    s3 = boto3.client("s3")
    rev = s3.get_object(Bucket=bucket, Key=f"{prefix}/LATEST")["Body"].read().decode().strip()
    dest = os.path.join(snapshots, rev)
    os.makedirs(dest, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")
    keys = [
        o["Key"]
        for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/{rev}/")
        for o in page.get("Contents", [])
    ]
    if not keys:
        raise RuntimeError(f"weights mirror {WEIGHTS_S3}/{rev} is empty")
    for key in keys:
        target = os.path.join(dest, os.path.relpath(key, f"{prefix}/{rev}"))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        s3.download_file(bucket, key, target)
    # refs/main lets bare model-name lookups resolve without the network
    refs = os.path.join(model_dir, "refs")
    os.makedirs(refs, exist_ok=True)
    with open(os.path.join(refs, "main"), "w") as f:
        f.write(rev)
    print(f"restored weights from {WEIGHTS_S3}@{rev} ({len(keys)} files)")


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

    _ensure_weights()
    # Gemma-2-9B bf16 weights are 17.2GiB of a 24GB A10G; at the defaults
    # (util 0.90, max_seq_len 8192 from the model config, CUDA-graph capture)
    # only ~0.75GiB is left for KV cache and the engine refuses to start
    # (needs 2.63GiB at 8192). Rows are ~512-token note chunks, so 4096 is
    # generous headroom; eager mode skips CUDA-graph memory, which matters
    # more than graph latency for batch-heavy guided decoding.
    return LLM(
        model=MODEL,
        max_model_len=int(os.environ.get("CHART_VLLM_MAX_MODEL_LEN", "4096")),
        gpu_memory_utilization=float(os.environ.get("CHART_VLLM_GPU_UTIL", "0.95")),
        enforce_eager=os.environ.get("CHART_VLLM_ENFORCE_EAGER", "1").strip().lower() in {"1", "true", "yes"},
    )


def digest(note_text: str) -> dict:
    """Tier 1 — the one GPU pass. Guided-decode the note to DIGEST_SCHEMA."""
    return digest_batch([note_text])[0]


def digest_batch(note_texts: list[str]) -> list[dict]:
    """Tier 1, batched — one vLLM `generate()` over the whole claim.

    vLLM's throughput lever is continuous batching: N prompts in one call share
    the forward passes, ~3-5x the notes/sec of calling generate() per note on
    the same GPU. The worker loop feeds an entire claimed batch through here.

    vLLM changed the Python structured-output API across releases: newer versions
    use `structured_outputs`, older versions use `guided_decoding`. Support both
    so the Function image can move independently of this repo. A note that fails
    to parse degrades to an empty digest rather than failing the batch — the
    `events Eq null`-filtered re-run picks it up again.
    """
    if not note_texts:
        return []
    empty = {"events": [], "summary": "", "diagnosis_category": "other", "specialty": "other"}
    # Rows are ~512-token chunks, but guard the engine's max_model_len (4096)
    # against outliers: ~12k chars ≈ 3k tokens leaves room for prompt + output.
    prompts = [_digest_prompt(t[:12000]) for t in note_texts]
    live = [i for i, t in enumerate(note_texts) if t]
    digests: list[dict] = [dict(empty) for _ in note_texts]
    if not live:
        return digests
    outs = _engine().generate([prompts[i] for i in live], _sampling_params())
    for i, out in zip(live, outs):
        try:
            digests[i] = _parse_digest(out.outputs[0].text)
        except (ValueError, json.JSONDecodeError):
            digests[i] = dict(empty)
    return digests


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


def derive_labels_v2(d: dict) -> dict:
    """Tier 2 v2 — only affirmed, quoted, sufficiently confident events become facets."""
    kept = [_normalize_event_v2(e) for e in _event_items_v2(d)]
    kept = [e for e in kept if e and _event_v2_facetable(e)]
    events = sorted({e["type"] for e in kept})
    groups = sorted({EVENT_GROUPS_V2[e] for e in events})
    confidence = {
        event_type: max(e["confidence"] for e in kept if e["type"] == event_type)
        for event_type in events
    }
    spans = {
        event_type: [e["span_quote"] for e in kept if e["type"] == event_type]
        for event_type in events
    }
    return {
        "events_v2": events,
        "event_groups_v2": groups,
        "event_confidence_v2": confidence,
        "event_spans_v2": spans,
        "has_treatment_change_v2": bool({"medication_started", "medication_dose_changed", "medication_stopped"} & set(events)),
        "has_treatment_response_v2": bool({"treatment_response", "treatment_failure_or_recurrence"} & set(events)),
        "has_complication_v2": bool({"adverse_drug_reaction", "procedure_complication", "non_drug_complication"} & set(events)),
        "has_care_transition_v2": "care_transition" in events,
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


def _event_items_v2(d: dict) -> list[dict]:
    events = d.get("events_v2", [])
    if not isinstance(events, list):
        return []
    return [event for event in events if isinstance(event, dict)]


def _clean_label(value: Any, *, default: str = "other") -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _digest_prompt(note: str) -> str:
    return _DIGEST_PROMPT.format(note=note, guards=json.dumps(prepass_guards(note), ensure_ascii=True))


def prepass_guards(note_text: str) -> list[dict]:
    """Cheap candidate spans and polarity hints for the model; not final labels."""
    guards = []
    for family, pattern in _GUARD_PATTERNS.items():
        for match in re.finditer(pattern, note_text, flags=re.I):
            start = max(0, match.start() - 80)
            end = min(len(note_text), match.end() + 80)
            window = note_text[start:end].strip()
            prefix = note_text[max(0, match.start() - 80):match.start()]
            polarity_hint = "affirmed"
            if _NEGATION_WINDOW.search(prefix):
                polarity_hint = "negated"
            elif _ROUTINE_WINDOW.search(prefix):
                polarity_hint = "historical_or_routine"
            guards.append(
                {
                    "family_hint": family,
                    "term": match.group(0),
                    "span": window,
                    "polarity_hint": polarity_hint,
                }
            )
    return guards[:24]


def _normalize_event_v2(event: dict) -> dict | None:
    event_type = event.get("type")
    if event_type not in EVENT_TYPES_V2:
        return None
    polarity = event.get("polarity")
    if polarity not in EVENT_POLARITIES_V2:
        polarity = "affirmed"
    span = event.get("span_quote")
    if not isinstance(span, str) or not span.strip():
        span = ""
    confidence = event.get("confidence")
    if not isinstance(confidence, int | float):
        confidence = 1.0
    normalized: dict[str, Any] = {
        "type": event_type,
        "polarity": polarity,
        "span_quote": span.strip(),
        "confidence": max(0.0, min(1.0, float(confidence))),
    }
    for field in ("drug_or_treatment", "procedure", "diagnosis", "temporal_context"):
        value = event.get(field)
        if isinstance(value, str) and value.strip():
            normalized[field] = value.strip()
    return normalized


def _event_v2_facetable(event: dict) -> bool:
    if event["polarity"] != "affirmed" or not event["span_quote"]:
        return False
    threshold = MIN_EVENT_CONFIDENCE_V2.get(event["type"], DEFAULT_MIN_EVENT_CONFIDENCE_V2)
    if event["confidence"] < threshold:
        return False
    if event["type"] == "medication_stopped" and not event.get("drug_or_treatment"):
        return False
    return True


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
    events_v2 = [
        normalized
        for event in _event_items_v2(parsed)
        if (normalized := _normalize_event_v2(event)) is not None
    ]
    digest = {
        "summary": _clean_label(parsed.get("summary"), default=""),
        "events": events,
        "diagnosis_category": _clean_label(parsed.get("diagnosis_category")),
        "specialty": _clean_label(parsed.get("specialty")),
    }
    if "events_v2" in parsed:
        digest["events_v2"] = events_v2
    return digest


# --- The UDF entrypoint. ------------------------------------------------------
# Primary output is `events` ([]string). The full Tier-2 label set is patched
# via one multi-row patch_columns per claim (columnar: N ids + aligned column
# arrays), settling RFC 0076's multi-write question as one cascade Function:
# one model pass, many denormalized labels.
#
# The per-item @udf spelling is kept for tests and CPU-side inspection, but the
# deployed worker runs `run_batched_worker` below: the client's run_udf_worker
# calls the UDF once per item, which serializes vLLM into N single-prompt
# generate() calls per claim. Speaking the documented claim/complete worker
# protocol directly (docs/kubernetes/function-crd § worker protocol) lets the
# whole claim go through one batched generate() + one writeback + one complete.
@udf(id="chart-classify-events", inputs=["id", "text"], output="events", batch_size=16)
async def classify_events_udf(
    id: object = "", text: object = "", tpuf: Any | None = None
) -> list[str]:
    note = "" if text is None else str(text)
    d = digest(note)
    labels = derive_labels(d)
    labels_v2 = derive_labels_v2(d)
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
        if "events_v2" in d:
            attrs.update(
                {
                    "events_v2": [labels_v2["events_v2"]],
                    "event_groups_v2": [labels_v2["event_groups_v2"]],
                    "event_confidence_v2": [labels_v2["event_confidence_v2"]],
                    "event_spans_v2": [labels_v2["event_spans_v2"]],
                    "has_treatment_change_v2": [labels_v2["has_treatment_change_v2"]],
                    "has_treatment_response_v2": [labels_v2["has_treatment_response_v2"]],
                    "has_complication_v2": [labels_v2["has_complication_v2"]],
                    "has_care_transition_v2": [labels_v2["has_care_transition_v2"]],
                }
            )
        await tpuf.patch_columns(Settings().namespace, [str(id)], attrs)
    return labels["events"]


def _batch_writeback_columns(digests: list[dict]) -> dict[str, list[Any]]:
    """Tier-2 labels for a whole claim as aligned column arrays (patch_columns'
    native shape: one write for N rows instead of N writes)."""
    columns: dict[str, list[Any]] = {field: [] for field in WRITEBACK_FIELDS}
    for d in digests:
        labels = derive_labels(d)
        labels_v2 = derive_labels_v2(d)
        columns["events"].append(labels["events"])
        columns["has_med_discontinuation"].append(labels["has_med_discontinuation"])
        columns["has_adverse_event"].append(labels["has_adverse_event"])
        columns["diagnosis_category"].append(labels["diagnosis_category"])
        columns["specialty"].append(labels["specialty"])
        columns["discontinuation_reason"].append(
            discontinuation_reason(d) if labels["has_med_discontinuation"] else None
        )
        columns["events_v2"].append(labels_v2["events_v2"])
        columns["event_groups_v2"].append(labels_v2["event_groups_v2"])
        columns["event_confidence_v2"].append(labels_v2["event_confidence_v2"])
        columns["event_spans_v2"].append(labels_v2["event_spans_v2"])
        columns["has_treatment_change_v2"].append(labels_v2["has_treatment_change_v2"])
        columns["has_treatment_response_v2"].append(labels_v2["has_treatment_response_v2"])
        columns["has_complication_v2"].append(labels_v2["has_complication_v2"])
        columns["has_care_transition_v2"].append(labels_v2["has_care_transition_v2"])
    return columns


async def run_batched_worker(*, once: bool = False, poll_interval: float = 2.0) -> None:
    """The deployed loop: claim → ONE batched generate() → ONE patch_columns →
    complete. Same protocol run_udf_worker speaks, minus the per-item fn calls."""
    import uuid

    from hevlayer import AsyncHevlayer

    settings = Settings()
    udf_id = os.environ.get("HEVLAYER_UDF_ID", "chart-classify-events")
    worker_id = os.environ.get("HEVLAYER_WORKER_ID") or f"{udf_id}-{uuid.uuid4().hex[:8]}"
    limit = int(os.environ.get("HEVLAYER_UDF_BATCH_SIZE", "16"))
    # The lease must outlast one whole batched generate() — scale it with the
    # claim size rather than relying on the client's 120s default.
    lease_seconds = int(os.environ.get("HEVLAYER_UDF_LEASE_SECONDS", "600"))
    base_url = os.environ.get("HEVLAYER_BASE_URL", "http://localhost:8080")

    async with AsyncHevlayer(api_key=settings.api_key, base_url=base_url) as client:
        while True:
            claimed = await client.claim_udf_items(
                udf_id, {"worker_id": worker_id, "limit": limit, "lease_seconds": lease_seconds}
            )
            if not claimed.items:
                if once:
                    return
                await asyncio.sleep(poll_interval)
                continue

            items = claimed.items
            digests = digest_batch(["" if i.input.get("text") is None else str(i.input.get("text")) for i in items])
            columns = _batch_writeback_columns(digests)

            # One multi-row patch per (namespace, claim) — items in a claim share
            # the target namespace in practice, but group defensively.
            by_namespace: dict[str, list[int]] = {}
            for idx, item in enumerate(items):
                by_namespace.setdefault(item.namespace, []).append(idx)
            for namespace, idxs in by_namespace.items():
                ids = [str(items[i].input.get("id") or items[i].id) for i in idxs]
                await client.patch_columns(
                    namespace, ids, {name: [values[i] for i in idxs] for name, values in columns.items()}
                )

            await client.complete_udf_items(
                udf_id,
                {
                    "worker_id": worker_id,
                    "items": [
                        {
                            "namespace": item.namespace,
                            "id": item.id,
                            "attributes": {"events": derive_labels(d)["events"]},
                        }
                        for item, d in zip(items, digests)
                    ],
                },
            )
            if once:
                return


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the chart clinical-event classifier UDF worker")
    parser.add_argument("--once", action="store_true", help="claim one batch and exit")
    parser.add_argument(
        "--per-item", action="store_true",
        help="use the client's per-item run_udf_worker loop instead of the batched loop",
    )
    args = parser.parse_args()
    if not Settings().api_key:
        raise SystemExit(
            "No gateway key. Set LAYER_GATEWAY_API_KEY in .env — it's the upstream "
            "Turbopuffer key (1Password: layer-turbopuffer / mesh-staging)."
        )
    once = args.once or os.environ.get("CHART_UDF_ONCE", "").strip().lower() in {"1", "true", "yes"}
    if args.per_item or os.environ.get("CHART_UDF_PER_ITEM", "").strip().lower() in {"1", "true", "yes"}:
        asyncio.run(run_udf_worker(classify_events_udf, udf_id="chart-classify-events", once=once))
    else:
        asyncio.run(run_batched_worker(once=once))


if __name__ == "__main__":
    main()
