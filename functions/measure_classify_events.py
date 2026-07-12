from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Callable

from chart_common.cli import non_negative_int, positive_float, positive_int
from chart_common.config import FULL_CORPUS_NOTES, Settings
from chart_common.runtime import runtime_error, runtime_report
from indexer.dataset import load_notes

from .classify_events import LEGACY_WRITEBACK_FIELDS, derive_labels, digest, discontinuation_reason

DISCONTINUATION_TERMS = (
    "discontinued",
    "discontinue",
    "stopped",
    "withdrawn",
    "cessation",
)


def estimate_full_run(
    *,
    notes: int,
    elapsed_seconds: float,
    full_notes: int = FULL_CORPUS_NOTES,
    gpu_hourly_usd: float | None = None,
) -> dict:
    if notes <= 0:
        raise ValueError("notes must be positive for a cost estimate")
    if elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds must be positive for a cost estimate")
    per_note = elapsed_seconds / notes if notes else 0.0
    estimate = {
        "notes": notes,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "per_note_seconds": round(per_note, 3),
        "full_notes": full_notes,
        "estimated_full_seconds": round(per_note * full_notes, 1),
        "estimated_full_hours": round((per_note * full_notes) / 3600, 2),
    }
    if gpu_hourly_usd is not None:
        estimate["gpu_hourly_usd"] = round(gpu_hourly_usd, 4)
        estimate["estimated_full_usd"] = round(estimate["estimated_full_hours"] * gpu_hourly_usd, 2)
    return estimate


def budget_report(
    estimate: dict,
    *,
    max_full_hours: float | None = None,
    max_full_usd: float | None = None,
) -> dict:
    checks = {}
    if max_full_hours is not None:
        checks["max_full_hours"] = {
            "limit": max_full_hours,
            "actual": estimate["estimated_full_hours"],
            "ok": estimate["estimated_full_hours"] <= max_full_hours,
        }
    if max_full_usd is not None:
        actual_usd = estimate.get("estimated_full_usd")
        checks["max_full_usd"] = {
            "limit": max_full_usd,
            "actual": actual_usd,
            "ok": actual_usd is not None and actual_usd <= max_full_usd,
        }
    return {"checks": checks, "accepted": all(check["ok"] for check in checks.values())}


def signal_report(
    sample: dict,
    *,
    min_med_discontinuations: int | None = None,
    examples: list[dict] | None = None,
    min_review_examples: int | None = None,
) -> dict:
    checks = {}
    if min_med_discontinuations is not None:
        actual = int(sample.get("med_discontinuation") or 0)
        checks["min_med_discontinuations"] = {
            "limit": min_med_discontinuations,
            "actual": actual,
            "ok": actual >= min_med_discontinuations,
        }
    if min_review_examples is not None:
        actual = len(examples or [])
        checks["min_review_examples"] = {
            "limit": min_review_examples,
            "actual": actual,
            "ok": actual >= min_review_examples,
        }
    return {"checks": checks, "accepted": all(check["ok"] for check in checks.values())}


def writeback_report() -> dict:
    return {
        "mode": "tpuf.patch_columns",
        "primary_output": "events",
        "model_passes_per_note": 1,
        "patched_fields": LEGACY_WRITEBACK_FIELDS,
        "settles_multi_write": True,
    }


def candidate_notes(settings: Settings, *, limit: int, discontinuation_only: bool) -> Iterable[str]:
    yielded = 0
    for record in load_notes(settings, include_similar_patient_ids=False):
        text = record.text or ""
        if discontinuation_only and not any(term in text.lower() for term in DISCONTINUATION_TERMS):
            continue
        yield text
        yielded += 1
        if yielded >= limit:
            return


DigestFn = Callable[[str], dict]


def run(
    *,
    limit: int,
    discontinuation_only: bool,
    max_chars: int | None,
    examples: int = 3,
    gpu_hourly_usd: float | None = None,
    accelerator: str = "unspecified",
    digest_fn: DigestFn = digest,
) -> dict:
    settings = Settings()
    counts = {
        "notes": 0,
        "med_discontinuation": 0,
        "adverse_event": 0,
        "events": {},
    }
    review_examples = []
    started = time.perf_counter()
    for note in candidate_notes(settings, limit=limit, discontinuation_only=discontinuation_only):
        if max_chars:
            note = note[:max_chars]
        d = digest_fn(note)
        labels = derive_labels(d)
        counts["notes"] += 1
        counts["med_discontinuation"] += int(labels["has_med_discontinuation"])
        counts["adverse_event"] += int(labels["has_adverse_event"])
        for event in labels["events"]:
            counts["events"][event] = counts["events"].get(event, 0) + 1
        if labels["has_med_discontinuation"] and len(review_examples) < examples:
            review_examples.append(
                {
                    "note_preview": note[:500],
                    "events": d.get("events", []),
                    "labels": labels,
                    "discontinuation_reason": discontinuation_reason(d) or "unspecified",
                }
            )

    if counts["notes"] != limit:
        raise RuntimeError(f"only found {counts['notes']} notes for requested limit {limit}")

    estimate = estimate_full_run(
        notes=counts["notes"],
        elapsed_seconds=time.perf_counter() - started,
        gpu_hourly_usd=gpu_hourly_usd,
    )
    return {
        "runtime": {
            "accelerator": accelerator,
        },
        "sample": counts,
        "estimate": estimate,
        "examples": review_examples,
        "writeback": writeback_report(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure chart Gemma cascade latency on a GPU box")
    parser.add_argument("--limit", type=positive_int, default=50)
    parser.add_argument(
        "--discontinuation-only",
        action="store_true",
        help="sample notes likely to contain discontinuation language",
    )
    parser.add_argument("--max-chars", type=positive_int, default=None, help="truncate each note for smoke runs")
    parser.add_argument(
        "--examples",
        type=non_negative_int,
        default=3,
        help="include this many discontinuation examples for hand verification",
    )
    parser.add_argument(
        "--gpu-hourly-usd",
        type=positive_float,
        default=None,
        help="include this GPU hourly rate in the full-run cost extrapolation",
    )
    parser.add_argument(
        "--accelerator",
        choices=["gpu", "cpu", "unspecified"],
        default=os.environ.get("CHART_MEASURE_ACCELERATOR", "unspecified"),
        help="hardware used for the timing sample; Phase-4/6 audit requires gpu",
    )
    parser.add_argument(
        "--max-full-hours",
        type=positive_float,
        default=None,
        help="exit non-zero if the estimated full classify run exceeds this many hours",
    )
    parser.add_argument(
        "--max-full-usd",
        type=positive_float,
        default=None,
        help="exit non-zero if the estimated full classify run exceeds this GPU budget",
    )
    parser.add_argument(
        "--min-med-discontinuations",
        type=non_negative_int,
        default=None,
        help="exit non-zero unless the sample finds at least this many medication discontinuations",
    )
    parser.add_argument(
        "--min-review-examples",
        type=non_negative_int,
        default=None,
        help="exit non-zero unless the report includes this many discontinuation examples for hand verification",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write the measurement report to this JSON path for Phase-4/6 audit gates",
    )
    args = parser.parse_args()
    runtime = runtime_report(args.accelerator)
    if error := runtime_error(runtime):
        failed = {
            "status": "failed",
            "error": error,
            "runtime": runtime,
            "writeback": writeback_report(),
        }
        rendered = json.dumps(failed, indent=2)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered + "\n")
        print(rendered)
        raise SystemExit(1)
    try:
        result = run(
            limit=args.limit,
            discontinuation_only=args.discontinuation_only,
            max_chars=args.max_chars,
            examples=args.examples,
            gpu_hourly_usd=args.gpu_hourly_usd,
            accelerator=args.accelerator,
        )
    except RuntimeError as exc:
        if args.out is not None:
            failed = {
                "status": "failed",
                "error": str(exc),
                "runtime": runtime,
                "writeback": writeback_report(),
            }
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(failed, indent=2) + "\n")
        if "vLLM is required for the Gemma classifier" in str(exc):
            raise SystemExit(str(exc)) from exc
        raise
    result["runtime"] = runtime
    result.setdefault("writeback", writeback_report())
    if args.max_full_hours is not None or args.max_full_usd is not None:
        result["budget"] = budget_report(
            result["estimate"],
            max_full_hours=args.max_full_hours,
            max_full_usd=args.max_full_usd,
        )
    if args.min_med_discontinuations is not None or args.min_review_examples is not None:
        result["signal"] = signal_report(
            result["sample"],
            min_med_discontinuations=args.min_med_discontinuations,
            examples=result["examples"],
            min_review_examples=args.min_review_examples,
        )
    rendered = json.dumps(result, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n")
    print(rendered)
    if result.get("budget") and not result["budget"]["accepted"]:
        raise SystemExit(1)
    if result.get("signal") and not result["signal"]["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
