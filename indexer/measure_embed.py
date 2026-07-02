from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from chart_common.cli import positive_float, positive_int
from chart_common.config import EMBED_DIM, FULL_CORPUS_NOTES, Settings
from chart_common.embed import Embedder
from chart_common.records import NoteRecord
from chart_common.runtime import runtime_error, runtime_report

from .dataset import load_notes
from .index import BATCH

EmbedFn = Callable[[list[str]], list[list[float]]]


def production_path_report() -> dict[str, Any]:
    return {
        "pipeline_cr": "chart-embed-gpu",
        "module": "indexer.embed",
        "compute_class": "gpu",
        "image": "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-embedder-plan-20260626-batchdocs1",
        "allow_full_cpu_index": False,
    }


def estimate_full_run(
    *,
    notes: int,
    elapsed_seconds: float,
    full_notes: int = FULL_CORPUS_NOTES,
    gpu_hourly_usd: float | None = None,
) -> dict[str, Any]:
    if notes <= 0:
        raise ValueError("notes must be positive for a cost estimate")
    if elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds must be positive for a cost estimate")
    per_note = elapsed_seconds / notes if notes else 0.0
    estimate: dict[str, Any] = {
        "notes": notes,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "per_note_seconds": round(per_note, 4),
        "notes_per_second": round(notes / elapsed_seconds, 3) if elapsed_seconds else 0.0,
        "full_notes": full_notes,
        "estimated_full_seconds": round(per_note * full_notes, 1),
        "estimated_full_hours": round((per_note * full_notes) / 3600, 2),
    }
    if gpu_hourly_usd is not None:
        estimate["gpu_hourly_usd"] = round(gpu_hourly_usd, 4)
        estimate["estimated_full_usd"] = round(estimate["estimated_full_hours"] * gpu_hourly_usd, 2)
    return estimate


def budget_report(
    estimate: dict[str, Any],
    *,
    max_full_hours: float | None = None,
    max_full_usd: float | None = None,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
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


def _batched_notes(notes: Iterable[NoteRecord], batch_size: int) -> Iterable[list[NoteRecord]]:
    batch: list[NoteRecord] = []
    for note in notes:
        batch.append(note)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def validate_embedding_batch(batch: list[NoteRecord], vectors: list[list[float]]) -> None:
    if len(vectors) != len(batch):
        ids = ", ".join(note.id for note in batch[:3])
        suffix = "..." if len(batch) > 3 else ""
        raise RuntimeError(
            f"embedder returned {len(vectors)} vectors for {len(batch)} notes in batch [{ids}{suffix}]"
        )
    for note, vector in zip(batch, vectors, strict=True):
        if len(vector) != EMBED_DIM:
            raise RuntimeError(f"{note.id}: expected {EMBED_DIM}-d embeddings, got {len(vector)}")


def run(
    *,
    limit: int,
    batch_size: int = BATCH,
    max_chars: int | None = None,
    gpu_hourly_usd: float | None = None,
    accelerator: str = "unspecified",
    embed_fn: EmbedFn | None = None,
) -> dict[str, Any]:
    settings = Settings()
    embed = embed_fn or Embedder(settings.embed_model).embed_passages
    total = 0
    vector_dim: int | None = None
    started = time.perf_counter()

    for batch in _batched_notes(
        load_notes(settings, limit=limit, include_similar_patient_ids=False),
        batch_size,
    ):
        texts = [note.text[:max_chars] if max_chars else note.text for note in batch]
        vectors = embed(texts)
        validate_embedding_batch(batch, vectors)
        total += len(vectors)
        if vectors and vector_dim is None:
            vector_dim = len(vectors[0])

    if total != limit:
        raise RuntimeError(f"only embedded {total} notes for requested limit {limit}")

    elapsed = time.perf_counter() - started
    return {
        "runtime": {
            "accelerator": accelerator,
        },
        "sample": {
            "notes": total,
            "batch_size": batch_size,
            "max_chars": max_chars,
            "vector_dim": vector_dim,
            "model": settings.embed_model,
        },
        "production_path": production_path_report(),
        "estimate": estimate_full_run(
            notes=total,
            elapsed_seconds=elapsed,
            gpu_hourly_usd=gpu_hourly_usd,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure chart embedding throughput on a bounded sample")
    parser.add_argument("--limit", type=positive_int, default=500)
    parser.add_argument("--batch-size", type=positive_int, default=BATCH)
    parser.add_argument("--max-chars", type=positive_int, default=None, help="truncate each note for smoke runs")
    parser.add_argument(
        "--gpu-hourly-usd",
        type=positive_float,
        default=None,
        help="include this GPU hourly rate in the full-index cost extrapolation",
    )
    parser.add_argument(
        "--accelerator",
        choices=["gpu", "cpu", "unspecified"],
        default=os.environ.get("CHART_MEASURE_ACCELERATOR", "unspecified"),
        help="hardware used for the timing sample; Phase-6 audit requires gpu",
    )
    parser.add_argument(
        "--max-full-hours",
        type=positive_float,
        default=None,
        help="exit non-zero if the estimated full index run exceeds this many hours",
    )
    parser.add_argument(
        "--max-full-usd",
        type=positive_float,
        default=None,
        help="exit non-zero if the estimated full index run exceeds this GPU budget",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write the measurement report to this JSON path for Phase-6 audit gates",
    )
    args = parser.parse_args()
    runtime = runtime_report(args.accelerator)
    if error := runtime_error(runtime):
        failed = {
            "status": "failed",
            "error": error,
            "runtime": runtime,
            "production_path": production_path_report(),
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
            batch_size=args.batch_size,
            max_chars=args.max_chars,
            gpu_hourly_usd=args.gpu_hourly_usd,
            accelerator=args.accelerator,
        )
    except RuntimeError as exc:
        if args.out is not None:
            failed = {
                "status": "failed",
                "error": str(exc),
                "runtime": runtime,
                "production_path": production_path_report(),
            }
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(failed, indent=2) + "\n")
        raise
    result["runtime"] = runtime
    result.setdefault("production_path", production_path_report())
    if args.max_full_hours is not None or args.max_full_usd is not None:
        result["budget"] = budget_report(
            result["estimate"],
            max_full_hours=args.max_full_hours,
            max_full_usd=args.max_full_usd,
        )
    rendered = json.dumps(result, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n")
    print(rendered)
    if result.get("budget") and not result["budget"]["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
