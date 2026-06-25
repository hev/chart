from __future__ import annotations

import asyncio
import itertools
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from chart_common.config import EMBED_DIM, Settings
from chart_common.embed import Embedder
from chart_common.gateway import (
    close_client,
    make_client,
    materialize_facet_snapshots,
    write_notes,
)
from chart_common.records import NoteRecord

from .dataset import load_notes

BATCH = 256
FULL_INDEX_ENV = "CHART_ALLOW_FULL_CPU_INDEX"


def _batched(it: Iterator[NoteRecord], n: int) -> Iterator[list[NoteRecord]]:
    while batch := list(itertools.islice(it, n)):
        yield batch


def validate_index_scope(*, limit: int | None, dry_run: bool) -> None:
    """Prevent accidental full-corpus CPU embedding.

    PLAN.md gates the full 167k run on a GPU embed path. The local indexer is the
    slice/smoke path, so an unbounded non-dry-run should be an explicit decision.
    """
    if limit is not None:
        return
    if os.environ.get(FULL_INDEX_ENV, "").strip().lower() in {"1", "true", "yes"}:
        return
    raise SystemExit(
        "Refusing an unbounded local index with the local embedder. Use --limit for "
        "the Phase-1 slice, or set CHART_ALLOW_FULL_CPU_INDEX=1 only after the "
        "Phase-6 full-index cost gate is accepted."
    )


def attach_vectors(batch: list[NoteRecord], vectors: list[list[float]]) -> None:
    if len(vectors) != len(batch):
        raise ValueError(f"embedder returned {len(vectors)} vectors for {len(batch)} records")
    for index, (record, vector) in enumerate(zip(batch, vectors, strict=True)):
        if len(vector) != EMBED_DIM:
            raise ValueError(f"{record.id}: expected {EMBED_DIM}-d vector at batch index {index}, got {len(vector)}")
        record.vector = vector


async def run(*, limit: int | None = None, dry_run: bool = False) -> dict[str, Any]:
    """Load → embed (Arctic) → upsert → materialize facet snapshots.

    The imperative twin of deploy/ (warehouse + pipeline + index). Embedding is
    inline here (the demo's CPU/GPU dev path); the declarative equivalent is the
    GPU embed stage sharing the pipelineId. The clinical facets the rail draws are
    UDF writeback (functions/) and fill in after enrichment runs — the snapshots
    here capture age_band / gender immediately and the rest as they land.
    """
    validate_index_scope(limit=limit, dry_run=dry_run)
    settings = Settings()
    embedder = Embedder(settings.embed_model)

    layer = None if dry_run else make_client(settings)
    total = 0
    facets_materialized = False
    schema = {
        "vector_dim": None,
        "rows_with_age_band": 0,
        "rows_with_gender": 0,
        "rows_with_similar_patient_ids": 0,
    }
    try:
        for batch in _batched(load_notes(settings, limit=limit), BATCH):
            vectors = embedder.embed_passages([r.text for r in batch])
            attach_vectors(batch, vectors)
            rows = [r.to_upsert() for r in batch]
            if vectors and schema["vector_dim"] is None:
                schema["vector_dim"] = len(vectors[0])
            schema["rows_with_age_band"] += sum(1 for row in rows if row.get("age_band"))
            schema["rows_with_gender"] += sum(1 for row in rows if row.get("gender"))
            schema["rows_with_similar_patient_ids"] += sum(
                1 for row in rows if row.get("similar_patient_ids")
            )
            if not dry_run:
                await write_notes(layer, settings.namespace, rows)
            total += len(rows)
            print(f"  upserted {total} notes", end="\r")

        print(f"\nindexed {total} notes into {settings.namespace}")
        if not dry_run:
            print("materializing facet snapshots…")
            await materialize_facet_snapshots(layer, settings.namespace)
            facets_materialized = True
            print("done.")
    finally:
        if layer is not None:
            await close_client(layer)
    return {
        "namespace": settings.namespace,
        "status": "completed",
        "limit": limit,
        "dry_run": dry_run,
        "indexed": total,
        "provenance": {
            "dataset_repo": settings.dataset_repo,
            "dataset_revision": settings.dataset_revision,
            "dataset_split": settings.dataset_split,
            "embed_model": settings.embed_model,
            "embed_dim": EMBED_DIM,
        },
        "schema": schema,
        "facet_snapshots_materialized": facets_materialized,
    }


def main(*, limit: int | None = None, dry_run: bool = False, out: Path | None = None) -> dict[str, Any]:
    report = asyncio.run(run(limit=limit, dry_run=dry_run))
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
    return report
