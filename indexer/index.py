from __future__ import annotations

import asyncio
import itertools
from collections.abc import Iterator

from chart_common.config import Settings
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


def _batched(it: Iterator[NoteRecord], n: int) -> Iterator[list[NoteRecord]]:
    while batch := list(itertools.islice(it, n)):
        yield batch


async def run(*, limit: int | None = None, dry_run: bool = False) -> None:
    """Load → embed (Arctic) → upsert → materialize facet snapshots.

    The imperative twin of deploy/ (warehouse + pipeline + index). Embedding is
    inline here (the demo's CPU/GPU dev path); the declarative equivalent is the
    GPU embed stage sharing the pipelineId. The clinical facets the rail draws are
    UDF writeback (functions/) and fill in after enrichment runs — the snapshots
    here capture age_band / gender immediately and the rest as they land.
    """
    settings = Settings()
    embedder = Embedder(settings.embed_model)

    layer = None if dry_run else make_client(settings)
    total = 0
    try:
        for batch in _batched(load_notes(settings, limit=limit), BATCH):
            vectors = embedder.embed_passages([r.text for r in batch])
            for record, vector in zip(batch, vectors):
                record.vector = vector
            rows = [r.to_upsert() for r in batch]
            if not dry_run:
                await write_notes(layer, settings.namespace, rows)
            total += len(rows)
            print(f"  upserted {total} notes", end="\r")

        print(f"\nindexed {total} notes into {settings.namespace}")
        if not dry_run:
            print("materializing facet snapshots…")
            await materialize_facet_snapshots(layer, settings.namespace)
            print("done.")
    finally:
        if layer is not None:
            await close_client(layer)


def main(*, limit: int | None = None, dry_run: bool = False) -> None:
    asyncio.run(run(limit=limit, dry_run=dry_run))
