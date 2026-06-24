from __future__ import annotations

from collections.abc import Iterator

from chart_common.config import Settings
from chart_common.records import NoteRecord


def load_notes(settings: Settings, *, limit: int | None = None) -> Iterator[NoteRecord]:
    """Stream PMC-Patients case-report notes as NoteRecords.

    Pinned to an exact revision (config.py) for RFC 0053 exact-replay discipline —
    and so the ReCDS qrels (eval/) match what was indexed. The declarative twin is
    deploy/pipeline.yaml's sourceRef; this is the imperative reader.
    """
    from datasets import load_dataset

    ds = load_dataset(
        settings.dataset_repo,
        split=settings.dataset_split,
        revision=settings.dataset_revision,
    )
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        try:
            yield NoteRecord.from_row(row)
        except (KeyError, TypeError):
            # A malformed row is a per-row skip, not a run failure.
            continue
