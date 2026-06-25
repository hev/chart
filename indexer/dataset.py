from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from chart_common.config import Settings
from chart_common.records import NoteRecord


def _download_qrels(settings: Settings, *, split: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            settings.recds_repo,
            f"PPR/qrels_{split}.tsv",
            repo_type="dataset",
            revision=settings.recds_revision,
        )
    )


def load_ppr_similar_patient_ids(
    settings: Settings, *, split: str = "train"
) -> dict[str, list[str]]:
    """Load non-eval PPR qrels as static similar-patient feature edges.

    PMC-Patients itself does not ship `similar_patients`; the relationship lives
    in ReCDS PPR qrels. Use train by default so dev/test eval splits remain held
    out from the indexed feature column.
    """
    similar: dict[str, set[str]] = {}
    with _download_qrels(settings, split=split).open() as f:
        header = next(f, "").strip().split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            row = dict(zip(header, parts, strict=True))
            query_id = row["query-id"]
            doc_id = row["corpus-id"]
            if query_id == doc_id:
                continue
            similar.setdefault(query_id, set()).add(doc_id)
            similar.setdefault(doc_id, set()).add(query_id)
    return {patient_id: sorted(ids) for patient_id, ids in similar.items()}


def load_notes(
    settings: Settings,
    *,
    limit: int | None = None,
    include_similar_patient_ids: bool = True,
) -> Iterator[NoteRecord]:
    """Stream PMC-Patients case-report notes as NoteRecords.

    Pinned to an exact revision (config.py) for RFC 0053 exact-replay discipline —
    and so the ReCDS qrels (eval/) match what was indexed. The declarative twin is
    deploy/pipeline.yaml's sourceRef; this is the imperative reader.
    """
    from datasets import load_dataset

    similar_by_id = (
        load_ppr_similar_patient_ids(settings, split="train")
        if include_similar_patient_ids
        else {}
    )
    ds = load_dataset(
        settings.dataset_repo,
        split=settings.dataset_split,
        revision=settings.dataset_revision,
    )
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        try:
            record = NoteRecord.from_row(row)
            record.similar_patient_ids = similar_by_id.get(record.id, record.similar_patient_ids)
            yield record
        except (KeyError, TypeError):
            # A malformed row is a per-row skip, not a run failure.
            continue
