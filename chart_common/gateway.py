from __future__ import annotations

import asyncio
import time
from typing import Any

from hevlayer import AsyncHevlayer

from .config import Settings

# The facet fields chart snapshots for the clinical rail. Declared declaratively
# in deploy/index.yaml (spec.snapshot.facetFields); materialized imperatively
# here against the shared gateway. age (continuous) is NOT a facet — age_band is.
# `events` is the Gemma cascade's output (functions/classify_events.py) — the
# clinical-event facet that lets "medication discontinued" compose with routing.
FACET_FIELDS = ["specialty", "age_band", "diagnosis_category", "gender", "events"]

# Explicit schema for the columns tpuf can't (or shouldn't) infer.
#  - text: the FTS + fuzzy field the Auto router ranks over (RFC 0022/0057) —
#    the keyword/fuzzy route for drug names, codes, abbreviations, and typos.
#  - title / source_url: display + deep-link; source_url non-filterable.
#  - the clinical facets are []string / string; pinned so the rail is stable
#    regardless of which row is seen first, and regardless of whether the
#    enrichment UDFs (functions/) have run yet.
#  - phi_flag: operational (scan-phi writeback), not a search facet.
# The vector column infers from the rows; distance_metric is set on the write.
SCHEMA: dict[str, Any] = {
    "text": {"type": "string", "full_text_search": True, "fuzzy": True},
    "title": {"type": "string"},
    "source_url": {"type": "string", "filterable": False},
    "pmid": {"type": "string"},
    "age": {"type": "int"},
    "age_band": {"type": "string"},
    "gender": {"type": "string"},
    # UDF writeback (functions/) — declared up front so the facet columns are
    # stable before/after enrichment runs.
    "specialty": {"type": "string"},
    "chief_complaint": {"type": "string"},
    "diagnosis_category": {"type": "string"},
    "body_system": {"type": "string"},
    "phi_flag": {"type": "bool"},
    # Gemma clinical-event cascade writeback (functions/classify_events.py).
    # `events` is the typed event list (a []string facet); the booleans are the
    # high-value filters — medication discontinuation is the headline event.
    "events": {"type": "[]string"},
    "has_med_discontinuation": {"type": "bool"},
    "has_adverse_event": {"type": "bool"},
    "discontinuation_reason": {"type": "string"},
    # Native PMC-Patients labels — power "find similar patients" and the
    # patient→article second act; also the ReCDS qrels for eval/.
    "similar_patient_ids": {"type": "[]string", "filterable": False},
    "relevant_article_pmids": {"type": "[]string", "filterable": False},
}


def make_client(settings: Settings) -> AsyncHevlayer:
    if not settings.api_key:
        raise SystemExit(
            "No gateway key. Set LAYER_GATEWAY_API_KEY in .env — it's the upstream "
            "Turbopuffer key (1Password: layer-turbopuffer / mesh-staging). "
            "Or run with --dry-run to skip the gateway."
        )
    return AsyncHevlayer(
        api_key=settings.api_key,
        base_url=settings.gateway_url,
        timeout=settings.http_timeout_seconds,
    )


async def close_client(layer: AsyncHevlayer) -> None:
    for name in ("aclose", "close"):
        closer = getattr(layer, name, None)
        if closer is None:
            continue
        result = closer()
        if hasattr(result, "__await__"):
            await result
        return


async def write_notes(layer: AsyncHevlayer, namespace: str, rows: list[dict]) -> Any:
    """Upsert a batch of note rows. Schema is sent inline (idempotent) so the
    text field is FTS+fuzzy-indexed and the vector column is cosine."""
    return await layer.write_namespace(
        namespace,
        {
            "upsert_rows": rows,
            "distance_metric": "cosine_distance",
            "schema": SCHEMA,
        },
    )


async def materialize_facet_snapshots(
    layer: AsyncHevlayer, namespace: str, *, fields: list[str] = FACET_FIELDS, timeout: float = 180.0
) -> None:
    """Materialize each facet histogram and wait for it to land — the imperative
    twin of deploy/index.yaml's snapshot.facetFields auto-writer. Used by the
    indexer because chart doesn't own the Index CR on the shared gateway."""
    for field_name in fields:
        job = await layer.create_snapshot(namespace, {"field": field_name, "source": "origin"})
        start = time.monotonic()
        while job.status == "running":
            if time.monotonic() - start > timeout:
                raise TimeoutError(f"snapshot {field_name} still running after {timeout:.0f}s")
            await asyncio.sleep(1.0)
            job = await layer.get_snapshot_job(namespace, job.id)
        if job.status != "completed":
            raise RuntimeError(f"snapshot {field_name} {job.status}: {job.error or 'no detail'}")


async def latest_facets(
    layer: AsyncHevlayer, namespace: str, *, field: str, limit: int = 14
) -> tuple[list[dict] | None, dict | None]:
    """Read the newest stored facet snapshot for `field` (value/count + sha
    provenance), or (None, None) when no snapshot body exists yet."""
    history = await layer.list_namespace_history(namespace, limit=1)
    if not history:
        return None, None
    body = await layer.get_namespace_snapshot(namespace, history[0].sha)
    column = next((f for f in body.fields if f.name == field), None)
    if column is None:
        return None, None
    top = sorted(column.values, key=lambda v: v.n, reverse=True)[:limit]
    facets = [{"value": v.v, "count": v.n} for v in top]
    provenance = {"sha": body.sha, "watermark_ms": body.watermark_ms, "row_count": body.row_count}
    return facets, provenance
