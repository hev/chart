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
SNAPSHOT_IN_PROGRESS = {"queued", "pending", "running"}

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
    require_gateway_key(settings)
    return AsyncHevlayer(
        api_key=settings.api_key,
        base_url=settings.gateway_url,
        timeout=settings.http_timeout_seconds,
    )


def require_gateway_key(settings: Settings) -> None:
    if not getattr(settings, "api_key", None):
        raise SystemExit(
            "No gateway key. Set LAYER_GATEWAY_API_KEY in .env — it's the upstream "
            "Turbopuffer key (1Password: layer-turbopuffer / mesh-staging). "
            "Or run with --dry-run to skip the gateway."
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
        while _read(job, "status") in SNAPSHOT_IN_PROGRESS:
            if time.monotonic() - start > timeout:
                raise TimeoutError(f"snapshot {field_name} still in progress after {timeout:.0f}s")
            await asyncio.sleep(1.0)
            job = await layer.get_snapshot_job(namespace, _read(job, "id"))
        if _read(job, "status") != "completed":
            raise RuntimeError(f"snapshot {field_name} {_read(job, 'status')}: {_read(job, 'error') or 'no detail'}")


def _read(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def count_selector(
    *, query: str | None = None, vector: list[float] | None = None, radius: float | None = None
) -> dict[str, Any]:
    """The Scans API selector that defines a search's match set:
      - keyword/fused query → an exact BM25 `fts` predicate over the text field;
      - semantic query → an `ann` ball of `radius` (cosine distance) around the
        query vector, since semantic relevance has no exact lexical match set.
        Approximate by ANN recall.
    A request carries at most one ranked selector; vector+radius wins over query.

    Gap: the `fts` selector is exact BM25 only — it cannot mirror the `hybrid_text`
    route's per-token fuzzy legs (RFC 0057), so a fuzzy/typo-surfaced query (afib,
    aspirn) the route retrieves counts as zero here. The UI guards against the
    resulting "count < hits shown" contradiction by withholding the live count; the
    real fix is a fuzzy-aware scan selector (see LAYER_IMPROVEMENTS.md)."""
    if vector is not None and radius is not None:
        return {"ann": {"field": "vector", "vector": vector, "radius": radius}}
    if query:
        return {"fts": {"field": "text", "query": query}}
    return {}


async def scan_count(
    layer: AsyncHevlayer,
    namespace: str,
    *,
    selector: dict[str, Any] | None = None,
    filters: Any | None = None,
    timeout_seconds: int = 25,
) -> int | None:
    """Count of rows matching the current search (Scans API, count mode — RFC
    docs/api/scans). `selector` is the fts/ann match set; `filters` is ANDed on.
    Slower than top-k (origin scatter/gather), so the UI calls this async, after
    results render. Returns None on any failure."""
    body: dict[str, Any] = {"mode": "count", "source": "auto", "timeout_seconds": timeout_seconds}
    if filters is not None:
        body["filters"] = filters
    body.update(selector or {})
    try:
        resp = await layer.create_scan(namespace, body)
    except Exception:
        return None
    count = _read(resp, "count")
    return int(count) if isinstance(count, int) else None


async def scan_facet_values(
    layer: AsyncHevlayer,
    namespace: str,
    *,
    field: str,
    selector: dict[str, Any] | None = None,
    filters: Any | None = None,
    limit: int = 24,
    timeout: float = 25.0,
) -> list[dict] | None:
    """The live per-search facet histogram for one field (Scans API, values mode):
    the distinct values of `field` over the rows the `selector` picks, each with its
    document count (`v`/`n`, the same vocabulary the snapshot rail uses). Returns
    None on failure so the caller can fall back to the corpus snapshot."""
    body: dict[str, Any] = {"mode": "values", "field": field, "source": "auto"}
    if filters is not None:
        body["filters"] = filters
    body.update(selector or {})
    try:
        job = await layer.scan(namespace, body, timeout=timeout)
        if _read(job, "status") != "completed":
            return None
        results = await layer.get_scan_results(namespace, _read(job, "id"), limit=limit)
    except Exception:
        return None
    values = _read(results, "values", []) or []
    return [{"value": _read(v, "v"), "count": _read(v, "n")} for v in values]


async def latest_facets(
    layer: AsyncHevlayer, namespace: str, *, field: str, limit: int = 14, history_limit: int = 20
) -> tuple[list[dict] | None, dict | None]:
    """Read the newest stored facet snapshot for `field` (value/count + sha
    provenance), or (None, None) when no snapshot body exists yet."""
    history = await layer.list_namespace_history(namespace, limit=history_limit)
    for snapshot in history or []:
        snapshot_sha = _read(snapshot, "sha")
        if not snapshot_sha:
            continue
        body = await layer.get_namespace_snapshot(namespace, snapshot_sha)
        column = next((f for f in _read(body, "fields", []) if _read(f, "name") == field), None)
        if column is None:
            continue
        top = sorted(_read(column, "values", []), key=lambda v: _read(v, "n", 0), reverse=True)[:limit]
        facets = [{"value": _read(v, "v"), "count": _read(v, "n")} for v in top]
        provenance = {
            "sha": _read(body, "sha") or snapshot_sha,
            "watermark_ms": _read(body, "watermark_ms"),
            "row_count": _read(body, "row_count"),
        }
        return facets, provenance
    return None, None
