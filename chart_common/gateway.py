from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from hevlayer import AsyncHevlayer
from pydantic import ValidationError

from .config import Settings

# The facet fields chart snapshots for the clinical rail. Declared declaratively
# in deploy/index.yaml (spec.snapshot.facetFields); materialized imperatively
# here against the shared gateway. age (continuous) is NOT a facet — age_band is.
# `events` is the Gemma cascade's output (functions/classify_events.py) — the
# clinical-event facet that lets "medication discontinued" compose with routing.
# `discontinuation_reason` is the headline event's structured "why" (a rail
# facet once the cascade backfills); `has_med_discontinuation` is a boolean the
# UI uses for classifier coverage (true+false counts = classified rows), not a
# rendered facet section.
FACET_FIELDS = [
    "specialty", "age_band", "diagnosis_category", "gender", "events",
    "discontinuation_reason", "has_med_discontinuation",
]
SNAPSHOT_IN_PROGRESS = {"queued", "pending", "running"}

# Explicit schema for the columns the backing store can't (or shouldn't) infer.
#  - text: the FTS + fuzzy field the Auto router ranks over (RFC 0022/0057) —
#    the keyword/fuzzy route for drug names, codes, abbreviations, and typos.
#  - the clinical facets are []string / string; pinned so the rail is stable
#    regardless of which row is seen first, and regardless of whether the
#    enrichment UDFs (functions/) have run yet.
#  - phi_flag: operational (scan-phi writeback), not a search facet.
#  - filterable only where a filter is actually planned (the rail facets, the
#    cascade booleans, phi_flag, and the `events Eq null` backfill claim).
#    Display/deep-link/plumbing columns (title, pmid, source_url, age, and the
#    chunk-model columns chunk_id/chunk_index/patient_uid) are pinned
#    filterable=False so the store doesn't index-by-default what nothing
#    filters on. age filters happen through age_band.
# The vector column infers from the rows; distance_metric is set on the write.
SCHEMA: dict[str, Any] = {
    "text": {"type": "string", "full_text_search": True, "fuzzy": True},
    "title": {"type": "string", "filterable": False},
    "source_url": {"type": "string", "filterable": False},
    "pmid": {"type": "string", "filterable": False},
    "age": {"type": "int", "filterable": False},
    "age_band": {"type": "string"},
    "gender": {"type": "string"},
    # Chunk-model plumbing (RFC 0056): identity/ordering for the embed pipeline,
    # never a filter surface.
    "chunk_id": {"type": "string", "filterable": False},
    "chunk_index": {"type": "int", "filterable": False},
    "patient_uid": {"type": "string", "filterable": False},
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
    "events_v2": {"type": "[]string"},
    "event_groups_v2": {"type": "[]string"},
    # Turbopuffer has no nested-object attribute type/value. These two maps are
    # canonical compact JSON strings at the write boundary; current search and
    # facet paths use events_v2/event_groups_v2 instead of filtering the maps.
    "event_confidence_v2": {"type": "string", "filterable": False},
    "event_spans_v2": {"type": "string", "filterable": False},
    "has_treatment_change_v2": {"type": "bool"},
    "has_treatment_response_v2": {"type": "bool"},
    "has_complication_v2": {"type": "bool"},
    "has_care_transition_v2": {"type": "bool"},
    # Native PMC-Patients labels — power "find similar patients" and the
    # patient→article second act; also the ReCDS qrels for eval/.
    "similar_patient_ids": {"type": "[]string", "filterable": False},
    "relevant_article_pmids": {"type": "[]string", "filterable": False},
}


# The late-interaction sibling namespace (Turbopuffer private beta, the
# rank_by ["tokens","ANN",[[...]]] surface RFC 0089 documents for hev search).
# Same display/native attributes as SCHEMA, but the ranked column is a token
# BAG — [][N]f32, one vector per document token, MaxSim-scored — instead of the
# single Arctic vector. No FTS/facet indexing here: this namespace exists to
# measure the token-bag write bill and rank the LI eval arm, not to re-run the
# router. N comes from the LI model (config.LI_EMBED_DIM).
def li_schema(dim: int) -> dict[str, Any]:
    return {
        "text": {"type": "string", "filterable": False},
        "title": {"type": "string", "filterable": False},
        "source_url": {"type": "string", "filterable": False},
        "pmid": {"type": "string", "filterable": False},
        "age": {"type": "int", "filterable": False},
        "age_band": {"type": "string"},
        "gender": {"type": "string"},
        "similar_patient_ids": {"type": "[]string", "filterable": False},
        "relevant_article_pmids": {"type": "[]string", "filterable": False},
        "tokens": {"type": f"[][{dim}]f32", "ann": {"late_interaction": True}},
    }


def billing_bytes_written(resp: Any) -> int:
    """billable_logical_bytes_written from a write response (model or dict) —
    the write-amplification instrument. 0 when the backend didn't echo billing."""
    billing = _read(resp, "billing")
    value = _read(billing, "billable_logical_bytes_written", 0) if billing else 0
    return int(value or 0)


async def write_li_notes(
    layer: AsyncHevlayer, namespace: str, rows: list[dict], *, dim: int
) -> Any:
    """Upsert note rows carrying `tokens` bags into the LI namespace. Turbopuffer
    requires distance_metric at the top level of the write (not inside the
    attribute's ann config); cosine matches the ColBERT family's normalized
    token vectors."""
    body = {
        "upsert_rows": rows,
        "distance_metric": "cosine_distance",
        "schema": li_schema(dim),
    }
    return await layer.write_namespace(namespace, body)


def make_client(settings: Settings, *, timeout: float | None = None) -> AsyncHevlayer:
    """Gateway client with the standard timeout, or an explicit override for
    calls whose server-side budget exceeds it (the Agent reasoning loop)."""
    require_gateway_key(settings)
    return AsyncHevlayer(
        api_key=settings.api_key,
        base_url=settings.gateway_url,
        timeout=settings.http_timeout_seconds if timeout is None else timeout,
    )


def require_gateway_key(settings: Settings) -> None:
    if not getattr(settings, "api_key", None):
        raise SystemExit(
            "No gateway key. Set LAYER_GATEWAY_API_KEY in .env — it is the "
            "Layer inbound key scoped to chart-notes. "
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
    body = {
        "upsert_rows": rows,
        "distance_metric": "cosine_distance",
        "schema": SCHEMA,
    }
    try:
        return await layer.write_namespace(namespace, body)
    except ValidationError as exc:
        missing = {
            ".".join(str(part) for part in error["loc"])
            for error in exc.errors()
            if error.get("type") == "missing"
        }
        if missing == {"message", "rows_affected", "billing"}:
            # kind=search currently returns {"status":"OK"} for writes, while
            # the generated Python client still expects the Turbopuffer write
            # shape. The write already succeeded server-side; keep indexing and
            # track the client mismatch in hev/layer#137.
            return {"status": "OK", "client_parse_warning": "hev/layer#137"}
        raise


async def materialize_facet_snapshots(
    layer: AsyncHevlayer, namespace: str, *, fields: list[str] = FACET_FIELDS, timeout: float = 180.0
) -> None:
    """Materialize each facet histogram and wait for it to land — the imperative
    twin of deploy/index.yaml's snapshot.facetFields auto-writer. Used by the
    indexer because chart doesn't own the Index CR on the shared gateway."""
    for field_name in fields:
        job = await layer.create_snapshot(
            namespace, {"field": field_name, "source": "origin", "page_size": 500}
        )
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
    """The Scans API selector that defines a search's match set, following the
    gateway's own routing decision:
      - keyword/fused query → an `fts` selector over the routed text field — the
        closest scan selector to the router's lexical legs. A `hybrid_text`
        selector (BM25 + per-token fuzzy, RFC 0057) would mirror the fused route
        exactly and count the typo-surfaced matches too — and chart-notes lives
        on Turbopuffer today (the kind=search cutover was never applied), where
        it is supported; `fts` is a conservative leftover from when the store
        was believed cut over to kind=search, which rejects it (hev/layer#141).
        Until it moves, fuzzy-surfaced hits are approximated by their exact
        lexical terms;
      - semantic query → an `ann` ball of `radius` (cosine distance) around the
        query vector, since semantic relevance has no exact lexical match set.
        Approximate by ANN recall.
    A request carries at most one ranked selector; vector+radius wins over query.
    The caller supplies vector+radius only for the semantic route, so a query
    without them is the keyword/fused route and counts via `fts`."""
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


def unnest_array_facets(values: list[dict]) -> list[dict]:
    """Explode serialized-array facet buckets into per-element counts.

    Snapshot histograms over a `[]string` column currently bucket by the whole
    JSON-serialized array ('["a","b"]' n=7) instead of per element (hev/layer#151
    — the live values-mode scan already unnests, the snapshot writer doesn't). A
    doc counts once per element it carries, so element count = Σ n over the
    buckets containing it: exploding here is exact, not an approximation. Scalar
    values pass through (merging with exploded elements, so a post-fix snapshot
    is a no-op) and empty arrays contribute nothing."""
    counts: dict[str, int] = {}
    for entry in values:
        value, count = entry.get("value"), entry.get("count") or 0
        elements = [value]
        if isinstance(value, str) and value.startswith("["):
            try:
                parsed = json.loads(value)
            except ValueError:
                parsed = None
            if isinstance(parsed, list):
                elements = [e for e in parsed if isinstance(e, str)]
        for element in elements:
            counts[element] = counts.get(element, 0) + count
    return [{"value": v, "count": n} for v, n in counts.items()]


async def latest_facets_many(
    layer: AsyncHevlayer,
    namespace: str,
    *,
    fields: list[str],
    limit: int = 14,
    history_limit: int = 20,
) -> dict[str, tuple[list[dict] | None, dict | None]]:
    """Newest stored snapshot values for several fields in ONE history walk —
    each snapshot body is fetched at most once and every wanted field it carries
    is taken from it (the same loop the Worker backend's facets() runs). The
    per-field spelling walks history once per field, which is what made the bare
    corpus rail slow. Fields absent from all of history map to (None, None)."""
    out: dict[str, tuple[list[dict] | None, dict | None]] = {f: (None, None) for f in fields}
    wanted = set(fields)
    history = await layer.list_namespace_history(namespace, limit=history_limit)
    for snapshot in history or []:
        if not wanted:
            break
        snapshot_sha = _read(snapshot, "sha")
        if not snapshot_sha:
            continue
        body = await layer.get_namespace_snapshot(namespace, snapshot_sha)
        provenance = {
            "sha": _read(body, "sha") or snapshot_sha,
            "watermark_ms": _read(body, "watermark_ms"),
            "row_count": _read(body, "row_count"),
        }
        for column in _read(body, "fields", []) or []:
            name = _read(column, "name")
            if name not in wanted:
                continue
            raw = [{"value": _read(v, "v"), "count": _read(v, "n", 0)} for v in _read(column, "values", [])]
            facets = sorted(unnest_array_facets(raw), key=lambda v: v["count"], reverse=True)[:limit]
            out[name] = (facets, provenance)
            wanted.discard(name)
    return out


async def latest_facets(
    layer: AsyncHevlayer, namespace: str, *, field: str, limit: int = 14, history_limit: int = 20
) -> tuple[list[dict] | None, dict | None]:
    """Read the newest stored facet snapshot for `field` (value/count + sha
    provenance), or (None, None) when no snapshot body exists yet."""
    result = await latest_facets_many(
        layer, namespace, fields=[field], limit=limit, history_limit=history_limit
    )
    return result[field]
