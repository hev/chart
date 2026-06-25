"""FastAPI dev search backend — the CPU-friendly local twin of the Cloudflare
Worker prod backend (src/worker.js). Both inject the gateway key SERVER-SIDE and
return the gateway's own routing echo; the static UI renders it. RFC 0076 § UX
("two backends, one UI", reimplement-nothing).
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from hevlayer import QueryRequest
from hevlayer.client import HevlayerError

from chart_common.config import EMBED_DIM, Settings
from chart_common.embed import Embedder
from chart_common.gateway import (
    FACET_FIELDS,
    close_client,
    count_selector,
    latest_facets,
    make_client,
    scan_count,
    scan_facet_values,
)

STATIC = Path(__file__).resolve().parent.parent / "web" / "static"
INCLUDE = [
    "id",
    "title",
    "text",
    "pmid",
    "source_url",
    "age",
    "age_band",
    "gender",
    "specialty",
    "diagnosis_category",
    "events",
    "has_med_discontinuation",
    "discontinuation_reason",
    "similar_patient_ids",
]
TRANSIENT = {502, 503, 504}

# The clickable facet rail narrows the search by attaching a turbolisp `filters`
# clause to the same routed query (RFC 0076 § Search experience — "filterable by
# specialty / age_band / diagnosis_category"). Scalars filter by equality; `events`
# is a []string, so it filters with the turbopuffer array `Contains`/`ContainsAny`.
SCALAR_FACETS = {"specialty", "age_band", "diagnosis_category", "gender"}
ARRAY_FACETS = {"events"}
FILTERABLE = SCALAR_FACETS | ARRAY_FACETS


def group_filters(raw: list[str] | None) -> dict[str, list[str]]:
    """Group repeated `f=field:value` params by field. Unknown fields are dropped so
    the UI can only filter on the snapshotted facets."""
    grouped: dict[str, list[str]] = {}
    for item in raw or []:
        field, sep, value = item.partition(":")
        value = value.strip()
        if not sep or field not in FILTERABLE or not value:
            continue
        bucket = grouped.setdefault(field, [])
        if value not in bucket:
            bucket.append(value)
    return grouped


def build_filters(grouped: dict[str, list[str]]) -> Any | None:
    """Turn grouped facet selections into one turbolisp filter. Scalars filter by
    equality (`Eq`/`In`); `events` is a []string, so it uses the array
    `Contains`/`ContainsAny` ops."""
    clauses: list = []
    for field, values in grouped.items():
        if field in ARRAY_FACETS:
            clauses.append([field, "Contains", values[0]] if len(values) == 1 else [field, "ContainsAny", values])
        else:
            clauses.append([field, "Eq", values[0]] if len(values) == 1 else [field, "In", values])
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else ["And", clauses]


def parse_filters(raw: list[str] | None) -> Any | None:
    return build_filters(group_filters(raw))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = Settings()
    app.state.embedder = Embedder(app.state.settings.embed_model)
    app.state.layer = make_client(app.state.settings)
    try:
        yield
    finally:
        await close_client(app.state.layer)


app = FastAPI(title="chart — clinical-notes routing demo", lifespan=lifespan)


def _dump(value):
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def _routing_route(routing) -> str | None:
    dumped = _dump(routing) or {}
    if isinstance(dumped, dict):
        return dumped.get("route") or dumped.get("strategy")
    return None


def _rows(rows):
    return [_dump(row) for row in rows or []]


async def _run_query(body: QueryRequest, *, require_routing: bool = False) -> dict:
    last_detail = "unknown error"
    for attempt in range(3):
        start = time.perf_counter()
        try:
            resp = await app.state.layer.query_namespace(app.state.settings.namespace, body)
        except HevlayerError as exc:
            if exc.status_code not in TRANSIENT:
                raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
            last_detail = exc.message
        else:
            if require_routing and not _routing_route(resp.routing):
                raise HTTPException(status_code=502, detail="gateway response did not include a routing decision")
            return {
                "rows": _rows(resp.rows),
                "routing": _dump(resp.routing),
                "hybrid": _dump(resp.hybrid),
                "took_ms": round((time.perf_counter() - start) * 1000),
            }
        if attempt < 2:
            await asyncio.sleep(0.4 * (attempt + 1))
    raise HTTPException(status_code=502, detail=f"gateway error after retries: {last_detail}")


def search_body(query: str, *, top_k: int = 20) -> QueryRequest:
    vector = app.state.embedder.embed_query(query)
    if len(vector) != EMBED_DIM:
        raise HTTPException(status_code=502, detail=f"embedder returned a non-{EMBED_DIM}-d vector")
    return QueryRequest(
        rank_by=["text", "Auto", query, {"vector": vector}],
        top_k=max(1, min(top_k, 50)),
        include_attributes=INCLUDE,
        include_leg_breakdown=True,
    )


def similar_body(patient_id: str, *, top_k: int = 20) -> QueryRequest:
    return QueryRequest(
        nearest_to_id=[patient_id],
        top_k=max(1, min(top_k, 50)),
        include_attributes=INCLUDE,
        include_leg_breakdown=True,
    )


def browse_body(filters: Any, *, top_k: int = 20) -> QueryRequest:
    """Filter-only listing for when a facet is clicked with an empty box — a plain
    ordered scan, no query, so the gateway returns no routing decision (honest:
    there is nothing to route). Ranked by id for a stable cohort view."""
    return QueryRequest(
        rank_by=["id", "asc"],
        filters=filters,
        top_k=max(1, min(top_k, 50)),
        include_attributes=INCLUDE,
    )


@app.get("/api/search")
async def search(q: str = "", top_k: int = 20, f: list[str] = Query(default=[])) -> JSONResponse:
    """Auto-routed search, optionally narrowed by the facet rail. Embed up front
    and hand the vector to Auto in the 4th tuple slot, matching shelf's live
    implementation; the gateway makes and echoes the route decision and applies any
    `filters`. An empty box with active filters becomes a cohort browse instead."""
    query = q.strip()
    filters = parse_filters(f)
    if not query:
        if filters is None:
            return JSONResponse({"rows": [], "routing": None, "hybrid": None, "query": ""})
        result = await _run_query(browse_body(filters, top_k=top_k))
        result["query"] = ""
        return JSONResponse(result)
    body = search_body(query, top_k=top_k)
    body.filters = filters
    result = await _run_query(body, require_routing=True)
    result["query"] = query
    return JSONResponse(result)


@app.get("/api/similar/{patient_id}")
async def similar(patient_id: str, top_k: int = 12) -> JSONResponse:
    patient_id = patient_id.strip()
    if not patient_id:
        return JSONResponse({"rows": [], "routing": None, "hybrid": None, "patient_id": ""})
    result = await _run_query(similar_body(patient_id, top_k=top_k))
    result["patient_id"] = patient_id
    return JSONResponse(result)


@app.get("/api/facets")
async def facets() -> JSONResponse:
    """The clinical facet rail, read from the latest snapshot bodies (corpus-wide
    counts + sha provenance), not a tally of returned rows."""
    out = {}
    for field_name in FACET_FIELDS:
        values, provenance = await latest_facets(
            app.state.layer, app.state.settings.namespace, field=field_name
        )
        if values is not None:
            out[field_name] = {"values": values, "provenance": provenance}
    return JSONResponse(out)


@app.get("/api/facet-counts")
async def facet_counts(q: str = "", f: list[str] = Query(default=[]), route: str = "") -> JSONResponse:
    """Live facet counts for the *current* search — the Scans API showcase. The
    matching total is a count scan; each facet's breakdown is a values scan over the
    same selector. Scans are slower than top-k (origin scatter/gather), so the UI
    calls this asynchronously, after the ranked results are already on screen.

    The selector follows the gateway's own routing decision (passed as `route`): a
    `semantic` query counts an `ann` ball of `settings.semantic_radius` around the
    query vector (relevance has no exact lexical match set), keyword/fused counts an
    exact `fts` predicate. ann counts are approximate by ANN recall.

    Drill-down semantics: a field's own active filter is excluded from its own
    breakdown, so the counts show what each alternative value *would* return."""
    query = q.strip()
    grouped = group_filters(f)
    namespace = app.state.settings.namespace
    layer = app.state.layer

    semantic = bool(query) and route == "semantic"
    selector = count_selector(
        query=query or None,
        vector=app.state.embedder.embed_query(query) if semantic else None,
        radius=app.state.settings.semantic_radius if semantic else None,
    )

    async def field_breakdown(field: str) -> tuple[str, list[dict] | None]:
        subset = {name: values for name, values in grouped.items() if name != field}
        values = await scan_facet_values(
            layer, namespace, field=field, selector=selector, filters=build_filters(subset)
        )
        return field, values

    total, *pairs = await asyncio.gather(
        scan_count(layer, namespace, selector=selector, filters=build_filters(grouped)),
        *(field_breakdown(field) for field in FACET_FIELDS),
    )
    fields = {field: values for field, values in pairs if values is not None}
    return JSONResponse({"fields": fields, "total": total, "approximate": semantic, "query": query})


@app.get("/api/config")
async def config() -> JSONResponse:
    return JSONResponse(
        {
            "namespace": app.state.settings.namespace,
            "gateway": app.state.settings.gateway_url.rstrip("/"),
            "field": "text",
        }
    )


app.mount("/", StaticFiles(directory=str(STATIC), html=True), name="static")
