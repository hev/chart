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

import httpx
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
    latest_facets_many,
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
    "body_system",
    "chief_complaint",
    "events",
    "has_med_discontinuation",
    "has_adverse_event",
    "discontinuation_reason",
    "similar_patient_ids",
]
TRANSIENT = {502, 503, 504}

# The agentic path runs the configured Agent's whole reasoning loop in one
# request — its server-side budget (deploy/agent.yaml budget.deadlineMs) is 60s,
# so the standard 60s client timeout can cut a legitimate full-deadline run.
# 2× the deadline leaves room for grading overhead and one transparent retry.
# Mirrored by AGENT_TIMEOUT_MS in src/worker.js and web/static/index.html.
AGENT_TIMEOUT_SECONDS = 120.0

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
    # The LI token-bag embedder loads lazily on the first ?li=1 query — it is a
    # try-out mode (Turbopuffer late-interaction beta), not the hot path, and
    # its ONNX download shouldn't gate readiness.
    app.state.li_embedder = None
    app.state.layer = make_client(app.state.settings)
    app.state.layer_agent = make_client(app.state.settings, timeout=AGENT_TIMEOUT_SECONDS)
    try:
        yield
    finally:
        await close_client(app.state.layer)
        await close_client(app.state.layer_agent)


app = FastAPI(title="chart — clinical-notes routing demo", lifespan=lifespan)


@app.middleware("http")
async def cache_policy(request, call_next):
    """Without Cache-Control, browsers heuristically cache off Last-Modified
    (~10% of the file's age) — after a deploy, users keep the stale UI for up
    to ~an hour with no revalidation. Static pages revalidate every time
    (no-cache still yields cheap 304s off StaticFiles' validators); API
    responses are per-search and never worth caching."""
    response = await call_next(request)
    if "cache-control" not in response.headers:
        is_api = request.url.path.startswith("/api/")
        response.headers["cache-control"] = "no-store" if is_api else "no-cache"
    return response


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


async def _run_query(
    body: QueryRequest, *, require_routing: bool = False, namespace: str | None = None
) -> dict:
    last_detail = "unknown error"
    for attempt in range(3):
        start = time.perf_counter()
        try:
            resp = await app.state.layer.query_namespace(
                namespace or app.state.settings.namespace, body
            )
        except HevlayerError as exc:
            if exc.status_code not in TRANSIENT:
                raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
            last_detail = exc.message
        except httpx.HTTPError as exc:
            # Transport-level failure (gateway restarting / unreachable — connect
            # error or timeout). Transient like a 502: retry, then surface a clean
            # 502 instead of a 500 traceback.
            last_detail = f"gateway unreachable: {exc.__class__.__name__}: {exc}"
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


# The LI sibling namespace carries only the native note attributes (no UDF
# writeback columns), so both the include list and the filterable facets are
# the native subset. Requesting a column the namespace doesn't have is a
# store error, not a silent null.
LI_INCLUDE = [
    "id", "title", "text", "pmid", "source_url",
    "age", "age_band", "gender", "similar_patient_ids",
]
LI_FILTERABLE = {"age_band", "gender"}


def _li_embedder():
    if app.state.li_embedder is None:
        from chart_common.embed import LateInteractionEmbedder

        app.state.li_embedder = LateInteractionEmbedder(app.state.settings.li_model)
    return app.state.li_embedder


async def _hydrate_li_rows(rows: list[dict], *, namespace: str) -> list[dict]:
    """Turbopuffer's late-interaction beta returns most ranked rows as bare
    $dist+id, dropping the requested attributes (observed 9/10 bare on direct
    upstream queries; single-vector ranked queries are unaffected — beta bug,
    reported upstream). Hydrate the display attributes with a follow-up filter
    query and merge by id, keeping the MaxSim order."""
    missing = [r.get("id") for r in rows if r.get("id") and not r.get("text")]
    if not missing:
        return rows
    resp = await app.state.layer.query_namespace(
        namespace,
        QueryRequest(
            rank_by=["id", "asc"],
            filters=["id", "In", missing],
            top_k=len(missing),
            include_attributes=LI_INCLUDE,
        ),
    )
    by_id = {row.get("id"): row for row in _rows(resp.rows) if row.get("id")}
    return [
        {**by_id[r["id"]], **r} if r.get("id") in by_id else r
        for r in rows
    ]


def li_body(query: str, *, top_k: int = 20) -> QueryRequest:
    """Late-interaction search (Turbopuffer private beta): rank the token-bag
    column by MaxSim — rank_by ["tokens","ANN",[<query token vectors>]]. A
    direct ANN rank, not an Auto route, so the gateway returns no routing echo;
    the caller synthesizes one for the badge."""
    return QueryRequest(
        rank_by=["tokens", "ANN", _li_embedder().embed_query(query)],
        top_k=max(1, min(top_k, 50)),
        include_attributes=LI_INCLUDE,
    )


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


async def _run_agent_query(query: str, *, top_k: int = 20) -> dict:
    """Agentic search (docs/api/agents): POST /v2/agents/{name}/query with the
    query, its Arctic embedding (BYO — Layer never embeds query text; the one
    vector serves every planned semantic leg), and top_k. The configured Agent
    (deploy/agent.yaml) owns the model, fan-out, and fusion; with provenance on,
    rows carry $agent scores and the response echoes the planned variants. The
    request carries no filters — the agent infers its own (visible in the echo)."""
    vector = app.state.embedder.embed_query(query)
    if len(vector) != EMBED_DIM:
        raise HTTPException(status_code=502, detail=f"embedder returned a non-{EMBED_DIM}-d vector")
    body = {"query": query, "vector": vector, "top_k": max(1, min(top_k, 50))}
    last_detail = "unknown error"
    for attempt in range(3):
        start = time.perf_counter()
        try:
            resp = await app.state.layer_agent.query_agent(app.state.settings.agent_name, body)
        except HevlayerError as exc:
            if exc.status_code not in TRANSIENT:
                raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
            last_detail = exc.message
        except httpx.HTTPError as exc:
            last_detail = f"gateway unreachable: {exc.__class__.__name__}: {exc}"
        else:
            return {
                "rows": _rows(resp.rows),
                "routing": None,
                "hybrid": None,
                "agent": _dump(resp.agent),
                "merge": _dump(resp.merge),
                "took_ms": round((time.perf_counter() - start) * 1000),
            }
        if attempt < 2:
            await asyncio.sleep(0.4 * (attempt + 1))
    raise HTTPException(status_code=502, detail=f"agent error after retries: {last_detail}")


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
async def search(
    q: str = "",
    top_k: int = 20,
    f: list[str] = Query(default=[]),
    agentic: int = 0,
    li: int = 0,
) -> JSONResponse:
    """Auto-routed search, optionally narrowed by the facet rail. Embed up front
    and hand the vector to Auto in the 4th tuple slot, matching shelf's live
    implementation; the gateway makes and echoes the route decision and applies any
    `filters`. An empty box with active filters becomes a cohort browse instead.
    With `agentic=1`, the query runs the configured reasoning loop instead
    (POST /v2/agents/{name}/query) — facet filters don't apply there; the agent
    infers its own, echoed in the response."""
    query = q.strip()
    filters = parse_filters(f)
    if agentic and query:
        result = await _run_agent_query(query, top_k=top_k)
        result["query"] = query
        return JSONResponse(result)
    if li and query:
        # Only the native facets exist in the LI namespace; drop the rest so a
        # stale rail selection doesn't turn into a store error.
        grouped = {k: v for k, v in group_filters(f).items() if k in LI_FILTERABLE}
        body = li_body(query, top_k=top_k)
        body.filters = build_filters(grouped)
        result = await _run_query(body, namespace=app.state.settings.li_namespace)
        result["rows"] = await _hydrate_li_rows(
            result["rows"], namespace=app.state.settings.li_namespace
        )
        result["routing"] = {
            "route": "late_interaction",
            "policy": "forced",
            "reason": "MaxSim over ColBERT token bags — Turbopuffer late-interaction beta",
        }
        result["query"] = query
        return JSONResponse(result)
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
async def facets(q: str = "", f: list[str] = Query(default=[]), route: str = "") -> JSONResponse:
    """The clinical facet rail, in two modes (the same contract as shelf's rail).

    With `q` (or active filters), per-search counts scoped by a values scan —
    delegates to the live-counts path below. Without either, the corpus-wide
    histogram from the latest snapshot bodies (counts + sha provenance), not a
    tally of returned rows. Either mode degrades to the other's absence
    gracefully."""
    if q.strip() or f:
        return await facet_counts(q=q, f=f, route=route)
    out = {}
    try:
        # One history walk for every rail field — each snapshot body fetched
        # once — so the corpus rail is up fast on a bare page load.
        results = await latest_facets_many(
            app.state.layer, app.state.settings.namespace, fields=FACET_FIELDS
        )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502, detail=f"gateway unreachable: {exc.__class__.__name__}: {exc}"
        ) from exc
    for field_name in FACET_FIELDS:
        values, provenance = results[field_name]
        if values is not None:
            out[field_name] = {"values": values, "provenance": provenance}
    return JSONResponse(out)


@app.get("/api/facet-counts")
async def facet_counts(q: str = "", f: list[str] = Query(default=[]), route: str = "") -> JSONResponse:
    """Live facet counts for the *current* search — the Scans API showcase. The
    matching total is a count scan; each facet's breakdown is a values scan over the
    same selector. Scans are slower than top-k (origin scatter/gather), so the UI
    calls this asynchronously, after the ranked results are already on screen.
    Also reachable as `/api/facets?q=` (the shelf-shaped spelling).

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
            "li_namespace": app.state.settings.li_namespace,
            "li_model": app.state.settings.li_model,
        }
    )


@app.get("/healthz")
async def healthz() -> JSONResponse:
    # Liveness/readiness for the in-cluster Deployment and the ALB target group.
    # The embedder loads in `lifespan` before uvicorn accepts traffic, so a 200
    # here means the Arctic model is resident and the gateway client is up.
    return JSONResponse({"ok": True})


app.mount("/", StaticFiles(directory=str(STATIC), html=True), name="static")
