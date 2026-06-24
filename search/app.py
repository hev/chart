"""FastAPI dev search backend — the CPU-friendly local twin of the Cloudflare
Worker prod backend (src/worker.js). Both inject the gateway key SERVER-SIDE and
return the gateway's own routing echo; the static UI renders it. RFC 0076 § UX
("two backends, one UI", reimplement-nothing).

SKELETON at the query seam (TODO): the gateway call shape mirrors shelf/search.
The load-bearing detail is the Arctic asymmetric query prefix on the embed path.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from chart_common.config import Settings
from chart_common.embed import Embedder
from chart_common.gateway import FACET_FIELDS, close_client, latest_facets, make_client

STATIC = Path(__file__).resolve().parent.parent / "web" / "static"


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


@app.get("/api/search")
async def search(q: str, top_k: int = 20) -> JSONResponse:
    """Auto-routed search. The gateway routes from the text (RFC 0044); on a
    vectorless semantic/fused route it returns the routing DECISION (executed:
    false), and we embed (with the Arctic query prefix) and re-issue with the
    route forced — the demo's one hop. The response carries the gateway's
    `routing`/`hybrid` echo, which the UI renders in the Routing inspector.

    TODO: call app.state.layer.query_namespace with rank_by=[text-field, "Auto", q],
    inspect the `routing` echo, and on a deferral re-issue with
    app.state.embedder.embed_query(q) + the forced route.
    """
    _settings: Settings = app.state.settings
    raise NotImplementedError("issue the Auto-routed query + the embed-on-deferral hop")


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


app.mount("/", StaticFiles(directory=str(STATIC), html=True), name="static")
