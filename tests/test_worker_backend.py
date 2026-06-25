import re
from pathlib import Path

from chart_common.config import ARCTIC_QUERY_PREFIX, EMBED_DIM
from chart_common.gateway import FACET_FIELDS
from search.app import INCLUDE


WORKER = Path(__file__).resolve().parent.parent / "src" / "worker.js"
WRANGLER_CONFIG = WORKER.parent.parent / "wrangler.jsonc"


def _worker_array(name: str) -> list[str]:
    source = WORKER.read_text()
    match = re.search(rf"const {name} = \[(.*?)\];", source, re.S)
    assert match, f"missing Worker {name}"
    return re.findall(r'"([^"]+)"', match.group(1))


def test_worker_query_and_facet_fields_match_python_backend_contract() -> None:
    assert _worker_array("INCLUDE") == INCLUDE
    assert _worker_array("FACET_FIELDS") == FACET_FIELDS


def test_worker_search_body_matches_gateway_auto_shape() -> None:
    source = WORKER.read_text()

    assert 'rank_by: ["text", "Auto", query, { vector }]' in source
    assert "top_k: boundedTopK(url.searchParams.get(\"top_k\"), 20, 50)" in source
    assert "include_attributes: INCLUDE" in source
    assert "include_leg_breakdown: true" in source


def test_worker_similar_body_uses_nearest_to_id() -> None:
    source = WORKER.read_text()

    assert "nearest_to_id: [id]" in source
    assert "top_k: boundedTopK(url.searchParams.get(\"top_k\"), 12, 50)" in source
    assert "include_attributes: INCLUDE" in source
    assert "include_leg_breakdown: true" in source


def test_worker_facets_scan_recent_snapshot_history() -> None:
    source = WORKER.read_text()

    assert "/history?limit=20" in source
    assert "const wanted = new Set(FACET_FIELDS)" in source
    assert "for (const snapshot of Array.isArray(history) ? history : [])" in source
    assert "sha: body.sha || snapshot.sha" in source
    assert "wanted.delete(field)" in source


def test_worker_requires_server_side_embedding_and_gateway_secrets() -> None:
    source = WORKER.read_text()

    assert "CHART_QUERY_EMBED_URL is required for Worker search" in source
    assert "LAYER_GATEWAY_API_KEY is required" in source
    assert "authorization: `Bearer ${env.LAYER_GATEWAY_API_KEY}`" in source
    assert f"vector.length !== {EMBED_DIM}" in source
    assert f"non-{EMBED_DIM}-d vector" in source


def test_wrangler_config_documents_production_search_secrets_without_plain_vars() -> None:
    config = WRANGLER_CONFIG.read_text()
    vars_block = config.split('"vars": {', 1)[1].split("\n  }", 1)[0]

    assert '"LAYER_GATEWAY_URL":' in vars_block
    assert '"CHART_NAMESPACE":' in vars_block
    assert '"CHART_EMBED_MODEL":' in vars_block
    assert "LAYER_GATEWAY_API_KEY as a Wrangler secret" in vars_block
    assert "CHART_QUERY_EMBED_URL as a Wrangler var or secret" in vars_block
    assert "CHART_QUERY_EMBED_KEY as a Wrangler secret" in vars_block
    assert '"LAYER_GATEWAY_API_KEY":' not in vars_block
    assert '"CHART_QUERY_EMBED_KEY":' not in vars_block


def test_python_and_worker_backends_share_embed_dimension_guard() -> None:
    worker = WORKER.read_text()
    python = (WORKER.parent.parent / "search" / "app.py").read_text()

    assert f"vector.length !== {EMBED_DIM}" in worker
    assert f"len(vector) != EMBED_DIM" in python
    assert f"non-{EMBED_DIM}-d vector" in worker
    assert f"non-{{EMBED_DIM}}-d vector" in python


def test_worker_query_prefix_matches_python_embedder() -> None:
    source = WORKER.read_text()

    assert f'const ARCTIC_QUERY_PREFIX = "{ARCTIC_QUERY_PREFIX}"' in source
    assert "text: ARCTIC_QUERY_PREFIX + query" in source


def test_worker_retries_transient_gateway_query_errors_like_python_backend() -> None:
    source = WORKER.read_text()

    assert "const TRANSIENT = new Set([502, 503, 504])" in source
    assert "for (let attempt = 0; attempt < 3; attempt += 1)" in source
    assert "if (!TRANSIENT.has(error.status) || attempt === 2) throw error" in source
    assert "await sleep(400 * (attempt + 1))" in source


def test_worker_search_requires_gateway_routing_decision_like_python_backend() -> None:
    source = WORKER.read_text()
    python = (WORKER.parent.parent / "search" / "app.py").read_text()

    assert "queryNamespace(env, body, { requireRouting: true });" in source
    assert "options.requireRouting && !route" in source
    assert "gateway response did not include a routing decision" in source
    assert "require_routing=True" in python


def test_worker_and_python_backends_share_facet_filter_and_browse_contract() -> None:
    source = WORKER.read_text()
    python = (WORKER.parent.parent / "search" / "app.py").read_text()

    # Scalars filter by equality/In; the events []string facet uses array ops.
    for backend in (source, python):
        assert '"Eq"' in backend and '"In"' in backend
        assert '"Contains"' in backend and '"ContainsAny"' in backend
    assert 'const SCALAR_FACETS = new Set(["specialty", "age_band", "diagnosis_category", "gender"]);' in source
    assert 'const ARRAY_FACETS = new Set(["events"]);' in source
    assert 'SCALAR_FACETS = {"specialty", "age_band", "diagnosis_category", "gender"}' in python
    assert 'ARRAY_FACETS = {"events"}' in python

    # An empty box with active filters is an honest filter-only cohort browse:
    # ordered scan, no query vector, so the gateway returns no routing decision.
    assert 'rank_by: ["id", "asc"]' in source
    assert 'rank_by=["id", "asc"]' in python


def test_worker_and_python_backends_share_facet_counts_scan_contract() -> None:
    source = WORKER.read_text()
    app_py = (WORKER.parent.parent / "search" / "app.py").read_text()
    gateway_py = (WORKER.parent.parent / "chart_common" / "gateway.py").read_text()

    # Both backends expose the live-counts endpoint…
    assert 'if (url.pathname === "/api/facet-counts") return facetCounts(request, env, url);' in source
    assert "async function facetCounts(" in source
    assert '@app.get("/api/facet-counts")' in app_py

    # …backed by the Scans API: a count-mode scan for the matching total and a
    # values-mode scan per facet, both with an `fts` selector over the text field.
    assert 'mode: "count"' in source and 'mode: "values"' in source
    assert '/v2/namespaces/${encodeURIComponent(namespace(env))}/scans' in source
    assert '"mode": "count"' in gateway_py and '"mode": "values"' in gateway_py
    assert '{"field": "text", "query": query}' in gateway_py


def test_worker_and_python_backends_count_semantic_queries_with_a_vector_radius() -> None:
    source = WORKER.read_text()
    app_py = (WORKER.parent.parent / "search" / "app.py").read_text()
    gateway_py = (WORKER.parent.parent / "chart_common" / "gateway.py").read_text()
    config_py = (WORKER.parent.parent / "chart_common" / "config.py").read_text()

    # A semantic query has no exact lexical match set, so its count is an `ann`
    # radius ball around the query vector; keyword/fused stays exact `fts`.
    assert '{ ann: { field: "vector", vector, radius } }' in source
    assert '{"ann": {"field": "vector", "vector": vector, "radius": radius}}' in gateway_py

    # The gateway's own route decision selects the scan selector in both backends.
    assert 'route === "semantic"' in source
    assert 'route == "semantic"' in app_py

    # A sane, tunable default radius (Arctic's asymmetric query embedding sits near
    # ~0.5 cosine distance, so the ball must open past that).
    assert "CHART_SEMANTIC_RADIUS" in config_py
    assert "0.6" in config_py
    assert "semanticRadius(env)" in source


def test_worker_rejects_non_get_methods_for_all_api_endpoints() -> None:
    source = WORKER.read_text()

    assert 'if (url.pathname === "/api/facets") return facets(request, env);' in source
    assert 'if (url.pathname === "/api/config")' in source
    assert 'async function facets(request, env)' in source
    assert 'assertMethod(request, "GET");' in source.split('if (url.pathname === "/api/config")')[1]
    assert 'assertMethod(request, "GET");' in source.split("async function facets(request, env)")[1]
