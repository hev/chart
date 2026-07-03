const INCLUDE = [
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
];

const FACET_FIELDS = ["specialty", "age_band", "diagnosis_category", "gender", "events", "discontinuation_reason", "has_med_discontinuation"];
// Clickable facet rail → a turbolisp `filters` clause on the same routed query.
// Scalars filter by equality; `events` is a []string, so it uses the turbopuffer
// array Contains/ContainsAny ops. Mirrors search/app.py.
const SCALAR_FACETS = new Set(["specialty", "age_band", "diagnosis_category", "gender"]);
const ARRAY_FACETS = new Set(["events"]);
const ARCTIC_QUERY_PREFIX = "Represent this sentence for searching relevant passages: ";
const TRANSIENT = new Set([502, 503, 504]);

function groupFilters(url) {
  const grouped = new Map();
  for (const item of url.searchParams.getAll("f")) {
    const idx = item.indexOf(":");
    if (idx < 0) continue;
    const field = item.slice(0, idx);
    const value = item.slice(idx + 1).trim();
    if (!value || (!SCALAR_FACETS.has(field) && !ARRAY_FACETS.has(field))) continue;
    if (!grouped.has(field)) grouped.set(field, []);
    if (!grouped.get(field).includes(value)) grouped.get(field).push(value);
  }
  return grouped;
}

function buildFilters(grouped) {
  const clauses = [];
  for (const [field, values] of grouped) {
    if (ARRAY_FACETS.has(field)) {
      clauses.push(values.length === 1 ? [field, "Contains", values[0]] : [field, "ContainsAny", values]);
    } else {
      clauses.push(values.length === 1 ? [field, "Eq", values[0]] : [field, "In", values]);
    }
  }
  if (!clauses.length) return null;
  return clauses.length === 1 ? clauses[0] : ["And", clauses];
}

function parseFilters(url) {
  return buildFilters(groupFilters(url));
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    try {
      if (url.pathname === "/api/search") return search(request, env, url);
      if (url.pathname.startsWith("/api/similar/")) return similar(request, env, url);
      if (url.pathname === "/api/facets") return facets(request, env, url);
      if (url.pathname === "/api/facet-counts") return facetCounts(request, env, url);
      if (url.pathname === "/api/config") {
        assertMethod(request, "GET");
        return json({
          namespace: namespace(env),
          gateway: gatewayUrl(env),
          field: "text",
        });
      }
      return env.ASSETS.fetch(request);
    } catch (error) {
      return json({ detail: error.message || "worker error" }, { status: error.status || 500 });
    }
  },
};

async function search(request, env, url) {
  assertMethod(request, "GET");
  const query = (url.searchParams.get("q") || "").trim();
  const filters = parseFilters(url);

  // Agentic search (docs/api/agents): the configured reasoning loop instead of
  // the single-hop Auto route. Facet filters don't apply — the agent infers its
  // own, echoed in the response. Mirrors search/app.py.
  if (url.searchParams.get("agentic") === "1" && query) {
    const result = await agentQuery(env, query, boundedTopK(url.searchParams.get("top_k"), 20, 50));
    result.query = query;
    return json(result);
  }

  if (!query) {
    // Empty box: filter-only cohort browse (no query → no routing decision), or
    // nothing at all when no facet is active.
    if (!filters) return json({ rows: [], routing: null, hybrid: null, query: "" });
    const result = await queryNamespace(env, {
      rank_by: ["id", "asc"],
      filters,
      top_k: boundedTopK(url.searchParams.get("top_k"), 20, 50),
      include_attributes: INCLUDE,
    });
    result.query = "";
    return json(result);
  }

  const vector = await embedQuery(env, query);
  const body = {
    rank_by: ["text", "Auto", query, { vector }],
    top_k: boundedTopK(url.searchParams.get("top_k"), 20, 50),
    include_attributes: INCLUDE,
    include_leg_breakdown: true,
  };
  if (filters) body.filters = filters;
  const result = await queryNamespace(env, body, { requireRouting: true });
  result.query = query;
  return json(result);
}

async function similar(request, env, url) {
  assertMethod(request, "GET");
  const id = decodeURIComponent(url.pathname.slice("/api/similar/".length)).trim();
  if (!id) return json({ rows: [], routing: null, hybrid: null, patient_id: "" });
  const result = await queryNamespace(env, {
    nearest_to_id: [id],
    top_k: boundedTopK(url.searchParams.get("top_k"), 12, 50),
    include_attributes: INCLUDE,
    include_leg_breakdown: true,
  });
  result.patient_id = id;
  return json(result);
}

// The clinical facet rail, in two modes (the same contract as shelf's rail):
// with `q` (or active filters) it delegates to the per-search live counts; bare,
// it reads the corpus-wide snapshot histograms. Mirrors search/app.py.
async function facets(request, env, url) {
  assertMethod(request, "GET");
  if ((url.searchParams.get("q") || "").trim() || url.searchParams.getAll("f").length) {
    return facetCounts(request, env, url);
  }
  const out = {};
  const wanted = new Set(FACET_FIELDS);
  const history = await gatewayFetch(env, `/v2/namespaces/${encodeURIComponent(namespace(env))}/history?limit=20`);
  for (const snapshot of Array.isArray(history) ? history : []) {
    if (!snapshot.sha || wanted.size === 0) continue;
    const body = await gatewayFetch(
      env,
      `/v2/namespaces/${encodeURIComponent(namespace(env))}/snapshots/${encodeURIComponent(snapshot.sha)}`,
    );
    for (const field of [...wanted]) {
      const column = (body.fields || []).find((candidate) => candidate.name === field);
      if (!column) continue;
      const values = [...(column.values || [])]
        .sort((a, b) => (b.n || 0) - (a.n || 0))
        .slice(0, 14)
        .map((value) => ({ value: value.v, count: value.n }));
      out[field] = {
        values,
        provenance: {
          sha: body.sha || snapshot.sha,
          watermark_ms: body.watermark_ms,
          row_count: body.row_count,
        },
      };
      wanted.delete(field);
    }
  }
  return json(out);
}

// Live facet counts for the current search — the Scans API showcase. Slower than
// top-k (origin scatter/gather), so the UI calls this asynchronously after the
// ranked results are on screen. Mirrors search/app.py's /api/facet-counts.
async function facetCounts(request, env, url) {
  assertMethod(request, "GET");
  const query = (url.searchParams.get("q") || "").trim();
  const route = url.searchParams.get("route") || "";
  const grouped = groupFilters(url);

  // Follow the gateway's routing decision: a semantic query has no exact lexical
  // match set, so count an `ann` radius ball around the query vector instead of fts.
  const semantic = Boolean(query) && route === "semantic";
  const selector = countSelector({
    query,
    vector: semantic ? await embedQuery(env, query) : null,
    radius: semantic ? semanticRadius(env) : null,
  });

  const total = scanCount(env, { selector, filters: buildFilters(grouped) });
  const breakdowns = FACET_FIELDS.map(async (field) => {
    // Drill-down: a field's own filter is excluded from its own breakdown, so the
    // counts show what each alternative value would return.
    const subset = new Map([...grouped].filter(([name]) => name !== field));
    return [field, await scanFacetValues(env, { field, selector, filters: buildFilters(subset) })];
  });

  const [totalCount, ...pairs] = await Promise.all([total, ...breakdowns]);
  const fields = {};
  for (const [field, values] of pairs) if (values) fields[field] = values;
  return json({ fields, total: totalCount, approximate: semantic, query });
}

// The Scans API match-set selector, following the gateway's routing decision: a
// semantic query counts an `ann` ball around the query vector (relevance has no
// exact lexical set), keyword/fused an `fts` selector over the routed text field.
// A `hybrid_text` selector (BM25 + per-token fuzzy, RFC 0057) would mirror the
// fused route exactly, but kind=search rejects it today (hev/layer#141), so the
// fuzzy/typo matches are approximated by their exact lexical terms. At most one
// ranked selector; vector+radius (semantic only) wins. Mirrors
// chart_common.gateway.count_selector.
function countSelector({ query, vector, radius }) {
  if (vector && radius != null) return { ann: { field: "vector", vector, radius } };
  if (query) return { fts: { field: "text", query } };
  return {};
}

function semanticRadius(env) {
  const r = Number.parseFloat(env.CHART_SEMANTIC_RADIUS);
  return Number.isFinite(r) ? r : 0.6;
}

async function scanCount(env, { selector, filters }) {
  const body = { mode: "count", source: "auto", timeout_seconds: 25, ...selector };
  if (filters) body.filters = filters;
  try {
    const resp = await gatewayFetch(env, `/v2/namespaces/${encodeURIComponent(namespace(env))}/scans`, {
      method: "POST",
      body: JSON.stringify(body),
      headers: { "content-type": "application/json" },
    });
    return typeof resp.count === "number" ? resp.count : null;
  } catch (_) {
    return null;
  }
}

async function scanFacetValues(env, { field, selector, filters, limit = 24 }) {
  const ns = encodeURIComponent(namespace(env));
  const body = { mode: "values", field, source: "auto", ...selector };
  if (filters) body.filters = filters;
  try {
    let job = await gatewayFetch(env, `/v2/namespaces/${ns}/scans`, {
      method: "POST",
      body: JSON.stringify(body),
      headers: { "content-type": "application/json" },
    });
    const started = Date.now();
    while (job.status && job.status !== "completed" && job.status !== "failed") {
      if (Date.now() - started > 25000) return null;
      await sleep(150);
      job = await gatewayFetch(env, `/v2/namespaces/${ns}/scans/${encodeURIComponent(job.id)}`);
    }
    if (job.status !== "completed") return null;
    const results = await gatewayFetch(
      env,
      `/v2/namespaces/${ns}/scans/${encodeURIComponent(job.id)}/results?limit=${limit}`,
    );
    return (results.values || []).map((value) => ({ value: value.v, count: value.n }));
  } catch (_) {
    return null;
  }
}

// POST /v2/agents/{name}/query with the query, its Arctic embedding (BYO — Layer
// never embeds query text; the one vector serves every planned semantic leg), and
// top_k. The Agent CR (deploy/agent.yaml) owns model, fan-out, and fusion; with
// provenance on, rows carry $agent scores and the response echoes the planned
// variants. Mirrors chart_common/search/app.py:_run_agent_query.
async function agentQuery(env, query, topK) {
  const vector = await embedQuery(env, query);
  const body = { query, vector, top_k: topK };
  const path = `/v2/agents/${encodeURIComponent(agentName(env))}/query`;
  const started = Date.now();
  let response;
  let lastError;
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      response = await gatewayFetch(env, path, {
        method: "POST",
        body: JSON.stringify(body),
        headers: { "content-type": "application/json" },
      });
      break;
    } catch (error) {
      if (!TRANSIENT.has(error.status) || attempt === 2) throw error;
      lastError = error;
      await sleep(400 * (attempt + 1));
    }
  }
  if (!response) throw lastError || new Error("agent query failed");
  return {
    rows: response.rows || [],
    routing: null,
    hybrid: null,
    agent: response.agent || null,
    merge: response.merge || null,
    took_ms: Date.now() - started,
  };
}

function agentName(env) {
  return env.CHART_AGENT_NAME || "chart-notes";
}

async function queryNamespace(env, body, options = {}) {
  const started = Date.now();
  let response;
  let lastError;
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      response = await gatewayFetch(
        env,
        `/v2/namespaces/${encodeURIComponent(namespace(env))}/query`,
        {
          method: "POST",
          body: JSON.stringify(body),
          headers: { "content-type": "application/json" },
        },
      );
      break;
    } catch (error) {
      if (!TRANSIENT.has(error.status) || attempt === 2) throw error;
      lastError = error;
      await sleep(400 * (attempt + 1));
    }
  }
  if (!response) throw lastError || new Error("gateway query failed");
  const routing = response.routing || null;
  const route = routing && (routing.route || routing.strategy);
  if (options.requireRouting && !route) {
    const error = new Error("gateway response did not include a routing decision");
    error.status = 502;
    throw error;
  }
  return {
    rows: response.rows || [],
    routing,
    hybrid: response.hybrid || null,
    took_ms: Date.now() - started,
  };
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function embedQuery(env, query) {
  if (!env.CHART_QUERY_EMBED_URL) {
    const error = new Error("CHART_QUERY_EMBED_URL is required for Worker search");
    error.status = 503;
    throw error;
  }
  const response = await fetch(env.CHART_QUERY_EMBED_URL, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      ...(env.CHART_QUERY_EMBED_KEY ? { authorization: `Bearer ${env.CHART_QUERY_EMBED_KEY}` } : {}),
    },
    body: JSON.stringify({
      text: ARCTIC_QUERY_PREFIX + query,
      model: env.CHART_EMBED_MODEL || "Snowflake/snowflake-arctic-embed-m-v1.5",
    }),
  });
  if (!response.ok) throw await upstreamError(response, "embedder");
  const data = await response.json();
  const vector = data.vector || data.embedding || data.data?.[0]?.embedding;
  if (!Array.isArray(vector) || vector.length !== 768) {
    const error = new Error("embedder returned a non-768-d vector");
    error.status = 502;
    throw error;
  }
  return vector;
}

async function gatewayFetch(env, path, init = {}) {
  if (!env.LAYER_GATEWAY_API_KEY) {
    const error = new Error("LAYER_GATEWAY_API_KEY is required");
    error.status = 503;
    throw error;
  }
  const response = await fetch(`${gatewayUrl(env)}${path}`, {
    ...init,
    headers: {
      ...(init.headers || {}),
      authorization: `Bearer ${env.LAYER_GATEWAY_API_KEY}`,
    },
  });
  if (!response.ok) throw await upstreamError(response, "gateway");
  return response.json();
}

async function upstreamError(response, name) {
  let detail = `${name} returned HTTP ${response.status}`;
  const clone = response.clone();
  try {
    const body = await response.json();
    detail = body.detail || body.message || body.error || detail;
  } catch (_) {
    const text = await clone.text();
    if (text) detail = text.slice(0, 240);
  }
  const error = new Error(detail);
  error.status = response.status >= 500 ? 502 : response.status;
  return error;
}

function assertMethod(request, method) {
  if (request.method !== method) {
    const error = new Error("method not allowed");
    error.status = 405;
    throw error;
  }
}

function boundedTopK(value, fallback, max) {
  const parsed = Number.parseInt(value || "", 10);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(1, Math.min(parsed, max));
}

function gatewayUrl(env) {
  return (env.LAYER_GATEWAY_URL || "https://aws-us-east-1.hevlayer.com").replace(/\/+$/, "");
}

function namespace(env) {
  return env.CHART_NAMESPACE || "chart-notes";
}

function json(body, init = {}) {
  return new Response(JSON.stringify(body), {
    ...init,
    headers: {
      "content-type": "application/json; charset=utf-8",
      ...(init.headers || {}),
    },
  });
}
