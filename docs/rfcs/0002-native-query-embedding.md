# RFC 0002: Auto routing without a query-embed endpoint — turbopuffer native embedding on the query side

Status: Draft (2026-07-14) — an architecture-correction RFC. Like RFC 0001 it
is a mapping, not a release commitment, and its enabler is new gateway
machinery — that part is filed upstream as a layer-pro RFC (gateway `Embed`
lowering, RFC 0044 phase 2), not code written here.

## Summary

turbopuffer now embeds at query time: `rank_by: ["text", "ANN",
["Embed", <query>]]` vectorizes the input inside the store, against the model
configured on the attribute (https://turbopuffer.com/docs/embedding). For
chart this closes the loop RFC 0044 left open:

- **`hybrid_text` never embeds** — unchanged, that's the router's whole point
  for short keyword traffic.
- **`semantic` / `fused` embed inside turbopuffer** — the gateway lowers the
  semantic leg to an `Embed` form instead of bouncing the query back for the
  application to vectorize.
- **Nobody else embeds.** The query-embed endpoint (`CHART_QUERY_EMBED_URL` /
  `CHART_QUERY_EMBED_KEY` and the deployment behind it) is deleted. The
  serving path — Worker and FastAPI backend alike — becomes a pure gateway
  client. `chart_common/embed.py` leaves the serving path entirely.

That is the headline: **the search service no longer has to exist as an
embedding host.** RFC 0044's fast-follow ("make short keyword traffic skip
embedding entirely") stops being work; the deferral posture becomes simply
how the store works.

## What stays: the write side and its GPU workers

The indexing pipeline keeps its own embedding workers
(`deploy/pipeline-embed.yaml`): the scale-to-zero GPU pool, warm windows, and
batch autoscaling are built and paid for, and Layer's Pipeline owns that
front. Native embedding is adopted **on the query side only**.

That works because the two sides must share one embedding space anyway, and
turbopuffer's catalog includes **open-weights models** — `baai/bge-m3` and
the `qwen/qwen3-embedding-{0p6b,4b,8b}` family — alongside the proprietary
voyage/cohere/gemini tiers. So:

- The corpus is re-embedded by our own GPU workers with a catalog
  open-weights model (candidate: `baai/bge-m3` at 1024 dims, or
  `qwen3-embedding-0p6b` for the small tier), replacing
  `snowflake-arctic-embed-m-v1.5` (768 dims, not in the catalog).
- Query-time `["Embed", q]` uses the same catalog model, so self-computed
  document vectors and store-computed query vectors land in the same space.
- The GPU workers keep doing exactly what they do today, with a different
  model string.

## Dependencies and open questions

1. **Gateway `Embed` lowering — layer-pro RFC (RFC 0044 phase 2).** The
   gateway's posture is "route first, embed lazily, never in the gateway."
   Native embedding keeps that posture and extends it: never in the
   application either. The router's decision logic is untouched; only the
   execution of a semantic/fused route changes — where today the gateway
   echoes the routing decision and waits for the app to embed and re-issue,
   it instead lowers the semantic leg to `["Embed", q]` on
   turbopuffer-backed VectorStores. The echo/deferral path stays for
   backends without native embedding (hev search is BYO-vector, layer-pro
   RFC 0086). Filed upstream; nothing in chart moves first.
2. **Mixed writes — confirm with turbopuffer.** The docs' migration example
   switches queries first, then writes, which implies self-supplied document
   vectors coexist with query-time `Embed` on the same attribute — but it is
   not stated outright. Confirm that (a) an attribute with `embed` configured
   still accepts pre-computed vectors on write, or (b) the `Embed` query
   function works against a schema-declared model without write-side native
   embedding enabled. One of the two must hold for the keep-our-GPU-workers
   plan; if neither does, the fallback is letting turbopuffer embed writes
   too, and the GPU pool keeps only the classifier cascade.
3. **Model swap is measurable.** Re-embed PMC-Patients with the chosen
   catalog model on a parallel namespace and re-run the ReCDS eval
   (`eval/`) against the arctic baseline. chart is the demo with real
   qrels; the eval blesses the swap or kills it. Exact-space fidelity
   (our GPU-computed doc vectors vs turbopuffer's query vectors of the same
   nominal model) gets a cheap sanity check: embed a handful of passages both
   ways, compare cosines, before committing the full corpus.

## Side note: this decouples the backend question

With embedding out of the serving path, the Cloudflare Worker no longer needs
`CHART_QUERY_EMBED_URL` — a thin gateway proxy runs anywhere. Whether prod
stays a Worker or moves in-cluster becomes a pure ops/identity choice,
separable from this RFC and decided on its own terms. (CI already moved to
self-hosted runners in #9; the twin-backend question can wait for the
routing work to land.)

## Order of work

1. File the layer-pro RFC (gateway `Embed` lowering for turbopuffer-backed
   stores, RFC 0044 phase 2). Confirm the mixed-writes question with
   turbopuffer in parallel.
2. Re-embed on a parallel namespace with the catalog model via the existing
   GPU pipeline; run the both-ways cosine sanity check; re-run ReCDS.
3. When the gateway lowering lands: flip chart's semantic/fused path to it,
   delete the query-embed endpoint and `CHART_QUERY_EMBED_URL/KEY`, drop
   `chart_common/embed.py` from the serving path (it stays for the indexer).
4. Update README: routing table unchanged, RFC 0044 line rewritten from
   "fast-follow" to "native"; badge behavior identical — the demo's story
   gets *stronger* (the router's cheap path is now free everywhere).
