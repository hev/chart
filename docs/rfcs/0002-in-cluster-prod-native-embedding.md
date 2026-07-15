# RFC 0002: chart is an in-cluster demo — and turbopuffer native embedding removes the embedder entirely

Status: Draft (2026-07-14) — an architecture-correction RFC. Like RFC 0001 it is
a mapping, not a release commitment, and one of its two legs (native embedding)
requires new gateway machinery — that part is a finding to file upstream in
layer-pro, not code to write here.

## Summary

Two related corrections to chart's prod architecture:

1. **chart's prod backend belongs in the cluster, not on Cloudflare.** Every
   feature chart demos is a gateway or Kubernetes feature — query routing, the
   `Function` GPU cascade, the `Agent` CRD, `Pipeline` indexing. The Cloudflare
   Worker (`src/worker.js`) exists only to inject the gateway key server-side,
   and because Workers can't run an embedding model it drags in a second
   deployment (`CHART_QUERY_EMBED_URL`) just to vectorize queries. The FastAPI
   backend (`search/app.py`) — currently labeled the "dev twin" — is already the
   more complete implementation and already has a `Dockerfile`. Promote it to
   prod, deployed in the `chart` namespace next to the gateway it demos.

2. **turbopuffer native embedding deletes embedding from the app.**
   turbopuffer now embeds on write (an `embed` model on a text attribute's
   schema) and on query (`rank_by: ["text", "ANN", ["Embed", <query>]]`) —
   https://turbopuffer.com/docs/embedding. If the gateway passes this through,
   chart stops shipping vectors at all:

   - `chart_common/embed.py` leaves the serving path; the search service is
     pure gateway client.
   - `CHART_QUERY_EMBED_URL` / `CHART_QUERY_EMBED_KEY` and the standalone
     query-embed endpoint are deleted.
   - The GPU embed pipeline (`deploy/pipeline-embed.yaml`) is no longer needed
     for indexing — writes embed on the fly. (The `Function` classifier
     cascade stays; that's a UDF demo, not plumbing.)
   - **RFC 0044 deferral falls out for free.** The `hybrid_text` route never
     touches an embedding — only semantic/fused queries carry an `Embed` form,
     and the vectorization happens inside turbopuffer. The README's promised
     fast-follow ("make short keyword traffic skip embedding entirely") stops
     being work and becomes the default.

## The honest tension

Native embedding *alone* would also rescue the Worker — with no embed endpoint
to call, a Worker that just proxies the gateway becomes viable again. So the
in-cluster argument doesn't rest on embedding; it rests on identity and ops:

- **Identity.** chart teaches the gateway's decision-making and its K8s surface
  (Agent, Function, Pipeline, VectorStore CRDs). A prod path that detours
  through Cloudflare demos someone else's edge, not our cluster.
- **Ops.** One deploy surface instead of three (cluster + Cloudflare +
  query-embed endpoint), no Cloudflare secrets/billing, and CI that runs
  entirely on our own runners (chart moved to self-hosted runners in #9 after
  hosted-runner billing broke every deploy).
- **Two backends, one UI** (Layer RFC 0076) collapses to **one backend, one
  UI** — the FastAPI service is dev *and* prod; the Worker and its drift
  surface go away.

## Dependencies (upstream findings, filed not fixed here)

1. **Gateway `Embed` pass-through — layer-pro RFC needed.** The gateway's Rust
   source has no turbopuffer `Embed` rank_by support today. Either turbolisp
   grows an `Embed` form the turbopuffer backend lowers, or the `VectorStore`
   CRD grows an embedding-model field the gateway consults. Note hev search
   (layer-pro RFC 0086) is BYO-vector — native embedding widens the gap the
   `kind: search` cutover has to bridge; the RFC there should say who owns
   embedding in that world.
2. **Model change + full re-index.** The corpus is embedded with
   `snowflake-arctic-embed-m-v1.5` (the notesearch model). turbopuffer's native
   catalog is voyage/cohere (S/M/L tiers, $0.02–$0.12 per 1M tokens). Adopting
   native embedding means re-embedding PMC-Patients with a catalog model and
   re-running the ReCDS eval (`eval/`) — chart is the one demo with real qrels,
   so the model swap is *measurable*, and the eval either blesses it or kills
   it. It does weaken the "public twin of notesearch" claim (different
   embedding model); the twin claim then rests on corpus + routing behavior,
   not vectors.

## Sketch of the order of work

1. File the layer-pro RFC for gateway-mediated native embedding (`Embed`
   pass-through or VectorStore-declared model). Nothing in chart moves first.
2. Behind that: re-index with a catalog model on a parallel namespace, re-run
   ReCDS, compare against the arctic baseline.
3. If the eval holds: promote `search/app.py` to prod in-cluster (Deployment +
   Service in `deploy/`), point DNS at the cluster ingress, retire
   `src/worker.js`, `wrangler.jsonc`, the deploy-worker workflow, and the
   query-embed endpoint.
4. Update README/RFC 0076 references: one backend, routing badge unchanged,
   RFC 0044 line rewritten from "fast-follow" to "native".
