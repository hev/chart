# chart

A public demo on hev layer: clinical patient-notes search (PMC-Patients) where
the gateway's query routing is the headline, plus a Gemma GPU classifier
running on Layer's Function runtime. The design of record is RFC 0076
(`../layer-pro/docs/rfcs/0076-clinical-notes-query-routing-demo.md`). `README.md` is the public tour; the gate run book
is `docs/operations.md`.

**This repo is public.** Never put client names, client systems or anything
from a client engagement in code, comments, docs or commit messages.

## You are a Layer customer

chart exists to use Layer the way a customer would and to report what it
hits. The demo working is table stakes; the report is the deliverable.

- **Reimplement nothing Layer owns:** routing, fusion, fuzzy matching, facet
  snapshots, the Function/UDF runtime, embedding. The app posts queries and
  renders what the gateway echoes. If you're writing fusion math or a
  tokenizer, the boundary is wrong.
- **Read the docs, don't invent API.** The request and response shapes are in
  `../layer-pro/site/src/content/docs/` and
  `../layer-pro/apps/layer-gateway/openapi.yaml`.
- **Report friction in Linear** (team `LYR`, via the `linear` skill / CLI):
  a bug or a wrong or missing doc is an issue; a missing capability is an RFC,
  written as a Linear project with an `RFC: <name>` document, with this
  workload as the motivating case. Old follow-ups cite `hev/layer-pro#NNN`
  GitHub issues; new ones go to Linear.
- **Layer operates itself.** Don't hand-tune scaling. If you must intervene
  to keep the demo up, the intervention gets an issue too.

## Run and test

```sh
uv sync --extra search --extra test
uv run pytest                          # 672 tests; needs ../layer-pro/clients/python
source scripts/lib/resolve_gateway_key.sh && resolve_gateway_key >/dev/null
uv run --extra search uvicorn search.app:app --reload   # http://localhost:8000
```

The gateway key comes from 1Password (`op://mesh-staging/layer-turbopuffer/credential`)
at run time. Never write it to a `.env` file or print it.

## Deploy

- **Web (prod):** a Cloudflare Worker (`src/worker.js`, `wrangler.jsonc`)
  serving `web/static/`. `.github/workflows/deploy-worker.yml` deploys on
  push to `main` when `src/`, `web/static/` or `wrangler.jsonc` change. Keep
  `src/worker.js` and the FastAPI dev backend (`search/`) in lockstep.
- **Cluster:** namespace `chart` on `layer-prod`. `deploy/` is the declarative
  bundle (VectorStore, Warehouse, Pipelines, Index, the GPU events Function,
  the Agent); see `deploy/README.md` for apply order.
- **Images** go to the mesh-account ECR (`scripts/build_gpu_images.sh`), never
  `ghcr.io`; a test enforces it.

## State (2026-09-27)

`chart-notes` is on Turbopuffer with ~11.4k of the 167k-note corpus indexed.
The ingest and GPU embed Pipelines are paused by choice, the classifier
backfill is complete, and everything is scaled to zero. Unpause the Pipelines
to grow the corpus; new writes re-trigger classification.
