# Agent Project Status

Last updated: 2026-06-25

## ⚠️ IMPORTANT — this repo is a Layer design-preview customer

This repo is a **design-preview customer of hev layer**, not part of the Layer
product. Its job is to *use* Layer the way a real customer would and **report
back** to the Layer team. That feedback loop is a primary responsibility of this
repo, not a side task — the demo working is table stakes; the signal we send the
Layer team is the deliverable.

**When you hit friction, do not fix Layer from here — report it:**

- **A bug, or docs that are wrong / unclear / missing** → file a **GitHub issue**
  on the Layer repo (`hev/layer`) with a minimal repro and the exact page or
  behavior at fault.
- **A missing feature or capability gap** → open an **RFC** in the Layer repo
  (`../layer/docs/rfcs/`), in the existing RFC shape, with this workload as the
  motivating / acceptance case.

**Operations are Layer's job.** This repo has operational access to the shared
Layer cluster, but the goal is that Layer operates *itself* — autoscaling,
scale-to-zero, scheduling, binpacking. Let it. Do **not** hand-tune what Layer is
meant to manage. The *deliverable* of any friction is a GH issue (bug) or an RFC
(capability gap) on `hev/layer` — never a local feedback log.

- When Layer falls short — autoscaling lags, a pipeline stalls, scale-to-zero
  misbehaves — it is OK to **intervene** to keep the demo healthy. But every
  intervention **must** produce a GitHub issue (bug) or an RFC (missing
  capability). An undocumented manual fix is a process failure: the intervention
  is the symptom, the report is the deliverable.
- **Shared namespace / binpacking.** This repo deploys to a namespace in the
  shared demo cluster alongside the other demos (shelf, shop, chart,
  hybrid-text-fusion-demo, label). Scheduling / binpacking contention may bite.
  Same rule: intervene to stay healthy if you must, but the result is a GH issue
  or an RFC documenting the shortfall — never a silent workaround.

The deliverable of any friction is always a **paper trail in `hev/layer`** (issue
or RFC) so the design-preview signal reaches the Layer team.

## Current state

The chart repo-side implementation is in place, including the local search app,
pipeline audit tooling, preflight/final-gate checks, Phase 4 guarded event smoke
helper, and Phase 6 status reporting.

The project is not complete end-to-end yet. The current final gate is blocked by
external Layer/runtime state rather than by a known local code task.

## Local test app

A local FastAPI web app is available from `search.app` and serves
`web/static/index.html`.

Run it with:

```sh
source scripts/lib/resolve_gateway_key.sh
resolve_gateway_key >/dev/null
uv run --extra search uvicorn search.app:app --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

The app injects the gateway key server-side. Do not expose or print the key in
browser-visible code or logs.

Verified locally:

- Static UI loads.
- `/api/config` responds for namespace `chart-notes`.
- `/api/facets` responds.
- `/api/search?q=metformin%20500mg&top_k=3` returns live gateway results.

Current data caveat: the latest observed facet snapshots were for about 2,000
rows. `age_band` and `gender` had values, while `specialty`,
`diagnosis_category`, and `events` were empty.

## Audit status

`scripts/plan_audit.sh --ready` currently reports:

```text
no ready next steps
```

`scripts/final_gate.sh` currently exits non-zero because the plan is not
complete.

Passing required gates observed before this handoff:

- `phase1_slice_index`
- `phase2_3_live_smoke`
- `phase3_facet_refresh`
- `phase4_classify_cost_signal`
- `phase5_holdout`
- `phase6_embed_cost`
- `phase6_gpu_images`
- `phase6_deploy_apply`
- `phase6_unpause_embed`

Failing or blocked required gates:

- `phase4_event_facet_smoke`: needs a Linux/vLLM Gemma classifier runtime and
  event facet writeback.
- `phase5_recds`: full retrieval corpus is missing; fused-dominance checks are
  not meaningful on the partial corpus.
- `phase6_runtime_status`: blocked by incomplete facets, incomplete corpus, UDF
  status mismatch, and Layer autoscaling readiness.
- `phase6_gate_complete`: blocked by the runtime status gate.

## In-flight state (2026-07-03, post-backfill)

- The classifier backfill is **COMPLETE**: 11,373/11,373 notes classified,
  zero failures, ~4h at 48-58 notes/min on two GPUs. Measured cost $8.2
  ($8.10 GPU + $0.13 Layer-metered; eval/out/classify-events-budget.json) —
  ~2x under the Haiku Batch baseline; README's cost table carries the
  measured numbers.
- Facet rail is fully live (events unnested per hev/layer#151 fix, verified;
  coverage line reads 10,678/11,373). Gemma weights are mirrored to
  s3://hevlayer-models-186219257916-us-east-1/google/gemma-2-9b-it/ and the
  worker restores from the mirror when the baked cache is absent.
- Everything is scaled to zero: classifier drained, and the `chart-embed-gpu`
  + `chart-ingest` Pipelines are **paused by choice** (2026-07-03 — "we have
  enough corpus"; ~11.4k of the 167k target indexed). Unpause both to resume
  corpus growth; new writes re-trigger classification via the write path.
- Image rolls still require the hev/layer#148 dance: apply CR → DELETE
  /v2/udfs/chart-classify-events → wait re-register → resume → discover.

## Main blockers

- The full PMC-Patients target is 167,000 notes; the latest tracked full-corpus
  status was still partial.
- The Phase 4 event facet smoke requires a Linux host with the classifier extra
  available, including vLLM/transformers runtime support.
- The `chart-classify-events` Kubernetes Function exists and is paused, but the
  gateway UDF status endpoint reports the UDF as missing (`hev/layer#99`); the
  `events`/`specialty`/`diagnosis_category` facets stay empty until the classifier
  backfill runs.

## Layer handling policy

Do not make Layer/platform changes from this repo unless explicitly asked.
Every Layer follow-up lands as a **GitHub issue (bug) or RFC (capability gap) on
`hev/layer`** (the design-preview contract above) — there is no local feedback
log. A finding that is not an issue or RFC on `hev/layer` has not been reported.

Open Layer follow-ups from chart:

- `hev/layer#137` — the Python client rejects the search-backed write response
  shape; `chart_common/gateway.py:write_notes` carries the same narrow catch
  shelf does until the gateway/client shapes are normalized.
- `hev/layer#138` — Layer does not build hev search FTS/ANN indexes on write;
  the cutover validation may require a manual `/fts-index` + `/index`
  intervention (as shelf's did).
- `hev/layer#141` — the `hybrid_text` scan selector is rejected on kind=search
  stores. It does **not** currently bind chart: `chart-notes` is live on
  Turbopuffer (the kind=search cutover was never applied — see
  `deploy/vectorstore.yaml`). The live facet counts still use `fts`, a leftover
  of the assumed cutover; moving them to `hybrid_text` is open.
- `hev/layer#99` — paused Function is unobservable (UDF status 404); root cause of
  the empty `events`/`specialty`/`diagnosis_category` facets, which stay empty
  until the classifier backfill runs.
- `hev/layer#101` — `scaling.warmWindowSeconds` documented on the API page but
  missing from the CRD reference (`scaling-crd`/`pipeline-crd`/`function-crd`).
- `hev/layer#143` — the ApiKey operator derives an invalid Secret name
  (`apikey_<name>` with underscores), so declarative ApiKey minting never
  reconciles; chart's key path stays imperative (`POST /v2/keys`) until it lands.
- `hev/layer#144` — the gateway only loads Agent CRs from its own namespace;
  the agentic-search Agent + `chart-openrouter` Secret are duplicated into ns
  `layer` on layer-prod (the `chart`-namespace copies are the declarative twin).
- `hev/layer#145` — entitlement namespace checks resolve against the default
  store, not `search-store`; a minted key needs a mirror grant on
  `vectorstore.turbopuffer-default` to read `chart-notes`.
- `hev/layer#146` — `agent.<name>: {}` (the documented form) is rejected;
  `scopes: [read]` is required (deploy/apikey.yaml carries the workaround).
- `hev/layer#147` — the Agent planner strict-parses OpenRouter content, but
  Anthropic-via-Bedrock ignores `response_format: json_object` and returns
  fenced JSON (502). chart's Agent is pinned to `openai/gpt-4o-mini` /
  `google/gemini-2.5-flash` until the parser is hardened.
- `hev/layer#148` — the operator cannot update a registered UDF (the gateway
  strips `worker.compute_class`, the CRD defaults it, and the 409 equality
  check never passes), so every Function spec change bricks reconciliation.
  chart's classifier is driven via the gateway API (resume/discover) and
  `deploy/functions-events.yaml` omits `computeClass`; the CR still reports
  GatewayRegistrationFailed, and `scaling.warmWindowSeconds` may not be
  synced until this lands. (Also bites moment's three Functions.)

Landed / consumed (kept for traceability):

- `hev/layer#100` (KEDA empty bearer token) — fixed and deployed; Layer now owns
  autoscaling, so manual `kubectl scale` is no longer a project behavior.
- `hev/layer#94` (cursor pagination for routed/fused queries) — shipped
  (`52a31d8`); enables a future move from client-side to server-side pagination.
- RFC 0081 P1 (pipeline claim-check coordination) — landed; staging no longer
  blanket-locks already-staged rows, so source + embed run concurrently (the
  external windowing loop is retired).
- RFC 0082 K1 (`warmWindowSeconds`) — landed and consumed on the GPU embed
  Pipeline (`deploy/pipeline-embed.yaml`); the classifier `Function` still needs it.
- Scans API `hybrid_text` selector (`hev/layer` `ed86668`) — landed;
  kind=search stores reject it (hev/layer#141), but `chart-notes` stayed on
  Turbopuffer (the cutover was never applied), so it is usable here. chart
  still counts the keyword/fused route via an `fts` selector (fuzzy-surfaced
  matches approximated by their exact lexical terms) — a leftover of the
  assumed cutover.

Important current Layer notes:

- Images should be pushed to ECR, not GHCR.
- Layer findings go to **GitHub issues / RFCs on `hev/layer`, never Linear** —
  the Linear MCP is connected, so it gets reached for by reflex, but tickets
  logged there miss the paper trail and have had to be re-filed as GH issues and
  deleted from Linear.
- Local cost measurement is not needed; Layer cost reporting is authoritative.
- Depot account details are available in `../layer`.
- Avoid manual scaling as a project behavior; Layer owns autoscaling now that the
  KEDA bearer-auth issue (`hev/layer#100`) is fixed and deployed.

## Useful commands

Run the ready audit:

```sh
./scripts/plan_audit.sh --ready
```

Run the final gate:

```sh
./scripts/final_gate.sh
```

Run the relevant local test suite:

```sh
uv run pytest
```

Run the guarded Phase 4 event smoke on a suitable Linux/vLLM runtime:

```sh
scripts/phase4_event_smoke.sh
```

Run the local web app:

```sh
source scripts/lib/resolve_gateway_key.sh
resolve_gateway_key >/dev/null
uv run --extra search uvicorn search.app:app --host 127.0.0.1 --port 8000
```
