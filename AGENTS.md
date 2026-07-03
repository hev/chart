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

## In-flight state (2026-07-02)

- The classifier backfill is ACTIVE: the `chart-classify-events` UDF is resumed
  with ~11.4k notes enqueued, driven via the gateway API because the operator
  cannot re-register a changed Function (hev/layer#148).
- The embed Pipeline `chart-embed-gpu` is **temporarily paused**
  (`spec.paused: true`, patched live) to free the single GPU node for the
  classifier — the AWS "Running On-Demand G and VT instances" vCPU quota (4)
  allows exactly one GPU node. **Resume embed when the backfill drains or the
  quota lands**: `kubectl patch pipeline chart-embed-gpu -n chart --type=merge
  -p '{"spec":{"paused":false}}'`.
- A quota increase 4→16 vCPUs is filed (request `114eb0bc…`, us-east-1,
  acct 186219257916) and sits in AWS support review (CASE_OPENED). Once it
  lands, embed + classify run concurrently and this section should be removed.
- `hev-shop-embed` (namespace hev-shop) is also **temporarily paused**: its
  worker was wedged 13h on three poison documents (chunks gone from cache/S3,
  hev/layer#149), holding the sole GPU with zero progress. Resume with
  `kubectl patch pipeline hev-shop-embed -n hev-shop --type=merge -p
  '{"spec":{"paused":false}}'` once #149 is addressed / the quota lands.
- vLLM-in-a-Function learnings (five environmental failures, the checklist,
  and the engine/worker/cluster gotchas) are captured in
  `docs/vllm-udf-runbook.md`; the Layer-side proposal distilled from it is
  RFC 0094 (`../layer/docs/rfcs/0094-gpu-inference-base-image.md`, a
  supported `hevlayer-inference` base image, first rung of RFC 0068 §4).
- The classifier worker downloads Gemma at startup: the image's weight bake
  silently skipped (gated repo, no build token), so the Function injects
  `HF_TOKEN` from the `chart-huggingface` Secret (1Password: "layer hugging
  face token"). Re-bake the image with the token to restore fast cold starts.

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
- `hev/layer#141` — the `hybrid_text` scan selector is rejected on kind=search;
  chart's live facet counts use `fts` until it lands.
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
- Scans API `hybrid_text` selector (`hev/layer` `ed86668`) — landed, but
  kind=search rejects it today (hev/layer#141), so since the hev search cutover
  chart counts the keyword/fused route via an `fts` selector (fuzzy-surfaced
  matches approximated by their exact lexical terms).

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
