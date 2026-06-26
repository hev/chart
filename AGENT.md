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
meant to manage. (This supersedes the older "record follow-ups in
`LAYER_IMPROVEMENTS.md`" note below: that log is fine as a scratchpad, but the
*deliverable* is a GH issue or RFC on `hev/layer`.)

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
helper, Phase 6 status reporting, and the Layer improvements log.

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

## Main blockers

- The full PMC-Patients target is 167,000 notes; the latest tracked full-corpus
  status was still partial.
- The Phase 4 event facet smoke requires a Linux host with the classifier extra
  available, including vLLM/transformers runtime support.
- Layer-managed KEDA autoscaling is not ready because the generated bearer auth
  reference points at a missing or empty `chart/layer` secret key
  `turbopuffer-api-key`.
- The `chart-classify-events` Kubernetes Function exists and is paused, but the
  gateway UDF status endpoint reports the UDF as missing.

## Layer handling policy

Do not make Layer/platform changes from this repo unless explicitly asked.
Record Layer follow-ups in `LAYER_IMPROVEMENTS.md` instead.

Important current Layer notes:

- Images should be pushed to ECR, not GHCR.
- Local cost measurement is not needed; Layer cost reporting is authoritative.
- Depot account details are available in `../layer`.
- Avoid manual scaling as a project behavior; Layer should own autoscaling once
  the KEDA bearer-auth issue is fixed.

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
