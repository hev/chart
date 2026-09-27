# Operations: indexing, gates and evidence

The run book for chart's live slice, cost gates and final audit. The public
overview is [README.md](../README.md); agent instructions are in
[AGENTS.md](../AGENTS.md).

## Local run

```bash
source scripts/lib/resolve_gateway_key.sh && resolve_gateway_key >/dev/null   # key from 1Password
uv sync --extra search
uv run --extra search python -m indexer --limit 2000     # smoke index a slice ( --dry-run to skip the gateway)
uv run --extra search uvicorn search.app:app --reload    # http://localhost:8000
scripts/smoke_live.sh                                    # verify routes/facets on the live slice
```

Applying the GPU classifier Function requires both `CHART_APPLY_CLASSIFIER=1` and
`CHART_ACCEPT_PHASE4_CLASSIFY_COST=1` after that classifier cost gate is accepted,
plus an accepted `CHART_PHASE4_CLASSIFY_REPORT` with budget and signal checks.
Layer supplies the authoritative deployed cost evidence; the local measurement
helpers are fallback/off-platform report producers, not the required path.
The audit prints these Layer cost report commands:

```bash
scripts/layer_cost_report.sh --kind classifier --accept --signal-reviewed --out eval/out/classify-events-budget.json
scripts/layer_cost_report.sh --kind embed --accept --out eval/out/embed-budget.json
```

For a live slice with the gateway key resolved from `LAYER_GATEWAY_API_KEY` or
1Password. The resolver first tries `op item get "layer turbopuffer" --vault
mesh-staging --field credential --reveal`, then falls back to the legacy
`layer-turbopuffer` item and the
`op://mesh-staging/layer turbopuffer/credential` reference; override with
`CHART_GATEWAY_KEY_OP_ITEM`, `CHART_GATEWAY_KEY_OP_VAULT`, `CHART_GATEWAY_KEY_OP_FIELD`,
or `CHART_GATEWAY_KEY_OP_REF` if your local item path differs.
`scripts/plan_audit.sh --requirements` reads persisted reports without resolving
secrets by default. Set `CHART_PLAN_AUDIT_PROBE_GATEWAY_KEY=1` when you want the
audit to run the same redacted resolver probe and report the gateway-key
requirement as present.
On non-Linux hosts, the local Gemma classifier step is reported as missing
`classifier_extra`, meaning the Linux/vLLM classifier runtime is not available
locally. Run that step in the GPU Function image or on a Linux GPU host, or set
`CHART_ASSUME_CLASSIFIER_EXTRA=1` when an external classifier runtime is already
provisioned.

```bash
scripts/preflight.sh
scripts/live_slice.sh
scripts/phase4_event_smoke.sh
scripts/refresh_facets.sh --fields age_band,gender
CHART_LIVE_SMOKE_REPORT=${CHART_LIVE_SMOKE_BASE_REPORT:-eval/out/live-smoke-base-report.json} scripts/smoke_live.sh
scripts/eval_live.sh
scripts/full_status.sh
scripts/gate_report.sh
scripts/final_gate.sh
```

`scripts/smoke_live.sh` writes `CHART_LIVE_SMOKE_REPORT`, capturing the slice
index-shape, routing-chip, nearest-neighbor, and facet checks. Use
`CHART_LIVE_SMOKE_BASE_REPORT`/`eval/out/live-smoke-base-report.json` for the
Phase 2/3 base smoke artifact, which requires age/gender facets but not the
post-classifier `events` facet. `scripts/live_slice.sh` also writes
`CHART_SLICE_INDEX_REPORT` for the preceding bounded index run.

For the Phase-5 retrieval gate after the full index exists, make holdout leakage,
query replay failures, and fused-ranking dominance hard failures:

```bash
CHART_EVAL_LIMIT=500 \
CHART_EVAL_TOP_K=1000 \
CHART_EVAL_HOLDOUT_MAX_OVERLAP=0 \
CHART_EVAL_REQUIRE_NO_FAILURES=1 \
CHART_EVAL_REQUIRE_FUSED_DOMINATES=1 \
CHART_EVAL_HOLDOUT_REPORT=eval/out/holdout-report.json \
CHART_EVAL_RECDS_REPORT=eval/out/recds-report.json \
scripts/eval_live.sh
```

After a Gemma cascade smoke has written `events` on the slice, rerun the live
smoke with:

```bash
scripts/refresh_facets.sh --fields age_band,gender,events
CHART_REQUIRE_EVENT_FACETS=1 scripts/smoke_live.sh
```

That requires the event facet snapshot, with SHA provenance, as well as the base
age/gender facets. `scripts/refresh_facets.sh` writes
`CHART_FACET_REFRESH_REPORT` with the refreshed field list.
The final audit prints the same Phase-4 event gate as:

```bash
scripts/phase4_event_smoke.sh
```

`scripts/preflight.sh` runs unit tests, compile checks, GPU image build dry-run,
a pinned Wrangler Worker dry-run, and deploy manifest dry-run locally. Set
`CHART_PREFLIGHT_DOCKER=1`
to include a Docker daemon check before building the GPU images. Set
`CHART_PREFLIGHT_LIVE=1` to make preflight also resolve the gateway key and run
the live smoke and Phase-6 gate wrappers, including their report artifacts; local
preflight does not require the key. Local dry-run reports are written under
`${TMPDIR:-/tmp}` so preflight does not overwrite accepted gate evidence in
`eval/out/`.
`scripts/full_status.sh` reports Pipeline progress, classifier UDF queue counts,
and facet snapshot visibility while the full index/classify gates are running.
Its JSON includes the target Layer `pipeline_id` (`chart-notes` by default),
which is distinct from the Kubernetes GPU embed Pipeline CR name
(`chart-embed-gpu`), plus the accepted embed/classifier cost baseline report
paths and estimates, and writes `CHART_PHASE6_STATUS_REPORT`.
`scripts/gate_report.sh` summarizes those checks into the Phase-6 gates,
including full-row snapshot coverage and SHA provenance for every configured
facet field and accepted embed/classifier cost baselines. The JSON includes a
`failures` array with the specific incomplete gate and count/provenance/cost
reason. Use
`scripts/gate_report.sh --require-complete` as the hard exit gate; it exits
non-zero until the full index, classifier, and facet snapshots are all complete
with zero queue failures, and writes `CHART_PHASE6_GATE_REPORT` for the final
audit trail.
`scripts/final_gate.sh` is the final phase-gate audit. It delegates to
`scripts/plan_audit.sh --requirements --require-complete`, reads
the slice, smoke, facet, budget, eval, build, deploy, unpause, and Phase-6 gate
reports from `eval/out/` (or their `CHART_*_REPORT` overrides), writes
`CHART_PLAN_AUDIT_REPORT`, and exits non-zero until every required report proves
its gate. Incomplete audit reports include `next_steps`, an ordered command list
for the missing or unsatisfied gates. Each step includes `ready` and `requires`;
blocked steps also include `blocked_by`, and present-but-invalid reports include
`details`. The hard gate includes local requirement diagnostics by default; set
`CHART_PLAN_AUDIT_PROBE_GATEWAY_KEY=1` when you also want it to prove the
1Password gateway-key path.
For example, base smoke is blocked by the slice index and base facet-refresh
gates until both reports pass, while event-facet visibility remains tied to the
classifier/facet gates. Use `scripts/plan_audit.sh --ready` to print only the
currently runnable next-step commands. Use
`scripts/plan_audit.sh --requirements` when you need local diagnostics for the
gateway key, full ReCDS retrieval corpus, Docker, kubectl, Layer autoscaling, and
operator-acceptance prerequisites listed by those steps.

The indexer refuses any unbounded local run by default. Full-corpus indexing is a
Phase-6 cost-gated operation; use a bounded `--limit` for smoke work, and only set
`CHART_ALLOW_FULL_CPU_INDEX=1` after accepting the full-index cost path.
The production full-index path is `deploy/pipeline-embed.yaml`, which runs
`python -m indexer.embed` on the GPU pool after the Phase-6 gate is accepted.
Layer owns source and embed scaling; the GPU embed Pipeline declares a warm
window so adjacent batches reuse the same warm node before returning to zero.

The Gemma cascade (`functions/`) and the ReCDS eval (`eval/`) are GPU- and
gateway-bound respectively; see their READMEs. The local seams are covered by
unit tests; the full live demo still requires the gateway key, indexed rows, and
the GPU classifier run.

For the production Worker, set `LAYER_GATEWAY_API_KEY` and `CHART_QUERY_EMBED_URL`
as Wrangler secrets/vars. Static assets are served through the Workers Static
Assets binding in `wrangler.jsonc`.
