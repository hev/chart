# chart

**Clinical patient-notes search that shows its routing.** One search box; the
gateway picks the retrieval strategy — keyword, fused, or semantic — and tells you
*why*. A query-routing demo on [hev layer](../layer), and the public twin of
clinical `notesearch`.

> Notes are published, de-identified case reports (PMC-Patients, CC-BY-NC-SA) —
> **not raw EHR, not for clinical use**. This is a search demo; it gives no medical
> advice. Backed by **RFC 0076** in `../layer/docs/rfcs`.

> **Name is provisional.** "chart" (a patient chart) is a working name in the
> `shelf`/`shop` family; RFC 0076 leaves the public name open. Renaming is a `mv`.

## Why clinical notes

Clinical search has the **sharpest bimodal query distribution** there is.
Clinicians search both by exact token and by clinical picture:

| You type | Tokens | Route | Why |
|---|---|---|---|
| `metformin 500mg` | 2 | `hybrid_text` | drug + dose; exact lexical, typo-tolerant |
| `CABG` · `afib` · `aspirn` | 1 | `hybrid_text` | abbreviations / typos — ANN noise, BM25+fuzzy win |
| `chest pain radiating to left arm` | 5 | `fused` | clinical phrase; both legs, RRF-merged |
| `elderly woman with progressive dyspnea and bilateral lower-extremity edema` | 9 | `semantic` | a clinical picture in prose; ANN over the embedding |

The routing badge renders the gateway's own decision (RFC 0044) — the demo
*teaches* the router. This v1 follows shelf and embeds up front so semantic/fused
routes execute in one hop; RFC 0044 deferral is the fast-follow that will make
short keyword traffic skip embedding entirely.

One rung above the router, the **Agentic search** toggle runs the same query
through a configured reasoning loop (`POST /v2/agents/chart-notes/query`,
RFC 0074 / `deploy/agent.yaml`): a model reformulates the query, fans out for
recall, grades the candidates, and returns the standard row shape. With
provenance on, the inspector shows the agent's *plan* — the reformulated
variants and inferred filters — and each hit carries its retrieval + relevance
scores. Same "show the decision" DNA, one level up. The backing store is the
gateway's default Turbopuffer store today; `deploy/vectorstore.yaml`
(`kind: search`, the first-party hev search engine) is the declared target but
has not been applied to the shared cluster yet — the cutover is pending.

## The features, and where they're documented

Everything visible in the UI is a gateway feature; the app composes them. The
same tour is served by the app itself at [`/help.html`](web/static/help.html).

| Feature | What you see | hev layer docs |
|---|---|---|
| **Query routing** | The `Auto` router picks `hybrid_text` / `fused` / `semantic` per query; the badge is the gateway's own decision, echoed back | [Query routing](https://hevlayer.com/docs/api/query/#query-routing) |
| **Hybrid text** | BM25 + per-token fuzzy fused in one lexical leg (`aspirn` finds aspirin, and the response says a fuzzy match surfaced it); `fused` merges it with the semantic leg via RRF; the rail's per-search counts come from the same machinery (Scans) | [Hybrid text fusion](https://hevlayer.com/docs/api/query/#hybrid-text-fusion) · [Scans](https://hevlayer.com/docs/api/scans/) |
| **Agentic search** | The toggle runs the query through a configured `Agent` (reasoning loop): reformulate → fan out → grade; the sidebar becomes the run inspector and each hit carries `$agent` provenance | [Agents API](https://hevlayer.com/docs/api/agents/) · [Agent CRD](https://hevlayer.com/docs/kubernetes/agent-crd/) |
| **UDF cascade (self-hosted model)** | `events`/`specialty`/`diagnosis_category` facets are written back by a Gemma classifier running on cluster GPUs via the `Function` runtime (scale-to-zero, guided decoding) — one GPU pass, many labels, run as a backfill | [Function CRD](https://hevlayer.com/docs/kubernetes/function-crd/) · [GPU classifier](https://hevlayer.com/docs/kubernetes/function-crd/#gpu-classifier) |

## Two things this demo proves

1. **Query routing, with a number.** chart is the first Layer demo with **real
   relevance judgments** — PMC-Patients **ReCDS** qrels — so the hybrid/routing
   claim is measured, not asserted (`eval/`). The honest split: ReCDS quantifies
   retrieval *quality* now; the *routing* claim needs a built bimodal query set
   (`eval/bimodal_queries.md`).
2. **A Gemma clinical-event cascade (the GPU showcase).** An open-weight Gemma
   cascade (vLLM, guided decoding, scale-to-zero) reads each note once and pulls
   out **clinical events** — *medication discontinuation* the headline — plus the
   facet labels in the same pass (`functions/classify_events.py`). One GPU pass,
   many labels (RFC 0072). It composes with routing: an `events` filter over a
   routed search — *"discontinued statins due to an adverse reaction"*.

### The cascade vs. the Batch API (the cost pitch)

The natural baseline for LLM classification at rest is a provider batch API —
Trio classifies with the **Claude Message Batches API on Haiku** today (50% off
realtime: $0.50 in / $2.50 out per MTok). The cascade's pitch is that a
self-hosted open-weight model on Layer's Function runtime beats that baseline
on marginal cost while keeping the data in-cluster:

| Path (per note ≈ 1.2k in / 300 out tokens) | 11.4k-note backfill | 167k full corpus |
|---|---|---|
| Haiku 4.5 realtime | ~$31 | ~$460 |
| Haiku 4.5 **Batch** (the baseline) | ~$16 | ~$235 |
| **Cascade** — Gemma-2-9B, 2× `g5.xlarge` ($1.01/hr each), **measured** | **$8.2** | **~$120** |

**Measured ~2× cheaper than Haiku Batch** (11,373 notes in ~4h at 48–58
notes/min across two GPUs, zero failures: $8.10 GPU + $0.13 Layer-metered
writes/scans/storage — the earlier ~$3–6 projection assumed larger claim
sizes; `batchSize` is still 16 pending hev/layer#148, so there's 2–3×
throughput headroom left on the table). Three structural reasons it holds up:

- **One pass, many labels.** The cascade derives `events`, `specialty`,
  `diagnosis_category`, `has_med_discontinuation`, and the discontinuation
  reason from a single digest — a per-label Batch-API pipeline pays per label.
- **Continuous batching.** The worker feeds each claimed batch through one
  vLLM `generate()` (`run_batched_worker`) and writes labels back as one
  multi-row `patch_columns` — the GPU stays saturated, not round-tripping.
- **Scale-to-zero.** Layer's Function runtime (KEDA on UDF queue depth) means
  the GPU exists only while the queue is non-empty; idle cost is $0, same as
  a batch API.

The costs above are projections from measured token counts and the on-demand
instance price; the authoritative number is Layer's own cost report
(`scripts/layer_cost_report.sh --kind classifier`), captured as part of the
Phase 4 gate once the backfill completes. Data locality is the non-cost half
of the pitch: notes never leave the cluster, which matters more on Trio's real
EHR corpus than on this public one. (For the runtime gap that would let Layer
drive provider batch APIs *natively* for customers who prefer them, see Layer
RFC 0093 — this workload is its acceptance case.)

## The Trio twin

This is the public stand-in for Trio's `notesearch` (Snowflake-fed clinical-note
search). Same Pipeline, same `Auto` read path, same UX; the only difference is one
block — `warehouse.yaml` swaps `kind: huggingface` → `kind: snowflake` to point at
Trio's real notes. Embedding is the **same model** notesearch runs
(`snowflake-arctic-embed-m-v1.5`), so retrieval behavior transfers. The public
demo proves the UX and the plumbing; routing quality on real EHR is validated only
on Trio's corpus, which never leaves their Snowflake.

## Layout

```
chart_common/   shared lib: config (Arctic + PMC pin), embed (Arctic query prefix),
                records (PMC-Patients → row), gateway (client, schema, snapshots)
indexer/        load PMC-Patients → embed → upsert → materialize facet snapshots
                (`indexer.embed` is the GPU Pipeline worker for the full run)
search/         FastAPI dev backend (Auto-routed; the Arctic query-prefix hop)
src/            Cloudflare Worker prod backend (twin of search/)
functions/      the transform-runtime act: classify_events (Gemma, GPU) +
                tag_specialty / extract_clinical_fields / scan_phi (CPU)
eval/           the ReCDS qrels harness — the routing/hybrid number
web/static/     the shelf-shaped single-page UI + the clinical routing chips
deploy/         the in-cluster declarative twin (VectorStore/Warehouse/Pipeline/
                Index + the GPU events Function) and GPU worker Dockerfile
```

## Run

```bash
cp .env.example .env          # set LAYER_GATEWAY_API_KEY (the upstream tpuf key)
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
`scripts/gate_report.sh` summarizes those checks into the PLAN.md Phase-6 gates,
including full-row snapshot coverage and SHA provenance for every configured
facet field and accepted embed/classifier cost baselines. The JSON includes a
`failures` array with the specific incomplete gate and count/provenance/cost
reason. Use
`scripts/gate_report.sh --require-complete` as the hard exit gate; it exits
non-zero until the full index, classifier, and facet snapshots are all complete
with zero queue failures, and writes `CHART_PHASE6_GATE_REPORT` for the final
audit trail.
`scripts/final_gate.sh` is the final PLAN.md audit gate. It delegates to
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
