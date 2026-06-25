# `deploy/` — the in-cluster declarative twin

The CR bundle that stands chart up *declaratively*, the equivalent of the
imperative `uv run python -m indexer` + the runtime gateway client. Both spellings
round-trip through one CRD/REST schema (CLAUDE.md § Design Bias); this is the YAML
side. It is also, one-to-one, the manifest set RFC 0076 specifies.

| File | Owns | Imperative twin |
|---|---|---|
| `vectorstore.yaml` | upstream Turbopuffer connection + `deriveFromStore` inbound auth | `chart_common/gateway.py:make_client()` |
| `warehouse.yaml` | the data source identity (`huggingface`, public/no-Secret) | `chart_common/config.py` dataset pin |
| `pipeline.yaml` | staged ingestion: source → chunk | `python -m indexer --dry-run --limit ...` load/chunk equivalent |
| `pipeline-embed.yaml` | GPU embedding for the full index | `python -m indexer.embed` worker path, cost-gated |
| `index.yaml` | operational policy: the facet snapshots, scan fan-out, consistency | `gateway.py:materialize_facet_snapshots()` |

Two things to know:

- **No new Warehouse kind, no new gateway machinery.** The whole bundle is the
  shipped `huggingface` Warehouse (RFC 0053), `Auto`/`HybridText` routing
  (RFC 0044/0057), and the RFC 0056 chunk model. That is the RFC 0076 thesis: a
  new vertical with zero new gateway code.
- **The Trio swap is one block.** `warehouse.yaml` carries the `kind: huggingface
  → snowflake` swap that points this exact Pipeline at Trio's real `notesearch`
  notes. Nothing else in the bundle changes.

Apply order: `namespace` → `vectorstore` → `warehouse` → `pipeline`
→ `pipeline-embed` → `index` → `functions-events`. Apply `pipeline-embed` only
after the Phase-6 full-index gate is accepted.

Use the ordered helper for validation/application:

```bash
scripts/deploy_apply.sh                # kubectl client-side dry-run
scripts/deploy_apply.sh --server-dry-run
scripts/deploy_apply.sh --apply        # real apply, same order
```

The helper writes `CHART_DEPLOY_APPLY_REPORT` with the mode, manifest order,
classifier apply decision, Kubernetes context, context-confirmation flag,
classifier cost-acceptance flag, and final status.
Refused `--apply` runs also persist an `error` field in that report so the final
audit can surface the exact missing context, Secret, or cost-gate refusal.

`--apply` installs the base stack and the GPU embed Pipeline. It validates
`functions-events.yaml`, but skips applying it unless
`CHART_APPLY_CLASSIFIER=1` and `CHART_ACCEPT_PHASE4_CLASSIFY_COST=1` are set,
and `CHART_PHASE4_CLASSIFY_REPORT` points at an accepted saved classifier report.
That Function has a `discovery` trigger and can start the full GPU classify
backfill, so only opt in after the Phase-4 smoke and cost gate are accepted:

```bash
CHART_K8S_CONTEXT_CONFIRM="$(kubectl config current-context)" \
CHART_APPLY_CLASSIFIER=1 \
CHART_ACCEPT_PHASE4_CLASSIFY_COST=1 \
scripts/deploy_apply.sh --apply
```

The final audit prints the same real apply as a compact next-step command:

```bash
CHART_APPLY_CLASSIFIER=1 CHART_ACCEPT_PHASE4_CLASSIFY_COST=1 CHART_K8S_CONTEXT_CONFIRM="$(kubectl config current-context)" scripts/deploy_apply.sh --apply
```

Before `--apply`, create the two runtime Secrets in namespace `chart`:

- `chart-turbopuffer`, key `credential` — upstream Turbopuffer key for the
  VectorStore.
- `chart-gateway`, key `LAYER_GATEWAY_API_KEY` — gateway key injected into the
  GPU embed/classifier workers.

Use `deploy/secrets.example.yaml` as the shape reference only; do not apply it
with placeholder values. The helper checks that both Secrets exist before applying
the namespaced runtime resources.

The full-index embed worker command is:

```bash
uv run python -m indexer.embed --once
```

In-cluster, the operator injects `HEVLAYER_PIPELINE_ID`; locally it defaults to
`chart-notes`.

Build GPU worker images from the repo root:

```bash
scripts/build_gpu_images.sh          # print exact buildx commands
scripts/build_gpu_images.sh --build  # local build, requires Docker daemon
scripts/build_gpu_images.sh --push   # build and push manifest image tags
```

The manifest image tags are pinned (`plan-20260624`) rather than `latest`, so a
deploy points at the same worker build that passed the local gates. Override tags
with `CHART_EMBED_IMAGE` / `CHART_CLASSIFIER_IMAGE`, and override the editable
hevlayer BuildKit context with `CHART_LAYER_CLIENT_CONTEXT`. Real build/push
modes fail before Docker starts if that Layer client path does not exist.
Set `CHART_ECR_REPOSITORY_URL` to push both workers to one ECR repository with
distinct tags, or set the two image variables directly. For ECR pushes the helper
runs `aws ecr get-login-password` unless `CHART_ECR_LOGIN=0`; `AWS_REGION`
defaults to `us-east-1`. GPU worker images default to `CHART_GPU_PLATFORM=linux/amd64`.
Set `CHART_GPU_BUILDER=depot` to use Depot CLI builds; `CHART_DEPOT_PROJECT_ID`
defaults to the Layer production project used by `../layer/scripts/deploy-layer.sh`.
Set `CHART_PRELOAD_EVENTS_MODEL=` to build the classifier image without baking a
Hugging Face model cache when model credentials are not available at build time.
`scripts/build_gpu_images.sh` writes `CHART_GPU_BUILD_REPORT` with the exact
buildx commands, image tags, mode, and final status. Failed real build/push
prerequisite checks and Docker build failures also write the report with
`status: failed` and an `error` field so the final audit does not rely on a stale
dry-run artifact.

Layer provides the authoritative full-index embed cost gate for the deployed GPU
Pipeline. Save that accepted Layer cost report to
`CHART_PHASE6_EMBED_BUDGET_REPORT` (`eval/out/embed-budget.json` by default)
before unpausing:

```bash
scripts/layer_cost_report.sh --kind embed --accept --out eval/out/embed-budget.json
```

For local/off-platform timing, the measurement helper can produce the same audit
JSON shape:

```bash
uv run --extra search python -m indexer.measure_embed \
  --limit 500 \
  --accelerator gpu \
  --gpu-hourly-usd 2.50 \
  --max-full-hours 3 \
  --max-full-usd 6 \
  --out eval/out/embed-budget.json
```

Supplying `--max-full-hours` or `--max-full-usd` makes the local helper a hard
gate for off-platform timing runs. The saved JSON report is the audit artifact
consumed by the unpause helper.

After Phases 1-5 are green and the Phase-6 embed cost gate is accepted, start the
full GPU embedding run explicitly:

```bash
scripts/phase6_unpause_embed.sh          # status only
CHART_K8S_CONTEXT_CONFIRM="$(kubectl config current-context)" \
CHART_ACCEPT_PHASE6_EMBED_COST=1 \
scripts/phase6_unpause_embed.sh --yes
scripts/phase6_prepare_embed_drain.sh --status
CHART_K8S_CONTEXT_CONFIRM="$(kubectl config current-context)" \
scripts/phase6_prepare_embed_drain.sh --yes
scripts/full_status.sh
scripts/gate_report.sh
scripts/gate_report.sh --require-complete
```

The final audit prints the same unpause as a compact next-step command:

```bash
CHART_ACCEPT_PHASE6_EMBED_COST=1 CHART_K8S_CONTEXT_CONFIRM="$(kubectl config current-context)" scripts/phase6_unpause_embed.sh --yes
```

The unpause helper checks `chart/chart-gateway` and the Pipeline CR before it
patches, and refuses `--yes` unless `CHART_K8S_CONTEXT_CONFIRM` exactly matches
the active `kubectl config current-context` and `CHART_ACCEPT_PHASE6_EMBED_COST=1` is set
after the embed cost gate is accepted and
`CHART_PHASE6_EMBED_BUDGET_REPORT` points at an accepted report with both hour
and dollar checks. Override the target with `CHART_K8S_NAMESPACE` and
`CHART_EMBED_PIPELINE_CR` only if the manifest names change.
It writes `CHART_PHASE6_UNPAUSE_REPORT` with the target Pipeline, budget report,
final status, and any refusal `error`.

Layer should own worker scale-up/down through its Pipeline `ScaledObject`s. The
window scripts below are a temporary Phase-6 backfill workaround for the current
cluster state where `scripts/full_status.sh` reports the Layer-managed
`chart-ingest-worker` and `chart-embed-gpu-worker` ScaledObjects as not ready
because KEDA bearer auth is generated with an empty token. Keep platform
follow-ups in `../LAYER_IMPROVEMENTS.md`; do not treat manual `kubectl scale` as
the intended steady-state operating model.

When that autoscaling issue is present, the full run can still drain in bounded
windows: let `chart-ingest-worker` stage work, then pause it before the GPU
embedder drains the pending queue. The gateway claim path uses
`FOR UPDATE SKIP LOCKED`; a continuously staging source worker can hold segment
locks long enough for claim checks to return no work while pending rows exist.
`scripts/phase6_prepare_embed_drain.sh --yes` scales `chart-ingest-worker` to
zero, clears stale idle pipeline-segment transactions, and verifies a
claim/release probe before the embedder drains. It writes
`CHART_PHASE6_DRAIN_REPORT`.

Stage the next source window only after the pending queue drains:

```bash
CHART_PHASE6_SOURCE_START_OFFSET=302 \
CHART_PHASE6_SOURCE_MAX_ROWS=1000 \
CHART_K8S_CONTEXT_CONFIRM="$(kubectl config current-context)" \
scripts/phase6_stage_source_window.sh --yes
```

The staging helper sets the source worker offset/window env, scales
`chart-ingest-worker` to one replica, and refuses to start when pending pipeline
documents already exist unless `CHART_PHASE6_ALLOW_STAGE_WITH_PENDING=1` is set.
After it stages the window, run `scripts/phase6_prepare_embed_drain.sh --yes`
again before letting the GPU embedder drain.

`scripts/full_status.sh` also records the accepted embed/classifier cost baseline
reports and estimates in `CHART_PHASE6_STATUS_REPORT`. It also records the
Layer-managed Kubernetes `Function` and `ScaledObject` status so the final audit
can distinguish a gateway UDF 404 from an installed-but-paused Function, and can
show whether KEDA is actually ready to own autoscaling. The final audit requires
the cost baselines so the full-run status is tied back to the cost gates that
were accepted before unpause/deploy.
The `--require-complete` gate exits non-zero until the full Pipeline and
classifier queues are drained with zero failures and every configured facet has a
full-corpus snapshot with SHA provenance, and both embed/classifier cost
baselines are accepted. Its JSON includes a `failures` array with the specific
incomplete gate and count/provenance/cost reason.
