# Layer Improvements Log

This captures Layer/platform follow-ups found while running the chart Phase 6 full-index pipeline.

## KEDA autoscaling cannot initialize with bearer auth

- Date observed: 2026-06-25
- Affected resources: `chart/chart-ingest-worker` and `chart/chart-embed-gpu-worker` `ScaledObject`s.
- Symptom: both Layer-managed `ScaledObject`s report `READY=False`, `ACTIVE=Unknown`.
- KEDA message:
  - `error parsing prometheus metadata: bearer token=<empty> is required when bearer auth is enabled`
- Current generated trigger shape:
  - `type: prometheus`
  - `serverAddress: http://layer-gateway.layer.svc.cluster.local:8080/v2/metrics`
  - `authModes: bearer`
  - `authenticationRef` points at per-worker auth resources.
- Current generated `TriggerAuthentication` shape:
  - `secretTargetRef.parameter: bearerToken`
  - `secretTargetRef.name: layer`
  - `secretTargetRef.key: turbopuffer-api-key`
- `scripts/full_status.sh` now records this as
  `kubernetes.trigger_authentications`, and `scripts/plan_audit.sh --requirements`
  includes `bearerTokenRef=layer/turbopuffer-api-key` in the Phase 6 runtime
  failure details.
- The chart final audit now treats this as a first-class `layer_autoscaling`
  requirement. Phase 6 runtime and final completion remain not ready until the
  Layer-managed `ScaledObject`s report `Ready=True`.
- Live redacted Secret status currently reports
  `secret_exists=False,key_exists=False,value_present=False` for
  `chart/layer` key `turbopuffer-api-key`. No Secret value is recorded.
- Impact: KEDA does not create/maintain a healthy HPA, so Layer autoscaling does not own worker replicas. Manual `kubectl scale` was used only as an operational workaround to make bounded progress.
- Suggested platform fix: ensure the generated `TriggerAuthentication` supplies a non-empty bearer token from the workload secret, or omit bearer auth when the in-cluster metrics endpoint does not require it.

## GPU node churn from manual scale-to-zero workaround

- Date observed: 2026-06-25
- Symptom: every manual embed drain after scaling `chart-embed-gpu-worker` to zero required a fresh `g4dn.xlarge` nodeclaim and a new ECR image pull.
- Typical Kubernetes warnings during scale-up:
  - `node(s) had untolerated taint(s)`
  - `Insufficient nvidia.com/gpu`
- These were scheduling/provisioning warnings, not application crashes. The embed pod ran with `RESTARTS=0` and drained successfully once scheduled.
- Impact: each drain paid avoidable Karpenter provisioning latency plus a ~7.8 GB image pull.
- Suggested platform fix: after KEDA auth is fixed, let Layer own autoscaling. For large backfills, consider a configured warm window or min-replica override so GPU nodes are not reclaimed between adjacent batches.

## Source and embed pipeline coordination

- Date observed: 2026-06-25
- The current chart docs note that continuous source staging can hold pipeline-segment locks long enough for embed claim checks to return no work while pending rows exist.
- Operational workaround used: stage bounded source windows, stop source, claim-check, then drain embed.
- Suggested platform fix: make this coordination a first-class Layer behavior for pull-dispatch pipelines that share a `pipelineId`, so users do not need external scripts to avoid claim starvation.

## Function CR exists but gateway UDF status is missing

- Date observed: 2026-06-25
- Affected resource: `chart/chart-classify-events`.
- Kubernetes state:
  - `Function` CR exists with `spec.paused: true`.
  - `status.conditions[Ready]` reports `Function spec.paused=true; deployment scaled to zero`.
  - `chart-classify-events-worker` deployment exists and is owned by the Function.
- Gateway state in `eval/out/phase6-status-report.json`:
  - `udf.error.status_code: 404`
  - `udf.error.message: UDF 'chart-classify-events' not found`
- Impact: Phase 6 gate reports `udf_installed=false` even though the Kubernetes Function object exists. This makes it hard to tell whether the missing piece is expected pause semantics, gateway registration, or the status endpoint only exposing active UDFs.
- Suggested platform fix: make Function registration/status semantics explicit. Either register paused Functions with a `paused` status in the gateway, or document that paused Functions intentionally return 404 and expose the install state through another status surface.

## Scans API has no hybrid_text/fuzzy selector (live counts can't mirror the keyword route)

- Date observed: 2026-06-25
- Affected surface: Scans API ranked selectors (`apps/layer-gateway/src/routes/scans.rs`), consumed by chart's `/api/facet-counts` live-count rail (`search/app.py`, `src/worker.js`, `chart_common.gateway.count_selector`).
- Symptom: a `hybrid_text`-routed query returns ranked results but the live rail reports **"0 matching cases"** with every facet count at 0. Reproduces on the demo's own chips — `afib`, `aspirn` (typo), `CABG`.
- Root cause: the Scans API exposes exactly two ranked selectors —
  - `fts` → `[field, "BM25", query]`, an **exact** BM25 predicate (`scans.rs` ~line 1263), and
  - `ann` → a cosine-distance radius ball around a query vector.
  The `hybrid_text` route (`routes/hybrid_text.rs`, RFC 0057) ranks one BM25 leg **plus one `Fuzzy(token)` leg per token**, plus a neutral-rank fuzzy fallback when BM25 is empty. So a query whose matches surface only through the fuzzy legs (an abbreviation or typo — exactly the queries `hybrid_text` exists to win) is retrieved by the route but counted as **zero** by the only lexical scan selector. The `ann` selector can't stand in either: `hybrid_text` is vectorless, so a semantic ball counts a different, route-irrelevant set.
- The same gap weakens `fused` counts: a `fused` result that ranked in via the semantic leg (BM25 = 0 for it) is also missed by the exact-`fts` count.
- chart-side mitigation (not a fix): the UI enforces a "total ≥ hits on screen" honesty invariant and withholds the live count + per-facet scans when violated, falling back to the corpus snapshot rail with a "live counts need a fuzzy-aware scan" note. This stops the contradiction but loses the live-count showcase for the keyword/fuzzy route.
- Suggested platform fix: give the Scans API a selector that mirrors `HybridText` retrieval — e.g. an `fts` option that enables the same per-token `Fuzzy` ladder the route uses (the `text` column is already declared `fuzzy: true`), or, more generally, let a scan count the match set of an `Auto`/`HybridText` `rank_by` expression directly so live counts always agree with what the route returns. Either lets the count selector follow the routing decision faithfully for all three routes, not just `semantic`.
