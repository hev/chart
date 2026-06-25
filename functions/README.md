# `functions/` — the transform-runtime act

Routing is chart's headline; the transform runtime is the second act — "one
primitive for every per-row job" over clinical notes (CLAUDE.md § Product Frame).
Each file is a `Function`: discovery, batching, claim leases, retries, and
`patch_columns` writeback are the gateway's, not ours.

| Function | Pool | Output | Role |
|---|---|---|---|
| `classify_events.py` | **gpu** | `events` + derived labels | **The GPU showcase.** A Gemma cascade (vLLM, guided decoding) finds clinical events — medication discontinuation the headline — and derives facet labels in the same pass. |
| `tag_specialty.py` | cpu | `specialty` | Legacy fallback; the Gemma cascade now derives this in the main GPU pass. |
| `extract_clinical_fields.py` | cpu | `diagnosis_category` (+more) | Legacy fallback; the Gemma cascade now derives the facet labels in the main GPU pass. |
| `scan_phi.py` | cpu | `phi_flag` | De-id *verification* — the per-row safety transform a real notesearch wants. |

## The new showcase: a Gemma clinical-event cascade

`classify_events.py` is the thing this demo adds beyond the routing story. It is a
**GPU, open-weight (Gemma), vLLM guided-decoding** cascade — the RFC 0072 shape
(designed for moment, whose deployed enrichment is still rule-based), built here
for the first time and specialized to **clinical events**:

- **Tier 1 — digest** (the one GPU pass): note → typed events `{type, drug, reason}`
  + summary, guided-decoded to a closed event taxonomy.
- **Tier 2 — derived labels** (no extra GPU): `events`, `has_med_discontinuation`,
  `has_adverse_event`, and the `diagnosis_category` / `specialty` facets read
  straight from the digest and are patched through the injected `tpuf` client —
  **one GPU pass, many labels**, subsuming the two CPU enrichers above.
- **Tier 3 — gated refine**: only on notes with a discontinuation, a targeted pass
  for the structured `discontinuation_reason`.

Why medication discontinuation? It is a concrete pharmacovigilance / cohort signal
a clinical customer (Trio) actually wants, and it composes with the routing
headline: an `events` filter over an `Auto`-routed search — *"discontinued statins
due to an adverse reaction"* — is a query neither keyword nor a flat note search
answers. `deploy/functions-events.yaml` is the `pool: gpu`, scale-to-zero Function.

`digest` runs vLLM structured output against `DIGEST_SCHEMA`; `refine_discontinuation`
is gated by the Tier-1 discontinuation event and returns the structured reason.

Layer provides the authoritative full-run cost gate for the deployed GPU
Function. Save that accepted Layer cost report to `CHART_PHASE4_CLASSIFY_REPORT`
(`eval/out/classify-events-budget.json` by default):

```bash
scripts/layer_cost_report.sh --kind classifier --accept --signal-reviewed --out eval/out/classify-events-budget.json
```

The `--signal-reviewed` flag is required when accepting the classifier report so
the audit records that medication-discontinuation examples and derived labels
were reviewed.

For local/off-platform smoke work, the timing helper can also produce the same
audit JSON shape:

```bash
uv run --extra classifier python -m functions.measure_classify_events \
  --limit 50 \
  --discontinuation-only \
  --examples 5 \
  --accelerator gpu \
  --gpu-hourly-usd 2.50 \
  --max-full-hours 4 \
  --max-full-usd 25 \
  --min-med-discontinuations 1 \
  --min-review-examples 1 \
  --out eval/out/classify-events-budget.json
```

The JSON output includes the sampled event counts, the 167k-note estimate fields
expected by the audit, and a bounded `examples` list with note previews plus
derived labels. Use those examples for the required hand check before accepting
the full classify cost. Supplying `--max-full-hours` or `--max-full-usd` turns the
local helper into a hard gate for off-platform timing runs.
`--min-med-discontinuations` makes the smoke fail if the sample does not exercise
the headline medication-discontinuation path. `--min-review-examples` also
requires reviewable discontinuation examples for the hand check before accepting
the full classify cost.
Applying `deploy/functions-events.yaml` for the discovery backfill requires both
`CHART_APPLY_CLASSIFIER=1` and `CHART_ACCEPT_PHASE4_CLASSIFY_COST=1` after this
gate is accepted, plus `CHART_PHASE4_CLASSIFY_REPORT` pointing at an accepted
saved report with budget and signal checks.

For a one-batch UDF smoke against the live queue, use the explicit one-shot flag:

```bash
uv run --extra classifier python -m functions.classify_events --once
```

The classifier runtime is Linux/vLLM/GPU-oriented. On macOS, `uv --extra
classifier` keeps `vllm` out of the local environment for CLI inspection, so run
the one-batch smoke inside the GPU Function image or on a Linux GPU host.

`CHART_UDF_ONCE=1` remains supported for the deployed worker environment, but
`--once` keeps local smoke commands inspectable with `--help`.

After a smoke classify run writes `events`, refresh that facet snapshot without
reindexing and make the live smoke require it. The guarded repo helper runs the
full sequence and refuses early when the Linux/vLLM classifier runtime is not
available:

```bash
scripts/phase4_event_smoke.sh
```

The underlying steps are:

```bash
uv run --extra classifier python -m functions.classify_events --once
scripts/refresh_facets.sh --fields age_band,gender,events
CHART_REQUIRE_EVENT_FACETS=1 scripts/smoke_live.sh
```
