# chart — implementation plan

From scaffold (RFC 0076, STUB/TODO seams) to a live demo: routing on a small
slice first, then the full 167k index + classify run. Each phase has a
**small-scale test** and an **exit gate**; the two full runs are the last, gated
step. The critical path to a *showable* demo is **0 → 1 → 2 → 3** (routing on ~2k
notes); the cascade (4) and eval (5) parallelize once the index exists.

Ground truth already confirmed from the sibling repos:

- The query call is `layer.query_namespace(ns, body)` with
  `rank_by = ["text", "Auto", query, {"vector": v}]` — `../shelf` embeds the query
  **up front** and hands the vector to `Auto` in the 4th slot so semantic/fused
  run in **one hop**, then reads `resp.routing` / `resp.hybrid`. Match that.
- write / snapshot / history client methods in `chart_common/gateway.py` already
  match shelf's working ones.
- The `@udf` + `run_udf_worker` wiring and the GPU `Function` CRD are pulled from
  `../moment`.

---

## Phase 0 — De-risk the seams (cheap, before any code)

Each is a known unknown that would otherwise blow up a later phase.

| Check | Confirm / decide | Fallback |
|---|---|---|
| **Client API** | `query_namespace` body shape; the `nearest_to_id` spelling for "similar patients"; `resp.routing`/`.hybrid` fields (read `../shelf/search/app.py`). | — |
| **Embedder** | Does `fastembed` list `Snowflake/snowflake-arctic-embed-m-v1.5`? Confirm 768-dim and the exact v1.5 query prefix. | Swap `embed.py` to `sentence-transformers` (torch) — the prefix logic is unchanged. |
| **Dataset** | Inspect PMC-Patients schema vs `records.py` assumptions (`patient_uid`, `patient`, `age` list, `gender`, `relevant_articles`, `similar_patients`); **pin a real commit SHA** in `config.py`. | — |
| **Gateway + key** | Smoke the live endpoint with the `mesh-staging` key (AGENTS.md command). | — |
| **GPU for the cascade** | Decide where Gemma runs for the smoke and the full run: the layer cluster GPU pool (the real Function) vs. a Modal/Colab box for the smoke (the CLIP-demo off-platform pattern). | — |

**Exit:** every seam has a confirmed shape; `config.py` SHA pinned; `tests/`
scaffolded (pytest, mirroring `../moment/tests`).

---

## Phase 1 — Small index

The indexer is already written; run it on a slice.

```bash
uv run python -m indexer --dry-run --limit 500     # load + embed, no gateway
uv run python -m indexer --limit 2000              # real upsert
```

- Add `tests/test_records.py` — age parsing (`[[54.0,'year']]` → 54 → `adult`),
  row shaping, missing-field tolerance. Pure functions, cheap, high-value.

**Exit:** ~2k notes queryable; schema correct, `vector` 768-d, `age_band`/`gender`
populated, `similar_patient_ids` present.

---

## Phase 2 — Routing, the headline (on the slice)

Wire `search/app.py:/api/search` to `query_namespace` with the
`["text","Auto",q,{vector}]` tuple; return the `routing`/`hybrid` echo. Wire "find
similar patients" (`nearest_to_id`).

- Add `tests/test_routing.py`: **each chip in `web/static/queries.json` lands on
  its expected route.** That assertion *is* the demo's correctness contract.
- **Decision — embed strategy:**
  - *v1 (recommended):* embed up front, one hop, like shelf. Simple; ships the
    routing badge.
  - *2b (fast-follow):* true RFC 0044 deferral — issue vectorless, read the
    routing decision, embed + re-issue only on semantic/fused — so the "short
    keyword traffic never pays embedding" cost story is literal, not narrated.

**Exit:** chips land on the right routes on 2k notes; the inspector shows the
gateway's real decision. The headline works.

---

## Phase 3 — Facets + UI (on the slice)

- `materialize_facet_snapshots` — `age_band`/`gender` land now;
  `specialty`/`diagnosis_category`/`events` fill in after Phase 4.
- Wire `/api/facets` + the static UI to the live backend (replace the
  preview-only `renderEcho` with a real `/api/search` call; draw the rail from
  `/api/facets` with `sha` provenance).

**Exit:** end-to-end UI — type → routed results + facet rail.

---

## Phase 4 — Gemma cascade smoke (small slice, GPU) ← cost gate

The new showcase. Wire `digest()` (`vLLM` with `guided_json=DIGEST_SCHEMA`); add
`tests/test_derive_labels.py` (`derive_labels` is already pure). Build the
`chart-classifier` image **or** run `vLLM` on a GPU box.

```bash
# over ~50–100 notes, ideally a slice known to contain discontinuations
uv run python -m functions.classify_events --once
```

- Hand-verify `events` + `has_med_discontinuation` on real discontinuation cases;
  confirm writeback lands; check a few `discontinuation_reason` (Tier 3).
- **Measure per-note latency → extrapolate the full 167k cost.** This is the gate
  before any full classify run.
- Settle the **multi-write shape** (RFC 0076 open question): one cascade Function
  multi-patching via the `tpuf` parameter (preferred — one pass, many labels) vs.
  splitting into sibling single-output UDFs (which would re-run the model).

**Exit:** sane events on a slice; full-classify cost/time known and accepted.

---

## Phase 5 — Eval, the number

Wire `eval/recds.py`: load **ReCDS-PPR** (`zhengyun21/PMC-Patients-ReCDS`, matching
revision), replay a query subset under `auto/semantic/bm25/fused`, score with
`ir_measures` vs. the published sparse+dense baseline.

```bash
uv run python -m eval.recds --task ppr --strategies auto,semantic,bm25,fused --limit 500
```

**Exit:** the hybrid-thesis number reproduces on a subset (sparse+dense ≥ either
leg alone). Note honestly that this is *retrieval quality*, not the routing claim
(see `eval/bimodal_queries.md`).

---

## Phase 6 — Full index + full classify run (the scale step)

Gated by Phases 1–5 green and Phase 4's cost sizing.

1. **Full index (167k).** Use the **GPU embed path**, not CPU fastembed (167k on
   CPU is hours). Idempotent re-upsert by `id` → resumable.
2. **Full classify.** Run the cascade Function over the whole namespace (discovery
   backfill; `filter: events Eq null` makes it resumable/idempotent). Watch
   throughput against the Phase-4 estimate; stop if cost diverges.
3. **Full facet snapshots** including `events`.

**Exit:** full namespace searchable, faceted, and event-classified — the live demo.

---

## Phase 7 — Productionize (later)

- `src/worker.js` prod Cloudflare Worker backend (the twin of `search/`) + deploy.
- Build the **bimodal eval set** (`eval/bimodal_queries.md`) to quantify the
  *routing* claim, not just retrieval quality.
- License / safety UI ("published case reports, not EHR; not for clinical use";
  CC-BY-NC-SA attribution), per RFC 0076.
- Lock all pins (dataset SHA, model, image tags) for the public repo.

---

## Cross-cutting discipline

- **Test the pure functions first:** `records` (age/shaping), `derive_labels`
  (events → labels), the routing-chip assertions. Cheapest correctness, earliest.
- **Idempotency everywhere:** re-upsert by `id`; the `events Eq null` filter; pin
  the dataset revision — so every full run is resumable and re-runnable.
- **Two hard cost gates:** size the embed (Phase 1 → 6.1) and the classify
  (Phase 4 → 6.2) on slices *before* committing the 167k runs. Don't let a full
  run be the first time you learn the per-row cost.
