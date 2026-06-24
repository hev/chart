# `functions/` — the transform-runtime act

Routing is chart's headline; the transform runtime is the second act — "one
primitive for every per-row job" over clinical notes (CLAUDE.md § Product Frame).
Each file is a `Function`: discovery, batching, claim leases, retries, and
`patch_columns` writeback are the gateway's, not ours.

| Function | Pool | Output | Role |
|---|---|---|---|
| `classify_events.py` | **gpu** | `events` + derived labels | **The GPU showcase.** A Gemma cascade (vLLM, guided decoding) finds clinical events — medication discontinuation the headline — and derives facet labels in the same pass. |
| `tag_specialty.py` | cpu | `specialty` | A specialty facet PMC-Patients lacks natively. |
| `extract_clinical_fields.py` | cpu | `diagnosis_category` (+more) | Structured fields → the richer facet rail. |
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
  straight from the digest — **one GPU pass, many labels**, subsuming the two CPU
  enrichers above.
- **Tier 3 — gated refine**: only on notes with a discontinuation, a targeted pass
  for the structured `discontinuation_reason`.

Why medication discontinuation? It is a concrete pharmacovigilance / cohort signal
a clinical customer (Trio) actually wants, and it composes with the routing
headline: an `events` filter over an `Auto`-routed search — *"discontinued statins
due to an adverse reaction"* — is a query neither keyword nor a flat note search
answers. `deploy/functions-events.yaml` is the `pool: gpu`, scale-to-zero Function.

The two `STUB` seams (`digest`, `refine_discontinuation`) are where the vLLM call
goes; the cascade structure and the writeback contract around them are the point.
