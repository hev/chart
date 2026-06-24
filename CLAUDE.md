# chart — agent context

Clinical patient-notes query-routing demo on hev layer. The public twin of Trio's
`notesearch`. Scaffolded from **RFC 0076** (`../layer/docs/rfcs/0076-clinical-notes-query-routing-demo.md`)
— read it first; it is the design of record. Naming rules for "hev layer" /
"Layer" are in `../layer/CLAUDE.md`.

## What this is

A **shelf-shaped** demo (pull UX from `../shelf`, reimplement nothing) where
**query routing is the headline**: the `Auto` router (RFC 0044) over a corpus —
PMC-Patients — whose bimodal query distribution makes the keyword/fused/semantic
decision visibly matter. Two things make it more than another vertical:

1. **Real qrels** (PMC-Patients ReCDS) → the routing/hybrid claim gets a number
   (`eval/`). Honest split in `eval/README.md`: retrieval quality now; the routing
   number needs a built bimodal query set.
2. **A Gemma clinical-event cascade** (`functions/classify_events.py`) — the GPU
   showcase, modeled on RFC 0072 / `../moment`, built here first. Headline event:
   medication discontinuation.

## Conventions (inherited from the demo family)

- **No new gateway machinery.** chart reuses the shipped `huggingface` Warehouse
  (RFC 0053), `Auto`/`HybridText` routing, the RFC 0056 chunk model, RFC 0011
  embedding, and the RFC 0004/0040/0068 Function runtime. If something here seems
  to need new gateway code, the boundary is wrong (RFC 0056 acceptance discipline).
- **Two authoring surfaces, one schema.** `deploy/` (CRs) and the imperative
  indexer/gateway calls are two spellings of the same objects. Keep them in sync.
- **Embedding = `Snowflake/snowflake-arctic-embed-m-v1.5`** to match `../notesearch`
  (768-d, 512-token window, asymmetric query prefix — applied explicitly in
  `chart_common/embed.py`). Don't swap the model; matching it is the point.
- **Honesty in the UI.** Published case reports ≠ raw EHR; non-commercial license;
  not for clinical use. These are UI requirements, not footnotes.

## Stack

Python (uv, run-as-app — `package = false`): `chart_common/` + `indexer/` +
`search/` (FastAPI dev) + `functions/` + `eval/`. Cloudflare Worker prod backend
(`src/`, TODO). Vanilla static UI (`web/static/`). `hevlayer` client sourced
editable from `../layer/clients/python`.

## State

Scaffold from RFC 0076. Many files carry `STUB`/`TODO` seams (the gateway query
calls, the vLLM cascade body, the ReCDS loader). The structure and the writeback/
routing contracts around the seams are the committed part.
