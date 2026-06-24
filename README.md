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
*teaches* the router. `metformin 500mg` never paid for an embedding; the long one
did, and the inspector shows it.

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
search/         FastAPI dev backend (Auto-routed; the Arctic query-prefix hop)
src/            Cloudflare Worker prod backend (twin of search/)   [TODO]
functions/      the transform-runtime act: classify_events (Gemma, GPU) +
                tag_specialty / extract_clinical_fields / scan_phi (CPU)
eval/           the ReCDS qrels harness — the routing/hybrid number
web/static/     the shelf-shaped single-page UI + the clinical routing chips
deploy/         the in-cluster declarative twin (VectorStore/Warehouse/Pipeline/
                Index + the GPU events Function)
```

## Run

```bash
cp .env.example .env          # set LAYER_GATEWAY_API_KEY (the upstream tpuf key)
uv sync --extra search
uv run python -m indexer --limit 2000     # smoke index a slice ( --dry-run to skip the gateway)
uv run uvicorn search.app:app --reload    # http://localhost:8000
```

The Gemma cascade (`functions/`) and the ReCDS eval (`eval/`) are GPU- and
qrels-bound respectively; see their READMEs. Several files have `STUB`/`TODO`
seams — chart is scaffolded from RFC 0076, not yet wired end to end.
