# chart

**Clinical patient-notes search that shows its routing.** One search box; the
gateway picks the retrieval strategy (keyword, fused or semantic) and tells you
*why*. A query-routing demo on [hev layer](https://hevlayer.com).

> Notes are published, de-identified case reports (PMC-Patients, CC-BY-NC-SA):
> **not raw EHR, not for clinical use**. This is a search demo; it gives no
> medical advice.

## Why clinical notes

Clinical search has the **sharpest bimodal query distribution** there is.
Clinicians search both by exact token and by clinical picture:

| You type | Tokens | Route | Why |
|---|---|---|---|
| `metformin 500mg` | 2 | `hybrid_text` | drug + dose; exact lexical, typo-tolerant |
| `CABG` · `afib` · `aspirn` | 1 | `hybrid_text` | abbreviations / typos: ANN noise, BM25+fuzzy win |
| `chest pain radiating to left arm` | 5 | `fused` | clinical phrase; both legs, RRF-merged |
| `elderly woman with progressive dyspnea and bilateral lower-extremity edema` | 9 | `semantic` | a clinical picture in prose; ANN over the embedding |

The routing badge renders the gateway's own decision, so the demo *teaches* the
router. One rung above it, the **Agentic search** toggle runs the same query
through a configured reasoning loop (`POST /v2/agents/chart-notes/query`,
`deploy/agent.yaml`): a model reformulates the query, fans out for recall,
grades the candidates, and returns the standard row shape. With provenance on,
the inspector shows the agent's plan, and each hit carries its retrieval and
relevance scores. The backing store is Turbopuffer (`deploy/vectorstore.yaml`).

## The features, and where they're documented

Everything visible in the UI is a gateway feature; the app composes them. The
same tour is served by the app at [`/help.html`](web/static/help.html).

| Feature | What you see | hev layer docs |
|---|---|---|
| **Query routing** | The `Auto` router picks `hybrid_text` / `fused` / `semantic` per query; the badge is the gateway's decision, echoed back | [Query routing](https://hevlayer.com/docs/api/query/#query-routing) |
| **Hybrid text** | BM25 + per-token fuzzy in one lexical leg (`aspirn` finds aspirin, and the response says a fuzzy match surfaced it); `fused` merges it with the semantic leg via RRF; the rail's per-search counts come from Scans | [Hybrid text fusion](https://hevlayer.com/docs/api/query/#hybrid-text-fusion) · [Scans](https://hevlayer.com/docs/api/scans/) |
| **Agentic search** | The toggle runs the query through an `Agent`: reformulate, fan out, grade; the sidebar becomes the run inspector | [Agents API](https://hevlayer.com/docs/api/agents/) · [Agent CRD](https://hevlayer.com/docs/kubernetes/agent-crd/) |
| **UDF cascade (self-hosted model)** | `events` / `specialty` / `diagnosis_category` facets are written back by a Gemma classifier on cluster GPUs via the `Function` runtime (scale-to-zero, guided decoding): one GPU pass, many labels | [Function CRD](https://hevlayer.com/docs/kubernetes/function-crd/) · [GPU classifier](https://hevlayer.com/docs/kubernetes/function-crd/#gpu-classifier) |

## Two things this demo proves

1. **Query routing, with a number.** chart has **real relevance judgments**
   (PMC-Patients ReCDS qrels), so the hybrid/routing claim is measured, not
   asserted (`eval/`). ReCDS quantifies retrieval *quality* today; the
   *routing* claim needs a bimodal query set (`eval/bimodal_queries.md`).
2. **A Gemma clinical-event cascade.** An open-weight Gemma cascade (vLLM,
   guided decoding, scale-to-zero) reads each note once and pulls out
   **clinical events**, with *medication discontinuation* the headline, plus
   the facet labels in the same pass (`functions/classify_events.py`). It
   composes with routing: an `events` filter over a routed search, e.g.
   *"discontinued statins due to an adverse reaction"*.

### The cascade vs. a batch API

The usual baseline for LLM classification at rest is a provider batch API,
such as the Claude Message Batches API on Haiku (50% off realtime). A
self-hosted open-weight model on Layer's Function runtime beats it on marginal
cost and keeps the data in the cluster:

| Path (per note ≈ 1.2k in / 300 out tokens) | 11.4k-note backfill | 167k full corpus |
|---|---|---|
| Haiku 4.5 realtime | ~$31 | ~$460 |
| Haiku 4.5 **Batch** (the baseline) | ~$16 | ~$235 |
| **Cascade**: Gemma-2-9B, 2× `g5.xlarge` ($1.01/hr each), **measured** | **$8.2** | **~$120** |

Measured: 11,373 notes in ~4h at 48–58 notes/min across two GPUs, zero
failures, $8.10 GPU + $0.13 Layer-metered writes, scans and storage. It holds
up for three reasons:

- **One pass, many labels.** `events`, `specialty`, `diagnosis_category`,
  `has_med_discontinuation` and the discontinuation reason come from a single
  digest; a per-label batch pipeline pays per label.
- **Continuous batching.** Each claimed batch goes through one vLLM
  `generate()` and writes back as one multi-row `patch_columns`.
- **Scale-to-zero.** The Function runtime scales on queue depth, so the GPU
  exists only while there is work; idle cost is $0.

Layer's own cost report (`scripts/layer_cost_report.sh --kind classifier`) is
the authoritative number.

## Layout

```
chart_common/   config, embedding (Arctic query prefix), records, gateway client
indexer/        load PMC-Patients → embed → upsert → materialize facet snapshots
search/         FastAPI dev backend
src/            Cloudflare Worker prod backend (twin of search/)
functions/      the transform-runtime UDFs: classify_events (Gemma, GPU) + CPU taggers
eval/           the ReCDS qrels harness
web/static/     the single-page UI
deploy/         the declarative bundle (VectorStore/Warehouse/Pipeline/Index,
                the GPU events Function, the Agent) and GPU worker images
docs/           operations run book, RFCs
```

## Run

```bash
uv sync --extra search
source scripts/lib/resolve_gateway_key.sh && resolve_gateway_key >/dev/null
uv run --extra search uvicorn search.app:app --reload    # http://localhost:8000
```

The gateway key comes from 1Password at run time; see
`scripts/lib/resolve_gateway_key.sh` for the item it reads and how to override
it. Indexing, cost gates, the classifier backfill and the final audit are in
[docs/operations.md](docs/operations.md).
