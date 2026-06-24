"""ReCDS eval harness — replay a judged query set through the gateway under each
retrieval strategy and score against PMC-Patients ReCDS qrels.

SKELETON. The metric computation and gateway query calls are stubbed at the seams
marked TODO; the structure is the contract: load qrels -> for each strategy, run
the judged queries -> score -> table. See eval/README.md for the two-claims
framing (retrieval quality is quantifiable now; the routing claim needs a built
bimodal set).
"""

from __future__ import annotations

import argparse
import asyncio

from chart_common.config import Settings
from chart_common.embed import Embedder
from chart_common.gateway import close_client, make_client

# rank_by spellings per strategy. `auto` is the headline — the gateway routes;
# the others force a fixed leg to compare against (RFC 0044/0057).
STRATEGIES = {
    "auto": "Auto",            # the router decides per query
    "semantic": "semantic",    # forced ANN over the Arctic vector
    "bm25": "hybrid_text",     # forced lexical (+ fuzzy)
    "fused": "fused",          # forced BM25 ⊕ semantic, RRF
}

# ReCDS metrics with published baselines to beat (RFC 0076 § Evaluation).
METRICS = ["MRR", "nDCG@10", "Recall@1000"]


def load_recds(task: str) -> tuple[list[dict], dict[str, dict[str, int]]]:
    """Load zhengyun21/PMC-Patients-ReCDS {queries, qrels} for `task` in
    {'ppr','par'}. TODO: read queries.jsonl + qrels.tsv at the pinned revision
    (datasets / huggingface_hub). Returns (queries, qrels[qid][docid]=grade)."""
    raise NotImplementedError("load ReCDS queries.jsonl + qrels.tsv (BEIR format)")


async def run_query(layer, namespace, *, text, strategy, embedder, top_k=1000) -> list[str]:
    """Issue one query under `strategy` and return ranked doc ids.

    Semantic/fused need a query vector: under RFC 0044 phase 1 the app embeds and
    supplies it — and MUST apply the Arctic query prefix (embed.py:embed_query).
    Auto on a vectorless long query returns the routing *decision* (executed:
    false); embed and re-issue with the route forced, exactly the demo's hop.
    TODO: call layer.query_namespace with the rank_by tuple for `strategy`.
    """
    needs_vector = strategy in ("semantic", "fused", "Auto")
    _vector = embedder.embed_query(text) if needs_vector else None
    raise NotImplementedError("issue the routed query; collect ranked ids")


def score(ranked: dict[str, list[str]], qrels: dict[str, dict[str, int]]) -> dict[str, float]:
    """TODO: compute METRICS over `ranked` against `qrels` (ir_measures)."""
    raise NotImplementedError("ir_measures over the ranked lists vs. qrels")


async def main() -> None:
    ap = argparse.ArgumentParser(description="ReCDS eval for the chart routing demo")
    ap.add_argument("--task", choices=["ppr", "par"], default="ppr")
    ap.add_argument("--strategies", default="auto,semantic,bm25,fused")
    ap.add_argument("--limit", type=int, default=500, help="judged queries to replay")
    args = ap.parse_args()

    settings = Settings()
    embedder = Embedder(settings.embed_model)
    queries, qrels = load_recds(args.task)
    queries = queries[: args.limit]

    layer = make_client(settings)
    try:
        for name in args.strategies.split(","):
            rank_by = STRATEGIES[name]
            ranked = {}
            for q in queries:
                ranked[q["id"]] = await run_query(
                    layer, settings.namespace, text=q["text"], strategy=rank_by, embedder=embedder
                )
            print(name, score(ranked, qrels))
    finally:
        await close_client(layer)


if __name__ == "__main__":
    asyncio.run(main())
