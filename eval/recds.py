"""ReCDS eval harness — replay a judged query set through the gateway under each
retrieval strategy and score against PMC-Patients ReCDS qrels."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from hevlayer import QueryRequest
from hevlayer.client import HevlayerError

from chart_common.cli import non_negative_int, positive_int
from chart_common.config import EMBED_DIM, Settings
from chart_common.embed import Embedder
from chart_common.gateway import close_client, make_client, require_gateway_key

# rank_by route overrides per strategy. `auto` is the headline — the gateway
# routes; the others force a fixed leg through Auto's comparison override.
STRATEGIES = {
    "auto": None,              # the router decides per query
    "semantic": "semantic",    # forced ANN over the Arctic vector
    "bm25": "hybrid_text",     # forced lexical (+ fuzzy)
    "fused": "fused",          # forced BM25 ⊕ semantic, RRF
    # Token-bag MaxSim over the LI sibling namespace (Turbopuffer private
    # beta) — a direct rank_by ["tokens","ANN",[[...]]], not an Auto route.
    # Caveat: ColBERT-family query encoders truncate to ~32 tokens, so long
    # ReCDS patient-note queries are heavily truncated on the query side.
    "late_interaction": "late_interaction",
}

# ReCDS metrics with published baselines to beat (RFC 0076 § Evaluation).
METRICS = ["MRR@10", "nDCG@10", "Recall@1000"]
TRANSIENT = {502, 503, 504}
FORCED_LEXICAL_TOKEN_LIMIT = 8

# Public PMC-Patients leaderboard values are percentages; store fractions to
# match `ir_measures` output. RRF is the sparse+dense thesis baseline; BM25 is
# the strong lexical baseline.
PUBLISHED_BASELINES = {
    "ppr": {
        "rrf": {"RR@10": 0.2776, "nDCG@10": 0.2412, "R@1000": 0.8514},
        "bm25": {"RR@10": 0.2286, "nDCG@10": 0.1829, "R@1000": 0.6966},
    },
    "par": {
        "rrf": {"RR@10": 0.2986, "nDCG@10": 0.1336, "R@1000": 0.4945},
        "bm25": {"RR@10": 0.1871, "nDCG@10": 0.0738, "R@1000": 0.2189},
    },
}


def parse_strategies(value: str) -> list[str]:
    strategies = [strategy.strip() for strategy in value.split(",") if strategy.strip()]
    unknown = sorted(set(strategies) - set(STRATEGIES))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown strategy(s): {', '.join(unknown)}; choose from {', '.join(STRATEGIES)}"
        )
    if not strategies:
        raise argparse.ArgumentTypeError("at least one strategy is required")
    return strategies


def validate_eval_gates(*, strategies: list[str], require_fused_dominates: bool) -> None:
    if require_fused_dominates:
        required = {"bm25", "semantic", "fused"}
        missing = sorted(required - set(strategies))
        if missing:
            raise SystemExit(
                "--require-fused-dominates requires strategies: "
                f"{', '.join(sorted(required))}; missing: {', '.join(missing)}"
            )


def validate_judged_queries(queries: list[dict], *, source: str, limit: int) -> None:
    if not queries:
        raise SystemExit(f"{source} produced no judged queries after applying --limit {limit}")


def _download(settings: Settings, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            settings.recds_repo,
            filename,
            repo_type="dataset",
            revision=settings.recds_revision,
        )
    )


def load_recds(
    task: str, *, settings: Settings | None = None, split: str = "dev"
) -> tuple[list[dict], dict[str, dict[str, int]]]:
    """Load zhengyun21/PMC-Patients-ReCDS {queries, qrels} for `task` in
    {'ppr','par'} at the pinned revision. Returns
    (queries, qrels[qid][docid]=grade)."""
    settings = settings or Settings()
    task_dir = task.upper()
    query_path = _download(settings, f"queries/{split}_queries.jsonl")
    qrels_path = _download(settings, f"{task_dir}/qrels_{split}.tsv")

    queries = []
    with query_path.open() as f:
        for line in f:
            row = json.loads(line)
            queries.append({"id": row["_id"], "text": row["text"]})

    qrels: dict[str, dict[str, int]] = {}
    with qrels_path.open() as f:
        header = next(f, "").strip().split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            row = dict(zip(header, parts, strict=True))
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])

    judged = {qid for qid, docs in qrels.items() if docs}
    return [q for q in queries if q["id"] in judged], qrels


def load_beir_dir(path: Path) -> tuple[list[dict], dict[str, dict[str, int]], dict]:
    """Load a BEIR-style directory with queries.jsonl, qrels.tsv, optional metadata."""
    query_path = path / "queries.jsonl"
    qrels_path = path / "qrels.tsv"
    if not query_path.exists() or not qrels_path.exists():
        raise FileNotFoundError(f"{path} must contain queries.jsonl and qrels.tsv")

    queries = []
    with query_path.open() as f:
        for line in f:
            row = json.loads(line)
            queries.append(
                {
                    "id": row.get("_id") or row["id"],
                    "text": row["text"],
                    "kind": row.get("kind"),
                }
            )

    qrels: dict[str, dict[str, int]] = {}
    with qrels_path.open() as f:
        header = next(f, "").strip().split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            row = dict(zip(header, parts, strict=True))
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])

    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    judged = {qid for qid, docs in qrels.items() if docs}
    return [q for q in queries if q["id"] in judged], qrels, metadata


async def run_query(
    layer, namespace, *, text, strategy, embedder, li_embedder=None, top_k=1000
) -> list[str]:
    """Issue one query under `strategy` and return ranked doc ids.

    Semantic/fused/Auto get a query vector in the 4th tuple slot, matching the
    demo path and avoiding vectorless deferral."""
    body = query_body(
        text, strategy=strategy, embedder=embedder, li_embedder=li_embedder, top_k=top_k
    )
    for attempt in range(3):
        try:
            resp = await layer.query_namespace(namespace, body)
            break
        except HevlayerError as exc:
            if exc.status_code not in TRANSIENT or attempt == 2:
                raise
            await asyncio.sleep(0.4 * (attempt + 1))
    rows = resp.rows or []
    ranked_ids = [_row_id(row) for row in rows]
    ranked_ids = [row_id for row_id in ranked_ids if row_id]
    if rows and not ranked_ids:
        raise ValueError("gateway returned rows without id/$id attributes")
    return ranked_ids


def _row_id(row: Any) -> str | None:
    if isinstance(row, dict):
        value = row.get("id") or row.get("$id")
    else:
        value = getattr(row, "id", None) or getattr(row, "$id", None)
    return str(value) if value else None


def _lexical_query_text(text: str, *, strategy: str | None) -> str:
    if strategy not in {"fused", "hybrid_text"}:
        return text
    tokens = text.split()
    if len(tokens) <= FORCED_LEXICAL_TOKEN_LIMIT:
        return text
    return " ".join(tokens[:FORCED_LEXICAL_TOKEN_LIMIT])


def query_body(
    text: str, *, strategy: str | None, embedder, li_embedder=None, top_k: int = 1000
) -> QueryRequest:
    if strategy == "late_interaction":
        if li_embedder is None:
            raise ValueError("late_interaction strategy requires an LI embedder")
        return QueryRequest(
            rank_by=["tokens", "ANN", li_embedder.embed_query(text)],
            top_k=max(1, min(top_k, 1000)),
            include_attributes=["id"],
        )
    vector = embedder.embed_query(text)
    if len(vector) != EMBED_DIM:
        raise ValueError(f"expected {EMBED_DIM}-d query vector, got {len(vector)}")
    lexical_text = _lexical_query_text(text, strategy=strategy)
    extra = {"vector": vector}
    if strategy is not None:
        extra["route"] = strategy
    return QueryRequest(
        rank_by=["text", "Auto", lexical_text, extra],
        top_k=max(1, min(top_k, 1000)),
        include_attributes=["id"],
    )


def score(ranked: dict[str, list[str]], qrels: dict[str, dict[str, int]]) -> dict[str, float]:
    """Compute aggregate ReCDS metrics over ranked ids."""
    import ir_measures
    from ir_measures import MRR, Recall, nDCG
    from ir_measures.util import Qrel, ScoredDoc

    if not any(docs for docs in qrels.values()):
        raise ValueError("cannot score without qrels")

    qrel_rows = [
        Qrel(query_id=qid, doc_id=docid, relevance=grade)
        for qid, docs in qrels.items()
        for docid, grade in docs.items()
    ]
    run_rows = [
        ScoredDoc(query_id=qid, doc_id=docid, score=1.0 / (rank + 1))
        for qid, docids in ranked.items()
        for rank, docid in enumerate(docids)
    ]
    measures = [MRR @ 10, nDCG @ 10, Recall @ 1000]
    return {str(measure): float(value) for measure, value in ir_measures.calc_aggregate(measures, qrel_rows, run_rows).items()}


def qrels_for_queries(qrels: dict[str, dict[str, int]], queries: list[dict]) -> dict[str, dict[str, int]]:
    query_ids = {query["id"] for query in queries}
    return {qid: docs for qid, docs in qrels.items() if qid in query_ids}


def validate_qrels_cover_queries(qrels: dict[str, dict[str, int]], queries: list[dict]) -> None:
    missing = [query["id"] for query in queries if not qrels.get(query["id"])]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise SystemExit(f"selected queries missing qrels: {preview}{suffix}")


def score_by_kind(
    ranked: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    queries: list[dict],
) -> dict[str, dict[str, float]]:
    out = {}
    for kind in sorted({query.get("kind") or "unknown" for query in queries}):
        query_ids = {query["id"] for query in queries if (query.get("kind") or "unknown") == kind}
        if not query_ids:
            continue
        kind_qrels = {qid: docs for qid, docs in qrels.items() if qid in query_ids}
        kind_ranked = {qid: docs for qid, docs in ranked.items() if qid in query_ids}
        if kind_qrels:
            out[kind] = score(kind_ranked, kind_qrels)
    return out


def baseline_report(metrics: dict[str, float], *, task: str, baseline: str = "rrf") -> dict:
    expected = PUBLISHED_BASELINES[task][baseline]
    deltas = {
        metric: round(metrics.get(metric, 0.0) - expected_value, 6)
        for metric, expected_value in expected.items()
    }
    return {
        "baseline": baseline,
        "baseline_metrics": expected,
        "delta": deltas,
        "meets_or_beats": all(value >= 0 for value in deltas.values()),
    }


def fused_dominance_report(summaries: list[dict], *, metrics: list[str] | None = None) -> dict:
    metrics = metrics or ["RR@10", "nDCG@10", "R@1000"]
    by_strategy = {summary["strategy"]: summary for summary in summaries}
    checks = []
    query_failures = []
    fused = by_strategy.get("fused")
    for strategy in ("bm25", "semantic", "fused"):
        summary = by_strategy.get(strategy) or {}
        failed = int(((summary.get("queries") or {}).get("failed")) or 0)
        if failed:
            query_failures.append({"strategy": strategy, "failed": failed})
    for metric in metrics:
        fused_value = (fused or {}).get("metrics", {}).get(metric)
        for baseline in ("bm25", "semantic"):
            baseline_value = by_strategy.get(baseline, {}).get("metrics", {}).get(metric)
            ok = fused_value is not None and baseline_value is not None and fused_value >= baseline_value
            checks.append(
                {
                    "metric": metric,
                    "baseline": baseline,
                    "fused": fused_value,
                    "baseline_value": baseline_value,
                    "ok": ok,
                }
            )
    return {
        "gate": "fused_dominates_legs",
        "checks": checks,
        "query_failures": query_failures,
        "accepted": not query_failures and all(check["ok"] for check in checks),
    }


def eval_summary(
    *,
    strategy: str,
    metrics: dict[str, float],
    queries: list[dict],
    task: str | None = None,
    baseline: str | None = None,
    dataset: dict | None = None,
    top_k: int = 1000,
    query_errors: list[dict] | None = None,
    metrics_by_kind: dict[str, dict[str, float]] | None = None,
) -> dict:
    by_kind = {}
    for query in queries:
        kind = query.get("kind") or "unknown"
        by_kind[kind] = by_kind.get(kind, 0) + 1
    out = {
        "strategy": strategy,
        "metrics": metrics,
        "queries": {
            "total": len(queries),
            "attempted": len(queries),
            "failed": len(query_errors or []),
            "succeeded": len(queries) - len(query_errors or []),
            "scored": len(queries),
            "by_kind": by_kind,
        },
        "top_k": top_k,
    }
    if metrics_by_kind:
        out["metrics_by_kind"] = metrics_by_kind
    if query_errors:
        out["query_errors"] = query_errors[:5]
        out["query_errors_truncated"] = max(0, len(query_errors) - 5)
    if dataset:
        out["dataset"] = dataset
    if task and baseline:
        out["published"] = baseline_report(metrics, task=task, baseline=baseline)
    return out


def report_provenance(settings: Settings, *, beir_dir: Path | None = None) -> dict:
    return {
        "recds_repo": getattr(settings, "recds_repo", None) if beir_dir is None else None,
        "recds_revision": getattr(settings, "recds_revision", None) if beir_dir is None else None,
        "embed_model": getattr(settings, "embed_model", None),
        "embed_dim": EMBED_DIM,
        "namespace": getattr(settings, "namespace", None),
        "li_model": getattr(settings, "li_model", None),
        "li_namespace": getattr(settings, "li_namespace", None),
    }


async def main() -> None:
    ap = argparse.ArgumentParser(description="ReCDS eval for the chart routing demo")
    ap.add_argument("--task", choices=["ppr", "par"], default="ppr")
    ap.add_argument("--split", choices=["dev", "test", "train"], default="dev")
    ap.add_argument("--strategies", type=parse_strategies, default=parse_strategies("auto,semantic,bm25,fused"))
    ap.add_argument("--limit", type=positive_int, default=500, help="judged queries to replay")
    ap.add_argument("--top-k", type=positive_int, default=1000, help="rows to request per query for scoring")
    ap.add_argument(
        "--progress-every",
        type=non_negative_int,
        default=0,
        help="print progress to stderr every N queries per strategy; 0 disables progress output",
    )
    ap.add_argument("--baseline", choices=["rrf", "bm25"], default="rrf")
    ap.add_argument(
        "--require-no-failures",
        action="store_true",
        help="exit non-zero if any replayed query fails",
    )
    ap.add_argument(
        "--require-fused-dominates",
        action="store_true",
        help="exit non-zero unless fused scores meet or beat bm25 and semantic on the headline metrics",
    )
    ap.add_argument(
        "--beir-dir",
        type=Path,
        default=None,
        help="evaluate a local BEIR-style queries.jsonl/qrels.tsv directory, e.g. eval/out/bimodal",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write an aggregate JSON eval report for Phase-5 audit gates",
    )
    args = ap.parse_args()
    validate_eval_gates(strategies=args.strategies, require_fused_dominates=args.require_fused_dominates)

    settings = Settings()
    dataset_meta = None
    if args.beir_dir:
        queries, qrels, dataset_meta = load_beir_dir(args.beir_dir)
        baseline_task = None
    else:
        queries, qrels = load_recds(args.task, settings=settings, split=args.split)
        baseline_task = args.task
    queries = queries[: args.limit]
    validate_judged_queries(
        queries,
        source=str(args.beir_dir) if args.beir_dir else f"ReCDS {args.task}/{args.split}",
        limit=args.limit,
    )
    scored_qrels = qrels_for_queries(qrels, queries)
    validate_qrels_cover_queries(scored_qrels, queries)

    require_gateway_key(settings)
    embedder = Embedder(settings.embed_model)
    li_embedder = None
    if "late_interaction" in args.strategies:
        from chart_common.embed import LateInteractionEmbedder

        li_embedder = LateInteractionEmbedder(settings.li_model)
    layer = make_client(settings)
    had_failures = False
    summaries = []
    try:
        for name in args.strategies:
            route = STRATEGIES[name]
            # The LI arm ranks the token-bag sibling namespace; every other arm
            # ranks chart-notes. Both hold the same note slice, so qrels apply.
            namespace = (
                settings.li_namespace if name == "late_interaction" else settings.namespace
            )
            ranked = {}
            query_errors = []
            for index, q in enumerate(queries, start=1):
                try:
                    ranked[q["id"]] = await run_query(
                        layer,
                        namespace,
                        text=q["text"],
                        strategy=route,
                        embedder=embedder,
                        li_embedder=li_embedder,
                        top_k=args.top_k,
                    )
                except HevlayerError as exc:
                    ranked[q["id"]] = []
                    query_errors.append(
                        {
                            "query_id": q["id"],
                            "status_code": exc.status_code,
                            "message": exc.message[:180],
                        }
                    )
                if args.progress_every and (index % args.progress_every == 0 or index == len(queries)):
                    print(
                        f"recds progress strategy={name} {index}/{len(queries)} failed={len(query_errors)}",
                        file=sys.stderr,
                        flush=True,
                    )
            had_failures = had_failures or bool(query_errors)
            metrics = score(ranked, scored_qrels)
            kind_metrics = score_by_kind(ranked, scored_qrels, queries)
            summary = eval_summary(
                strategy=name,
                metrics=metrics,
                queries=queries,
                task=baseline_task,
                baseline=args.baseline if baseline_task else None,
                dataset=dataset_meta,
                top_k=args.top_k,
                query_errors=query_errors,
                metrics_by_kind=kind_metrics,
            )
            summaries.append(summary)
            print(json.dumps(summary))
    finally:
        await close_client(layer)
    dominance = None
    if args.require_fused_dominates:
        dominance = fused_dominance_report(summaries)
        print(json.dumps(dominance))
    report = {
        "task": args.task if not args.beir_dir else None,
        "split": args.split if not args.beir_dir else None,
        "beir_dir": str(args.beir_dir) if args.beir_dir else None,
        "provenance": report_provenance(settings, beir_dir=args.beir_dir),
        "strategies": args.strategies,
        "limit": args.limit,
        "top_k": args.top_k,
        "summaries": summaries,
        "gates": {
            "no_failures": {
                "required": args.require_no_failures,
                "accepted": not had_failures,
            },
            "fused_dominates": dominance,
        },
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    if dominance is not None and not dominance["accepted"]:
        raise SystemExit(1)
    if args.require_no_failures and had_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
