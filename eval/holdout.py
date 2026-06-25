from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from chart_common.cli import non_negative_int, positive_int
from chart_common.config import Settings
from chart_common.records import NoteRecord
from indexer.dataset import load_notes

from .recds import load_recds


def edge(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((str(a), str(b))))


def feature_edges(records: Iterable[NoteRecord]) -> set[tuple[str, str]]:
    edges = set()
    for record in records:
        for similar_id in record.similar_patient_ids:
            if similar_id != record.id:
                edges.add(edge(record.id, similar_id))
    return edges


def qrel_edges(qrels: dict[str, dict[str, int]]) -> set[tuple[str, str]]:
    edges = set()
    for query_id, docs in qrels.items():
        for doc_id in docs:
            if doc_id != query_id:
                edges.add(edge(query_id, doc_id))
    return edges


def overlap_report(
    *,
    feature: set[tuple[str, str]],
    qrels: set[tuple[str, str]],
    examples: int = 10,
) -> dict[str, Any]:
    overlap = sorted(feature & qrels)
    return {
        "feature_edges": len(feature),
        "qrel_edges": len(qrels),
        "overlap_edges": len(overlap),
        "overlap_fraction_of_qrels": round(len(overlap) / len(qrels), 6) if qrels else 0.0,
        "examples": [{"patient_a": a, "patient_b": b} for a, b in overlap[:examples]],
    }


def holdout_gate(report: dict[str, Any], *, max_overlap_edges: int | None = None) -> dict[str, Any]:
    checks = {
        "feature_edges_present": {"actual": report["feature_edges"], "ok": report["feature_edges"] > 0},
        "qrel_edges_present": {"actual": report["qrel_edges"], "ok": report["qrel_edges"] > 0},
    }
    if max_overlap_edges is not None:
        checks["max_overlap_edges"] = {
            "limit": max_overlap_edges,
            "actual": report["overlap_edges"],
            "ok": report["overlap_edges"] <= max_overlap_edges,
        }
    return {"gate": "recds_holdout_overlap", "checks": checks, "accepted": all(check["ok"] for check in checks.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ReCDS PPR overlap with live similar-patient feature edges")
    parser.add_argument("--split", choices=["dev", "test", "train"], default="dev")
    parser.add_argument("--notes-limit", type=positive_int, default=None)
    parser.add_argument("--examples", type=non_negative_int, default=10)
    parser.add_argument(
        "--max-overlap-edges",
        type=non_negative_int,
        default=None,
        help="exit non-zero if feature/qrel overlap exceeds this many undirected edges",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write the holdout leakage report to this JSON path for Phase-5 audit gates",
    )
    args = parser.parse_args()

    settings = Settings()
    _queries, qrels = load_recds("ppr", settings=settings, split=args.split)
    records = load_notes(settings, limit=args.notes_limit)
    report = {
        "split": args.split,
        "notes_limit": args.notes_limit,
        **overlap_report(
            feature=feature_edges(records),
            qrels=qrel_edges(qrels),
            examples=args.examples,
        ),
    }
    if args.max_overlap_edges is not None:
        report["gate"] = holdout_gate(report, max_overlap_edges=args.max_overlap_edges)
    rendered = json.dumps(report, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n")
    print(rendered)
    if report.get("gate") and not report["gate"]["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
