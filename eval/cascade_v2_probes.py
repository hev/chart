"""Qrels-style query probes for cascade v2 event-filtered search."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from hevlayer import QueryRequest

from chart_common.config import EMBED_DIM, Settings
from chart_common.embed import Embedder
from chart_common.gateway import close_client, make_client, require_gateway_key
from eval.cascade_v2 import FAMILIES, read_json, write_json
from eval.recds import score

PROBES = [
    ("q01", "immune checkpoint pneumonitis treatment stopped", "medication_stopped"),
    ("q02", "procedure complication bleeding", "procedure_complication"),
    ("q03", "incidental adrenal mass", "incidental_finding"),
    ("q04", "drug induced rash adverse reaction", "adverse_drug_reaction"),
    ("q05", "relapsed cancer treatment failure", "treatment_failure_or_recurrence"),
    ("q06", "biopsy confirmed diagnosis", "diagnostic_workup"),
    ("q07", "dose tapered after improvement", "medication_dose_changed"),
    ("q08", "hospital admission discharge follow up", "care_transition"),
    ("q09", "postoperative respiratory failure complication", "non_drug_complication"),
    ("q10", "palliative care patient died", "severe_outcome_or_death"),
]


def build_probe_set(gold: dict[str, Any]) -> dict[str, Any]:
    labels = gold["labels"]
    queries = [{"_id": qid, "text": text, "kind": "cascade_v2_probe", "facet": family} for qid, text, family in PROBES]
    qrels = []
    for qid, _text, family in PROBES:
        if family not in FAMILIES:
            raise ValueError(f"unknown family: {family}")
        for note_id, note_labels in labels.items():
            if note_labels.get(family) == "affirmed":
                qrels.append({"query-id": qid, "corpus-id": note_id, "score": 1})
    return {
        "metadata": {
            "shape": "BEIR/TREC qrels-style event-filter probes",
            "source": "cascade-v2 gold labels; no note text included",
            "queries": len(queries),
            "qrels": len(qrels),
        },
        "queries": queries,
        "qrels": qrels,
    }


def write_beir_dir(path: Path, probe_set: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "queries.jsonl").open("w") as f:
        for query in probe_set["queries"]:
            f.write(json.dumps(query, sort_keys=True) + "\n")
    with (path / "qrels.tsv").open("w") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for row in probe_set["qrels"]:
            f.write(f"{row['query-id']}\t{row['corpus-id']}\t{row['score']}\n")
    write_json(path / "metadata.json", probe_set["metadata"])


def query_body(text: str, *, vector: list[float], top_k: int, facet_field: str | None, facet: str | None) -> QueryRequest:
    if len(vector) != EMBED_DIM:
        raise ValueError(f"expected {EMBED_DIM}-d query vector, got {len(vector)}")
    body = QueryRequest(
        rank_by=["text", "Auto", text, {"vector": vector}],
        top_k=max(1, min(top_k, 1000)),
        include_attributes=["id"],
    )
    if facet_field and facet:
        body.filters = [facet_field, "Contains", facet]
    return body


def row_id(row: Any) -> str | None:
    if isinstance(row, dict):
        value = row.get("id") or row.get("$id")
    else:
        value = getattr(row, "id", None) or getattr(row, "$id", None)
    return str(value) if value else None


async def run_probe_eval(probe_dir: Path, *, top_k: int, facet_field: str) -> dict[str, Any]:
    settings = Settings()
    require_gateway_key(settings)
    embedder = Embedder(settings.embed_model)
    layer = make_client(settings)
    queries = [json.loads(line) for line in (probe_dir / "queries.jsonl").read_text().splitlines() if line.strip()]
    qrels: dict[str, dict[str, int]] = {}
    with (probe_dir / "qrels.tsv").open() as f:
        header = next(f).strip().split("\t")
        for line in f:
            row = dict(zip(header, line.rstrip("\n").split("\t"), strict=True))
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    modes = {"unfiltered": None, "filtered": facet_field}
    ranked = {mode: {} for mode in modes}
    try:
        for query in queries:
            vector = embedder.embed_query(query["text"])
            for mode, field in modes.items():
                resp = await layer.query_namespace(
                    settings.namespace,
                    query_body(query["text"], vector=vector, top_k=top_k, facet_field=field, facet=query["facet"]),
                )
                ranked[mode][query["_id"]] = [rid for row in (resp.rows or []) if (rid := row_id(row))]
    finally:
        await close_client(layer)
    return {
        "top_k": top_k,
        "facet_field": facet_field,
        "metrics": {mode: score(run, qrels) for mode, run in ranked.items()},
        "ranked_counts": {mode: {qid: len(rows) for qid, rows in run.items()} for mode, run in ranked.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build or run cascade v2 search-usefulness probes")
    sub = ap.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build")
    build.add_argument("--gold", type=Path, required=True)
    build.add_argument("--out-dir", type=Path, default=Path("eval/out/cascade-v2-probes"))
    run = sub.add_parser("run")
    run.add_argument("--probe-dir", type=Path, default=Path("eval/out/cascade-v2-probes"))
    run.add_argument("--top-k", type=int, default=50)
    run.add_argument("--facet-field", default="events_v2")
    run.add_argument("--out", type=Path, default=Path("eval/out/cascade-v2-query-probes-report.json"))
    args = ap.parse_args()
    if args.cmd == "build":
        probe_set = build_probe_set(read_json(args.gold))
        write_beir_dir(args.out_dir, probe_set)
    elif args.cmd == "run":
        report = asyncio.run(run_probe_eval(args.probe_dir, top_k=args.top_k, facet_field=args.facet_field))
        write_json(args.out, report)
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
