from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from chart_common.cli import non_negative_int, positive_float, positive_int
from chart_common.config import Settings
from indexer.dataset import load_notes

from .recds import load_recds

DRUG_TERMS = {
    "aspirin",
    "atorvastatin",
    "ceftriaxone",
    "clopidogrel",
    "furosemide",
    "heparin",
    "insulin",
    "lisinopril",
    "metformin",
    "prednisone",
    "statin",
    "warfarin",
}

ABBREVIATIONS = {
    "AF",
    "AKI",
    "CABG",
    "CHF",
    "COPD",
    "CT",
    "DVT",
    "ECG",
    "MI",
    "MRI",
    "PE",
    "SLE",
}

DOSE_RE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|g|ml|units?)\b", re.I)
ICD_RE = re.compile(r"\b[A-Z]\d{2}(?:\.\d+)?\b")
TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z-]{2,}\b")
STOPWORDS = {
    "and",
    "are",
    "for",
    "had",
    "has",
    "her",
    "his",
    "not",
    "patient",
    "presented",
    "reported",
    "she",
    "showed",
    "that",
    "the",
    "there",
    "this",
    "was",
    "were",
    "with",
}


def perturb_typo(text: str) -> str:
    """Deterministic one-edit typo for fuzzy-route evaluation."""
    clean = text.strip()
    if len(clean) < 4:
        return clean[:-1] if len(clean) > 1 else clean
    return clean[:2] + clean[3:]


def typo_fraction(value: str) -> float:
    parsed = positive_float(value)
    if parsed > 1:
        raise argparse.ArgumentTypeError(f"expected value <= 1, got {parsed}")
    return parsed


def extract_short_queries(text: str, *, max_per_note: int = 4) -> list[str]:
    lower = text.lower()
    candidates: list[str] = []

    drug_hits = []
    for drug in DRUG_TERMS:
        match = re.search(rf"\b{re.escape(drug)}\b", lower)
        if match:
            drug_hits.append((match.start(), drug))
    for _pos, drug in sorted(drug_hits):
        candidates.append(drug)

    doses = DOSE_RE.findall(text)
    if doses and candidates:
        candidates.append(f"{candidates[0]} {doses[0].replace(' ', '')}")

    for abbr in sorted(ABBREVIATIONS):
        if re.search(rf"\b{re.escape(abbr)}\b", text):
            candidates.append(abbr)

    candidates.extend(ICD_RE.findall(text))

    # Fallback: stable high-signal title-case-ish clinical terms, not stopwords.
    if not candidates:
        counts = Counter(token.lower() for token in TOKEN_RE.findall(text))
        for token, _n in counts.most_common(12):
            if token not in STOPWORDS:
                candidates.append(token)
            if len(candidates) >= max_per_note:
                break

    deduped = []
    seen = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate.lower() in seen:
            continue
        seen.add(candidate.lower())
        deduped.append(candidate)
        if len(deduped) >= max_per_note:
            break
    return deduped


def build_short_set(settings: Settings, *, notes: int, typo_fraction: float) -> tuple[list[dict], dict[str, dict[str, int]]]:
    queries: list[dict] = []
    qrels: dict[str, dict[str, int]] = {}
    typo_every = int(1 / typo_fraction) if typo_fraction > 0 else 0

    for record in load_notes(settings, limit=notes, include_similar_patient_ids=False):
        for query in extract_short_queries(record.text):
            qid = f"short:{record.id}:{len(queries)}"
            queries.append({"id": qid, "text": query, "kind": "short"})
            qrels[qid] = {record.id: 1}

            if typo_every and len(queries) % typo_every == 0:
                typo = perturb_typo(query)
                if typo and typo != query:
                    typo_id = f"typo:{record.id}:{len(queries)}"
                    queries.append({"id": typo_id, "text": typo, "kind": "typo", "source": query})
                    qrels[typo_id] = {record.id: 1}
    return queries, qrels


def build_bimodal_set(
    *,
    settings: Settings,
    short_notes: int,
    long_limit: int,
    typo_fraction: float,
    split: str,
) -> tuple[list[dict], dict[str, dict[str, int]], dict]:
    short_queries, short_qrels = build_short_set(
        settings, notes=short_notes, typo_fraction=typo_fraction
    )
    long_queries, long_qrels = load_recds("ppr", settings=settings, split=split)
    long_queries = [
        {"id": f"long:{q['id']}", "text": q["text"], "kind": "long", "source_id": q["id"]}
        for q in long_queries[:long_limit]
    ]
    qrels = dict(short_qrels)
    for query in long_queries:
        qrels[query["id"]] = long_qrels[query["source_id"]]

    queries = short_queries + long_queries
    meta = {
        "short": len(short_queries),
        "long": len(long_queries),
        "typo": sum(1 for q in short_queries if q["kind"] == "typo"),
        "split": split,
        "short_notes": short_notes,
        "long_limit": long_limit,
        "typo_fraction": typo_fraction,
        "dataset_repo": settings.dataset_repo,
        "dataset_revision": settings.dataset_revision,
        "dataset_split": settings.dataset_split,
        "recds_repo": settings.recds_repo,
        "recds_revision": settings.recds_revision,
        "weak_label": "short query source note is relevant doc; long queries use ReCDS-PPR qrels",
    }
    return queries, qrels, meta


def validate_bimodal_set(queries: list[dict], qrels: dict[str, dict[str, int]], meta: dict) -> None:
    if int(meta.get("short") or 0) <= 0:
        raise SystemExit("bimodal set has no short keyword/fuzzy queries")
    if int(meta.get("long") or 0) <= 0:
        raise SystemExit("bimodal set has no long semantic ReCDS queries")
    missing_qrels = [query["id"] for query in queries if not qrels.get(query["id"])]
    if missing_qrels:
        preview = ", ".join(missing_qrels[:5])
        suffix = "..." if len(missing_qrels) > 5 else ""
        raise SystemExit(f"bimodal set has queries without qrels: {preview}{suffix}")


def write_beir(queries: Iterable[dict], qrels: dict[str, dict[str, int]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "queries.jsonl").open("w") as f:
        for query in queries:
            f.write(json.dumps({"_id": query["id"], "text": query["text"], "kind": query.get("kind")}) + "\n")

    with (out_dir / "qrels.tsv").open("w") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for qid, docs in qrels.items():
            for docid, score_value in docs.items():
                f.write(f"{qid}\t{docid}\t{score_value}\n")


def write_review_csv(
    queries: Iterable[dict],
    qrels: dict[str, dict[str, int]],
    out_path: Path,
    *,
    limit_per_kind: int = 25,
) -> dict[str, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    written = 0
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query_id",
                "kind",
                "query",
                "relevant_doc_id",
                "weak_score",
                "human_judgment",
                "review_notes",
            ],
        )
        writer.writeheader()
        for query in queries:
            kind = str(query.get("kind") or "unknown")
            if counts[kind] >= limit_per_kind:
                continue
            for doc_id, score_value in qrels.get(query["id"], {}).items():
                writer.writerow(
                    {
                        "query_id": query["id"],
                        "kind": kind,
                        "query": query["text"],
                        "relevant_doc_id": doc_id,
                        "weak_score": score_value,
                        "human_judgment": "",
                        "review_notes": "",
                    }
                )
                counts[kind] += 1
                written += 1
                break
    counts["total"] = written
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build weakly judged bimodal routing eval set")
    parser.add_argument("--short-notes", type=positive_int, default=500)
    parser.add_argument("--long-limit", type=positive_int, default=500)
    parser.add_argument("--typo-fraction", type=typo_fraction, default=0.2)
    parser.add_argument("--split", choices=["dev", "test", "train"], default="dev")
    parser.add_argument("--out", type=Path, default=Path("eval/out/bimodal"))
    parser.add_argument(
        "--review-limit-per-kind",
        type=non_negative_int,
        default=25,
        help="write this many weak labels per query kind to review.csv for hand judgment",
    )
    args = parser.parse_args()

    settings = Settings()
    queries, qrels, meta = build_bimodal_set(
        settings=settings,
        short_notes=args.short_notes,
        long_limit=args.long_limit,
        typo_fraction=args.typo_fraction,
        split=args.split,
    )
    validate_bimodal_set(queries, qrels, meta)
    write_beir(queries, qrels, args.out)
    review_counts = write_review_csv(
        queries,
        qrels,
        args.out / "review.csv",
        limit_per_kind=args.review_limit_per_kind,
    )
    (args.out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), **meta, "review": review_counts}))


if __name__ == "__main__":
    main()
