"""Late-interaction indexer — the token-bag twin of indexer/index.py.

Writes the SAME PMC-Patients slice chart-notes holds into the LI sibling
namespace (settings.li_namespace), swapping the single Arctic vector for a
ColBERT-style `tokens` bag ([][N]f32, Turbopuffer late-interaction private
beta). Every write response's billable_logical_bytes_written is summed so the
run doubles as the write-amplification measurement.

`--baseline` instead writes the same rows in the standard single-vector shape
to a throwaway namespace and records ITS billing, so the amplification ratio is
measured on identical rows, same day, same rates — then deletes the namespace.

Run:  uv run python -m indexer.index_li --limit 11373 --out eval/out/li_index_report.json
      uv run python -m indexer.index_li --baseline --limit 1000 --out eval/out/li_baseline_report.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx

from chart_common.cli import non_negative_int, positive_int
from chart_common.config import LI_EMBED_DIM, Settings
from chart_common.embed import Embedder, LateInteractionEmbedder
from chart_common.gateway import (
    billing_bytes_written,
    close_client,
    make_client,
    write_li_notes,
    write_notes,
)
from chart_common.records import NoteRecord

from .dataset import load_notes
from .index import _batched, attach_vectors

# Embedding dominates wall-clock; small batches keep request payloads (JSON
# floats are ~3x the logical f32 bytes) and progress checkpoints reasonable.
BATCH = 64
BASELINE_NAMESPACE_SUFFIX = "-b0"


def li_rows(batch: list[NoteRecord], bags: list[list[list[float]]]) -> list[dict]:
    if len(bags) != len(batch):
        raise ValueError(f"embedder returned {len(bags)} bags for {len(batch)} records")
    rows = []
    for record, bag in zip(batch, bags, strict=True):
        if not bag or len(bag[0]) != LI_EMBED_DIM:
            raise ValueError(
                f"{record.id}: expected non-empty {LI_EMBED_DIM}-d token bag, "
                f"got {len(bag)}x{len(bag[0]) if bag else 0}"
            )
        row = record.to_upsert()
        row.pop("vector", None)
        # Round to 6 significant decimals: halves the JSON payload; f32 keeps
        # ~7 digits anyway, so the stored vectors are unchanged in practice.
        row["tokens"] = [[round(x, 6) for x in vec] for vec in bag]
        rows.append(row)
    return rows


async def run(
    *,
    limit: int | None,
    skip: int = 0,
    baseline: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    settings = Settings()
    namespace = (
        settings.li_namespace + BASELINE_NAMESPACE_SUFFIX if baseline else settings.li_namespace
    )
    embedder = Embedder(settings.embed_model) if baseline else None
    li_embedder = None if baseline else LateInteractionEmbedder(settings.li_model)

    layer = None if dry_run else make_client(settings, timeout=300.0)
    started = time.time()
    totals = {"docs": 0, "tokens": 0, "billable_logical_bytes_written": 0, "batches": 0}
    try:
        notes = load_notes(settings, limit=None if limit is None else limit + skip)
        for _ in range(skip):
            next(notes, None)
        for batch in _batched(notes, BATCH):
            if baseline:
                vectors = embedder.embed_passages([r.text for r in batch])
                attach_vectors(batch, vectors)
                rows = [r.to_upsert() for r in batch]
            else:
                bags = li_embedder.embed_passages([r.text for r in batch])
                rows = li_rows(batch, bags)
                totals["tokens"] += sum(len(row["tokens"]) for row in rows)
            if not dry_run:
                resp = (
                    await write_notes(layer, namespace, rows)
                    if baseline
                    else await write_li_notes(layer, namespace, rows, dim=LI_EMBED_DIM)
                )
                totals["billable_logical_bytes_written"] += billing_bytes_written(resp)
            totals["docs"] += len(rows)
            totals["batches"] += 1
            rate = totals["docs"] / max(time.time() - started, 1e-9)
            print(
                f"  {namespace}: {totals['docs']} docs "
                f"({totals['billable_logical_bytes_written']:,} billable bytes, "
                f"{rate:.1f} docs/s)",
                flush=True,
            )
    finally:
        if layer is not None:
            await close_client(layer)

    report: dict[str, Any] = {
        "namespace": namespace,
        "mode": "baseline-single-vector" if baseline else "late-interaction",
        "limit": limit,
        "skip": skip,
        "dry_run": dry_run,
        "elapsed_seconds": round(time.time() - started, 1),
        "totals": totals,
        "provenance": {
            "dataset_repo": settings.dataset_repo,
            "dataset_revision": settings.dataset_revision,
            "model": settings.embed_model if baseline else settings.li_model,
            "dim": LI_EMBED_DIM if not baseline else None,
        },
    }
    if totals["docs"]:
        report["bytes_per_doc"] = round(
            totals["billable_logical_bytes_written"] / totals["docs"], 1
        )
    if totals["tokens"]:
        report["tokens_per_doc"] = round(totals["tokens"] / totals["docs"], 1)
        report["bytes_per_token"] = round(
            totals["billable_logical_bytes_written"] / totals["tokens"], 1
        )
    if not dry_run:
        report["namespace_metadata"] = await namespace_metadata(settings, namespace)
    return report


async def namespace_metadata(settings: Settings, namespace: str) -> dict | None:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{settings.gateway_url.rstrip('/')}/v1/namespaces/{namespace}/metadata",
            headers={"Authorization": f"Bearer {settings.api_key}"},
        )
        if resp.status_code != 200:
            return None
        body = resp.json()
        return {
            "approx_row_count": body.get("approx_row_count"),
            "approx_logical_bytes": body.get("approx_logical_bytes"),
        }


async def delete_namespace(settings: Settings, namespace: str) -> int:
    """v2 namespace delete (the v1 spelling 404s through the gateway)."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.delete(
            f"{settings.gateway_url.rstrip('/')}/v2/namespaces/{namespace}",
            headers={"Authorization": f"Bearer {settings.api_key}"},
        )
        return resp.status_code


async def amain(args: argparse.Namespace) -> None:
    report = await run(
        limit=args.limit, skip=args.skip, baseline=args.baseline, dry_run=args.dry_run
    )
    if args.baseline and args.delete_after:
        settings = Settings()
        status = await delete_namespace(settings, report["namespace"])
        report["baseline_namespace_deleted"] = status == 200
    print(json.dumps(report, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Index PMC-Patients token bags into the late-interaction namespace"
    )
    ap.add_argument("--limit", type=positive_int, default=None, help="cap notes")
    ap.add_argument(
        "--skip", type=non_negative_int, default=0, help="skip the first N notes (resume)"
    )
    ap.add_argument(
        "--baseline",
        action="store_true",
        help="write the SAME rows in single-vector shape to a throwaway namespace instead",
    )
    ap.add_argument(
        "--delete-after",
        action="store_true",
        help="with --baseline: delete the throwaway namespace after measuring",
    )
    ap.add_argument("--dry-run", action="store_true", help="embed but skip the gateway")
    ap.add_argument("--out", type=Path, default=None, help="write the report JSON here")
    args = ap.parse_args()
    if args.limit is None and not args.dry_run:
        raise SystemExit("refusing an unbounded run; pass --limit (chart-notes holds 11373)")
    asyncio.run(amain(args))
