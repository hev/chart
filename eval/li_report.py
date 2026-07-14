"""Combine the LI indexing run, the single-vector baseline run, and live
namespace metadata into the write-amplification report.

Run:  uv run python -m eval.li_report --out eval/out/li_write_amplification.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import httpx

from chart_common.config import Settings

LI_REPORT = Path("eval/out/li_index_report.json")
BASELINE_REPORT = Path("eval/out/li_baseline_report.json")


async def metadata(settings: Settings, namespace: str) -> dict[str, Any] | None:
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


def per_doc(report: dict) -> float | None:
    totals = report.get("totals") or {}
    docs = totals.get("docs") or 0
    if not docs:
        return None
    return totals.get("billable_logical_bytes_written", 0) / docs


async def main() -> None:
    ap = argparse.ArgumentParser(description="LI write-amplification report")
    ap.add_argument("--li-report", type=Path, default=LI_REPORT)
    ap.add_argument("--baseline-report", type=Path, default=BASELINE_REPORT)
    ap.add_argument("--out", type=Path, default=Path("eval/out/li_write_amplification.json"))
    args = ap.parse_args()

    li = json.loads(args.li_report.read_text())
    baseline = json.loads(args.baseline_report.read_text()) if args.baseline_report.exists() else None

    settings = Settings()
    notes_meta = await metadata(settings, settings.namespace)
    li_meta = await metadata(settings, settings.li_namespace)

    li_bytes_doc = per_doc(li)
    baseline_bytes_doc = per_doc(baseline) if baseline else None
    # Fallback baseline: the live chart-notes footprint (text+attrs+768-d vector).
    stored_baseline_doc = (
        notes_meta["approx_logical_bytes"] / notes_meta["approx_row_count"]
        if notes_meta and notes_meta.get("approx_row_count")
        else None
    )

    out: dict[str, Any] = {
        "li_namespace": settings.li_namespace,
        "notes_namespace": settings.namespace,
        "li_run": {
            "docs": li["totals"]["docs"],
            "tokens": li["totals"].get("tokens"),
            "billable_logical_bytes_written": li["totals"]["billable_logical_bytes_written"],
            "bytes_per_doc": li_bytes_doc,
            "tokens_per_doc": li.get("tokens_per_doc"),
            "bytes_per_token": li.get("bytes_per_token"),
            "model": li["provenance"].get("model"),
            "dim": li["provenance"].get("dim"),
        },
        "baseline_run": None
        if baseline is None
        else {
            "docs": baseline["totals"]["docs"],
            "billable_logical_bytes_written": baseline["totals"][
                "billable_logical_bytes_written"
            ],
            "bytes_per_doc": baseline_bytes_doc,
            "model": baseline["provenance"].get("model"),
        },
        "stored": {"chart_notes": notes_meta, "chart_notes_li": li_meta},
    }
    if li_bytes_doc and baseline_bytes_doc:
        out["write_amplification_vs_single_vector_row"] = round(
            li_bytes_doc / baseline_bytes_doc, 1
        )
    if li_bytes_doc and stored_baseline_doc:
        out["write_amplification_vs_stored_chart_notes"] = round(
            li_bytes_doc / stored_baseline_doc, 1
        )
    if (
        li_meta
        and notes_meta
        and li_meta.get("approx_logical_bytes")
        and notes_meta.get("approx_logical_bytes")
    ):
        out["storage_amplification_stored"] = round(
            li_meta["approx_logical_bytes"] / notes_meta["approx_logical_bytes"], 1
        )

    print(json.dumps(out, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
