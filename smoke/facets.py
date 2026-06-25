from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from chart_common.cli import positive_float
from chart_common.config import Settings
from chart_common.gateway import FACET_FIELDS, close_client, latest_facets, make_client, materialize_facet_snapshots


async def refresh_facets(*, fields: list[str], timeout: float) -> dict:
    settings = Settings()
    layer = make_client(settings)
    try:
        await materialize_facet_snapshots(layer, settings.namespace, fields=fields, timeout=timeout)
        snapshots = {}
        for field in fields:
            values, provenance = await latest_facets(layer, settings.namespace, field=field)
            snapshots[field] = {
                "values": len(values or []),
                "sha": (provenance or {}).get("sha"),
                "row_count": (provenance or {}).get("row_count"),
                "watermark_ms": (provenance or {}).get("watermark_ms"),
            }
    finally:
        await close_client(layer)
    return {"namespace": settings.namespace, "status": "completed", "fields": fields, "snapshots": snapshots}


def _parse_fields(value: str) -> list[str]:
    fields = list(dict.fromkeys(field.strip() for field in value.split(",") if field.strip()))
    if not fields:
        raise argparse.ArgumentTypeError("at least one facet field is required")
    unknown = sorted(set(fields) - set(FACET_FIELDS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown facet field(s): {', '.join(unknown)}; choose from {', '.join(FACET_FIELDS)}"
        )
    return fields


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh chart facet snapshots without reindexing")
    parser.add_argument(
        "--fields",
        type=_parse_fields,
        default=FACET_FIELDS,
        help=f"comma-separated facet fields to refresh; default: {','.join(FACET_FIELDS)}",
    )
    parser.add_argument("--timeout", type=positive_float, default=180.0)
    parser.add_argument("--out", type=Path, default=None, help="write the facet refresh report to this JSON path")
    args = parser.parse_args()
    report = asyncio.run(refresh_facets(fields=args.fields, timeout=args.timeout))
    rendered = json.dumps(report, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
