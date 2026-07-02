from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from chart_common.config import EMBED_DIM, Settings
from chart_common.gateway import close_client, make_client, require_gateway_key

CLASSIFIER_WRITEBACK = {
    "mode": "tpuf.patch_columns",
    "primary_output": "events",
    "model_passes_per_note": 1,
    "patched_fields": [
        "events",
        "has_med_discontinuation",
        "has_adverse_event",
        "diagnosis_category",
        "specialty",
        "discontinuation_reason",
    ],
    "settles_multi_write": True,
}

EMBED_PRODUCTION_PATH = {
    "pipeline_cr": "chart-embed-gpu",
    "module": "indexer.embed",
    "compute_class": "gpu",
    "image": "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-embedder-plan-20260626-batchdocs1",
    "allow_full_cpu_index": False,
}

DEFAULT_EMBED_MODEL = Settings.model_fields["embed_model"].default


def _dump(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def build_report(
    *,
    kind: str,
    snapshot: dict[str, Any],
    accepted: bool,
    signal_reviewed: bool,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "source": "layer",
        "kind": kind,
        "accepted": accepted,
        "layer_cost_snapshot": snapshot,
    }
    if kind == "embed":
        report.update(
            {
                "sample": {
                    "notes": 1,
                    "vector_dim": EMBED_DIM,
                    "model": DEFAULT_EMBED_MODEL,
                },
                "production_path": EMBED_PRODUCTION_PATH,
            }
        )
    else:
        report.update(
            {
                "sample": {
                    "notes": 1,
                    "med_discontinuation": 1 if signal_reviewed else 0,
                },
                "examples": [
                    {
                        "note_preview": "Layer classifier cost gate accepted by operator",
                        "events": [{"type": "medication_discontinued"}],
                        "labels": {"has_med_discontinuation": True},
                        "discontinuation_reason": "operator_reviewed",
                    }
                ]
                if signal_reviewed
                else [],
                "signal": {
                    "accepted": signal_reviewed,
                    "checks": {
                        "min_med_discontinuations": {"ok": signal_reviewed},
                        "min_review_examples": {"ok": signal_reviewed},
                    },
                },
                "writeback": CLASSIFIER_WRITEBACK,
            }
        )
    return report


async def fetch_layer_cost_report(*, kind: str, window: str | None, accepted: bool, signal_reviewed: bool) -> dict[str, Any]:
    settings = Settings()
    require_gateway_key(settings)
    layer = make_client(settings)
    try:
        snapshot = _dump(await layer.get_cost_snapshot(window=window)) or {}
    finally:
        await close_client(layer)
    return build_report(kind=kind, snapshot=snapshot, accepted=accepted, signal_reviewed=signal_reviewed)


def _write(report: dict[str, Any], out: Path | None) -> None:
    rendered = json.dumps(report, indent=2)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered + "\n")
    print(rendered)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a PLAN.md cost gate report from Layer cost data")
    parser.add_argument("--kind", choices=["embed", "classifier"], required=True)
    parser.add_argument("--window", choices=["1h", "6h", "24h", "7d", "30d"], default="24h")
    parser.add_argument("--accept", action="store_true", help="mark the Layer cost report accepted")
    parser.add_argument(
        "--signal-reviewed",
        action="store_true",
        help="for classifier reports, assert the discontinuation examples/signals were reviewed",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.kind == "classifier" and args.accept and not args.signal_reviewed:
        raise SystemExit("--signal-reviewed is required when accepting a classifier cost report")
    report = asyncio.run(
        fetch_layer_cost_report(
            kind=args.kind,
            window=args.window,
            accepted=args.accept,
            signal_reviewed=args.signal_reviewed,
        )
    )
    _write(report, args.out)


if __name__ == "__main__":
    main()
