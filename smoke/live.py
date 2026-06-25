from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from hevlayer import QueryRequest

from chart_common.cli import positive_int
from chart_common.config import EMBED_DIM, Settings
from chart_common.embed import Embedder
from chart_common.gateway import FACET_FIELDS, close_client, latest_facets, make_client, require_gateway_key
from search.app import INCLUDE

QUERIES = Path(__file__).resolve().parent.parent / "web" / "static" / "queries.json"
ROUTES = {"hybrid_text", "fused", "semantic"}


class FacetCheckError(AssertionError):
    def __init__(self, message: str, facets: dict) -> None:
        super().__init__(message)
        self.facets = facets


def policy_route(text: str) -> str:
    tokens = text.split()
    if len(tokens) <= 2:
        return "hybrid_text"
    if len(tokens) >= 8:
        return "semantic"
    return "fused"


def expected_route(example: dict[str, Any]) -> str:
    declared = example.get("expected_route")
    if declared not in ROUTES:
        raise ValueError(f"{example.get('text')!r}: expected_route must be one of {sorted(ROUTES)}")
    policy = policy_route(str(example["text"]))
    if declared != policy:
        raise ValueError(f"{example['text']!r}: expected_route {declared} does not match v1 policy {policy}")
    return declared


def _dump(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


async def _query(layer, namespace: str, body: QueryRequest):
    return await layer.query_namespace(namespace, body)


def _query_vector(embedder: Embedder, text: str) -> list[float]:
    vector = embedder.embed_query(text)
    if len(vector) != EMBED_DIM:
        raise AssertionError(f"{text!r}: expected {EMBED_DIM}-d query vector, got {len(vector)}")
    return vector


def _row_id(row: Any) -> str | None:
    if isinstance(row, dict):
        value = row.get("id") or row.get("$id")
    else:
        value = getattr(row, "id", None) or getattr(row, "$id", None)
    return str(value) if value else None


def _row_ids(rows: list[Any], *, context: str) -> list[str]:
    ids = [_row_id(row) for row in rows]
    if rows and not all(ids):
        raise AssertionError(f"{context} returned rows without ids")
    return [row_id for row_id in ids if row_id]


def _route(routing: dict[str, Any]) -> str | None:
    return routing.get("route") or routing.get("strategy")


async def check_routes(layer, settings: Settings, embedder: Embedder, *, top_k: int) -> list[dict]:
    examples = json.loads(QUERIES.read_text())["examples"]
    results = []
    for example in examples:
        text = example["text"]
        vector = _query_vector(embedder, text)
        resp = await _query(
            layer,
            settings.namespace,
            QueryRequest(
                rank_by=["text", "Auto", text, {"vector": vector}],
                top_k=top_k,
                include_attributes=INCLUDE,
                include_leg_breakdown=True,
            ),
        )
        routing = _dump(resp.routing) or {}
        route = _route(routing)
        expected = expected_route(example)
        if route != expected:
            raise AssertionError(f"{text!r}: expected {expected}, got {route}")
        hybrid = _dump(resp.hybrid)
        if route == "fused" and not hybrid:
            raise AssertionError(f"{text!r}: gateway response did not include a fused hybrid echo")
        rows = resp.rows or []
        if not rows:
            raise AssertionError(f"{text!r}: gateway returned no rows")
        _row_ids(rows, context=repr(text))
        results.append(
            {
                "query": text,
                "route": route,
                "routing": routing,
                "hybrid": hybrid,
                "rows": len(rows),
            }
        )
    return results


async def check_index_shape(layer, settings: Settings, embedder: Embedder) -> dict:
    vector = _query_vector(embedder, "metformin 500mg")

    resp = await _query(
        layer,
        settings.namespace,
        QueryRequest(
            rank_by=["text", "Auto", "metformin 500mg", {"vector": vector}],
            top_k=20,
            include_attributes=["id", "age_band", "gender", "similar_patient_ids"],
        ),
    )
    rows = resp.rows or []
    if not rows:
        raise AssertionError("index shape check returned no rows")
    _row_ids(rows, context="index shape check")
    if not any(row.get("age_band") for row in rows):
        raise AssertionError("index shape check found no populated age_band values")
    if not any(row.get("gender") for row in rows):
        raise AssertionError("index shape check found no populated gender values")
    if not any(isinstance(row.get("similar_patient_ids"), list) and row.get("similar_patient_ids") for row in rows):
        raise AssertionError("index shape check found no populated similar_patient_ids list")
    return {"vector_dim": len(vector), "rows": len(rows)}


async def check_similar(
    layer, settings: Settings, embedder: Embedder, patient_id: str | None, *, top_k: int
) -> dict:
    if patient_id is None:
        seed = await _query(
            layer,
            settings.namespace,
            QueryRequest(
                rank_by=[
                    "text",
                    "Auto",
                    "metformin 500mg",
                    {"vector": _query_vector(embedder, "metformin 500mg")},
                ],
                top_k=1,
                include_attributes=["id", "similar_patient_ids"],
            ),
        )
        rows = seed.rows or []
        seed_ids = _row_ids(rows, context="nearest_to_id seed lookup")
        if not seed_ids:
            raise AssertionError("could not find a seed row for nearest_to_id")
        patient_id = seed_ids[0]

    resp = await _query(
        layer,
        settings.namespace,
        QueryRequest(
            nearest_to_id=[patient_id],
            top_k=top_k,
            include_attributes=INCLUDE,
        ),
    )
    rows = resp.rows or []
    if not rows:
        raise AssertionError(f"nearest_to_id returned no rows for {patient_id!r}")
    neighbor_ids = _row_ids(rows, context="nearest_to_id")
    if not any(row_id != patient_id for row_id in neighbor_ids):
        raise AssertionError(f"nearest_to_id returned no neighbor rows for {patient_id!r}")
    return {"patient_id": patient_id, "rows": len(rows), "neighbors": len(set(neighbor_ids) - {patient_id})}


async def check_facets(layer, settings: Settings, *, require_events: bool = False) -> dict:
    out = {}

    def fail(message: str) -> None:
        raise FacetCheckError(message, out)

    for field in FACET_FIELDS:
        values, provenance = await latest_facets(layer, settings.namespace, field=field)
        if values or provenance:
            out[field] = {
                "values": len(values or []),
                "sha": (provenance or {}).get("sha"),
                "row_count": (provenance or {}).get("row_count"),
                "watermark_ms": (provenance or {}).get("watermark_ms"),
            }
    for required in ("age_band", "gender"):
        if required not in out:
            fail(f"missing materialized facet snapshot for {required!r}")
        if not out[required].get("sha"):
            fail(f"materialized facet snapshot for {required!r} is missing sha provenance")
    if require_events and "events" not in out:
        fail("missing materialized facet snapshot for 'events'")
    if require_events and not out["events"].get("sha"):
        fail("materialized facet snapshot for 'events' is missing sha provenance")
    if require_events and out["events"].get("values", 0) <= 0:
        fail("materialized facet snapshot for 'events' has no values")
    return out


def _write_report(report: dict, out: Path | None) -> None:
    rendered = json.dumps(report, indent=2)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered + "\n")
    print(rendered)


async def run(
    *,
    top_k: int,
    similar_id: str | None,
    skip_facets: bool,
    require_event_facets: bool,
    out: Path | None = None,
) -> dict:
    settings = Settings()
    require_gateway_key(settings)
    embedder = Embedder(settings.embed_model)
    layer = make_client(settings)
    report = {
        "ok": False,
        "status": "failed",
        "requirements": {
            "facets": not skip_facets,
            "event_facets": require_event_facets,
        },
        "index_shape": None,
        "routes": None,
        "similar": None,
        "facets": None,
    }
    try:
        report["index_shape"] = await check_index_shape(layer, settings, embedder)
        report["routes"] = await check_routes(layer, settings, embedder, top_k=top_k)
        report["similar"] = await check_similar(layer, settings, embedder, similar_id, top_k=min(top_k, 12))
        report["facets"] = (
            None if skip_facets else await check_facets(layer, settings, require_events=require_event_facets)
        )
    except FacetCheckError as exc:
        report["facets"] = exc.facets
        report["error"] = str(exc)
        _write_report(report, out)
        raise
    except Exception as exc:
        report["error"] = str(exc)
        _write_report(report, out)
        raise
    finally:
        await close_client(layer)

    report["ok"] = True
    report["status"] = "completed"
    _write_report(report, out)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Live PLAN.md smoke for indexed chart namespace")
    parser.add_argument("--top-k", type=positive_int, default=5)
    parser.add_argument("--similar-id", default=None)
    parser.add_argument("--skip-facets", action="store_true")
    parser.add_argument(
        "--require-event-facets",
        action="store_true",
        help="also require the Gemma events facet snapshot to be visible",
    )
    parser.add_argument("--out", type=Path, default=None, help="write the live smoke report to this JSON path")
    args = parser.parse_args()
    asyncio.run(
        run(
            top_k=args.top_k,
            similar_id=args.similar_id,
            skip_facets=args.skip_facets,
            require_event_facets=args.require_event_facets,
            out=args.out,
        )
    )


if __name__ == "__main__":
    main()
