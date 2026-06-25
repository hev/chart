from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from chart_common.config import FULL_CORPUS_NOTES
from chart_common.gateway import FACET_FIELDS

from .full_status import collect_status


def _int_value(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _invalid_int_fields(section: dict[str, Any], fields: tuple[str, ...], *, counts_fields: tuple[str, ...] = ()) -> list[str]:
    invalid = []
    counts = section.get("counts") or {}
    for field in fields:
        value = section.get(field)
        if value is not None:
            try:
                int(value)
            except (TypeError, ValueError):
                invalid.append(field)
    for field in counts_fields:
        value = counts.get(field)
        if value is not None:
            try:
                int(value)
            except (TypeError, ValueError):
                invalid.append(f"counts.{field}")
    return invalid


def _ready_count(section: dict[str, Any]) -> int:
    if "error" in section:
        return 0
    counts = section.get("counts") or {}
    return _int_value(counts.get("indexed") or counts.get("completed") or section.get("indexed_count") or 0)


def _installed(section: dict[str, Any]) -> bool:
    return bool(section) and "error" not in section


def _is_clear(section: dict[str, Any]) -> bool:
    if "error" in section:
        return False
    counts = section.get("counts") or {}
    if _invalid_int_fields(
        section,
        ("pending_count", "processing_count", "failed_count"),
        counts_fields=("failed",),
    ):
        return False
    failed_count = _int_value(section.get("failed_count") or counts.get("failed") or 0)
    return (
        _int_value(section.get("pending_count") or 0) == 0
        and _int_value(section.get("processing_count") or 0) == 0
        and failed_count == 0
    )


def _facet_row_count(section: dict[str, Any]) -> int:
    return _int_value(section.get("row_count") or 0)


def _facet_values(section: dict[str, Any]) -> int:
    return _int_value(section.get("values") or 0)


def _facet_has_snapshot(section: dict[str, Any]) -> bool:
    return bool(section.get("sha"))


def _full_facet_shas(facets: dict[str, Any]) -> set[str]:
    return {str((facets.get(field) or {}).get("sha")) for field in FACET_FIELDS if (facets.get(field) or {}).get("sha")}


def _facet_snapshots_aligned(facets: dict[str, Any]) -> bool:
    return len(_full_facet_shas(facets)) == 1


def _facet_failure_reasons(section: dict[str, Any], *, target: int) -> list[str]:
    if "error" in section:
        error = section.get("error") or {}
        status_code = error.get("status_code")
        message = error.get("message")
        detail = f"HTTP {status_code}" if status_code else "gateway error"
        return [f"{detail}: {message}" if message else detail]
    reasons = []
    invalid = _invalid_int_fields(section, ("row_count", "values"))
    if invalid:
        reasons.append(f"invalid numeric field(s): {', '.join(invalid)}")
    if _facet_row_count(section) < target:
        reasons.append(f"row_count={_facet_row_count(section)}/{target}")
    if _facet_values(section) <= 0:
        reasons.append("no values")
    if not _facet_has_snapshot(section):
        reasons.append("missing sha")
    return reasons


def _section_failure_reason(section: dict[str, Any], *, target: int) -> str | None:
    return _section_failure_reason_for_id(section, target=target, id_field=None, expected_id=None)


def _section_failure_reason_for_id(
    section: dict[str, Any], *, target: int, id_field: str | None, expected_id: str | None
) -> str | None:
    if not section:
        return "status body is empty"
    if "error" in section:
        error = section.get("error") or {}
        status_code = error.get("status_code")
        message = error.get("message")
        detail = f"HTTP {status_code}" if status_code else "gateway error"
        return f"{detail}: {message}" if message else detail
    if id_field and expected_id and section.get(id_field) != expected_id:
        return f"{id_field}={section.get(id_field)!r}, expected {expected_id!r}"
    counts = section.get("counts") or {}
    invalid = _invalid_int_fields(
        section,
        ("pending_count", "processing_count", "failed_count", "indexed_count"),
        counts_fields=("indexed", "completed", "failed"),
    )
    if invalid:
        return f"invalid numeric field(s): {', '.join(invalid)}"
    pending = _int_value(section.get("pending_count") or 0)
    processing = _int_value(section.get("processing_count") or 0)
    failed = _int_value(section.get("failed_count") or counts.get("failed") or 0)
    ready = _ready_count(section)
    if failed:
        return f"{failed} failed"
    if pending or processing:
        return f"{pending} pending, {processing} processing"
    if ready < target:
        return f"{ready}/{target} ready"
    return None


def _target_notes(status: dict[str, Any]) -> int:
    targets = status.get("targets") or {}
    return int(targets.get("full_corpus_notes") or FULL_CORPUS_NOTES)


def _cost_baselines_accepted(status: dict[str, Any]) -> bool:
    baselines = status.get("cost_baselines") or {}
    return (
        (baselines.get("embed") or {}).get("accepted") is True
        and (baselines.get("classifier") or {}).get("accepted") is True
    )


def gate_failures(status: dict[str, Any], gates: dict[str, bool]) -> list[dict[str, Any]]:
    target = _target_notes(status)
    targets = status.get("targets") or {}
    expected_pipeline_id = targets.get("pipeline_id")
    expected_udf_id = targets.get("udf_id")
    pipeline = status.get("pipeline") or {}
    udf = status.get("udf") or {}
    facets = status.get("facets") or {}
    failures: list[dict[str, Any]] = []

    if not gates["pipeline_installed"]:
        failures.append({"gate": "pipeline_installed", "reason": _section_failure_reason(pipeline, target=target)})
    if not gates["udf_installed"]:
        failures.append({"gate": "udf_installed", "reason": _section_failure_reason(udf, target=target)})
    if gates["pipeline_installed"] and not gates["full_index_complete"]:
        failure = {
            "gate": "full_index_complete",
            "reason": _section_failure_reason_for_id(
                pipeline,
                target=target,
                id_field="pipeline_id",
                expected_id=expected_pipeline_id,
            ),
        }
        kubernetes = status.get("kubernetes") or {}
        embed_pods = kubernetes.get("embed_pods") or []
        gpu_pods = kubernetes.get("gpu_pods") or []
        if embed_pods or gpu_pods:
            failure["kubernetes"] = {
                "embed_pods": embed_pods,
                "gpu_pods": gpu_pods,
            }
        failures.append(failure)
    if gates["udf_installed"] and not gates["full_classify_complete"]:
        failures.append(
            {
                "gate": "full_classify_complete",
                "reason": _section_failure_reason_for_id(
                    udf,
                    target=target,
                    id_field="udf_id",
                    expected_id=expected_udf_id,
                ),
            }
        )
    if not gates["base_facets_visible"]:
        missing = [field for field in ("age_band", "gender") if _facet_values(facets.get(field) or {}) <= 0]
        failures.append({"gate": "base_facets_visible", "reason": f"missing values for: {', '.join(missing)}"})
    if not gates["event_facets_visible"]:
        failures.append({"gate": "event_facets_visible", "reason": "missing values for: events"})
    if not gates["full_facets_complete"]:
        incomplete = []
        aligned = _facet_snapshots_aligned(facets)
        for field in FACET_FIELDS:
            section = facets.get(field) or {}
            reasons = _facet_failure_reasons(section, target=target)
            if not aligned and _facet_has_snapshot(section):
                reasons.append("snapshot sha differs from other full facets")
            if reasons:
                incomplete.append({"field": field, "reasons": reasons})
        failures.append({"gate": "full_facets_complete", "reason": "incomplete facet snapshots", "facets": incomplete})
    if not gates["cost_baselines_accepted"]:
        baselines = status.get("cost_baselines") or {}
        incomplete = []
        for name in ("embed", "classifier"):
            baseline = baselines.get(name) or {}
            reasons = []
            if baseline.get("accepted") is not True:
                reasons.append("not accepted")
            if baseline.get("error"):
                reasons.append(str(baseline.get("error")))
            if not baseline.get("report"):
                reasons.append("missing report path")
            if not (baseline.get("estimate") or baseline.get("layer_cost_snapshot") or {}):
                reasons.append("missing estimate")
            if reasons:
                incomplete.append({"baseline": name, "reasons": reasons})
        failures.append({"gate": "cost_baselines_accepted", "reason": "incomplete cost baselines", "baselines": incomplete})
    return failures


def summarize_gates(status: dict[str, Any]) -> dict[str, Any]:
    pipeline = status.get("pipeline") or {}
    udf = status.get("udf") or {}
    facets = status.get("facets") or {}
    targets = status.get("targets") or {}
    target = _target_notes(status)
    expected_pipeline_id = targets.get("pipeline_id")
    expected_udf_id = targets.get("udf_id")
    event_facet = facets.get("events") or {}
    age_band = facets.get("age_band") or {}
    gender = facets.get("gender") or {}
    pipeline_matches = not expected_pipeline_id or pipeline.get("pipeline_id") == expected_pipeline_id
    udf_matches = not expected_udf_id or udf.get("udf_id") == expected_udf_id

    gates = {
        "pipeline_installed": _installed(pipeline),
        "udf_installed": _installed(udf),
        "full_index_complete": _installed(pipeline)
        and pipeline_matches
        and _is_clear(pipeline)
        and _ready_count(pipeline) >= target,
        "full_classify_complete": _installed(udf)
        and udf_matches
        and _is_clear(udf)
        and _ready_count(udf) >= target,
        "base_facets_visible": (age_band.get("values") or 0) > 0 and (gender.get("values") or 0) > 0,
        "event_facets_visible": (event_facet.get("values") or 0) > 0,
        "cost_baselines_accepted": _cost_baselines_accepted(status),
    }
    gates["full_facets_complete"] = all(
        _facet_row_count(facets.get(field) or {}) >= target
        and _facet_values(facets.get(field) or {}) > 0
        and _facet_has_snapshot(facets.get(field) or {})
        for field in FACET_FIELDS
    ) and _facet_snapshots_aligned(facets)
    gates["phase6_complete"] = all(
        gates[name]
        for name in (
            "pipeline_installed",
            "udf_installed",
            "full_index_complete",
            "full_classify_complete",
            "base_facets_visible",
            "event_facets_visible",
            "full_facets_complete",
            "cost_baselines_accepted",
        )
    )
    failures = gate_failures(status, gates)
    return {
        "namespace": status.get("namespace"),
        "targets": status.get("targets") or {
            "namespace": status.get("namespace"),
            "full_corpus_notes": FULL_CORPUS_NOTES,
        },
        "gates": gates,
        "failures": failures,
        "status": status,
    }


async def collect_gate_report(*, pipeline_id: str | None = None, udf_id: str = "chart-classify-events") -> dict[str, Any]:
    return summarize_gates(await collect_status(pipeline_id=pipeline_id, udf_id=udf_id))


def main() -> None:
    parser = argparse.ArgumentParser(description="PLAN.md gate report for chart")
    parser.add_argument("--pipeline-id", default=None)
    parser.add_argument("--udf-id", default="chart-classify-events")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="exit non-zero unless every Phase-6 gate is complete",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write the Phase-6 gate report to this JSON path",
    )
    args = parser.parse_args()
    report = asyncio.run(collect_gate_report(pipeline_id=args.pipeline_id, udf_id=args.udf_id))
    rendered = json.dumps(report, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n")
    print(rendered)
    if args.require_complete and not report["gates"]["phase6_complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
