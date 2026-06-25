from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

from chart_common.config import FULL_CORPUS_NOTES

DEFAULT_REPORTS = {
    "slice_index": ("CHART_SLICE_INDEX_REPORT", "eval/out/slice-index-report.json"),
    "live_smoke_base": ("CHART_LIVE_SMOKE_BASE_REPORT", "eval/out/live-smoke-base-report.json"),
    "live_smoke": ("CHART_LIVE_SMOKE_REPORT", "eval/out/live-smoke-report.json"),
    "facet_refresh": ("CHART_FACET_REFRESH_REPORT", "eval/out/facet-refresh-report.json"),
    "embed_budget": ("CHART_PHASE6_EMBED_BUDGET_REPORT", "eval/out/embed-budget.json"),
    "classify_budget": ("CHART_PHASE4_CLASSIFY_REPORT", "eval/out/classify-events-budget.json"),
    "holdout": ("CHART_EVAL_HOLDOUT_REPORT", "eval/out/holdout-report.json"),
    "recds": ("CHART_EVAL_RECDS_REPORT", "eval/out/recds-report.json"),
    "bimodal": ("CHART_EVAL_BIMODAL_REPORT", "eval/out/bimodal-report.json"),
    "phase6_status": ("CHART_PHASE6_STATUS_REPORT", "eval/out/phase6-status-report.json"),
    "phase6_gate": ("CHART_PHASE6_GATE_REPORT", "eval/out/phase6-gate-report.json"),
    "phase6_unpause": ("CHART_PHASE6_UNPAUSE_REPORT", "eval/out/phase6-unpause-report.json"),
    "gpu_build": ("CHART_GPU_BUILD_REPORT", "eval/out/gpu-build-report.json"),
    "deploy_apply": ("CHART_DEPLOY_APPLY_REPORT", "eval/out/deploy-apply-report.json"),
}


def report_path(name: str, *, env: dict[str, str] | None = None) -> Path:
    var, default = DEFAULT_REPORTS[name]
    return Path((env or os.environ).get(var, default))


def load_report(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, "missing"
    try:
        value = json.loads(path.read_text())
    except Exception as exc:
        return None, f"invalid json: {exc}"
    if not isinstance(value, dict):
        return None, "report is not a JSON object"
    return value, None


def _object_section(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _accepted_checks(section: dict[str, Any], required: tuple[str, ...]) -> bool:
    if section.get("accepted") is not True:
        return False
    checks = section.get("checks") or {}
    return all((checks.get(name) or {}).get("ok") is True for name in required)


def _budget_accepted(report: dict[str, Any]) -> bool:
    return _accepted_checks(report.get("budget") or {}, ("max_full_hours", "max_full_usd"))


def _signal_accepted(report: dict[str, Any]) -> bool:
    return _accepted_checks(report.get("signal") or {}, ("min_med_discontinuations", "min_review_examples"))


def _accepted_check_failures(section: dict[str, Any], required: tuple[str, ...], *, label: str) -> list[str]:
    failures = []
    if section.get("accepted") is not True:
        failures.append(f"{label}.accepted={section.get('accepted')!r}, expected true")
    checks = section.get("checks") or {}
    for name in required:
        check = checks.get(name) or {}
        if check.get("ok") is not True:
            failures.append(f"{label}.checks.{name}.ok={check.get('ok')!r}, expected true")
    return failures


def _cost_estimate_complete(report: dict[str, Any]) -> bool:
    estimate = report.get("estimate") or {}
    return (
        _positive_int(estimate.get("full_notes")) == FULL_CORPUS_NOTES
        and _positive_number(estimate.get("estimated_full_seconds")) > 0
        and _positive_number(estimate.get("estimated_full_hours")) > 0
        and _positive_number(estimate.get("estimated_full_usd")) > 0
        and _positive_number(estimate.get("gpu_hourly_usd")) > 0
    )


def _layer_cost_snapshot(report: dict[str, Any]) -> dict[str, Any]:
    snapshot = report.get("layer_cost_snapshot") or report.get("cost_snapshot") or {}
    return snapshot if isinstance(snapshot, dict) else {}


def _layer_cost_evidence_complete(report: dict[str, Any]) -> bool:
    snapshot = _layer_cost_snapshot(report)
    totals = snapshot.get("totals") or {}
    lines = snapshot.get("lines") or []
    return (
        report.get("accepted") is True
        and report.get("source") in {"layer", "layer_cost"}
        and _positive_int(snapshot.get("as_of_ms")) > 0
        and _positive_int(snapshot.get("window_seconds")) > 0
        and "total_usd" in totals
        and _positive_number(totals.get("total_usd")) >= 0
        and isinstance(lines, list)
        and any(isinstance(line, dict) and line.get("basis") in {"metered", "invoice"} for line in lines)
    )


def _cost_gate_accepted(report: dict[str, Any]) -> bool:
    return (
        _budget_accepted(report)
        and _cost_estimate_complete(report)
    ) or _layer_cost_evidence_complete(report)


def _cost_gate_failures(report: dict[str, Any], *, label: str = "budget") -> list[str]:
    if _layer_cost_evidence_complete(report):
        return []
    if report.get("source") in {"layer", "layer_cost"} or report.get("layer_cost_snapshot") or report.get("cost_snapshot"):
        snapshot = _layer_cost_snapshot(report)
        totals = snapshot.get("totals") or {}
        lines = snapshot.get("lines") or []
        failures = []
        if report.get("accepted") is not True:
            failures.append("accepted must be true for Layer cost report")
        if report.get("source") not in {"layer", "layer_cost"}:
            failures.append("source must be 'layer' or 'layer_cost'")
        if _positive_int(snapshot.get("as_of_ms")) <= 0:
            failures.append("layer_cost_snapshot.as_of_ms must be positive")
        if _positive_int(snapshot.get("window_seconds")) <= 0:
            failures.append("layer_cost_snapshot.window_seconds must be positive")
        if "total_usd" not in totals or _positive_number(totals.get("total_usd")) < 0:
            failures.append("layer_cost_snapshot.totals.total_usd must be non-negative")
        if not isinstance(lines, list) or not any(
            isinstance(line, dict) and line.get("basis") in {"metered", "invoice"} for line in lines
        ):
            failures.append("layer_cost_snapshot.lines must include a metered or invoice line")
        return failures
    failures = _accepted_check_failures(report.get("budget") or {}, ("max_full_hours", "max_full_usd"), label=label)
    failures.extend(_cost_estimate_failures(report))
    return failures


def _cost_estimate_failures(report: dict[str, Any]) -> list[str]:
    failures = []
    estimate = report.get("estimate") or {}
    if _positive_int(estimate.get("full_notes")) != FULL_CORPUS_NOTES:
        failures.append(f"estimate.full_notes={estimate.get('full_notes')!r}, expected {FULL_CORPUS_NOTES}")
    for name in ("estimated_full_seconds", "estimated_full_hours", "estimated_full_usd", "gpu_hourly_usd"):
        if _positive_number(estimate.get(name)) <= 0:
            failures.append(f"estimate.{name} must be positive")
    return failures


def _has_positive_sample(report: dict[str, Any]) -> bool:
    sample = report.get("sample") or {}
    try:
        return int(sample.get("notes") or 0) > 0
    except (TypeError, ValueError):
        return False


def _gpu_runtime(report: dict[str, Any]) -> bool:
    runtime = report.get("runtime") or {}
    return runtime.get("accelerator") == "gpu" and bool(runtime.get("gpu_device"))


def _embed_sample_complete(report: dict[str, Any]) -> bool:
    sample = report.get("sample") or {}
    return (
        _has_positive_sample(report)
        and _positive_int(sample.get("vector_dim")) == 768
        and sample.get("model") == "Snowflake/snowflake-arctic-embed-m-v1.5"
    )


def _embed_production_path_complete(report: dict[str, Any]) -> bool:
    path = report.get("production_path") or {}
    return (
        path.get("pipeline_cr") == "chart-embed-gpu"
        and path.get("module") == "indexer.embed"
        and path.get("compute_class") == "gpu"
        and path.get("image") == "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-embedder-plan-20260624-dedupe2"
        and path.get("allow_full_cpu_index") is False
    )


def _classify_sample_complete(report: dict[str, Any]) -> bool:
    sample = report.get("sample") or {}
    examples = report.get("examples") or []
    return (
        _has_positive_sample(report)
        and _positive_int(sample.get("med_discontinuation")) > 0
        and isinstance(examples, list)
        and len(examples) > 0
        and any(
            isinstance(example, dict)
            and (example.get("labels") or {}).get("has_med_discontinuation") is True
            and bool(example.get("discontinuation_reason"))
            for example in examples
        )
    )


def _classify_writeback_complete(report: dict[str, Any]) -> bool:
    writeback = report.get("writeback") or {}
    patched_fields = set(writeback.get("patched_fields") or [])
    required_fields = {
        "events",
        "has_med_discontinuation",
        "has_adverse_event",
        "diagnosis_category",
        "specialty",
        "discontinuation_reason",
    }
    return (
        writeback.get("mode") == "tpuf.patch_columns"
        and writeback.get("primary_output") == "events"
        and _positive_int(writeback.get("model_passes_per_note")) == 1
        and writeback.get("settles_multi_write") is True
        and required_fields.issubset(patched_fields)
    )


def _embed_budget_failures(report: dict[str, Any]) -> list[str]:
    failures = _cost_gate_failures(report, label="budget")
    if report.get("error"):
        failures.append(f"error={report.get('error')!r}")
    sample = report.get("sample") or {}
    if not _has_positive_sample(report):
        failures.append("sample.notes must be positive")
    if _positive_int(sample.get("vector_dim")) != 768:
        failures.append(f"sample.vector_dim={sample.get('vector_dim')!r}, expected 768")
    expected_model = "Snowflake/snowflake-arctic-embed-m-v1.5"
    if sample.get("model") != expected_model:
        failures.append(f"sample.model={sample.get('model')!r}, expected {expected_model!r}")
    path = report.get("production_path") or {}
    if path.get("pipeline_cr") != "chart-embed-gpu":
        failures.append(f"production_path.pipeline_cr={path.get('pipeline_cr')!r}, expected 'chart-embed-gpu'")
    if path.get("module") != "indexer.embed":
        failures.append(f"production_path.module={path.get('module')!r}, expected 'indexer.embed'")
    if path.get("compute_class") != "gpu":
        failures.append(f"production_path.compute_class={path.get('compute_class')!r}, expected 'gpu'")
    if path.get("image") != "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-embedder-plan-20260624-dedupe2":
        failures.append(
            f"production_path.image={path.get('image')!r}, expected '186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-embedder-plan-20260624-dedupe2'"
        )
    if path.get("allow_full_cpu_index") is not False:
        failures.append("production_path.allow_full_cpu_index must be false")
    return failures


def _classify_budget_failures(report: dict[str, Any]) -> list[str]:
    failures = _cost_gate_failures(report, label="budget")
    if report.get("error"):
        failures.append(f"error={report.get('error')!r}")
    failures.extend(
        _accepted_check_failures(
            report.get("signal") or {},
            ("min_med_discontinuations", "min_review_examples"),
            label="signal",
        )
    )
    sample = report.get("sample") or {}
    if not _has_positive_sample(report):
        failures.append("sample.notes must be positive")
    if _positive_int(sample.get("med_discontinuation")) <= 0:
        failures.append("sample.med_discontinuation must be positive")
    if not isinstance(report.get("examples") or [], list) or not report.get("examples"):
        failures.append("examples must include at least one review example")
    elif not any(
        isinstance(example, dict)
        and (example.get("labels") or {}).get("has_med_discontinuation") is True
        and bool(example.get("discontinuation_reason"))
        for example in report.get("examples") or []
    ):
        failures.append("examples must include a medication discontinuation review example with discontinuation_reason")
    writeback = report.get("writeback") or {}
    patched_fields = set(writeback.get("patched_fields") or [])
    required_fields = {
        "events",
        "has_med_discontinuation",
        "has_adverse_event",
        "diagnosis_category",
        "specialty",
        "discontinuation_reason",
    }
    if writeback.get("mode") != "tpuf.patch_columns":
        failures.append(f"writeback.mode={writeback.get('mode')!r}, expected 'tpuf.patch_columns'")
    if writeback.get("primary_output") != "events":
        failures.append(f"writeback.primary_output={writeback.get('primary_output')!r}, expected 'events'")
    if _positive_int(writeback.get("model_passes_per_note")) != 1:
        failures.append(
            f"writeback.model_passes_per_note={writeback.get('model_passes_per_note')!r}, expected 1"
        )
    if writeback.get("settles_multi_write") is not True:
        failures.append("writeback.settles_multi_write must be true")
    missing_fields = sorted(required_fields - patched_fields)
    if missing_fields:
        failures.append(f"writeback.patched_fields missing: {', '.join(missing_fields)}")
    return failures


def _positive_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _positive_number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _live_smoke_complete(report: dict[str, Any], *, require_event_facets: bool) -> bool:
    index_shape = report.get("index_shape") or {}
    routes = report.get("routes") or []
    similar = report.get("similar") or {}
    facets = report.get("facets") or {}
    requirements = report.get("requirements") or {}
    route_names = {route.get("route") for route in routes if isinstance(route, dict)}
    route_echoes_ok = all(
        isinstance(route, dict)
        and isinstance(route.get("routing"), dict)
        and (route["routing"].get("route") or route["routing"].get("strategy")) == route.get("route")
        and (route.get("route") != "fused" or bool(route.get("hybrid")))
        for route in routes
        if isinstance(route, dict)
    )
    required_facets = ("age_band", "gender", "events") if require_event_facets else ("age_band", "gender")
    facets_ok = all(
        _positive_int((facets.get(field) or {}).get("values")) > 0 and (facets.get(field) or {}).get("sha")
        for field in required_facets
    )
    return (
        report.get("ok") is True
        and _completed_status(report)
        and _positive_int(index_shape.get("vector_dim")) == 768
        and _positive_int(index_shape.get("rows")) > 0
        and requirements.get("event_facets") is require_event_facets
        and {"hybrid_text", "fused", "semantic"}.issubset(route_names)
        and route_echoes_ok
        and all(_positive_int(route.get("rows")) > 0 for route in routes if isinstance(route, dict))
        and _positive_int(similar.get("neighbors")) > 0
        and facets_ok
    )


def _live_smoke_failures(report: dict[str, Any], *, require_event_facets: bool) -> list[str]:
    failures = []
    index_shape = report.get("index_shape") or {}
    routes = report.get("routes") or []
    similar = report.get("similar") or {}
    facets = report.get("facets") or {}
    requirements = report.get("requirements") or {}
    route_names = {route.get("route") for route in routes if isinstance(route, dict)}
    if report.get("error"):
        failures.append(f"error={report.get('error')!r}")
    if report.get("ok") is not True:
        failures.append(f"ok={report.get('ok')!r}, expected true")
    if not _completed_status(report):
        failures.append(f"status={report.get('status')!r}, expected 'completed'")
    if _positive_int(index_shape.get("vector_dim")) != 768:
        failures.append(f"index_shape.vector_dim={index_shape.get('vector_dim')!r}, expected 768")
    if _positive_int(index_shape.get("rows")) <= 0:
        failures.append("index_shape.rows must be positive")
    if requirements.get("event_facets") is not require_event_facets:
        failures.append(f"requirements.event_facets must be {str(require_event_facets).lower()}")
    missing_routes = sorted({"hybrid_text", "fused", "semantic"} - route_names)
    if missing_routes:
        failures.append(f"missing route checks: {', '.join(missing_routes)}")
    missing_echoes = sorted(
        str(route.get("route") or route.get("strategy"))
        for route in routes
        if isinstance(route, dict)
        and (
            not isinstance(route.get("routing"), dict)
            or (route["routing"].get("route") or route["routing"].get("strategy")) != route.get("route")
            or (route.get("route") == "fused" and not route.get("hybrid"))
        )
    )
    if missing_echoes:
        failures.append(f"route checks missing gateway routing/fused-hybrid echo: {', '.join(missing_echoes)}")
    empty_routes = sorted(
        str(route.get("route") or route.get("strategy"))
        for route in routes
        if isinstance(route, dict) and _positive_int(route.get("rows")) <= 0
    )
    if empty_routes:
        failures.append(f"route checks returned no rows: {', '.join(empty_routes)}")
    if _positive_int(similar.get("neighbors")) <= 0:
        failures.append("similar.neighbors must be positive")
    required_facets = ("age_band", "gender", "events") if require_event_facets else ("age_band", "gender")
    for field in required_facets:
        facet = facets.get(field) or {}
        if _positive_int(facet.get("values")) <= 0:
            failures.append(f"facet {field!r} has no values")
        if not facet.get("sha"):
            failures.append(f"facet {field!r} missing sha")
    return failures


def _facet_refresh_complete(report: dict[str, Any]) -> bool:
    fields = set(report.get("fields") or [])
    snapshots = report.get("snapshots") or {}
    required = {"age_band", "gender"}
    return (
        _completed_status(report)
        and required.issubset(fields)
        and all(
            _positive_int((snapshots.get(field) or {}).get("values")) > 0
            and bool((snapshots.get(field) or {}).get("sha"))
            for field in required
        )
    )


def _facet_refresh_failures(report: dict[str, Any]) -> list[str]:
    failures = []
    fields = set(report.get("fields") or [])
    snapshots = report.get("snapshots") or {}
    if report.get("error"):
        failures.append(f"error={report.get('error')!r}")
    if not _completed_status(report):
        failures.append(f"status={report.get('status')!r}, expected 'completed'")
    required = {"age_band", "gender"}
    missing = sorted(required - fields)
    if missing:
        failures.append(f"missing refreshed fields: {', '.join(missing)}")
    for field in sorted(required & fields):
        snapshot = snapshots.get(field) or {}
        if _positive_int(snapshot.get("values")) <= 0:
            failures.append(f"snapshot {field!r} has no values")
        if not snapshot.get("sha"):
            failures.append(f"snapshot {field!r} missing sha")
    return failures


def _event_facet_refresh_complete(report: dict[str, Any]) -> bool:
    fields = set(report.get("fields") or [])
    snapshots = report.get("snapshots") or {}
    event = snapshots.get("events") or {}
    return (
        _completed_status(report)
        and "events" in fields
        and _positive_int(event.get("values")) > 0
        and bool(event.get("sha"))
    )


def _event_facet_refresh_failures(report: dict[str, Any]) -> list[str]:
    failures = []
    fields = set(report.get("fields") or [])
    snapshots = report.get("snapshots") or {}
    event = snapshots.get("events") or {}
    if report.get("error"):
        failures.append(f"error={report.get('error')!r}")
    if not _completed_status(report):
        failures.append(f"status={report.get('status')!r}, expected 'completed'")
    if "events" not in fields:
        failures.append("missing refreshed fields: events")
    if _positive_int(event.get("values")) <= 0:
        failures.append("snapshot 'events' has no values")
    if not event.get("sha"):
        failures.append("snapshot 'events' missing sha")
    return failures


def _published_baseline_complete(report: dict[str, Any]) -> bool:
    by_strategy = {
        summary.get("strategy"): summary for summary in report.get("summaries") or [] if isinstance(summary, dict)
    }
    fused = by_strategy.get("fused") or {}
    published = fused.get("published") or {}
    baseline_metrics = published.get("baseline_metrics") or {}
    delta = published.get("delta") or {}
    return (
        published.get("baseline") == "rrf"
        and published.get("meets_or_beats") is True
        and all(_positive_number(baseline_metrics.get(metric)) > 0 for metric in ("RR@10", "nDCG@10", "R@1000"))
        and all(metric in delta for metric in ("RR@10", "nDCG@10", "R@1000"))
    )


def _published_baseline_failures(report: dict[str, Any]) -> list[str]:
    failures = []
    by_strategy = {
        summary.get("strategy"): summary for summary in report.get("summaries") or [] if isinstance(summary, dict)
    }
    fused = by_strategy.get("fused") or {}
    published = fused.get("published") or {}
    baseline_metrics = published.get("baseline_metrics") or {}
    delta = published.get("delta") or {}
    if not published:
        return ["fused summary missing published baseline comparison"]
    if published.get("baseline") != "rrf":
        failures.append(f"fused.published.baseline={published.get('baseline')!r}, expected 'rrf'")
    if published.get("meets_or_beats") is not True:
        failures.append("fused.published.meets_or_beats must be true")
    for metric in ("RR@10", "nDCG@10", "R@1000"):
        if _positive_number(baseline_metrics.get(metric)) <= 0:
            failures.append(f"fused.published.baseline_metrics.{metric} must be positive")
        if metric not in delta:
            failures.append(f"fused.published.delta.{metric} missing")
    return failures


def _fused_dominance_checks_complete(fused_dominates: dict[str, Any]) -> bool:
    checks = fused_dominates.get("checks") or []
    expected = {(metric, baseline) for metric in ("RR@10", "nDCG@10", "R@1000") for baseline in ("bm25", "semantic")}
    seen = set()
    for check in checks:
        if not isinstance(check, dict):
            return False
        key = (check.get("metric"), check.get("baseline"))
        if key in expected:
            seen.add(key)
            if (
                check.get("ok") is not True
                or check.get("fused") is None
                or check.get("baseline_value") is None
                or _positive_number(check.get("fused")) < _positive_number(check.get("baseline_value"))
            ):
                return False
    return seen == expected


def _fused_dominance_check_failures(fused_dominates: dict[str, Any]) -> list[str]:
    failures = []
    checks = fused_dominates.get("checks") or []
    expected = {(metric, baseline) for metric in ("RR@10", "nDCG@10", "R@1000") for baseline in ("bm25", "semantic")}
    seen = set()
    malformed = []
    failing = []
    for check in checks:
        if not isinstance(check, dict):
            malformed.append(repr(check))
            continue
        key = (check.get("metric"), check.get("baseline"))
        if key not in expected:
            continue
        seen.add(key)
        if check.get("ok") is not True:
            failing.append(f"{key[0]} vs {key[1]} ok={check.get('ok')!r}")
        elif check.get("fused") is None or check.get("baseline_value") is None:
            failing.append(f"{key[0]} vs {key[1]} missing fused/baseline_value")
        elif _positive_number(check.get("fused")) < _positive_number(check.get("baseline_value")):
            failing.append(
                f"{key[0]} vs {key[1]} fused={check.get('fused')!r} < baseline_value={check.get('baseline_value')!r}"
            )
    missing = sorted(expected - seen)
    if missing:
        failures.append(
            "gates.fused_dominates.checks missing headline comparisons: "
            + ", ".join(f"{metric} vs {baseline}" for metric, baseline in missing)
        )
    if malformed:
        failures.append("gates.fused_dominates.checks contains malformed checks")
    if failing:
        failures.append("gates.fused_dominates.checks failing headline comparisons: " + "; ".join(failing))
    return failures


def _eval_report_complete(report: dict[str, Any], *, require_published: bool = False) -> bool:
    required_strategies = {"auto", "semantic", "bm25", "fused"}
    provenance = report.get("provenance") or {}
    if report.get("task") != "ppr" or report.get("split") != "dev" or report.get("beir_dir") is not None:
        return False
    if (
        provenance.get("recds_repo") != "zhengyun21/PMC-Patients-ReCDS"
        or provenance.get("recds_revision") != "a27717bb27679cf0860305997685547ca01b3dd1"
        or provenance.get("embed_model") != "Snowflake/snowflake-arctic-embed-m-v1.5"
        or _positive_int(provenance.get("embed_dim")) != 768
    ):
        return False
    if not required_strategies.issubset(set(report.get("strategies") or [])):
        return False
    if _positive_int(report.get("limit")) < 500 or _positive_int(report.get("top_k")) < 1000:
        return False
    summaries = report.get("summaries") or []
    if not isinstance(summaries, list) or not summaries:
        return False
    by_strategy = {summary.get("strategy"): summary for summary in summaries if isinstance(summary, dict)}
    if not required_strategies.issubset(set(by_strategy)):
        return False
    if any(_positive_int((summary.get("queries") or {}).get("failed")) for summary in by_strategy.values()):
        return False
    gates = report.get("gates") or {}
    no_failures = gates.get("no_failures") or {}
    fused_dominates = gates.get("fused_dominates") or {}
    complete = (
        no_failures.get("required") is True
        and no_failures.get("accepted") is True
        and fused_dominates.get("accepted") is True
        and _fused_dominance_checks_complete(fused_dominates)
        and not fused_dominates.get("query_failures")
    )
    if require_published:
        complete = complete and _published_baseline_complete(report)
    return complete


def _eval_report_failures(report: dict[str, Any], *, require_published: bool = False) -> list[str]:
    failures = []
    required_strategies = {"auto", "semantic", "bm25", "fused"}
    provenance = report.get("provenance") or {}
    if report.get("error"):
        failures.append(f"error={report.get('error')!r}")
    if report.get("task") != "ppr":
        failures.append(f"task={report.get('task')!r}, expected 'ppr'")
    if report.get("split") != "dev":
        failures.append(f"split={report.get('split')!r}, expected 'dev'")
    if report.get("beir_dir") is not None:
        failures.append(f"beir_dir={report.get('beir_dir')!r}, expected null")
    expected_provenance = {
        "recds_repo": "zhengyun21/PMC-Patients-ReCDS",
        "recds_revision": "a27717bb27679cf0860305997685547ca01b3dd1",
        "embed_model": "Snowflake/snowflake-arctic-embed-m-v1.5",
    }
    for name, expected in expected_provenance.items():
        if provenance.get(name) != expected:
            failures.append(f"provenance.{name}={provenance.get(name)!r}, expected {expected!r}")
    if _positive_int(provenance.get("embed_dim")) != 768:
        failures.append(f"provenance.embed_dim={provenance.get('embed_dim')!r}, expected 768")
    missing_strategies = sorted(required_strategies - set(report.get("strategies") or []))
    if missing_strategies:
        failures.append(f"missing strategies: {', '.join(missing_strategies)}")
    if _positive_int(report.get("limit")) < 500:
        failures.append(f"limit={report.get('limit')!r}, expected at least 500")
    if _positive_int(report.get("top_k")) < 1000:
        failures.append(f"top_k={report.get('top_k')!r}, expected at least 1000")
    summaries = report.get("summaries") or []
    if not isinstance(summaries, list) or not summaries:
        failures.append("summaries missing")
        by_strategy = {}
    else:
        by_strategy = {summary.get("strategy"): summary for summary in summaries if isinstance(summary, dict)}
    missing_summary = sorted(required_strategies - set(by_strategy))
    if missing_summary:
        failures.append(f"missing summaries: {', '.join(missing_summary)}")
    failed = sorted(
        str(strategy)
        for strategy, summary in by_strategy.items()
        if _positive_int((summary.get("queries") or {}).get("failed"))
    )
    if failed:
        failures.append(f"query failures present for strategies: {', '.join(failed)}")
    gates = report.get("gates") or {}
    no_failures = gates.get("no_failures") or {}
    fused_dominates = gates.get("fused_dominates") or {}
    if no_failures.get("required") is not True:
        failures.append("gates.no_failures.required must be true")
    if no_failures.get("accepted") is not True:
        failures.append("gates.no_failures.accepted must be true")
    if fused_dominates.get("accepted") is not True:
        failures.append("gates.fused_dominates.accepted must be true")
    checks = fused_dominates.get("checks") or []
    if not checks:
        failures.append("gates.fused_dominates.checks must be non-empty")
    else:
        failures.extend(_fused_dominance_check_failures(fused_dominates))
    if fused_dominates.get("query_failures"):
        failures.append("gates.fused_dominates.query_failures must be empty")
    if require_published:
        failures.extend(_published_baseline_failures(report))
    return failures


def _bimodal_report_complete(report: dict[str, Any]) -> bool:
    required_strategies = {"auto", "semantic", "bm25", "fused"}
    if report.get("task") is not None or report.get("split") is not None or not report.get("beir_dir"):
        return False
    if not required_strategies.issubset(set(report.get("strategies") or [])):
        return False
    if _positive_int(report.get("limit")) <= 0 or _positive_int(report.get("top_k")) <= 0:
        return False
    summaries = report.get("summaries") or []
    if not isinstance(summaries, list) or not summaries:
        return False
    by_strategy = {summary.get("strategy"): summary for summary in summaries if isinstance(summary, dict)}
    if not required_strategies.issubset(set(by_strategy)):
        return False
    if any(_positive_int((summary.get("queries") or {}).get("failed")) for summary in by_strategy.values()):
        return False
    if any((summary.get("published") for summary in by_strategy.values())):
        return False
    for summary in by_strategy.values():
        dataset = summary.get("dataset") or {}
        metrics_by_kind = summary.get("metrics_by_kind") or {}
        if _positive_int(dataset.get("short")) <= 0 or _positive_int(dataset.get("long")) <= 0:
            return False
        if (
            dataset.get("dataset_repo") != "zhengyun21/PMC-Patients"
            or dataset.get("dataset_revision") != "28d8836518f86d4f1e6358ea8ec09977023e5766"
            or dataset.get("dataset_split") != "train"
            or dataset.get("recds_repo") != "zhengyun21/PMC-Patients-ReCDS"
            or dataset.get("recds_revision") != "a27717bb27679cf0860305997685547ca01b3dd1"
        ):
            return False
        if "short" not in metrics_by_kind or "long" not in metrics_by_kind:
            return False
    gates = report.get("gates") or {}
    no_failures = gates.get("no_failures") or {}
    fused_dominates = gates.get("fused_dominates") or {}
    return (
        no_failures.get("required") is True
        and no_failures.get("accepted") is True
        and fused_dominates.get("accepted") is True
        and _fused_dominance_checks_complete(fused_dominates)
        and not fused_dominates.get("query_failures")
    )


def _bimodal_report_failures(report: dict[str, Any]) -> list[str]:
    failures = []
    required_strategies = {"auto", "semantic", "bm25", "fused"}
    if report.get("error"):
        failures.append(f"error={report.get('error')!r}")
    if report.get("task") is not None:
        failures.append(f"task={report.get('task')!r}, expected null for bimodal BEIR eval")
    if report.get("split") is not None:
        failures.append(f"split={report.get('split')!r}, expected null for bimodal BEIR eval")
    if not report.get("beir_dir"):
        failures.append("beir_dir must point at the bimodal BEIR directory")
    missing_strategies = sorted(required_strategies - set(report.get("strategies") or []))
    if missing_strategies:
        failures.append(f"missing strategies: {', '.join(missing_strategies)}")
    if _positive_int(report.get("limit")) <= 0:
        failures.append("limit must be positive")
    if _positive_int(report.get("top_k")) <= 0:
        failures.append("top_k must be positive")
    summaries = report.get("summaries") or []
    if not isinstance(summaries, list) or not summaries:
        failures.append("summaries missing")
        by_strategy = {}
    else:
        by_strategy = {summary.get("strategy"): summary for summary in summaries if isinstance(summary, dict)}
    missing_summary = sorted(required_strategies - set(by_strategy))
    if missing_summary:
        failures.append(f"missing summaries: {', '.join(missing_summary)}")
    failed = sorted(
        str(strategy)
        for strategy, summary in by_strategy.items()
        if _positive_int((summary.get("queries") or {}).get("failed"))
    )
    if failed:
        failures.append(f"query failures present for strategies: {', '.join(failed)}")
    for strategy, summary in sorted(by_strategy.items()):
        if summary.get("published"):
            failures.append(f"{strategy} summary must not include published baseline for bimodal BEIR eval")
        dataset = summary.get("dataset") or {}
        metrics_by_kind = summary.get("metrics_by_kind") or {}
        if _positive_int(dataset.get("short")) <= 0 or _positive_int(dataset.get("long")) <= 0:
            failures.append(f"{strategy}.dataset must include positive short and long counts")
        expected_dataset = {
            "dataset_repo": "zhengyun21/PMC-Patients",
            "dataset_revision": "28d8836518f86d4f1e6358ea8ec09977023e5766",
            "dataset_split": "train",
            "recds_repo": "zhengyun21/PMC-Patients-ReCDS",
            "recds_revision": "a27717bb27679cf0860305997685547ca01b3dd1",
        }
        for name, expected in expected_dataset.items():
            if dataset.get(name) != expected:
                failures.append(f"{strategy}.dataset.{name}={dataset.get(name)!r}, expected {expected!r}")
        missing_kinds = sorted({"short", "long"} - set(metrics_by_kind))
        if missing_kinds:
            failures.append(f"{strategy}.metrics_by_kind missing: {', '.join(missing_kinds)}")
    gates = report.get("gates") or {}
    no_failures = gates.get("no_failures") or {}
    fused_dominates = gates.get("fused_dominates") or {}
    if no_failures.get("required") is not True:
        failures.append("gates.no_failures.required must be true")
    if no_failures.get("accepted") is not True:
        failures.append("gates.no_failures.accepted must be true")
    if fused_dominates.get("accepted") is not True:
        failures.append("gates.fused_dominates.accepted must be true")
    checks = fused_dominates.get("checks") or []
    if not checks:
        failures.append("gates.fused_dominates.checks must be non-empty")
    else:
        failures.extend(_fused_dominance_check_failures(fused_dominates))
    if fused_dominates.get("query_failures"):
        failures.append("gates.fused_dominates.query_failures must be empty")
    return failures


def _holdout_complete(report: dict[str, Any]) -> bool:
    gate = report.get("gate") or {}
    return bool(gate) and _accepted_checks(gate, ("feature_edges_present", "qrel_edges_present", "max_overlap_edges"))


def _holdout_failures(report: dict[str, Any]) -> list[str]:
    gate = report.get("gate") or {}
    if not gate:
        return ["gate missing"]
    return _accepted_check_failures(
        gate,
        ("feature_edges_present", "qrel_edges_present", "max_overlap_edges"),
        label="gate",
    )


def _completed_status(report: dict[str, Any]) -> bool:
    return report.get("status") == "completed"


def _live_slice_index_complete(report: dict[str, Any]) -> bool:
    indexed = _positive_int(report.get("indexed"))
    limit = _positive_int(report.get("limit"))
    schema = report.get("schema") or {}
    provenance = report.get("provenance") or {}
    return (
        _completed_status(report)
        and report.get("dry_run") is False
        and indexed >= 2000
        and (limit == 0 or limit == indexed)
        and provenance.get("dataset_repo") == "zhengyun21/PMC-Patients"
        and provenance.get("dataset_revision") == "28d8836518f86d4f1e6358ea8ec09977023e5766"
        and provenance.get("dataset_split") == "train"
        and provenance.get("embed_model") == "Snowflake/snowflake-arctic-embed-m-v1.5"
        and _positive_int(provenance.get("embed_dim")) == 768
        and _positive_int(schema.get("vector_dim")) == 768
        and _positive_int(schema.get("rows_with_age_band")) > 0
        and _positive_int(schema.get("rows_with_gender")) > 0
        and _positive_int(schema.get("rows_with_similar_patient_ids")) > 0
        and report.get("facet_snapshots_materialized") is True
    )


def _live_slice_index_failures(report: dict[str, Any]) -> list[str]:
    failures = []
    indexed = _positive_int(report.get("indexed"))
    limit = _positive_int(report.get("limit"))
    schema = report.get("schema") or {}
    provenance = report.get("provenance") or {}
    if report.get("error"):
        failures.append(f"error={report.get('error')!r}")
    if not _completed_status(report):
        failures.append(f"status={report.get('status')!r}, expected 'completed'")
    if report.get("dry_run") is not False:
        failures.append(f"dry_run={report.get('dry_run')!r}, expected false")
    if indexed < 2000:
        failures.append(f"indexed={indexed}, expected at least 2000")
    if limit and limit != indexed:
        failures.append(f"limit={limit} does not match indexed={indexed}")
    expected_provenance = {
        "dataset_repo": "zhengyun21/PMC-Patients",
        "dataset_revision": "28d8836518f86d4f1e6358ea8ec09977023e5766",
        "dataset_split": "train",
        "embed_model": "Snowflake/snowflake-arctic-embed-m-v1.5",
    }
    for name, expected in expected_provenance.items():
        if provenance.get(name) != expected:
            failures.append(f"provenance.{name}={provenance.get(name)!r}, expected {expected!r}")
    if _positive_int(provenance.get("embed_dim")) != 768:
        failures.append(f"provenance.embed_dim={provenance.get('embed_dim')!r}, expected 768")
    if _positive_int(schema.get("vector_dim")) != 768:
        failures.append(f"schema.vector_dim={schema.get('vector_dim')!r}, expected 768")
    for name in ("rows_with_age_band", "rows_with_gender", "rows_with_similar_patient_ids"):
        if _positive_int(schema.get(name)) <= 0:
            failures.append(f"schema.{name} must be positive")
    if report.get("facet_snapshots_materialized") is not True:
        failures.append("facet_snapshots_materialized must be true")
    return failures


def _gpu_build_complete(report: dict[str, Any]) -> bool:
    embed_image = str(report.get("embed_image") or "")
    source_image = str(report.get("source_image") or "")
    classifier_image = str(report.get("classifier_image") or "")
    embed_command = report.get("embed_command") or []
    source_command = report.get("source_command") or []
    classifier_command = report.get("classifier_command") or []
    targets = set(report.get("targets") or ["embed", "classifier"])
    return (
        report.get("status") == "completed"
        and not report.get("error")
        and report.get("mode") == "push"
        and bool(embed_image)
        and ("source" not in targets or bool(source_image))
        and bool(classifier_image)
        and bool(targets)
        and targets <= {"embed", "source", "classifier"}
        and not embed_image.endswith(":latest")
        and ("source" not in targets or not source_image.endswith(":latest"))
        and not classifier_image.endswith(":latest")
        and ("embed" not in targets or (embed_image in embed_command and "--push" in embed_command))
        and ("source" not in targets or (source_image in source_command and "--push" in source_command))
        and (
            "classifier" not in targets
            or (classifier_image in classifier_command and "--push" in classifier_command)
        )
    )


def _gpu_build_failures(report: dict[str, Any]) -> list[str]:
    failures = []
    embed_image = str(report.get("embed_image") or "")
    source_image = str(report.get("source_image") or "")
    classifier_image = str(report.get("classifier_image") or "")
    embed_command = report.get("embed_command") or []
    source_command = report.get("source_command") or []
    classifier_command = report.get("classifier_command") or []
    targets = set(report.get("targets") or ["embed", "classifier"])
    if report.get("status") != "completed":
        failures.append(f"status={report.get('status')!r}, expected 'completed'")
    if report.get("error"):
        failures.append(f"error={report.get('error')!r}")
    if report.get("mode") != "push":
        failures.append(f"mode={report.get('mode')!r}, expected 'push'")
    if not embed_image:
        failures.append("embed_image missing")
    if "source" in targets and not source_image:
        failures.append("source_image missing")
    if not classifier_image:
        failures.append("classifier_image missing")
    if not targets:
        failures.append("targets missing")
    if not targets <= {"embed", "source", "classifier"}:
        failures.append(f"unsupported targets: {', '.join(sorted(targets - {'embed', 'source', 'classifier'}))}")
    if embed_image.endswith(":latest"):
        failures.append("embed_image must not use :latest")
    if "source" in targets and source_image.endswith(":latest"):
        failures.append("source_image must not use :latest")
    if classifier_image.endswith(":latest"):
        failures.append("classifier_image must not use :latest")
    if "embed" in targets and embed_image not in embed_command:
        failures.append("embed_command does not include embed_image")
    if "source" in targets and source_image not in source_command:
        failures.append("source_command does not include source_image")
    if "classifier" in targets and classifier_image not in classifier_command:
        failures.append("classifier_command does not include classifier_image")
    if "embed" in targets and "--push" not in embed_command:
        failures.append("embed_command missing --push")
    if "source" in targets and "--push" not in source_command:
        failures.append("source_command missing --push")
    if "classifier" in targets and "--push" not in classifier_command:
        failures.append("classifier_command missing --push")
    return failures


def _deploy_apply_complete(report: dict[str, Any], *, expected_classifier_report: str | None = None) -> bool:
    manifests = report.get("manifests") or []
    runtime_manifests = report.get("runtime_manifests") or []
    expected_manifests = [
        "deploy/namespace.yaml",
        "deploy/vectorstore.yaml",
        "deploy/warehouse.yaml",
        "deploy/pipeline.yaml",
        "deploy/pipeline-embed.yaml",
        "deploy/index.yaml",
        "deploy/functions-events.yaml",
    ]
    expected_runtime = [
        "deploy/vectorstore.yaml",
        "deploy/warehouse.yaml",
        "deploy/pipeline.yaml",
        "deploy/pipeline-embed.yaml",
        "deploy/index.yaml",
    ]
    return (
        report.get("status") == "completed"
        and not report.get("error")
        and report.get("mode") == "apply"
        and bool(report.get("namespace"))
        and bool(report.get("kube_context"))
        and report.get("kube_context_confirmed") is True
        and report.get("classifier_cost_accepted") is True
        and report.get("classifier") == "applied"
        and bool(report.get("classifier_report"))
        and (expected_classifier_report is None or report.get("classifier_report") == expected_classifier_report)
        and manifests == expected_manifests
        and runtime_manifests == expected_runtime
    )


def _deploy_apply_failures(report: dict[str, Any], *, expected_classifier_report: str | None = None) -> list[str]:
    failures = []
    manifests = report.get("manifests") or []
    runtime_manifests = report.get("runtime_manifests") or []
    expected_manifests = [
        "deploy/namespace.yaml",
        "deploy/vectorstore.yaml",
        "deploy/warehouse.yaml",
        "deploy/pipeline.yaml",
        "deploy/pipeline-embed.yaml",
        "deploy/index.yaml",
        "deploy/functions-events.yaml",
    ]
    expected_runtime = [
        "deploy/vectorstore.yaml",
        "deploy/warehouse.yaml",
        "deploy/pipeline.yaml",
        "deploy/pipeline-embed.yaml",
        "deploy/index.yaml",
    ]
    if report.get("status") != "completed":
        failures.append(f"status={report.get('status')!r}, expected 'completed'")
    if report.get("error"):
        failures.append(f"error={report.get('error')!r}")
    if report.get("mode") != "apply":
        failures.append(f"mode={report.get('mode')!r}, expected 'apply'")
    if not report.get("namespace"):
        failures.append("namespace missing")
    if not report.get("kube_context"):
        failures.append("kube_context missing")
    if report.get("kube_context_confirmed") is not True:
        failures.append("kube_context_confirmed must be true")
    if report.get("classifier_cost_accepted") is not True:
        failures.append("classifier_cost_accepted must be true")
    if report.get("classifier") != "applied":
        failures.append(f"classifier={report.get('classifier')!r}, expected 'applied'")
    if not report.get("classifier_report"):
        failures.append("classifier_report missing")
    elif expected_classifier_report is not None and report.get("classifier_report") != expected_classifier_report:
        failures.append(
            f"classifier_report={report.get('classifier_report')!r}, expected {expected_classifier_report!r}"
        )
    if manifests != expected_manifests:
        failures.append("manifests do not match full ordered apply set")
    if runtime_manifests != expected_runtime:
        failures.append("runtime_manifests do not match expected runtime set")
    return failures


def _phase6_unpause_complete(report: dict[str, Any], *, expected_budget_report: str | None = None) -> bool:
    return (
        report.get("status") == "unpaused"
        and not report.get("error")
        and report.get("mode") == "unpause"
        and bool(report.get("namespace"))
        and bool(report.get("pipeline_cr"))
        and bool(report.get("expected_pipeline_id"))
        and report.get("actual_pipeline_id") == report.get("expected_pipeline_id")
        and bool(report.get("budget_report"))
        and (expected_budget_report is None or report.get("budget_report") == expected_budget_report)
    )


def _phase6_unpause_failures(report: dict[str, Any], *, expected_budget_report: str | None = None) -> list[str]:
    failures = []
    if report.get("status") != "unpaused":
        failures.append(f"status={report.get('status')!r}, expected 'unpaused'")
    if report.get("error"):
        failures.append(f"error={report.get('error')!r}")
    if report.get("mode") != "unpause":
        failures.append(f"mode={report.get('mode')!r}, expected 'unpause'")
    if not report.get("namespace"):
        failures.append("namespace missing")
    if not report.get("pipeline_cr"):
        failures.append("pipeline_cr missing")
    if not report.get("expected_pipeline_id"):
        failures.append("expected_pipeline_id missing")
    if report.get("actual_pipeline_id") != report.get("expected_pipeline_id"):
        failures.append(
            f"actual_pipeline_id={report.get('actual_pipeline_id')!r}, expected {report.get('expected_pipeline_id')!r}"
        )
    if not report.get("budget_report"):
        failures.append("budget_report missing")
    elif expected_budget_report is not None and report.get("budget_report") != expected_budget_report:
        failures.append(f"budget_report={report.get('budget_report')!r}, expected {expected_budget_report!r}")
    return failures


def _cost_baselines_complete(
    report: dict[str, Any],
    *,
    expected_embed_report: str | None = None,
    expected_classifier_report: str | None = None,
) -> bool:
    baselines = report.get("cost_baselines") or {}
    embed = baselines.get("embed") or {}
    classifier = baselines.get("classifier") or {}
    return (
        embed.get("accepted") is True
        and classifier.get("accepted") is True
        and (expected_embed_report is None or embed.get("report") == expected_embed_report)
        and (expected_classifier_report is None or classifier.get("report") == expected_classifier_report)
        and (
            _cost_estimate_complete({"estimate": embed.get("estimate") or {}})
            or _layer_cost_evidence_complete(embed)
        )
        and (
            _cost_estimate_complete({"estimate": classifier.get("estimate") or {}})
            or _layer_cost_evidence_complete(classifier)
        )
    )


def _runtime_status_complete(
    report: dict[str, Any],
    *,
    expected_embed_report: str | None = None,
    expected_classifier_report: str | None = None,
) -> bool:
    targets = report.get("targets") or {}
    pipeline = report.get("pipeline") or {}
    udf = report.get("udf") or {}
    facets = report.get("facets") or {}
    required_facets = {"specialty", "age_band", "diagnosis_category", "gender", "events"}
    target_rows = _positive_int(targets.get("full_corpus_notes"))
    facet_snapshots_complete = all(
        _positive_int((facets.get(field) or {}).get("values")) > 0
        and _positive_int((facets.get(field) or {}).get("row_count")) >= target_rows
        and bool((facets.get(field) or {}).get("sha"))
        for field in required_facets
    )
    return (
        bool(report.get("namespace"))
        and not report.get("error")
        and report.get("namespace") == targets.get("namespace")
        and bool(targets.get("pipeline_id"))
        and bool(targets.get("udf_id"))
        and targets.get("embed_pipeline_cr") == "chart-embed-gpu"
        and targets.get("embed_compute_class") == "gpu"
        and targets.get("classifier_compute_class") == "gpu"
        and targets.get("embed_image") == "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-embedder-plan-20260624-dedupe2"
        and targets.get("classifier_image") == "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-classifier-plan-20260624"
        and target_rows >= 167000
        and pipeline.get("pipeline_id") == targets.get("pipeline_id")
        and udf.get("udf_id") == targets.get("udf_id")
        and required_facets.issubset(set(facets))
        and facet_snapshots_complete
        and not any((facets.get(field) or {}).get("error") for field in required_facets)
        and "error" not in pipeline
        and "error" not in udf
        and _cost_baselines_complete(
            report,
            expected_embed_report=expected_embed_report,
            expected_classifier_report=expected_classifier_report,
        )
    )


def _cost_baseline_failures(
    report: dict[str, Any],
    *,
    expected_embed_report: str | None = None,
    expected_classifier_report: str | None = None,
) -> list[str]:
    failures = []
    baselines = report.get("cost_baselines") or {}
    expected = {
        "embed": expected_embed_report,
        "classifier": expected_classifier_report,
    }
    for name in ("embed", "classifier"):
        baseline = baselines.get(name) or {}
        if baseline.get("accepted") is not True:
            failures.append(f"cost_baselines.{name}.accepted must be true")
        if baseline.get("error"):
            failures.append(f"cost_baselines.{name}.error={baseline.get('error')!r}")
        if expected[name] is not None and baseline.get("report") != expected[name]:
            failures.append(f"cost_baselines.{name}.report={baseline.get('report')!r}, expected {expected[name]!r}")
        if not (
            _cost_estimate_complete({"estimate": baseline.get("estimate") or {}})
            or _layer_cost_evidence_complete(baseline)
        ):
            estimate_failures = _cost_estimate_failures({"estimate": baseline.get("estimate") or {}})
            layer_failures = _cost_gate_failures(baseline)
            if layer_failures and not estimate_failures:
                failures.extend(f"cost_baselines.{name}.{failure}" for failure in layer_failures)
            else:
                failures.extend(f"cost_baselines.{name}.{failure}" for failure in estimate_failures)
    return failures


def _runtime_status_failures(
    report: dict[str, Any],
    *,
    expected_embed_report: str | None = None,
    expected_classifier_report: str | None = None,
) -> list[str]:
    failures = []
    targets = report.get("targets") or {}
    pipeline = report.get("pipeline") or {}
    udf = report.get("udf") or {}
    facets = report.get("facets") or {}
    kubernetes = report.get("kubernetes") or {}
    function_status = kubernetes.get("function_status") or {}
    scaled_objects = kubernetes.get("scaled_objects") or {}
    trigger_authentications = kubernetes.get("trigger_authentications") or {}
    required_facets = {"specialty", "age_band", "diagnosis_category", "gender", "events"}
    target_rows = _positive_int(targets.get("full_corpus_notes"))
    if report.get("error"):
        failures.append(f"error={report.get('error')!r}")
    if not report.get("namespace"):
        failures.append("namespace missing")
    elif report.get("namespace") != targets.get("namespace"):
        failures.append(f"namespace={report.get('namespace')!r} does not match targets.namespace={targets.get('namespace')!r}")
    if not targets.get("pipeline_id"):
        failures.append("targets.pipeline_id missing")
    if not targets.get("udf_id"):
        failures.append("targets.udf_id missing")
    if targets.get("embed_pipeline_cr") != "chart-embed-gpu":
        failures.append(f"targets.embed_pipeline_cr={targets.get('embed_pipeline_cr')!r}, expected 'chart-embed-gpu'")
    if targets.get("embed_compute_class") != "gpu":
        failures.append(f"targets.embed_compute_class={targets.get('embed_compute_class')!r}, expected 'gpu'")
    if targets.get("classifier_compute_class") != "gpu":
        failures.append(f"targets.classifier_compute_class={targets.get('classifier_compute_class')!r}, expected 'gpu'")
    if targets.get("embed_image") != "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-embedder-plan-20260624-dedupe2":
        failures.append(
            f"targets.embed_image={targets.get('embed_image')!r}, expected '186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-embedder-plan-20260624-dedupe2'"
        )
    if targets.get("classifier_image") != "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-classifier-plan-20260624":
        failures.append(
            "targets.classifier_image="
            f"{targets.get('classifier_image')!r}, expected '186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-classifier-plan-20260624'"
        )
    if target_rows < 167000:
        failures.append(f"targets.full_corpus_notes={targets.get('full_corpus_notes')!r}, expected at least 167000")
    if pipeline.get("pipeline_id") != targets.get("pipeline_id"):
        failures.append("pipeline.pipeline_id does not match targets.pipeline_id")
    if udf.get("udf_id") != targets.get("udf_id"):
        failures.append("udf.udf_id does not match targets.udf_id")
    missing = sorted(required_facets - set(facets))
    if missing:
        failures.append(f"missing facets: {', '.join(missing)}")
    for field in sorted(required_facets & set(facets)):
        facet = facets.get(field) or {}
        field_failures = []
        if _positive_int(facet.get("values")) <= 0:
            field_failures.append("values<=0")
        if _positive_int(facet.get("row_count")) < target_rows:
            field_failures.append(f"row_count={_positive_int(facet.get('row_count'))}/{target_rows}")
        if not facet.get("sha"):
            field_failures.append("missing sha")
        if facet.get("error"):
            field_failures.append(str(facet.get("error")))
        if field_failures:
            failures.append(f"facet {field}: {', '.join(field_failures)}")
    if "error" in pipeline:
        failures.append(f"pipeline error: {pipeline.get('error')}")
    if "error" in udf:
        failures.append(f"udf error: {udf.get('error')}")
        udf_id = targets.get("udf_id")
        function = function_status.get(udf_id) if udf_id else None
        if isinstance(function, dict):
            paused = function.get("paused")
            conditions = function.get("conditions") or []
            ready = next((condition for condition in conditions if condition.get("type") == "Ready"), {})
            detail = f"kubernetes function {udf_id} exists"
            if paused is not None:
                detail += f", paused={paused}"
            if ready:
                reason = ready.get("reason")
                message = ready.get("message")
                if reason:
                    detail += f", ready_reason={reason}"
                if message:
                    detail += f", ready_message={message!r}"
            failures.append(detail)
    for name, scaled_object in sorted(scaled_objects.items()):
        if not isinstance(scaled_object, dict):
            continue
        ready = next(
            (condition for condition in scaled_object.get("conditions") or [] if condition.get("type") == "Ready"),
            {},
        )
        if ready and ready.get("status") != "True":
            reason = ready.get("reason") or "not ready"
            message = ready.get("message")
            detail = f"scaledobject {name} not ready: {reason}"
            if message:
                detail += f": {message}"
            trigger_auth = trigger_authentications.get(name)
            if isinstance(trigger_auth, dict):
                refs = trigger_auth.get("secret_target_refs") or []
                bearer_ref = next((ref for ref in refs if ref.get("parameter") == "bearerToken"), None)
                if bearer_ref:
                    detail += (
                        "; bearerTokenRef="
                        f"{bearer_ref.get('name')}/{bearer_ref.get('key')}"
                    )
                    ref_status = bearer_ref.get("status") or {}
                    if ref_status:
                        detail += (
                            "; bearerTokenStatus="
                            f"secret_exists={ref_status.get('secret_exists')},"
                            f"key_exists={ref_status.get('key_exists')},"
                            f"value_present={ref_status.get('value_present')}"
                        )
            failures.append(detail)
    failures.extend(
        _cost_baseline_failures(
            report,
            expected_embed_report=expected_embed_report,
            expected_classifier_report=expected_classifier_report,
        )
    )
    return failures


def _phase6_gate_complete(report: dict[str, Any], *, status_report: dict[str, Any] | None = None) -> bool:
    gates = report.get("gates") or {}
    targets = report.get("targets") or {}
    status = _object_section(report.get("status"))
    status_targets = _object_section(status.get("targets"))
    pipeline = _object_section(status.get("pipeline"))
    udf = _object_section(status.get("udf"))
    required = (
        "pipeline_installed",
        "udf_installed",
        "full_index_complete",
        "full_classify_complete",
        "base_facets_visible",
        "event_facets_visible",
        "full_facets_complete",
        "cost_baselines_accepted",
        "phase6_complete",
    )
    complete = (
        bool(report.get("namespace"))
        and not report.get("error")
        and all(gates.get(name) is True for name in required)
        and report.get("failures") == []
        and bool(targets.get("pipeline_id"))
        and bool(targets.get("udf_id"))
        and targets.get("pipeline_id") == status_targets.get("pipeline_id")
        and targets.get("udf_id") == status_targets.get("udf_id")
        and pipeline.get("pipeline_id") == targets.get("pipeline_id")
        and udf.get("udf_id") == targets.get("udf_id")
    )
    if not complete:
        return False
    if status_report is None:
        return True
    status_report_targets = status_report.get("targets") or {}
    return (
        status_report.get("namespace") == report.get("namespace")
        and status_report_targets.get("pipeline_id") == targets.get("pipeline_id")
        and status_report_targets.get("udf_id") == targets.get("udf_id")
    )


def _phase6_gate_failures(report: dict[str, Any], *, status_report: dict[str, Any] | None = None) -> list[str]:
    failures = []
    gates = report.get("gates") or {}
    targets = report.get("targets") or {}
    status = _object_section(report.get("status"))
    status_targets = _object_section(status.get("targets"))
    pipeline = _object_section(status.get("pipeline"))
    udf = _object_section(status.get("udf"))
    required = (
        "pipeline_installed",
        "udf_installed",
        "full_index_complete",
        "full_classify_complete",
        "base_facets_visible",
        "event_facets_visible",
        "full_facets_complete",
        "cost_baselines_accepted",
        "phase6_complete",
    )
    if report.get("error"):
        failures.append(f"error={report.get('error')!r}")
    if not report.get("namespace"):
        failures.append("namespace missing")
    for name in required:
        if gates.get(name) is not True:
            failures.append(f"gates.{name}={gates.get(name)!r}, expected true")
    if report.get("failures") != []:
        for failure in report.get("failures") or []:
            if isinstance(failure, dict):
                gate = failure.get("gate") or "unknown"
                reason = failure.get("reason") or "gate incomplete"
                failures.append(f"failure {gate}: {reason}")
            else:
                failures.append(f"failure={failure!r}")
    if not targets.get("pipeline_id"):
        failures.append("targets.pipeline_id missing")
    if not targets.get("udf_id"):
        failures.append("targets.udf_id missing")
    if targets.get("pipeline_id") != status_targets.get("pipeline_id"):
        failures.append("targets.pipeline_id does not match status.targets.pipeline_id")
    if targets.get("udf_id") != status_targets.get("udf_id"):
        failures.append("targets.udf_id does not match status.targets.udf_id")
    if pipeline.get("pipeline_id") != targets.get("pipeline_id"):
        failures.append("status.pipeline.pipeline_id does not match targets.pipeline_id")
    if udf.get("udf_id") != targets.get("udf_id"):
        failures.append("status.udf.udf_id does not match targets.udf_id")
    if status_report is not None:
        status_report_targets = status_report.get("targets") or {}
        if status_report.get("namespace") != report.get("namespace"):
            failures.append("phase6 status namespace does not match gate namespace")
        if status_report_targets.get("pipeline_id") != targets.get("pipeline_id"):
            failures.append("phase6 status pipeline_id does not match gate targets.pipeline_id")
        if status_report_targets.get("udf_id") != targets.get("udf_id"):
            failures.append("phase6 status udf_id does not match gate targets.udf_id")
    return failures


CHECKS = {
    "phase1_slice_index": {
        "reports": ("slice_index",),
        "ok": lambda reports: _live_slice_index_complete(reports["slice_index"]),
    },
    "phase2_3_live_smoke": {
        "reports": ("live_smoke_base",),
        "ok": lambda reports: _live_smoke_complete(reports["live_smoke_base"], require_event_facets=False),
    },
    "phase3_facet_refresh": {
        "reports": ("facet_refresh",),
        "ok": lambda reports: _facet_refresh_complete(reports["facet_refresh"]),
    },
    "phase4_classify_cost_signal": {
        "reports": ("classify_budget",),
        "ok": lambda reports: _cost_gate_accepted(reports["classify_budget"])
        and _signal_accepted(reports["classify_budget"])
        and _classify_sample_complete(reports["classify_budget"])
        and _classify_writeback_complete(reports["classify_budget"])
    },
    "phase4_event_facet_smoke": {
        "reports": ("facet_refresh", "live_smoke"),
        "ok": lambda reports: _event_facet_refresh_complete(reports["facet_refresh"])
        and _live_smoke_complete(reports["live_smoke"], require_event_facets=True),
    },
    "phase5_holdout": {
        "reports": ("holdout",),
        "ok": lambda reports: _holdout_complete(reports["holdout"]),
    },
    "phase5_recds": {
        "reports": ("recds",),
        "ok": lambda reports: _eval_report_complete(reports["recds"], require_published=True),
    },
    "phase5_bimodal_if_present": {
        "reports": ("bimodal",),
        "optional": True,
        "ok": lambda reports: _bimodal_report_complete(reports["bimodal"]),
    },
    "phase6_embed_cost": {
        "reports": ("embed_budget",),
        "ok": lambda reports: _cost_gate_accepted(reports["embed_budget"])
        and _embed_sample_complete(reports["embed_budget"])
        and _embed_production_path_complete(reports["embed_budget"]),
    },
    "phase6_gpu_images": {
        "reports": ("gpu_build",),
        "ok": lambda reports: _gpu_build_complete(reports["gpu_build"]),
    },
    "phase6_deploy_apply": {
        "reports": ("deploy_apply",),
        "ok": lambda reports: _deploy_apply_complete(
            reports["deploy_apply"],
            expected_classifier_report=(reports.get("_report_paths") or {}).get("classify_budget"),
        ),
    },
    "phase6_unpause_embed": {
        "reports": ("phase6_unpause",),
        "ok": lambda reports: _phase6_unpause_complete(
            reports["phase6_unpause"],
            expected_budget_report=(reports.get("_report_paths") or {}).get("embed_budget"),
        ),
    },
    "phase6_runtime_status": {
        "reports": ("phase6_status",),
        "ok": lambda reports: _runtime_status_complete(
            reports["phase6_status"],
            expected_embed_report=(reports.get("_report_paths") or {}).get("embed_budget"),
            expected_classifier_report=(reports.get("_report_paths") or {}).get("classify_budget"),
        ),
    },
    "phase6_gate_complete": {
        "reports": ("phase6_gate", "phase6_status"),
        "ok": lambda reports: _phase6_gate_complete(
            reports["phase6_gate"],
            status_report=reports.get("phase6_status"),
        ),
    },
}

NEXT_ACTIONS = {
    "phase1_slice_index": "scripts/live_slice.sh",
    "phase2_3_live_smoke": (
        "CHART_LIVE_SMOKE_REPORT=${CHART_LIVE_SMOKE_BASE_REPORT:-eval/out/live-smoke-base-report.json} "
        "scripts/smoke_live.sh"
    ),
    "phase3_facet_refresh": "scripts/refresh_facets.sh --fields age_band,gender",
    "phase4_classify_cost_signal": (
        "scripts/layer_cost_report.sh --kind classifier --accept --signal-reviewed "
        "--out eval/out/classify-events-budget.json"
    ),
    "phase4_event_facet_smoke": "scripts/phase4_event_smoke.sh",
    "phase5_recds": (
        "CHART_EVAL_LIMIT=500 CHART_EVAL_TOP_K=1000 "
        "CHART_EVAL_REQUIRE_NO_FAILURES=1 CHART_EVAL_REQUIRE_FUSED_DOMINATES=1 "
        "scripts/eval_live.sh"
    ),
    "phase5_holdout": (
        "uv run --extra eval python -m eval.holdout --split dev "
        "--max-overlap-edges 0 --out eval/out/holdout-report.json"
    ),
    "phase6_embed_cost": (
        "scripts/layer_cost_report.sh --kind embed --accept --out eval/out/embed-budget.json"
    ),
    "phase6_gpu_images": "scripts/build_gpu_images.sh --push",
    "phase6_deploy_apply": (
        "CHART_APPLY_CLASSIFIER=1 CHART_ACCEPT_PHASE4_CLASSIFY_COST=1 "
        'CHART_K8S_CONTEXT_CONFIRM="$(kubectl config current-context)" scripts/deploy_apply.sh --apply'
    ),
    "phase6_unpause_embed": (
        'CHART_ACCEPT_PHASE6_EMBED_COST=1 CHART_K8S_CONTEXT_CONFIRM="$(kubectl config current-context)" '
        "scripts/phase6_unpause_embed.sh --yes"
    ),
    "phase6_runtime_status": "scripts/full_status.sh",
    "phase6_gate_complete": "scripts/gate_report.sh --require-complete",
}

NEXT_PREREQUISITES = {
    "phase2_3_live_smoke": ("phase1_slice_index", "phase3_facet_refresh"),
    "phase4_event_facet_smoke": ("phase4_classify_cost_signal",),
    "phase5_recds": ("phase1_slice_index", "phase5_holdout"),
    "phase6_gpu_images": ("phase4_classify_cost_signal", "phase6_embed_cost"),
    "phase6_deploy_apply": ("phase4_classify_cost_signal", "phase6_gpu_images"),
    "phase6_unpause_embed": ("phase6_deploy_apply", "phase6_embed_cost"),
    "phase6_runtime_status": ("phase6_deploy_apply", "phase6_unpause_embed"),
    "phase6_gate_complete": ("phase6_runtime_status",),
}

NEXT_REQUIREMENTS = {
    "phase1_slice_index": ("gateway_key",),
    "phase2_3_live_smoke": ("gateway_key",),
    "phase3_facet_refresh": ("gateway_key",),
    "phase4_classify_cost_signal": ("classifier_cost_report",),
    "phase4_event_facet_smoke": ("gateway_key", "classifier_extra"),
    "phase5_recds": ("gateway_key", "full_retrieval_corpus"),
    "phase5_holdout": ("eval_extra",),
    "phase6_embed_cost": ("embed_cost_report",),
    "phase6_gpu_images": ("docker_daemon", "registry_push"),
    "phase6_deploy_apply": ("kubectl_context", "k8s_secrets", "classifier_cost_acceptance"),
    "phase6_unpause_embed": ("kubectl_context", "k8s_secrets", "embed_cost_acceptance"),
    "phase6_runtime_status": ("gateway_key", "kubectl_context", "layer_autoscaling"),
    "phase6_gate_complete": ("gateway_key", "kubectl_context", "layer_autoscaling"),
}

REQUIREMENT_DESCRIPTIONS = {
    "gateway_key": "LAYER_GATEWAY_API_KEY from the environment, .env, or the live script's 1Password fallback",
    "gpu": "GPU runtime visible to commands that explicitly run model timing locally",
    "classifier_extra": "Linux/vLLM Gemma classifier runtime available for Phase-4 event writeback",
    "search_extra": "Python dependencies installed with uv --extra search",
    "eval_extra": "Python dependencies installed with uv --extra eval",
    "classifier_cost_report": "accepted Layer classifier cost/signal report at CHART_PHASE4_CLASSIFY_REPORT",
    "embed_cost_report": "accepted Layer embed cost report at CHART_PHASE6_EMBED_BUDGET_REPORT",
    "full_retrieval_corpus": "full PMC-Patients corpus indexed for ReCDS retrieval scoring",
    "docker_daemon": "Docker CLI and a reachable Docker daemon",
    "registry_push": "Registry credentials that can push the configured GPU image tags",
    "kubectl_context": "kubectl configured with the target Kubernetes context",
    "layer_autoscaling": "Layer-managed KEDA ScaledObjects are ready to own worker replicas",
    "k8s_secrets": "Kubernetes secrets and namespace prerequisites for deploy/apply",
    "classifier_cost_acceptance": "operator acceptance of the Phase-4 classifier cost gate",
    "embed_cost_acceptance": "operator acceptance of the Phase-6 embed cost gate",
}


def _env_file_has_gateway_key(cwd: Path) -> bool:
    env_path = cwd / ".env"
    if not env_path.exists():
        return False
    try:
        lines = env_path.read_text().splitlines()
    except OSError:
        return False
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or not stripped.startswith("LAYER_GATEWAY_API_KEY="):
            continue
        value = stripped.split("=", 1)[1].split("#", 1)[0].strip().strip("\"'")
        if value:
            return True
    return False


def _indexed_rows_from_report(path: Path) -> int:
    try:
        report = json.loads(path.read_text())
    except Exception:
        return 0
    if not isinstance(report, dict):
        return 0
    return _positive_int(report.get("indexed"))


def _indexed_rows_from_phase6_status(path: Path) -> int:
    try:
        report = json.loads(path.read_text())
    except Exception:
        return 0
    if not isinstance(report, dict):
        return 0
    pipeline = report.get("pipeline") or {}
    counts = pipeline.get("counts") or {}
    return _positive_int(counts.get("indexed"))


def _layer_autoscaling_status_from_phase6(path: Path) -> tuple[bool, str]:
    try:
        report = json.loads(path.read_text())
    except Exception as exc:
        return False, f"{path}: {exc}"
    if not isinstance(report, dict):
        return False, f"{path}: report is not a JSON object"
    kubernetes = report.get("kubernetes") or {}
    scaled_objects = kubernetes.get("scaled_objects") or {}
    trigger_authentications = kubernetes.get("trigger_authentications") or {}
    if not isinstance(scaled_objects, dict) or not scaled_objects:
        return False, f"{path}: no Layer ScaledObject status recorded"
    unhealthy: list[str] = []
    for name, scaled_object in sorted(scaled_objects.items()):
        if not isinstance(scaled_object, dict):
            continue
        ready = next(
            (condition for condition in scaled_object.get("conditions") or [] if condition.get("type") == "Ready"),
            {},
        )
        if ready and ready.get("status") == "True":
            continue
        reason = ready.get("reason") or "not ready"
        detail = f"{name}: {reason}"
        trigger_auth = trigger_authentications.get(name)
        if isinstance(trigger_auth, dict):
            refs = trigger_auth.get("secret_target_refs") or []
            bearer_ref = next((ref for ref in refs if ref.get("parameter") == "bearerToken"), None)
            if bearer_ref:
                ref_status = bearer_ref.get("status") or {}
                detail += (
                    f" bearerTokenRef={bearer_ref.get('name')}/{bearer_ref.get('key')}"
                    f" secret_exists={ref_status.get('secret_exists')}"
                    f" key_exists={ref_status.get('key_exists')}"
                    f" value_present={ref_status.get('value_present')}"
                )
        unhealthy.append(detail)
    if unhealthy:
        return False, "; ".join(unhealthy)
    return True, f"{path}: Layer ScaledObjects are ready"


def _probe_command(command: list[str]) -> bool:
    try:
        return subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode == 0
    except OSError:
        return False


def _probe_gateway_key_resolver(env: dict[str, str], cwd: Path) -> bool:
    resolver = cwd / "scripts" / "lib" / "resolve_gateway_key.sh"
    if not resolver.exists():
        return False
    probe_env = dict(env)
    probe_env.setdefault("CHART_OP_TIMEOUT_SECONDS", "15")
    try:
        return (
            subprocess.run(
                ["bash", "-lc", "source scripts/lib/resolve_gateway_key.sh; resolve_gateway_key >/dev/null"],
                cwd=cwd,
                env=probe_env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=70,
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def requirement_statuses(
    requirements: list[str] | None = None,
    *,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> dict[str, dict[str, str]]:
    env = env or os.environ
    cwd = cwd or Path.cwd()
    names = sorted(set(requirements or REQUIREMENT_DESCRIPTIONS))
    statuses: dict[str, dict[str, str]] = {}
    for name in names:
        status = {
            "state": "unknown",
            "description": REQUIREMENT_DESCRIPTIONS.get(name, "operator-provided prerequisite"),
        }
        if name == "gateway_key":
            if env.get("LAYER_GATEWAY_API_KEY") or _env_file_has_gateway_key(cwd):
                status.update({"state": "present", "reason": "gateway key is configured locally"})
            elif env.get("CHART_PLAN_AUDIT_PROBE_GATEWAY_KEY") == "1":
                if _probe_gateway_key_resolver(env, cwd):
                    status.update({"state": "present", "reason": "gateway key resolved by redacted live resolver probe"})
                else:
                    status.update({"state": "missing", "reason": "redacted live resolver probe could not resolve gateway key"})
            elif shutil.which("op"):
                status.update(
                    {
                        "state": "unknown",
                        "reason": "1Password fallback is available; set CHART_PLAN_AUDIT_PROBE_GATEWAY_KEY=1 to probe it",
                    }
                )
            else:
                status.update({"state": "missing", "reason": "LAYER_GATEWAY_API_KEY is not configured locally"})
        elif name == "gpu":
            if shutil.which("nvidia-smi") and _probe_command(["nvidia-smi"]):
                status.update({"state": "present", "reason": "nvidia-smi succeeded"})
            else:
                status.update({"state": "missing", "reason": "no working nvidia-smi probe"})
        elif name == "docker_daemon":
            if not shutil.which("docker"):
                status.update({"state": "missing", "reason": "docker CLI is not installed"})
            elif _probe_command(["docker", "info"]):
                status.update({"state": "present", "reason": "docker info succeeded"})
            else:
                status.update({"state": "missing", "reason": "docker daemon is not reachable"})
        elif name == "kubectl_context":
            if not shutil.which("kubectl"):
                status.update({"state": "missing", "reason": "kubectl is not installed"})
            elif _probe_command(["kubectl", "config", "current-context"]):
                status.update({"state": "present", "reason": "kubectl current-context succeeded"})
            else:
                status.update({"state": "missing", "reason": "kubectl has no current context"})
        elif name == "layer_autoscaling":
            ok, reason = _layer_autoscaling_status_from_phase6(report_path("phase6_status", env=env))
            status.update({"state": "present" if ok else "missing", "reason": reason})
        elif name == "classifier_cost_acceptance":
            if env.get("CHART_ACCEPT_PHASE4_CLASSIFY_COST") == "1":
                status.update({"state": "present", "reason": "CHART_ACCEPT_PHASE4_CLASSIFY_COST=1"})
            else:
                status.update({"state": "missing", "reason": "set CHART_ACCEPT_PHASE4_CLASSIFY_COST=1 for apply"})
        elif name == "embed_cost_acceptance":
            if env.get("CHART_ACCEPT_PHASE6_EMBED_COST") == "1":
                status.update({"state": "present", "reason": "CHART_ACCEPT_PHASE6_EMBED_COST=1"})
            else:
                status.update({"state": "missing", "reason": "set CHART_ACCEPT_PHASE6_EMBED_COST=1 for unpause"})
        elif name == "classifier_cost_report":
            path = report_path("classify_budget", env=env)
            report, error = load_report(path)
            if error:
                status.update({"state": "missing", "reason": f"{path}: {error}"})
            elif (
                _cost_gate_accepted(report or {})
                and _signal_accepted(report or {})
                and _classify_sample_complete(report or {})
                and _classify_writeback_complete(report or {})
            ):
                status.update({"state": "present", "reason": f"{path}: accepted"})
            else:
                failures = _classify_budget_failures(report or {})
                status.update({"state": "missing", "reason": f"{path}: " + "; ".join(failures[:3])})
        elif name == "embed_cost_report":
            path = report_path("embed_budget", env=env)
            report, error = load_report(path)
            if error:
                status.update({"state": "missing", "reason": f"{path}: {error}"})
            elif (
                _cost_gate_accepted(report or {})
                and _embed_sample_complete(report or {})
                and _embed_production_path_complete(report or {})
            ):
                status.update({"state": "present", "reason": f"{path}: accepted"})
            else:
                failures = _embed_budget_failures(report or {})
                status.update({"state": "missing", "reason": f"{path}: " + "; ".join(failures[:3])})
        elif name == "full_retrieval_corpus":
            indexed = _indexed_rows_from_phase6_status(report_path("phase6_status", env=env))
            source = "phase6 status"
            if indexed <= 0:
                indexed = _indexed_rows_from_report(report_path("slice_index", env=env))
                source = "slice index report"
            if indexed >= FULL_CORPUS_NOTES:
                status.update({"state": "present", "reason": f"{source}: indexed rows={indexed}"})
            else:
                status.update(
                    {
                        "state": "missing",
                        "reason": f"{source}: indexed rows={indexed}, expected at least {FULL_CORPUS_NOTES}",
                    }
                )
        elif name == "classifier_extra":
            if env.get("CHART_ASSUME_CLASSIFIER_EXTRA") == "1":
                status.update({"state": "present", "reason": "CHART_ASSUME_CLASSIFIER_EXTRA=1"})
            elif platform.system() != "Linux":
                status.update(
                    {
                        "state": "missing",
                        "reason": "Gemma classifier runtime requires Linux/vLLM; run on a GPU Function image or Linux GPU host",
                    }
                )
            elif _module_available("vllm") and _module_available("transformers"):
                status.update({"state": "present", "reason": "vllm and transformers are importable"})
            else:
                status.update({"state": "missing", "reason": "vllm and transformers are required for classifier runtime"})
        elif name in {"search_extra", "eval_extra"}:
            status.update({"state": "unknown", "reason": "resolved by the corresponding uv --extra command"})
        elif name in {"registry_push", "k8s_secrets"}:
            status.update({"state": "unknown", "reason": "operator/account state; audit cannot prove it locally"})
        statuses[name] = status
    return statuses


def apply_requirement_readiness(report: dict[str, Any], requirements: dict[str, dict[str, str]]) -> dict[str, Any]:
    for step in report.get("next_steps") or []:
        missing = [
            requirement
            for requirement in step.get("requires", [])
            if (requirements.get(requirement) or {}).get("state") == "missing"
        ]
        if (
            "gateway_key" in step.get("requires", [])
            and any("LAYER_GATEWAY_API_KEY is required" in detail for detail in step.get("details", []))
            and "gateway_key" not in missing
        ):
            missing.append("gateway_key")
        if missing:
            step["ready"] = False
            step["missing_requirements"] = missing
    return report


def _next_step_details(name: str, loaded: dict[str, dict[str, Any]]) -> list[str]:
    if name == "phase1_slice_index" and "slice_index" in loaded:
        return _live_slice_index_failures(loaded["slice_index"])
    if name == "phase2_3_live_smoke" and "live_smoke_base" in loaded:
        return _live_smoke_failures(loaded["live_smoke_base"], require_event_facets=False)
    if name == "phase3_facet_refresh" and "facet_refresh" in loaded:
        return _facet_refresh_failures(loaded["facet_refresh"])
    if name == "phase4_classify_cost_signal" and "classify_budget" in loaded:
        return _classify_budget_failures(loaded["classify_budget"])
    if name == "phase4_event_facet_smoke":
        details = []
        if "facet_refresh" in loaded:
            details.extend(_event_facet_refresh_failures(loaded["facet_refresh"]))
        if "live_smoke" in loaded:
            details.extend(_live_smoke_failures(loaded["live_smoke"], require_event_facets=True))
        return details
    if name == "phase5_recds" and "recds" in loaded:
        return _eval_report_failures(loaded["recds"], require_published=True)
    if name == "phase5_bimodal_if_present" and "bimodal" in loaded:
        return _bimodal_report_failures(loaded["bimodal"])
    if name == "phase5_holdout" and "holdout" in loaded:
        return _holdout_failures(loaded["holdout"])
    if name == "phase6_embed_cost" and "embed_budget" in loaded:
        return _embed_budget_failures(loaded["embed_budget"])
    if name == "phase6_gpu_images" and "gpu_build" in loaded:
        return _gpu_build_failures(loaded["gpu_build"])
    if name == "phase6_deploy_apply" and "deploy_apply" in loaded:
        return _deploy_apply_failures(
            loaded["deploy_apply"],
            expected_classifier_report=(loaded.get("_report_paths") or {}).get("classify_budget"),
        )
    if name == "phase6_unpause_embed" and "phase6_unpause" in loaded:
        return _phase6_unpause_failures(
            loaded["phase6_unpause"],
            expected_budget_report=(loaded.get("_report_paths") or {}).get("embed_budget"),
        )
    if name == "phase6_runtime_status" and "phase6_status" in loaded:
        return _runtime_status_failures(
            loaded["phase6_status"],
            expected_embed_report=(loaded.get("_report_paths") or {}).get("embed_budget"),
            expected_classifier_report=(loaded.get("_report_paths") or {}).get("classify_budget"),
        )
    if name == "phase6_gate_complete" and "phase6_gate" in loaded:
        return _phase6_gate_failures(loaded["phase6_gate"], status_report=loaded.get("phase6_status"))
    return []


def summarize(*, env: dict[str, str] | None = None) -> dict[str, Any]:
    loaded: dict[str, dict[str, Any]] = {}
    report_entries: dict[str, dict[str, Any]] = {}
    for name in DEFAULT_REPORTS:
        path = report_path(name, env=env)
        report, error = load_report(path)
        report_entries[name] = {"path": str(path), "present": error is None}
        if error:
            report_entries[name]["error"] = error
        else:
            loaded[name] = report or {}
    loaded["_report_paths"] = {name: entry["path"] for name, entry in report_entries.items()}

    checks = {}
    for name, spec in CHECKS.items():
        required = spec["reports"]
        optional = bool(spec.get("optional"))
        missing = [report for report in required if report not in loaded]
        if missing:
            skipped = optional and all(report_entries[report]["error"] == "missing" for report in missing)
            checks[name] = {
                "ok": skipped,
                "optional": optional,
                "reports": list(required),
                "reason": "optional report missing" if skipped else f"missing report(s): {', '.join(missing)}",
            }
            continue
        try:
            ok = bool(spec["ok"](loaded))
        except Exception as exc:
            ok = False
            reason = f"check failed: {exc}"
        else:
            reason = None if ok else "report content does not satisfy gate"
        checks[name] = {"ok": ok, "optional": optional, "reports": list(required)}
        if reason:
            checks[name]["reason"] = reason
            if optional:
                details = _next_step_details(name, loaded)
                if details:
                    checks[name]["details"] = details

    required_checks = {name: check for name, check in checks.items() if not check["optional"]}
    next_steps = []
    for name, check in required_checks.items():
        if check["ok"]:
            continue
        step = {
            "check": name,
            "reason": check.get("reason", "gate incomplete"),
            "command": NEXT_ACTIONS[name],
            "requires": list(NEXT_REQUIREMENTS.get(name, ())),
        }
        details = _next_step_details(name, loaded)
        if details:
            step["details"] = details
        blocked_by = [prereq for prereq in NEXT_PREREQUISITES.get(name, ()) if not checks[prereq]["ok"]]
        step["ready"] = not blocked_by
        if blocked_by:
            step["blocked_by"] = blocked_by
        next_steps.append(step)
    return {
        "complete": all(check["ok"] for check in required_checks.values()),
        "checks": checks,
        "next_steps": next_steps,
        "reports": report_entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit PLAN.md completion from persisted gate reports")
    parser.add_argument("--out", type=Path, default=None, help="write the audit report to this JSON path")
    parser.add_argument("--ready", action="store_true", help="print only currently runnable next-step commands")
    parser.add_argument("--requirements", action="store_true", help="include local requirement diagnostics in the JSON report")
    parser.add_argument("--require-complete", action="store_true", help="exit non-zero unless every required gate passes")
    args = parser.parse_args()
    report = summarize()
    if args.requirements or args.ready:
        needed = [requirement for step in report["next_steps"] for requirement in step.get("requires", [])]
        requirements = requirement_statuses(needed)
        apply_requirement_readiness(report, requirements)
        if args.requirements:
            report["requirements"] = requirements
    rendered = json.dumps(report, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n")
    if args.ready:
        ready_steps = [step for step in report["next_steps"] if step.get("ready")]
        if ready_steps:
            for step in ready_steps:
                print(f"{step['check']}: {step['command']}")
        else:
            print("no ready next steps")
    else:
        print(rendered)
    if args.require_complete and not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
