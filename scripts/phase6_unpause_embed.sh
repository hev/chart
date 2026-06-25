#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="status"
if [[ "${1:-}" == "--yes" ]]; then
  MODE="unpause"
  shift
elif [[ "${1:-}" == "--status" ]]; then
  shift
fi

if [[ $# -gt 0 ]]; then
  echo "usage: scripts/phase6_unpause_embed.sh [--status|--yes]" >&2
  exit 2
fi

pipeline_name="${CHART_EMBED_PIPELINE_CR:-chart-embed-gpu}"
namespace="${CHART_K8S_NAMESPACE:-chart}"
expected_pipeline_id="${CHART_EMBED_PIPELINE_ID:-chart-notes}"
budget_report="${CHART_PHASE6_EMBED_BUDGET_REPORT:-eval/out/embed-budget.json}"
unpause_report="${CHART_PHASE6_UNPAUSE_REPORT:-eval/out/phase6-unpause-report.json}"

write_report() {
  local status="$1"
  local error="${2:-}"
  local actual_pipeline_id="${3:-}"
  local report_tmp="${unpause_report}.tmp.$$"
  mkdir -p "$(dirname "$unpause_report")"
  python3 - "$MODE" "$status" "$namespace" "$pipeline_name" "$expected_pipeline_id" "$budget_report" "$actual_pipeline_id" "$error" >"$report_tmp" <<'PY'
import json
import sys

mode, status, namespace, pipeline_cr, expected_pipeline_id, budget_report, actual_pipeline_id, error = sys.argv[1:9]
report = {
    "mode": mode,
    "status": status,
    "namespace": namespace,
    "pipeline_cr": pipeline_cr,
    "expected_pipeline_id": expected_pipeline_id,
    "budget_report": budget_report,
}
if actual_pipeline_id:
    report["actual_pipeline_id"] = actual_pipeline_id
if error:
    report["error"] = error
print(json.dumps(report, indent=2))
PY
  mv "$report_tmp" "$unpause_report"
}

fail_report() {
  local message="$1"
  local actual_pipeline_id="${2:-}"
  write_report "failed" "$message" "$actual_pipeline_id"
  echo "$message" >&2
  exit 1
}

if ! command -v kubectl >/dev/null 2>&1; then
  fail_report "missing required command: kubectl"
fi

if [[ "$MODE" == "status" ]]; then
  kubectl -n "$namespace" get pipeline "$pipeline_name" \
    -o 'custom-columns=NAME:.metadata.name,PIPELINE_ID:.spec.pipelineId,PAUSED:.spec.paused,IMAGE:.spec.worker.image,POOL:.spec.scaling.pool'
  write_report "checked"
  exit 0
fi

current_context="$(kubectl config current-context 2>/dev/null || true)"
if [[ -z "$current_context" ]]; then
  fail_report "refusing to unpause without an active kubectl context"
fi
if [[ "${CHART_K8S_CONTEXT_CONFIRM:-}" != "$current_context" ]]; then
  fail_report "refusing to unpause $namespace/$pipeline_name on Kubernetes context $current_context without CHART_K8S_CONTEXT_CONFIRM=$current_context"
fi

if [[ "${CHART_ACCEPT_PHASE6_EMBED_COST:-}" != "1" ]]; then
  fail_report "refusing to unpause $namespace/$pipeline_name without CHART_ACCEPT_PHASE6_EMBED_COST=1. Run indexer.measure_embed with accepted --max-full-hours/--max-full-usd budgets first."
fi
if [[ ! -f "$budget_report" ]]; then
  fail_report "refusing to unpause $namespace/$pipeline_name without embed budget report: $budget_report. Run indexer.measure_embed with accepted --max-full-hours/--max-full-usd budgets first."
fi
budget_error=""
if ! budget_error="$(python3 - "$budget_report" 2>&1 <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    report = json.loads(path.read_text())
except Exception as exc:
    raise SystemExit(f"invalid embed budget report {path}: {exc}") from exc
budget = report.get("budget") or {}
snapshot = report.get("layer_cost_snapshot") or report.get("cost_snapshot") or {}
layer_accepted = (
    report.get("accepted") is True
    and report.get("source") in {"layer", "layer_cost"}
    and isinstance(snapshot, dict)
    and snapshot.get("as_of_ms")
    and snapshot.get("window_seconds")
    and (snapshot.get("totals") or {}).get("total_usd") is not None
)
if layer_accepted:
    try:
        total_usd = float((snapshot.get("totals") or {}).get("total_usd"))
    except (TypeError, ValueError):
        raise SystemExit(f"embed Layer cost report has invalid total_usd: {path}")
    if total_usd < 0:
        raise SystemExit(f"embed Layer cost report has negative total_usd: {path}")
else:
    if budget.get("accepted") is not True:
        raise SystemExit(f"embed budget report is not accepted: {path}")
    checks = budget.get("checks") or {}
    missing = [name for name in ("max_full_hours", "max_full_usd") if name not in checks]
    if missing:
        raise SystemExit(f"embed budget report is missing required budget checks: {', '.join(missing)}")
    failed = [name for name in ("max_full_hours", "max_full_usd") if (checks.get(name) or {}).get("ok") is not True]
    if failed:
        raise SystemExit(f"embed budget report has failing budget checks: {', '.join(failed)}")
    runtime = report.get("runtime") or {}
    if runtime.get("accelerator") != "gpu":
        raise SystemExit(f"embed budget report must be measured on gpu accelerator: {path}")
PY
)"; then
  fail_report "$budget_error"
fi

if ! kubectl -n "$namespace" get secret chart-gateway >/dev/null; then
  fail_report "missing required secret $namespace/chart-gateway"
fi
if ! kubectl -n "$namespace" get pipeline "$pipeline_name" >/dev/null; then
  fail_report "missing required pipeline $namespace/$pipeline_name"
fi
actual_pipeline_id="$(kubectl -n "$namespace" get pipeline "$pipeline_name" -o jsonpath='{.spec.pipelineId}')"
if [[ "$actual_pipeline_id" != "$expected_pipeline_id" ]]; then
  fail_report "refusing to unpause $namespace/$pipeline_name: spec.pipelineId=$actual_pipeline_id, expected $expected_pipeline_id" "$actual_pipeline_id"
fi
kubectl -n "$namespace" patch pipeline "$pipeline_name" --type=merge -p '{"spec":{"paused":false}}'
kubectl -n "$namespace" get pipeline "$pipeline_name" \
  -o 'custom-columns=NAME:.metadata.name,PIPELINE_ID:.spec.pipelineId,PAUSED:.spec.paused,IMAGE:.spec.worker.image,POOL:.spec.scaling.pool'
write_report "unpaused" "" "$actual_pipeline_id"
