#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="status"
if [[ "${1:-}" == "--yes" ]]; then
  MODE="stage"
  shift
elif [[ "${1:-}" == "--status" ]]; then
  shift
fi

if [[ $# -gt 0 ]]; then
  echo "usage: scripts/phase6_stage_source_window.sh [--status|--yes]" >&2
  exit 2
fi

namespace="${CHART_K8S_NAMESPACE:-chart}"
pipeline_id="${CHART_EMBED_PIPELINE_ID:-chart-notes}"
ingest_deployment="${CHART_INGEST_DEPLOYMENT:-chart-ingest-worker}"
gateway_namespace="${CHART_LAYER_NAMESPACE:-layer}"
gateway_pod="${CHART_LAYER_GATEWAY_POD:-layer-gateway-0}"
postgres_container="${CHART_LAYER_POSTGRES_CONTAINER:-postgres}"
start_offset="${CHART_PHASE6_SOURCE_START_OFFSET:-}"
max_rows="${CHART_PHASE6_SOURCE_MAX_ROWS:-1000}"
page_size="${CHART_PHASE6_SOURCE_PAGE_SIZE:-100}"
report_path="${CHART_PHASE6_STAGE_REPORT:-eval/out/phase6-stage-report.json}"

write_report() {
  local status="$1"
  local pending="${2:-unknown}"
  local replicas="${3:-unknown}"
  local error="${4:-}"
  local report_tmp="${report_path}.tmp.$$"
  mkdir -p "$(dirname "$report_path")"
  python3 - "$MODE" "$status" "$namespace" "$pipeline_id" "$ingest_deployment" "$start_offset" "$max_rows" "$page_size" "$pending" "$replicas" "$error" >"$report_tmp" <<'PY'
import json
import sys

(
    mode,
    status,
    namespace,
    pipeline_id,
    ingest_deployment,
    start_offset,
    max_rows,
    page_size,
    pending,
    replicas,
    error,
) = sys.argv[1:12]

def parse_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

report = {
    "mode": mode,
    "status": status,
    "namespace": namespace,
    "pipeline_id": pipeline_id,
    "ingest_deployment": ingest_deployment,
    "start_offset": parse_int(start_offset),
    "max_rows": parse_int(max_rows),
    "page_size": parse_int(page_size),
    "pending_count": parse_int(pending),
    "replicas": parse_int(replicas),
}
if error:
    report["error"] = error
print(json.dumps(report, indent=2))
PY
  mv "$report_tmp" "$report_path"
}

fail_report() {
  local message="$1"
  write_report "failed" "unknown" "unknown" "$message"
  echo "$message" >&2
  exit 1
}

if ! command -v kubectl >/dev/null 2>&1; then
  fail_report "missing required command: kubectl"
fi

pending_count() {
  kubectl -n "$gateway_namespace" exec "$gateway_pod" -c "$postgres_container" -- \
    psql -U gateway -d gateway -At -c \
    "select coalesce(sum(row_count), 0) from pipeline_segments where pipeline_id='${pipeline_id}' and stage='pending';"
}

current_replicas() {
  kubectl -n "$namespace" get deployment "$ingest_deployment" -o jsonpath='{.spec.replicas}'
}

pending="$(pending_count | tr -d '[:space:]')"
replicas="$(current_replicas | tr -d '[:space:]')"

if [[ "$MODE" == "status" ]]; then
  kubectl -n "$namespace" get deployment "$ingest_deployment" \
    -o 'custom-columns=NAME:.metadata.name,READY:.status.readyReplicas,REPLICAS:.spec.replicas,IMAGE:.spec.template.spec.containers[0].image'
  echo "pipeline_id=$pipeline_id pending=$pending replicas=$replicas"
  write_report "checked" "$pending" "$replicas"
  exit 0
fi

current_context="$(kubectl config current-context 2>/dev/null || true)"
if [[ -z "$current_context" ]]; then
  fail_report "refusing to stage source window without an active kubectl context"
fi
if [[ "${CHART_K8S_CONTEXT_CONFIRM:-}" != "$current_context" ]]; then
  fail_report "refusing to stage source window on Kubernetes context $current_context without CHART_K8S_CONTEXT_CONFIRM=$current_context"
fi
if [[ -z "$start_offset" ]]; then
  fail_report "CHART_PHASE6_SOURCE_START_OFFSET is required for deterministic source staging"
fi
case "$start_offset" in
  ''|*[!0-9]*) fail_report "CHART_PHASE6_SOURCE_START_OFFSET must be a non-negative integer" ;;
esac
case "$max_rows" in
  ''|*[!0-9]*) fail_report "CHART_PHASE6_SOURCE_MAX_ROWS must be a positive integer" ;;
esac
case "$page_size" in
  ''|*[!0-9]*) fail_report "CHART_PHASE6_SOURCE_PAGE_SIZE must be a positive integer" ;;
esac
if [[ "$max_rows" -le 0 ]]; then
  fail_report "CHART_PHASE6_SOURCE_MAX_ROWS must be > 0"
fi
if [[ "$page_size" -le 0 ]]; then
  fail_report "CHART_PHASE6_SOURCE_PAGE_SIZE must be > 0"
fi
if [[ "${pending:-0}" != "0" && "${CHART_PHASE6_ALLOW_STAGE_WITH_PENDING:-}" != "1" ]]; then
  fail_report "refusing to stage source window while pipeline $pipeline_id has $pending pending documents; drain first or set CHART_PHASE6_ALLOW_STAGE_WITH_PENDING=1"
fi

kubectl -n "$namespace" set env deployment/"$ingest_deployment" \
  CHART_HF_SOURCE_START_OFFSET="$start_offset" \
  CHART_HF_SOURCE_MAX_ROWS="$max_rows" \
  CHART_HF_SOURCE_PAGE_SIZE="$page_size"
kubectl -n "$namespace" scale deployment/"$ingest_deployment" --replicas=1
replicas="$(current_replicas | tr -d '[:space:]')"
pending="$(pending_count | tr -d '[:space:]')"
kubectl -n "$namespace" get deployment "$ingest_deployment" \
  -o 'custom-columns=NAME:.metadata.name,READY:.status.readyReplicas,REPLICAS:.spec.replicas,IMAGE:.spec.template.spec.containers[0].image'
echo "pipeline_id=$pipeline_id start_offset=$start_offset max_rows=$max_rows page_size=$page_size pending=$pending replicas=$replicas"
write_report "staging" "$pending" "$replicas"
