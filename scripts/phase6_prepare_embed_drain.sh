#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source scripts/lib/resolve_gateway_key.sh

MODE="status"
if [[ "${1:-}" == "--yes" ]]; then
  MODE="prepare"
  shift
elif [[ "${1:-}" == "--status" ]]; then
  shift
fi

if [[ $# -gt 0 ]]; then
  echo "usage: scripts/phase6_prepare_embed_drain.sh [--status|--yes]" >&2
  exit 2
fi

namespace="${CHART_K8S_NAMESPACE:-chart}"
pipeline_id="${CHART_EMBED_PIPELINE_ID:-chart-notes}"
ingest_deployment="${CHART_INGEST_DEPLOYMENT:-chart-ingest-worker}"
gateway_namespace="${CHART_LAYER_NAMESPACE:-layer}"
gateway_pod="${CHART_LAYER_GATEWAY_POD:-layer-gateway-0}"
postgres_container="${CHART_LAYER_POSTGRES_CONTAINER:-postgres}"
report_path="${CHART_PHASE6_DRAIN_REPORT:-eval/out/phase6-drain-report.json}"
claim_worker_id="${CHART_PHASE6_DRAIN_WORKER_ID:-chart-phase6-drain-check}"
claim_limit="${CHART_PHASE6_DRAIN_CLAIM_LIMIT:-2}"

write_report() {
  local status="$1"
  local claimable="${2:-unknown}"
  local pending="${3:-unknown}"
  local terminated="${4:-0}"
  local error="${5:-}"
  local report_tmp="${report_path}.tmp.$$"
  mkdir -p "$(dirname "$report_path")"
  python3 - "$MODE" "$status" "$namespace" "$pipeline_id" "$ingest_deployment" "$gateway_namespace" "$gateway_pod" "$claimable" "$pending" "$terminated" "$error" >"$report_tmp" <<'PY'
import json
import sys

(
    mode,
    status,
    namespace,
    pipeline_id,
    ingest_deployment,
    gateway_namespace,
    gateway_pod,
    claimable,
    pending,
    terminated,
    error,
) = sys.argv[1:12]

def parse_bool(value):
    if value == "true":
        return True
    if value == "false":
        return False
    return None

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
    "gateway_namespace": gateway_namespace,
    "gateway_pod": gateway_pod,
    "claimable": parse_bool(claimable),
    "pending_count": parse_int(pending),
    "terminated_idle_transactions": parse_int(terminated) or 0,
}
if error:
    report["error"] = error
print(json.dumps(report, indent=2))
PY
  mv "$report_tmp" "$report_path"
}

fail_report() {
  local message="$1"
  write_report "failed" "unknown" "unknown" "0" "$message"
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

terminate_idle_segment_transactions() {
  kubectl -n "$gateway_namespace" exec "$gateway_pod" -c "$postgres_container" -- \
    psql -U gateway -d gateway -At -c \
    "with terminated as (
       select pg_terminate_backend(pid) ok
       from pg_stat_activity
       where datname='gateway'
         and state='idle in transaction'
         and query like 'SELECT id, stage, manifest_key FROM pipeline_segments%'
     )
     select count(*) from terminated where ok;"
}

claim_check() {
  if [[ -z "${LAYER_GATEWAY_API_KEY:-}" ]]; then
    LAYER_GATEWAY_API_KEY="$(
      kubectl -n "$namespace" get secret chart-gateway \
        -o jsonpath='{.data.LAYER_GATEWAY_API_KEY}' 2>/dev/null | base64 --decode
    )"
    export LAYER_GATEWAY_API_KEY
  fi
  if [[ -z "${LAYER_GATEWAY_API_KEY:-}" ]]; then
    resolve_gateway_key >/dev/null
  fi
  LAYER_GATEWAY_API_KEY="$LAYER_GATEWAY_API_KEY" uv run python - "$pipeline_id" "$claim_worker_id" "$claim_limit" <<'PY'
import asyncio
import sys

from chart_common.config import Settings
from chart_common.gateway import close_client, make_client

pipeline_id, worker_id, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])

async def main():
    layer = make_client(Settings())
    docs = []
    try:
        claim = await layer.claim_documents(
            pipeline_id,
            {"worker_id": worker_id, "limit": limit, "lease_seconds": 30},
        )
        docs = list(claim.documents)
        if docs:
            await layer.release_documents(
                pipeline_id,
                docs,
                from_stage=claim.claim_stage,
                worker_id=worker_id,
            )
    finally:
        await close_client(layer)
    print("true" if docs else "false")

asyncio.run(main())
PY
}

pending="$(pending_count | tr -d '[:space:]')"
if [[ "$MODE" == "status" ]]; then
  claimable="$(claim_check | tail -n 1 | tr -d '[:space:]')"
  kubectl -n "$namespace" get deployment "$ingest_deployment" \
    -o 'custom-columns=NAME:.metadata.name,READY:.status.readyReplicas,REPLICAS:.spec.replicas,IMAGE:.spec.template.spec.containers[0].image'
  echo "pipeline_id=$pipeline_id pending=$pending claimable=$claimable"
  write_report "checked" "$claimable" "$pending" "0"
  exit 0
fi

current_context="$(kubectl config current-context 2>/dev/null || true)"
if [[ -z "$current_context" ]]; then
  fail_report "refusing to prepare embed drain without an active kubectl context"
fi
if [[ "${CHART_K8S_CONTEXT_CONFIRM:-}" != "$current_context" ]]; then
  fail_report "refusing to prepare embed drain on Kubernetes context $current_context without CHART_K8S_CONTEXT_CONFIRM=$current_context"
fi

kubectl -n "$namespace" scale deployment/"$ingest_deployment" --replicas=0
terminated="$(terminate_idle_segment_transactions | tr -d '[:space:]')"
claimable="$(claim_check | tail -n 1 | tr -d '[:space:]')"
pending="$(pending_count | tr -d '[:space:]')"
if [[ "$claimable" != "true" && "${pending:-0}" != "0" ]]; then
  fail_report "pipeline $pipeline_id has pending documents but claim check returned no documents"
fi
kubectl -n "$namespace" get deployment "$ingest_deployment" \
  -o 'custom-columns=NAME:.metadata.name,READY:.status.readyReplicas,REPLICAS:.spec.replicas,IMAGE:.spec.template.spec.containers[0].image'
echo "pipeline_id=$pipeline_id pending=$pending claimable=$claimable terminated_idle_transactions=$terminated"
write_report "prepared" "$claimable" "$pending" "$terminated"
