#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source scripts/lib/resolve_gateway_key.sh
source scripts/lib/write_failure_report.sh

LIMIT="${CHART_SLICE_LIMIT:-2000}"
INDEX_REPORT="${CHART_SLICE_INDEX_REPORT:-eval/out/slice-index-report.json}"

if ! resolve_gateway_key; then
  write_failure_report "$INDEX_REPORT" "failed" "LAYER_GATEWAY_API_KEY is required"
  exit 1
fi

echo "Indexing ${LIMIT} PMC-Patients notes into ${CHART_NAMESPACE:-chart-notes}..."
if ! uv run --extra search python -m indexer --limit "$LIMIT" --out "$INDEX_REPORT"; then
  write_failure_report "$INDEX_REPORT" "failed" "slice index command failed"
  exit 1
fi

echo "Verifying routed chips, nearest_to_id, and facets..."
scripts/smoke_live.sh

echo "Live slice is queryable."
