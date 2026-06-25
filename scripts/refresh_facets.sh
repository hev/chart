#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source scripts/lib/resolve_gateway_key.sh
source scripts/lib/write_failure_report.sh

args=("$@")
has_out=0
if [[ "${#args[@]}" -gt 0 ]]; then
  for arg in "${args[@]}"; do
    if [[ "$arg" == "--out" || "$arg" == --out=* ]]; then
      has_out=1
      break
    fi
  done
fi
if [[ "$has_out" == "0" ]]; then
  args+=(--out "${CHART_FACET_REFRESH_REPORT:-eval/out/facet-refresh-report.json}")
fi
REPORT_PATH="${CHART_FACET_REFRESH_REPORT:-eval/out/facet-refresh-report.json}"
for ((i = 0; i < ${#args[@]}; i++)); do
  arg="${args[$i]}"
  if [[ "$arg" == "--out" && $((i + 1)) -lt ${#args[@]} ]]; then
    REPORT_PATH="${args[$((i + 1))]}"
    break
  elif [[ "$arg" == --out=* ]]; then
    REPORT_PATH="${arg#--out=}"
    break
  fi
done

if ! resolve_gateway_key; then
  write_failure_report "$REPORT_PATH" "failed" "LAYER_GATEWAY_API_KEY is required"
  exit 1
fi

if ! uv run --extra search python -m smoke.facets "${args[@]}"; then
  write_failure_report "$REPORT_PATH" "failed" "facet refresh command failed"
  exit 1
fi
