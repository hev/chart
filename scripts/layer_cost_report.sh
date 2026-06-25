#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source scripts/lib/resolve_gateway_key.sh
source scripts/lib/write_failure_report.sh

kind=""
out=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  arg="${args[$i]}"
  if [[ "$arg" == "--kind" && $((i + 1)) -lt ${#args[@]} ]]; then
    kind="${args[$((i + 1))]}"
  elif [[ "$arg" == --kind=* ]]; then
    kind="${arg#--kind=}"
  elif [[ "$arg" == "--out" && $((i + 1)) -lt ${#args[@]} ]]; then
    out="${args[$((i + 1))]}"
  elif [[ "$arg" == --out=* ]]; then
    out="${arg#--out=}"
  fi
done

if [[ -z "$kind" ]]; then
  echo "usage: scripts/layer_cost_report.sh --kind embed|classifier --out PATH [--accept] [--signal-reviewed]" >&2
  exit 2
fi
if [[ -z "$out" ]]; then
  if [[ "$kind" == "embed" ]]; then
    out="${CHART_PHASE6_EMBED_BUDGET_REPORT:-eval/out/embed-budget.json}"
    args+=(--out "$out")
  elif [[ "$kind" == "classifier" ]]; then
    out="${CHART_PHASE4_CLASSIFY_REPORT:-eval/out/classify-events-budget.json}"
    args+=(--out "$out")
  else
    echo "unknown cost report kind: $kind" >&2
    exit 2
  fi
fi
has_window=0
for arg in "${args[@]}"; do
  if [[ "$arg" == "--window" || "$arg" == --window=* ]]; then
    has_window=1
    break
  fi
done
if [[ "$has_window" == "0" ]]; then
  args+=(--window "${CHART_LAYER_COST_WINDOW:-24h}")
fi

if ! resolve_gateway_key; then
  write_failure_report "$out" "failed" "LAYER_GATEWAY_API_KEY is required"
  exit 1
fi

rm -f "$out"
if ! uv run --extra search python -m smoke.layer_cost "${args[@]}"; then
  if [[ ! -s "$out" ]]; then
    write_failure_report "$out" "failed" "Layer cost report command failed"
  fi
  exit 1
fi
