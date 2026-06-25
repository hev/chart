#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

has_out=0
for arg in "$@"; do
  if [[ "$arg" == "--out" || "$arg" == --out=* ]]; then
    has_out=1
    break
  fi
done
args=("$@")
if [[ "$has_out" == "0" ]]; then
  args+=(--out "${CHART_PLAN_AUDIT_REPORT:-eval/out/plan-audit-report.json}")
fi

uv run python -m smoke.plan_audit "${args[@]}"
