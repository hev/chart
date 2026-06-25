#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source scripts/lib/write_failure_report.sh

LIVE_REPORT="${CHART_LIVE_SMOKE_REPORT:-eval/out/live-smoke-report.json}"
FACET_REPORT="${CHART_FACET_REFRESH_REPORT:-eval/out/facet-refresh-report.json}"

write_phase4_failure() {
  local error="$1"
  write_failure_report "$LIVE_REPORT" "failed" "$error"
  write_failure_report "$FACET_REPORT" "failed" "$error"
}

if [[ "${CHART_ASSUME_CLASSIFIER_EXTRA:-0}" != "1" ]]; then
  if [[ "$(uname -s)" != "Linux" ]]; then
    write_phase4_failure "Gemma classifier runtime requires Linux/vLLM; run on a GPU Function image or Linux GPU host"
    exit 1
  fi
  if ! uv run --extra classifier python - <<'PY'
import importlib.util
import sys

missing = [name for name in ("vllm", "transformers") if importlib.util.find_spec(name) is None]
if missing:
    print(f"missing classifier runtime module(s): {', '.join(missing)}", file=sys.stderr)
    raise SystemExit(1)
PY
  then
    write_phase4_failure "vllm and transformers are required for classifier runtime"
    exit 1
  fi
fi

if ! uv run --extra classifier python -m functions.classify_events --once; then
  write_phase4_failure "classifier event smoke command failed"
  exit 1
fi
scripts/refresh_facets.sh --fields age_band,gender,events
CHART_REQUIRE_EVENT_FACETS=1 scripts/smoke_live.sh
