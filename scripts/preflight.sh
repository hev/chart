#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    return 1
  fi
}

need_cmd uv
need_cmd bash
need_cmd node
need_cmd npx

while IFS= read -r script; do
  bash -n "$script"
done < <(find scripts -name '*.sh' -type f | sort)

CHART_GPU_BUILD_REPORT="${TMPDIR:-/tmp}/chart-preflight-gpu-build-report.json" \
  scripts/build_gpu_images.sh --dry-run >/dev/null

if [[ "${CHART_PREFLIGHT_DOCKER:-0}" == "1" ]]; then
  need_cmd docker
  docker info >/dev/null
fi

uv run --extra search --extra eval --extra test pytest
python -m compileall chart_common indexer search functions eval smoke tests
node --check src/worker.js
npx wrangler@4.104.0 deploy --dry-run --outdir /tmp/chart-worker-dry-run
CHART_DEPLOY_APPLY_REPORT="${TMPDIR:-/tmp}/chart-preflight-deploy-apply-report.json" \
  scripts/deploy_apply.sh --dry-run
CHART_PLAN_AUDIT_REPORT="${TMPDIR:-/tmp}/chart-preflight-plan-audit-report.json" \
  scripts/plan_audit.sh --requirements

if [[ "${CHART_PREFLIGHT_LIVE:-0}" == "1" ]]; then
  scripts/smoke_live.sh
  scripts/gate_report.sh
fi

echo "preflight passed"
