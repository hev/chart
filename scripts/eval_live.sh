#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source scripts/lib/resolve_gateway_key.sh
source scripts/lib/write_failure_report.sh

LIMIT="${CHART_EVAL_LIMIT:-500}"
TOP_K="${CHART_EVAL_TOP_K:-1000}"
PROGRESS_EVERY="${CHART_EVAL_PROGRESS_EVERY:-50}"
RECDS_OUT="${CHART_EVAL_RECDS_REPORT:-eval/out/recds-report.json}"
BIMODAL_OUT="${CHART_EVAL_BIMODAL_REPORT:-eval/out/bimodal-report.json}"
HOLDOUT_OUT="${CHART_EVAL_HOLDOUT_REPORT:-eval/out/holdout-report.json}"
FAIL_ARG=()
if [[ "${CHART_EVAL_REQUIRE_NO_FAILURES:-0}" == "1" ]]; then
  FAIL_ARG=(--require-no-failures)
fi
DOMINANCE_ARG=()
if [[ "${CHART_EVAL_REQUIRE_FUSED_DOMINATES:-0}" == "1" ]]; then
  DOMINANCE_ARG=(--require-fused-dominates)
fi
if [[ -n "${CHART_EVAL_HOLDOUT_MAX_OVERLAP:-}" ]]; then
  uv run --extra eval python -m eval.holdout \
    --split "${CHART_EVAL_HOLDOUT_SPLIT:-dev}" \
    --examples "${CHART_EVAL_HOLDOUT_EXAMPLES:-10}" \
    --out "$HOLDOUT_OUT" \
    --max-overlap-edges "$CHART_EVAL_HOLDOUT_MAX_OVERLAP"
fi

if ! resolve_gateway_key; then
  write_failure_report "$RECDS_OUT" "failed" "LAYER_GATEWAY_API_KEY is required"
  exit 1
fi

recds_cmd=(uv run --extra eval python -m eval.recds \
  --task ppr \
  --strategies "${CHART_EVAL_STRATEGIES:-auto,semantic,bm25,fused}" \
  --limit "$LIMIT" \
  --top-k "$TOP_K" \
  --progress-every "$PROGRESS_EVERY" \
  --out "$RECDS_OUT")
if [[ "${#FAIL_ARG[@]}" -gt 0 ]]; then
  recds_cmd+=("${FAIL_ARG[@]}")
fi
if [[ "${#DOMINANCE_ARG[@]}" -gt 0 ]]; then
  recds_cmd+=("${DOMINANCE_ARG[@]}")
fi
rm -f "$RECDS_OUT"
if ! "${recds_cmd[@]}"; then
  if [[ ! -s "$RECDS_OUT" ]]; then
    write_failure_report "$RECDS_OUT" "failed" "ReCDS eval command failed"
  fi
  exit 1
fi

if [[ -d eval/out/bimodal ]]; then
  bimodal_cmd=(uv run --extra eval python -m eval.recds \
    --beir-dir eval/out/bimodal \
    --strategies "${CHART_EVAL_STRATEGIES:-auto,semantic,bm25,fused}" \
    --limit "$LIMIT" \
    --top-k "$TOP_K" \
    --progress-every "$PROGRESS_EVERY" \
    --out "$BIMODAL_OUT")
  if [[ "${#FAIL_ARG[@]}" -gt 0 ]]; then
    bimodal_cmd+=("${FAIL_ARG[@]}")
  fi
  if [[ "${#DOMINANCE_ARG[@]}" -gt 0 ]]; then
    bimodal_cmd+=("${DOMINANCE_ARG[@]}")
  fi
  rm -f "$BIMODAL_OUT"
  if ! "${bimodal_cmd[@]}"; then
    if [[ ! -s "$BIMODAL_OUT" ]]; then
      write_failure_report "$BIMODAL_OUT" "failed" "bimodal eval command failed"
    fi
    exit 1
  fi
fi
