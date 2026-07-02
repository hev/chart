#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="client-dry-run"
if [[ "${1:-}" == "--apply" ]]; then
  MODE="apply"
  shift
elif [[ "${1:-}" == "--server-dry-run" ]]; then
  MODE="server-dry-run"
  shift
elif [[ "${1:-}" == "--dry-run" ]]; then
  shift
fi

if [[ $# -gt 0 ]]; then
  echo "usage: scripts/deploy_apply.sh [--dry-run|--server-dry-run|--apply]" >&2
  exit 2
fi

manifests=(
  deploy/namespace.yaml
  deploy/vectorstore.yaml
  deploy/warehouse.yaml
  deploy/pipeline.yaml
  deploy/pipeline-embed.yaml
  deploy/index.yaml
  deploy/functions-events.yaml
)

runtime_manifests=(
  deploy/vectorstore.yaml
  deploy/warehouse.yaml
  deploy/pipeline.yaml
  deploy/pipeline-embed.yaml
  deploy/index.yaml
)
classify_report="${CHART_PHASE4_CLASSIFY_REPORT:-eval/out/classify-events-budget.json}"
deploy_report="${CHART_DEPLOY_APPLY_REPORT:-eval/out/deploy-apply-report.json}"
namespace="${CHART_K8S_NAMESPACE:-chart}"
kube_context=""
kube_context_confirmed="false"
render_dir=""

cleanup() {
  if [[ -n "$render_dir" ]]; then
    rm -rf "$render_dir"
  fi
}
trap cleanup EXIT

render_manifest() {
  local manifest="$1"
  local image_var=""
  local default_image=""
  if [[ "$manifest" == "deploy/pipeline-embed.yaml" && -n "${CHART_EMBED_IMAGE:-}" ]]; then
    image_var="$CHART_EMBED_IMAGE"
    default_image="186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-embedder-plan-20260626-batchdocs1"
  elif [[ "$manifest" == "deploy/functions-events.yaml" && -n "${CHART_CLASSIFIER_IMAGE:-}" ]]; then
    image_var="$CHART_CLASSIFIER_IMAGE"
    default_image="186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-classifier-plan-20260624"
  else
    printf '%s\n' "$manifest"
    return 0
  fi
  if [[ -z "$render_dir" ]]; then
    render_dir="$(mktemp -d)"
  fi
  local rendered="$render_dir/${manifest#deploy/}"
  python3 - "$manifest" "$rendered" "$default_image" "$image_var" <<'PY'
from pathlib import Path
import sys

src, dst, old, new = sys.argv[1:5]
text = Path(src).read_text()
if old not in text:
    raise SystemExit(f"{src}: expected image {old!r} not found")
Path(dst).write_text(text.replace(old, new))
PY
  printf '%s\n' "$rendered"
}

kubectl_apply_manifest() {
  local manifest="$1"
  shift
  local rendered
  rendered="$(render_manifest "$manifest")"
  kubectl apply "$@" -f "$rendered"
}

apply_or_fail() {
  local manifest="$1"
  shift
  if ! kubectl_apply_manifest "$manifest" "$@"; then
    fail_report "failed to apply $manifest in $MODE mode"
  fi
}

write_report() {
  local status="$1"
  local classifier_status="${2:-not-requested}"
  local error="${3:-}"
  local classifier_cost_accepted="false"
  local report_tmp="${deploy_report}.tmp.$$"
  if [[ "${CHART_ACCEPT_PHASE4_CLASSIFY_COST:-}" == "1" ]]; then
    classifier_cost_accepted="true"
  fi
  mkdir -p "$(dirname "$deploy_report")"
  python3 - "$MODE" "$status" "$namespace" "$classifier_status" "$classify_report" "$error" "$kube_context" "$kube_context_confirmed" "$classifier_cost_accepted" "${manifests[@]}" -- "${runtime_manifests[@]}" >"$report_tmp" <<'PY'
import json
import sys

mode, status, namespace, classifier, classifier_report, error, kube_context, kube_context_confirmed, classifier_cost_accepted = sys.argv[1:10]
separator = sys.argv.index("--", 10)
report = {
    "mode": mode,
    "status": status,
    "namespace": namespace,
    "classifier": classifier,
    "classifier_report": classifier_report,
    "kube_context": kube_context or None,
    "kube_context_confirmed": kube_context_confirmed == "true",
    "classifier_cost_accepted": classifier_cost_accepted == "true",
    "manifests": sys.argv[10:separator],
    "runtime_manifests": sys.argv[separator + 1 :],
}
if error:
    report["error"] = error
print(json.dumps(report, indent=2))
PY
  mv "$report_tmp" "$deploy_report"
}

fail_report() {
  local message="$1"
  local classifier_status="${2:-pending}"
  write_report "failed" "$classifier_status" "$message"
  echo "$message" >&2
  exit 1
}

require_secret() {
  local name="$1"
  if ! kubectl -n "$namespace" get secret "$name" >/dev/null 2>&1; then
    fail_report "missing required secret $namespace/$name; see deploy/secrets.example.yaml"
  fi
}

require_kube_context_confirm() {
  kube_context="$(kubectl config current-context 2>/dev/null || true)"
  if [[ -z "$kube_context" ]]; then
    fail_report "refusing to apply without an active kubectl context"
  fi
  if [[ "${CHART_K8S_CONTEXT_CONFIRM:-}" != "$kube_context" ]]; then
    fail_report "refusing to apply to Kubernetes context $kube_context without CHART_K8S_CONTEXT_CONFIRM=$kube_context"
  fi
  kube_context_confirmed="true"
}

require_classifier_report() {
  if [[ ! -f "$classify_report" ]]; then
    fail_report "refusing to apply deploy/functions-events.yaml without classifier report: $classify_report. Run functions.measure_classify_events with accepted --max-full-hours/--max-full-usd budgets first."
  fi
  local classifier_error
  if ! classifier_error="$(python3 - "$classify_report" 2>&1 <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    report = json.loads(path.read_text())
except Exception as exc:
    raise SystemExit(f"invalid classifier report {path}: {exc}") from exc

def positive_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

def positive_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0

def layer_cost_complete(report):
    snapshot = report.get("layer_cost_snapshot") or report.get("cost_snapshot") or {}
    totals = snapshot.get("totals") or {}
    lines = snapshot.get("lines") or []
    return (
        report.get("accepted") is True
        and report.get("source") in {"layer", "layer_cost"}
        and positive_number(snapshot.get("as_of_ms")) > 0
        and positive_number(snapshot.get("window_seconds")) > 0
        and positive_number(totals.get("total_usd")) >= 0
        and any(isinstance(line, dict) and line.get("basis") in {"metered", "invoice"} for line in lines)
    )

failures = []
budget = report.get("budget") or {}
if not layer_cost_complete(report):
    if budget.get("accepted") is not True:
        failures.append("budget.accepted must be true")
    budget_checks = budget.get("checks") or {}
    for name in ("max_full_hours", "max_full_usd"):
        if name not in budget_checks:
            failures.append(f"classifier report is missing required budget check: {name}")
        elif (budget_checks.get(name) or {}).get("ok") is not True:
            failures.append(f"budget.checks.{name}.ok={(budget_checks.get(name) or {}).get('ok')!r}, expected true")
signal = report.get("signal") or {}
if signal.get("accepted") is not True:
    failures.append("classifier signal report is not accepted")
signal_checks = signal.get("checks") or {}
for name in ("min_med_discontinuations", "min_review_examples"):
    if name not in signal_checks:
        failures.append(f"classifier report is missing required signal check: {name}")
    elif (signal_checks.get(name) or {}).get("ok") is not True:
        failures.append(f"signal.checks.{name}.ok={(signal_checks.get(name) or {}).get('ok')!r}, expected true")
sample = report.get("sample") or {}
if positive_int(sample.get("notes")) <= 0:
    failures.append("sample.notes must be positive")
if positive_int(sample.get("med_discontinuation")) <= 0:
    failures.append("sample.med_discontinuation must be positive")
examples = report.get("examples") or []
if not isinstance(examples, list) or not examples:
    failures.append("examples must include at least one review example")
writeback = report.get("writeback") or {}
patched_fields = set(writeback.get("patched_fields") or [])
required_fields = {
    "events",
    "has_med_discontinuation",
    "has_adverse_event",
    "diagnosis_category",
    "specialty",
    "discontinuation_reason",
}
if writeback.get("mode") != "tpuf.patch_columns":
    failures.append("writeback.mode must be tpuf.patch_columns")
if writeback.get("primary_output") != "events":
    failures.append("writeback.primary_output must be events")
if positive_int(writeback.get("model_passes_per_note")) != 1:
    failures.append("writeback.model_passes_per_note must be 1")
if writeback.get("settles_multi_write") is not True:
    failures.append("writeback.settles_multi_write must be true")
missing_fields = sorted(required_fields - patched_fields)
if missing_fields:
    failures.append(f"writeback.patched_fields missing: {', '.join(missing_fields)}")
if failures:
    raise SystemExit("; ".join(failures))
PY
  )"; then
    fail_report "$classifier_error"
  fi
}

if ! command -v kubectl >/dev/null 2>&1; then
  fail_report "missing required command: kubectl"
fi

if [[ "$MODE" == "client-dry-run" ]]; then
  write_report "started" "validated"
  for manifest in "${manifests[@]}"; do
    apply_or_fail "$manifest" --dry-run=client --validate=false
  done
  write_report "completed" "validated"
  echo "deploy client dry-run passed"
elif [[ "$MODE" == "server-dry-run" ]]; then
  write_report "started" "validated"
  for manifest in "${manifests[@]}"; do
    apply_or_fail "$manifest" --dry-run=server
  done
  write_report "completed" "validated"
  echo "deploy server dry-run passed"
else
  write_report "started" "pending"
  require_kube_context_confirm
  apply_or_fail deploy/namespace.yaml
  require_secret chart-hevsearch
  require_secret chart-layer-inbound
  require_secret chart-gateway
  for manifest in "${runtime_manifests[@]}"; do
    apply_or_fail "$manifest"
  done
  if [[ "${CHART_APPLY_CLASSIFIER:-}" == "1" ]]; then
    if [[ "${CHART_ACCEPT_PHASE4_CLASSIFY_COST:-}" != "1" ]]; then
      fail_report "refusing to apply deploy/functions-events.yaml without CHART_ACCEPT_PHASE4_CLASSIFY_COST=1"
    fi
    require_classifier_report
    apply_or_fail deploy/functions-events.yaml
    write_report "completed" "applied"
  else
    apply_or_fail deploy/functions-events.yaml --dry-run=client --validate=false
    write_report "completed" "validated-skipped"
    echo "skipped deploy/functions-events.yaml; set CHART_APPLY_CLASSIFIER=1 and CHART_ACCEPT_PHASE4_CLASSIFY_COST=1 after the Phase-4 cost gate to apply it"
  fi
  echo "deploy applied"
fi
