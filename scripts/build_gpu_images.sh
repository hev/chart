#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="dry-run"
if [[ "${1:-}" == "--build" ]]; then
  MODE="build"
  shift
elif [[ "${1:-}" == "--push" ]]; then
  MODE="push"
  shift
elif [[ "${1:-}" == "--dry-run" ]]; then
  shift
fi

if [[ $# -gt 0 ]]; then
  echo "usage: scripts/build_gpu_images.sh [--dry-run|--build|--push]" >&2
  exit 2
fi

default_embed_tag="186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-embedder-plan-20260626-batchdocs1"
default_source_tag="186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-huggingface-source-plan-20260624-concurrent"
default_classifier_tag="186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-classifier-plan-20260624"
if [[ -n "${CHART_ECR_REPOSITORY_URL:-}" ]]; then
  default_embed_tag="${CHART_ECR_REPOSITORY_URL}:chart-embedder-plan-20260626-batchdocs1"
  default_source_tag="${CHART_ECR_REPOSITORY_URL}:chart-huggingface-source-plan-20260624-concurrent"
  default_classifier_tag="${CHART_ECR_REPOSITORY_URL}:chart-classifier-plan-20260624"
fi

embed_tag="${CHART_EMBED_IMAGE:-$default_embed_tag}"
source_tag="${CHART_SOURCE_IMAGE:-$default_source_tag}"
classifier_tag="${CHART_CLASSIFIER_IMAGE:-$default_classifier_tag}"
layer_context="${CHART_LAYER_CLIENT_CONTEXT:-../layer/clients/python}"
build_report="${CHART_GPU_BUILD_REPORT:-eval/out/gpu-build-report.json}"
platform="${CHART_GPU_PLATFORM:-linux/amd64}"
builder="${CHART_GPU_BUILDER:-docker}"
depot_project="${CHART_DEPOT_PROJECT_ID:-8zfcn2cf80}"
build_targets="${CHART_GPU_BUILD_TARGETS:-embed,source,classifier}"
preload_embed_model="${CHART_PRELOAD_EMBED_MODEL-Snowflake/snowflake-arctic-embed-m-v1.5}"
preload_events_model="${CHART_PRELOAD_EVENTS_MODEL-google/gemma-2-9b-it}"

if [[ "$builder" == "depot" ]]; then
  cmd_embed=(
    depot build
    --project "$depot_project"
    -f deploy/Dockerfile.gpu
    --target embedder
    --platform "$platform"
    --build-context "layer_client=${layer_context}"
    --build-arg "PRELOAD_EMBED_MODEL=${preload_embed_model}"
    -t "$embed_tag"
  )
  cmd_source=(
    depot build
    --project "$depot_project"
    -f deploy/Dockerfile.gpu
    --target huggingface-source
    --platform "$platform"
    --build-context "layer_client=${layer_context}"
    -t "$source_tag"
  )
  cmd_classifier=(
    depot build
    --project "$depot_project"
    -f deploy/Dockerfile.gpu
    --target classifier
    --platform "$platform"
    --build-context "layer_client=${layer_context}"
    --build-arg "PRELOAD_EVENTS_MODEL=${preload_events_model}"
    -t "$classifier_tag"
  )
elif [[ "$builder" == "docker" ]]; then
  cmd_embed=(
    docker buildx build
    -f deploy/Dockerfile.gpu
    --target embedder
    --platform "$platform"
    --build-context "layer_client=${layer_context}"
    --build-arg "PRELOAD_EMBED_MODEL=${preload_embed_model}"
    -t "$embed_tag"
  )
  cmd_source=(
    docker buildx build
    -f deploy/Dockerfile.gpu
    --target huggingface-source
    --platform "$platform"
    --build-context "layer_client=${layer_context}"
    -t "$source_tag"
  )
  cmd_classifier=(
    docker buildx build
    -f deploy/Dockerfile.gpu
    --target classifier
    --platform "$platform"
    --build-context "layer_client=${layer_context}"
    --build-arg "PRELOAD_EVENTS_MODEL=${preload_events_model}"
    -t "$classifier_tag"
  )
else
  echo "CHART_GPU_BUILDER must be 'docker' or 'depot'" >&2
  exit 2
fi

if [[ "$MODE" == "push" ]]; then
  cmd_embed+=(--push)
  cmd_source+=(--push)
  cmd_classifier+=(--push)
elif [[ "$MODE" == "build" ]]; then
  cmd_embed+=(--load)
  cmd_source+=(--load)
  cmd_classifier+=(--load)
fi

cmd_embed+=(.)
cmd_source+=(.)
cmd_classifier+=(.)

print_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

json_string() {
  python3 -c 'import json, sys; print(json.dumps(sys.argv[1]))' "$1"
}

json_array() {
  python3 -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "$@"
}

ecr_registry_for() {
  local image="$1"
  local registry="${image%%/*}"
  if [[ "$registry" == *".dkr.ecr."*".amazonaws.com" ]]; then
    printf '%s\n' "$registry"
  fi
}

ecr_login() {
  if [[ "${CHART_ECR_LOGIN:-auto}" == "0" || "${CHART_ECR_LOGIN:-auto}" == "false" ]]; then
    return 0
  fi
  local registry="${CHART_ECR_LOGIN_REGISTRY:-}"
  if [[ -z "$registry" ]]; then
    registry="$(ecr_registry_for "$embed_tag")"
  fi
  if [[ -z "$registry" ]]; then
    registry="$(ecr_registry_for "$source_tag")"
  fi
  if [[ -z "$registry" ]]; then
    registry="$(ecr_registry_for "$classifier_tag")"
  fi
  if [[ -z "$registry" ]]; then
    return 0
  fi
  if ! command -v aws >/dev/null 2>&1; then
    fail_report "missing required command: aws"
  fi
  if ! aws ecr get-login-password --region "${AWS_REGION:-us-east-1}" \
    | docker login --username AWS --password-stdin "$registry"; then
    fail_report "ECR login failed for $registry"
  fi
}

write_report() {
  local status="$1"
  local error="${2:-}"
  local report_tmp="${build_report}.tmp.$$"
  mkdir -p "$(dirname "$build_report")"
  python3 - "$MODE" "$status" "$embed_tag" "$source_tag" "$classifier_tag" "$layer_context" "$platform" "$builder" "$depot_project" "$build_targets" "$preload_embed_model" "$preload_events_model" "$error" "${cmd_embed[@]}" --source "${cmd_source[@]}" --classifier "${cmd_classifier[@]}" >"$report_tmp" <<'PY'
import json
import sys

(
    mode,
    status,
    embed_image,
    source_image,
    classifier_image,
    layer_context,
    platform,
    builder,
    depot_project,
    build_targets,
    preload_embed_model,
    preload_events_model,
    error,
) = sys.argv[1:14]
source_separator = sys.argv.index("--source", 14)
classifier_separator = sys.argv.index("--classifier", source_separator + 1)
targets = [target.strip() for target in build_targets.split(",") if target.strip()]
report = {
    "mode": mode,
    "status": status,
    "targets": targets,
    "embed_image": embed_image,
    "source_image": source_image,
    "classifier_image": classifier_image,
    "layer_context": layer_context,
    "platform": platform,
    "builder": builder,
    "preload_embed_model": preload_embed_model or None,
    "preload_events_model": preload_events_model or None,
    "embed_command": sys.argv[14:source_separator],
    "source_command": sys.argv[source_separator + 1 : classifier_separator],
    "classifier_command": sys.argv[classifier_separator + 1 :],
}
if builder == "depot":
    report["depot_project"] = depot_project
if error:
    report["error"] = error
print(json.dumps(report, indent=2))
PY
  mv "$report_tmp" "$build_report"
}

fail_report() {
  local message="$1"
  write_report "failed" "$message"
  echo "$message" >&2
  exit 1
}

if [[ "$MODE" == "dry-run" ]]; then
  write_report "dry-run"
  print_cmd "${cmd_embed[@]}"
  print_cmd "${cmd_source[@]}"
  print_cmd "${cmd_classifier[@]}"
  exit 0
fi

if [[ "$builder" == "docker" ]] && ! command -v docker >/dev/null 2>&1; then
  fail_report "missing required command: docker"
fi
if [[ "$builder" == "depot" ]] && ! command -v depot >/dev/null 2>&1; then
  fail_report "missing required command: depot"
fi

if [[ ! -d "$layer_context" ]]; then
  fail_report "missing Layer Python client build context: $layer_context. Set CHART_LAYER_CLIENT_CONTEXT to the checkout path for layer/clients/python."
fi

case ",$build_targets," in
  *",embed,"*|*",source,"*|*",classifier,"*) ;;
  *) fail_report "CHART_GPU_BUILD_TARGETS must include embed, source, classifier, or a comma-separated combination" ;;
esac

if [[ "$builder" == "docker" ]] && ! docker info >/dev/null; then
  fail_report "docker daemon is not reachable"
fi
if [[ "$MODE" == "push" ]]; then
  ecr_login
fi
write_report "started"
if [[ ",$build_targets," == *",embed,"* ]]; then
  if ! "${cmd_embed[@]}"; then
    fail_report "embed image build failed"
  fi
fi
if [[ ",$build_targets," == *",source,"* ]]; then
  if ! "${cmd_source[@]}"; then
    fail_report "source image build failed"
  fi
fi
if [[ ",$build_targets," == *",classifier,"* ]]; then
  if ! "${cmd_classifier[@]}"; then
    fail_report "classifier image build failed"
  fi
fi
write_report "completed"
