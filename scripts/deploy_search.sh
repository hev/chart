#!/usr/bin/env bash
# Build + push the chart serving image and (re)deploy the in-cluster Helm release.
#
# The serving layer is one FastAPI process (search/app.py) that serves the static
# UI, the /api/* routes, and embeds queries in-process with Arctic v1.5. The
# ingestion side (Pipelines/Function/Warehouse/VectorStore + the chart-gateway
# Secret) is owned by the Layer operator and is NOT touched here.
#
# Prereqs: AWS creds (AWS_PROFILE=mesh-916) for the ECR push, kubectl context =
# layer-prod, helm, and either depot or docker. TLS is the wildcard
# *.hevlayer.com ACM cert already on the shared "hev-public" ALB, auto-discovered
# from the host rule — no cert ARN required. Set CERT_ARN only to pin one.
#
# Usage:
#   AWS_PROFILE=mesh-916 scripts/deploy_search.sh
#   BUILD=0 scripts/deploy_search.sh                     # skip build, just helm upgrade
set -euo pipefail

cd "$(dirname "$0")/.."

RELEASE="${RELEASE:-chart}"
NAMESPACE="${NAMESPACE:-chart}"
IMAGE_TAG="${IMAGE_TAG:-chart-search-20260702b}"
ECR_REPOSITORY="${ECR_REPOSITORY:-186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh}"
IMAGE="${ECR_REPOSITORY}:${IMAGE_TAG}"
BUILD="${BUILD:-1}"
BUILDER="${BUILDER:-depot}"
AWS_REGION="${AWS_REGION:-us-east-1}"
DEPOT_PROJECT_ID="${DEPOT_PROJECT_ID:-8zfcn2cf80}"
LAYER_CLIENT_CONTEXT="${LAYER_CLIENT_CONTEXT:-../layer/clients/python}"
CERT_ARN="${CERT_ARN:-}"
INGRESS_HOST="${INGRESS_HOST:-chart.hevlayer.com}"

log() { echo "==> $*"; }
err() { echo "ERROR: $*" >&2; exit 1; }
require_tool() { command -v "$1" >/dev/null 2>&1 || err "Missing required tool: $1"; }

require_tool kubectl
require_tool helm
[[ -d "$LAYER_CLIENT_CONTEXT" ]] || err "LAYER_CLIENT_CONTEXT not found: ${LAYER_CLIENT_CONTEXT}"

if [[ "$BUILD" == "1" ]]; then
  require_tool aws
  log "Logging in to ECR ${ECR_REPOSITORY%/*}..."
  aws ecr get-login-password --region "$AWS_REGION" \
    | docker login --username AWS --password-stdin "${ECR_REPOSITORY%/*}"

  log "Building and pushing ${IMAGE} (builder=${BUILDER})..."
  case "$BUILDER" in
    depot)
      require_tool depot
      DEPOT_PROJECT_ID="$DEPOT_PROJECT_ID" DEPOT_DISABLE_OTEL=1 depot build \
        --project "$DEPOT_PROJECT_ID" \
        --platform linux/amd64 \
        --build-context "layer_client=${LAYER_CLIENT_CONTEXT}" \
        -f search/Dockerfile -t "$IMAGE" --push .
      ;;
    docker)
      require_tool docker
      docker buildx build \
        --platform linux/amd64 \
        --build-context "layer_client=${LAYER_CLIENT_CONTEXT}" \
        -f search/Dockerfile -t "$IMAGE" --push .
      ;;
    *) err "Unsupported BUILDER=${BUILDER}; use depot or docker." ;;
  esac
fi

log "Deploying ${RELEASE} with Helm..."
helm_set=(
  --set "image.repository=${ECR_REPOSITORY}"
  --set "image.tag=${IMAGE_TAG}"
  --set "ingress.host=${INGRESS_HOST}"
)
[[ -n "$CERT_ARN" ]] && helm_set+=(--set "ingress.certificateArn=${CERT_ARN}")
helm upgrade --install "$RELEASE" ./helm/chart \
  --namespace "$NAMESPACE" \
  "${helm_set[@]}"

log "Waiting for rollout..."
kubectl rollout status "deployment/${RELEASE}-search" -n "$NAMESPACE" --timeout=300s

log "Done. ALB hostname (point ${INGRESS_HOST} at this, DNS-only CNAME):"
kubectl get ingress -n "$NAMESPACE" "$RELEASE" \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}{"\n"}' || true
log "Checks:"
log "  kubectl get pods -n ${NAMESPACE} -l app.kubernetes.io/component=search"
log "  kubectl port-forward -n ${NAMESPACE} svc/${RELEASE}-search 18080:8080"
