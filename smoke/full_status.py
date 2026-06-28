from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from hevlayer.client import HevlayerError
from pydantic import ValidationError

from chart_common.config import FULL_CORPUS_NOTES, Settings
from chart_common.gateway import FACET_FIELDS, close_client, latest_facets, make_client, require_gateway_key

DEFAULT_UDF_ID = "chart-classify-events"
DEFAULT_PIPELINE_ID = "chart-notes"
DEFAULT_EMBED_PIPELINE_CR = "chart-embed-gpu"
DEFAULT_EMBED_IMAGE = "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-embedder-plan-20260624-dedupe2"
DEFAULT_CLASSIFIER_IMAGE = "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-classifier-plan-20260624"
DEFAULT_GPU_COMPUTE_CLASS = "gpu"
DEFAULT_EMBED_BUDGET_REPORT = "eval/out/embed-budget.json"
DEFAULT_CLASSIFY_BUDGET_REPORT = "eval/out/classify-events-budget.json"
DEFAULT_K8S_NAMESPACE = "chart"
DEFAULT_INGEST_DEPLOYMENT = "chart-ingest-worker"
DEFAULT_EMBED_DEPLOYMENT = "chart-embed-gpu-worker"
DEFAULT_CLASSIFIER_DEPLOYMENT = "chart-classify-events-worker"
DEFAULT_CLASSIFIER_FUNCTION = "chart-classify-events"


def phase6_targets(settings: Settings, *, pipeline_id: str, udf_id: str) -> dict[str, Any]:
    return {
        "namespace": settings.namespace,
        "pipeline_id": pipeline_id,
        "udf_id": udf_id,
        "full_corpus_notes": FULL_CORPUS_NOTES,
        "embed_pipeline_cr": DEFAULT_EMBED_PIPELINE_CR,
        "embed_compute_class": DEFAULT_GPU_COMPUTE_CLASS,
        "embed_image": DEFAULT_EMBED_IMAGE,
        "classifier_image": DEFAULT_CLASSIFIER_IMAGE,
        "classifier_compute_class": DEFAULT_GPU_COMPUTE_CLASS,
    }


def _dump(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def _status_body(value: Any, *, id_field: str, expected_id: str) -> dict[str, Any]:
    dumped = _dump(value)
    if isinstance(dumped, dict) and dumped:
        return dumped
    return {
        id_field: expected_id,
        "error": {"status_code": None, "message": "status body is empty"},
    }


async def pipeline_status(layer, pipeline_id: str) -> dict[str, Any]:
    try:
        status = await layer.get_pipeline_status(pipeline_id)
    except HevlayerError as exc:
        return {"pipeline_id": pipeline_id, "error": {"status_code": exc.status_code, "message": exc.message}}
    except ValidationError as exc:
        raw = await _raw_pipeline_status(layer, pipeline_id)
        if raw:
            raw.setdefault("pipeline_id", pipeline_id)
            raw["schema_error"] = str(exc)
            return raw
        return {
            "pipeline_id": pipeline_id,
            "error": {
                "status_code": None,
                "message": f"pipeline status response did not match client schema: {exc}",
            },
        }
    return _status_body(status, id_field="pipeline_id", expected_id=pipeline_id)


async def _raw_pipeline_status(layer, pipeline_id: str) -> dict[str, Any] | None:
    client = getattr(layer, "_client", None)
    if client is None:
        return None
    try:
        response = await client.get(f"/v2/pipelines/{pipeline_id}/status")
        response.raise_for_status()
        data = response.json()
    except Exception:
        return None
    return data if isinstance(data, dict) else None


async def udf_status(layer, udf_id: str) -> dict[str, Any]:
    try:
        status = await layer.get_udf_status(udf_id)
    except HevlayerError as exc:
        return {"udf_id": udf_id, "error": {"status_code": exc.status_code, "message": exc.message}}
    return _status_body(status, id_field="udf_id", expected_id=udf_id)


async def facet_status(layer, namespace: str) -> dict[str, Any]:
    out = {}
    for field in FACET_FIELDS:
        try:
            values, provenance = await latest_facets(layer, namespace, field=field)
        except HevlayerError as exc:
            out[field] = {
                "values": 0,
                "sha": None,
                "row_count": None,
                "error": {"status_code": exc.status_code, "message": exc.message},
            }
            continue
        out[field] = {
            "values": len(values or []),
            "sha": (provenance or {}).get("sha"),
            "row_count": (provenance or {}).get("row_count"),
            "watermark_ms": (provenance or {}).get("watermark_ms"),
        }
    return out


def cost_baseline(path: str) -> dict[str, Any]:
    report_path = Path(path)
    out: dict[str, Any] = {"report": path}
    if not report_path.exists():
        out["accepted"] = False
        out["error"] = "missing"
        return out
    try:
        report = json.loads(report_path.read_text())
    except Exception as exc:
        out["accepted"] = False
        out["error"] = f"invalid json: {exc}"
        return out
    budget = report.get("budget") or {}
    source = report.get("source")
    layer_snapshot = report.get("layer_cost_snapshot") or report.get("cost_snapshot")
    out["accepted"] = budget.get("accepted") is True or (
        report.get("accepted") is True and source in {"layer", "layer_cost"} and isinstance(layer_snapshot, dict)
    )
    out["estimate"] = report.get("estimate") or {}
    if source:
        out["source"] = source
    if isinstance(layer_snapshot, dict):
        out["layer_cost_snapshot"] = layer_snapshot
    return out


def cost_baselines() -> dict[str, Any]:
    embed_report = os.environ.get("CHART_PHASE6_EMBED_BUDGET_REPORT", DEFAULT_EMBED_BUDGET_REPORT)
    classify_report = os.environ.get("CHART_PHASE4_CLASSIFY_REPORT", DEFAULT_CLASSIFY_BUDGET_REPORT)
    return {
        "embed": cost_baseline(embed_report),
        "classifier": cost_baseline(classify_report),
    }


def _kubectl_json(args: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        ["kubectl", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout or "{}")


def _container_gpu_request(container: dict[str, Any]) -> str | None:
    resources = container.get("resources") or {}
    requests = resources.get("requests") or {}
    limits = resources.get("limits") or {}
    value = requests.get("nvidia.com/gpu") or limits.get("nvidia.com/gpu")
    return str(value) if value is not None else None


def _pod_summary(pod: dict[str, Any]) -> dict[str, Any]:
    metadata = pod.get("metadata") or {}
    spec = pod.get("spec") or {}
    status = pod.get("status") or {}
    containers = spec.get("containers") or []
    waiting = []
    for container_status in status.get("containerStatuses") or []:
        state = container_status.get("state") or {}
        if state.get("waiting"):
            waiting.append(
                {
                    "name": container_status.get("name"),
                    "reason": (state.get("waiting") or {}).get("reason"),
                    "message": (state.get("waiting") or {}).get("message"),
                }
            )
    return {
        "namespace": metadata.get("namespace"),
        "name": metadata.get("name"),
        "phase": status.get("phase"),
        "node": spec.get("nodeName"),
        "nominated_node": status.get("nominatedNodeName"),
        "conditions": [
            {
                "type": condition.get("type"),
                "status": condition.get("status"),
                "reason": condition.get("reason"),
                "message": condition.get("message"),
            }
            for condition in status.get("conditions") or []
        ],
        "waiting": waiting,
        "gpu": next((_container_gpu_request(container) for container in containers if _container_gpu_request(container)), None),
        "image": containers[0].get("image") if containers else None,
    }


def _pod_event_messages(namespace: str, pod_name: str) -> list[str]:
    try:
        events = _kubectl_json(
            [
                "-n",
                namespace,
                "get",
                "events",
                "--field-selector",
                f"involvedObject.kind=Pod,involvedObject.name={pod_name}",
                "-o",
                "json",
            ]
        )
    except Exception:
        return []
    messages = []
    for item in events.get("items") or []:
        message = item.get("message")
        if message:
            messages.append(message)
    return messages[-8:]


def _function_summary(function: dict[str, Any]) -> dict[str, Any]:
    metadata = function.get("metadata") or {}
    spec = function.get("spec") or {}
    status = function.get("status") or {}
    worker = spec.get("worker") or {}
    scaling = spec.get("scaling") or {}
    return {
        "name": metadata.get("name"),
        "paused": spec.get("paused", False),
        "version": spec.get("version"),
        "target_namespaces": spec.get("targetNamespaces") or [],
        "image": worker.get("image"),
        "compute_class": worker.get("computeClass"),
        "scaling": {
            "mode": scaling.get("mode"),
            "pool": scaling.get("pool"),
            "replicas": scaling.get("replicas") or {},
        },
        "active_namespaces": status.get("activeNamespaces") or [],
        "queue_depth": status.get("queueDepth"),
        "conditions": [
            {
                "type": condition.get("type"),
                "status": condition.get("status"),
                "reason": condition.get("reason"),
                "message": condition.get("message"),
            }
            for condition in status.get("conditions") or []
        ],
    }


def _scaled_object_summary(scaled_object: dict[str, Any]) -> dict[str, Any]:
    metadata = scaled_object.get("metadata") or {}
    spec = scaled_object.get("spec") or {}
    status = scaled_object.get("status") or {}
    scale_target = spec.get("scaleTargetRef") or {}
    triggers = []
    for trigger in spec.get("triggers") or []:
        trigger_metadata = trigger.get("metadata") or {}
        auth_ref = trigger.get("authenticationRef") or {}
        triggers.append(
            {
                "type": trigger.get("type"),
                "authentication_ref": auth_ref.get("name"),
                "auth_modes": trigger_metadata.get("authModes"),
                "metric_name": trigger_metadata.get("metricName"),
                "threshold": trigger_metadata.get("threshold"),
                "server_address": trigger_metadata.get("serverAddress"),
            }
        )
    return {
        "name": metadata.get("name"),
        "scale_target": scale_target.get("name"),
        "min_replicas": spec.get("minReplicaCount"),
        "max_replicas": spec.get("maxReplicaCount"),
        "triggers": triggers,
        "conditions": [
            {
                "type": condition.get("type"),
                "status": condition.get("status"),
                "reason": condition.get("reason"),
                "message": condition.get("message"),
            }
            for condition in status.get("conditions") or []
        ],
    }


def _secret_ref_status(namespace: str, name: str | None, key: str | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "secret_exists": False,
        "key_exists": False,
        "value_present": False,
    }
    if not name or not key:
        out["error"] = "secret name or key missing"
        return out
    try:
        secret = _kubectl_json(["-n", namespace, "get", "secret", name, "-o", "json"])
    except Exception as exc:
        out["error"] = str(exc)
        return out
    data = secret.get("data") or {}
    value = data.get(key)
    out["secret_exists"] = True
    out["key_exists"] = key in data
    out["value_present"] = bool(value)
    return out


def _trigger_authentication_summary(namespace: str, trigger_authentication: dict[str, Any]) -> dict[str, Any]:
    metadata = trigger_authentication.get("metadata") or {}
    spec = trigger_authentication.get("spec") or {}
    status = trigger_authentication.get("status") or {}
    return {
        "name": metadata.get("name"),
        "secret_target_refs": [
            {
                "parameter": ref.get("parameter"),
                "name": ref.get("name"),
                "key": ref.get("key"),
                "status": _secret_ref_status(namespace, ref.get("name"), ref.get("key")),
            }
            for ref in spec.get("secretTargetRef") or []
        ],
        "scaled_objects": status.get("scaledobjects"),
    }


def k8s_runtime_status() -> dict[str, Any]:
    namespace = os.environ.get("CHART_K8S_NAMESPACE", DEFAULT_K8S_NAMESPACE)
    embed_deployment = os.environ.get("CHART_EMBED_DEPLOYMENT", DEFAULT_EMBED_DEPLOYMENT)
    ingest_deployment = os.environ.get("CHART_INGEST_DEPLOYMENT", DEFAULT_INGEST_DEPLOYMENT)
    classifier_deployment = os.environ.get("CHART_CLASSIFIER_DEPLOYMENT", DEFAULT_CLASSIFIER_DEPLOYMENT)
    classifier_function = os.environ.get("CHART_CLASSIFIER_FUNCTION", DEFAULT_CLASSIFIER_FUNCTION)
    out: dict[str, Any] = {
        "namespace": namespace,
        "deployments": {
            "ingest": ingest_deployment,
            "embed": embed_deployment,
            "classifier": classifier_deployment,
        },
        "functions": {
            "classifier": classifier_function,
        },
    }
    try:
        deployments = _kubectl_json(
            [
                "-n",
                namespace,
                "get",
                "deployment",
                ingest_deployment,
                embed_deployment,
                classifier_deployment,
                "-o",
                "json",
            ]
        )
        out["deployment_status"] = {
            item.get("metadata", {}).get("name"): {
                "replicas": (item.get("spec") or {}).get("replicas"),
                "ready_replicas": (item.get("status") or {}).get("readyReplicas", 0),
                "image": (((item.get("spec") or {}).get("template") or {}).get("spec") or {}).get("containers", [{}])[0].get("image"),
            }
            for item in deployments.get("items") or []
        }
    except FileNotFoundError:
        out["error"] = "kubectl not found"
        return out
    except Exception as exc:
        out["deployment_error"] = str(exc)

    try:
        function = _kubectl_json(["-n", namespace, "get", "function", classifier_function, "-o", "json"])
        out["function_status"] = {
            classifier_function: _function_summary(function),
        }
    except Exception as exc:
        out["function_error"] = str(exc)

    try:
        scaled_objects = _kubectl_json(
            [
                "-n",
                namespace,
                "get",
                "scaledobject",
                ingest_deployment,
                embed_deployment,
                "-o",
                "json",
            ]
        )
        out["scaled_objects"] = {
            item.get("metadata", {}).get("name"): _scaled_object_summary(item)
            for item in scaled_objects.get("items") or []
        }
    except Exception as exc:
        out["scaled_objects_error"] = str(exc)

    try:
        trigger_authentications = _kubectl_json(
            [
                "-n",
                namespace,
                "get",
                "triggerauthentication",
                ingest_deployment,
                embed_deployment,
                "-o",
                "json",
            ]
        )
        out["trigger_authentications"] = {
            item.get("metadata", {}).get("name"): _trigger_authentication_summary(namespace, item)
            for item in trigger_authentications.get("items") or []
        }
    except Exception as exc:
        out["trigger_authentications_error"] = str(exc)

    try:
        embed_pods = _kubectl_json(
            [
                "-n",
                namespace,
                "get",
                "pods",
                "-l",
                "hevlayer.com/pipeline=chart-embed-gpu",
                "-o",
                "json",
            ]
        )
        summaries = [_pod_summary(item) for item in embed_pods.get("items") or []]
        for summary in summaries:
            if summary.get("namespace") and summary.get("name"):
                summary["events"] = _pod_event_messages(str(summary["namespace"]), str(summary["name"]))
        out["embed_pods"] = summaries
    except Exception as exc:
        out["embed_pods_error"] = str(exc)

    try:
        all_pods = _kubectl_json(["get", "pods", "-A", "-o", "json"])
        gpu_pods = []
        for pod in all_pods.get("items") or []:
            summary = _pod_summary(pod)
            if summary.get("gpu"):
                gpu_pods.append(summary)
        out["gpu_pods"] = gpu_pods
    except Exception as exc:
        out["gpu_pods_error"] = str(exc)
    return out


async def collect_status(*, pipeline_id: str | None = None, udf_id: str = DEFAULT_UDF_ID) -> dict[str, Any]:
    settings = Settings()
    require_gateway_key(settings)
    pipeline_id = pipeline_id or DEFAULT_PIPELINE_ID
    layer = make_client(settings)
    try:
        pipeline, udf, facets = await asyncio.gather(
            pipeline_status(layer, pipeline_id),
            udf_status(layer, udf_id),
            facet_status(layer, settings.namespace),
        )
    finally:
        await close_client(layer)
    return {
        "namespace": settings.namespace,
        "targets": phase6_targets(settings, pipeline_id=pipeline_id, udf_id=udf_id),
        "cost_baselines": cost_baselines(),
        "pipeline": pipeline,
        "udf": udf,
        "facets": facets,
        "kubernetes": k8s_runtime_status(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-run status for chart Phase 6")
    parser.add_argument("--pipeline-id", default=None)
    parser.add_argument("--udf-id", default=DEFAULT_UDF_ID)
    parser.add_argument("--out", type=Path, default=None, help="write the full status report to this JSON path")
    args = parser.parse_args()
    report = asyncio.run(collect_status(pipeline_id=args.pipeline_id, udf_id=args.udf_id))
    rendered = json.dumps(report, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
