from types import SimpleNamespace
import json

import pytest

from smoke import full_status
from smoke.full_status import collect_status, cost_baseline, facet_status, k8s_runtime_status, pipeline_status, udf_status


class FakeModel:
    def __init__(self, **values) -> None:
        self.values = values

    def model_dump(self):
        return dict(self.values)


@pytest.mark.anyio
async def test_pipeline_and_udf_status_dump_models() -> None:
    class FakeLayer:
        async def get_pipeline_status(self, pipeline_id):
            return FakeModel(pipeline_id=pipeline_id, pending_count=1)

        async def get_udf_status(self, udf_id):
            return FakeModel(udf_id=udf_id, failed_count=2)

    layer = FakeLayer()

    assert await pipeline_status(layer, "chart-notes") == {"pipeline_id": "chart-notes", "pending_count": 1}
    assert await udf_status(layer, "chart-classify-events") == {
        "udf_id": "chart-classify-events",
        "failed_count": 2,
    }


@pytest.mark.anyio
async def test_pipeline_and_udf_status_report_empty_bodies_as_errors() -> None:
    class FakeLayer:
        async def get_pipeline_status(self, pipeline_id):
            return None

        async def get_udf_status(self, udf_id):
            return {}

    layer = FakeLayer()

    assert await pipeline_status(layer, "chart-notes") == {
        "pipeline_id": "chart-notes",
        "error": {"status_code": None, "message": "status body is empty"},
    }
    assert await udf_status(layer, "chart-classify-events") == {
        "udf_id": "chart-classify-events",
        "error": {"status_code": None, "message": "status body is empty"},
    }


@pytest.mark.anyio
async def test_facet_status_reports_counts_and_provenance(monkeypatch) -> None:
    async def fake_latest_facets(layer, namespace, *, field, limit=14):
        if field == "age_band":
            return [{"value": "adult", "count": 10}], {"sha": "abc", "row_count": 20, "watermark_ms": 123456}
        return None, None

    monkeypatch.setattr("smoke.full_status.latest_facets", fake_latest_facets)

    status = await facet_status(SimpleNamespace(), "chart-notes")

    assert status["age_band"] == {"values": 1, "sha": "abc", "row_count": 20, "watermark_ms": 123456}
    assert status["events"] == {"values": 0, "sha": None, "row_count": None, "watermark_ms": None}


@pytest.mark.anyio
async def test_facet_status_reports_gateway_errors_per_field(monkeypatch) -> None:
    async def fake_latest_facets(layer, namespace, *, field, limit=14):
        if field == "age_band":
            raise full_status.HevlayerError(502, "bad gateway")
        return None, None

    monkeypatch.setattr("smoke.full_status.latest_facets", fake_latest_facets)

    status = await facet_status(SimpleNamespace(), "chart-notes")

    assert status["age_band"] == {
        "values": 0,
        "sha": None,
        "row_count": None,
        "error": {"status_code": 502, "message": "bad gateway"},
    }
    assert status["gender"] == {"values": 0, "sha": None, "row_count": None, "watermark_ms": None}


@pytest.mark.anyio
async def test_collect_status_reports_explicit_phase6_targets(monkeypatch) -> None:
    layer = object()
    calls = []

    monkeypatch.setattr(
        full_status,
        "Settings",
        lambda: SimpleNamespace(namespace="chart-notes", api_key="key"),
    )
    monkeypatch.setattr(full_status, "make_client", lambda settings: layer)

    async def fake_pipeline_status(layer_arg, pipeline_id):
        calls.append(("pipeline", layer_arg, pipeline_id))
        return {"pipeline_id": pipeline_id}

    async def fake_udf_status(layer_arg, udf_id):
        calls.append(("udf", layer_arg, udf_id))
        return {"udf_id": udf_id}

    async def fake_facet_status(layer_arg, namespace):
        calls.append(("facets", layer_arg, namespace))
        return {}

    async def fake_close(layer_arg):
        calls.append(("close", layer_arg))

    monkeypatch.setattr(full_status, "pipeline_status", fake_pipeline_status)
    monkeypatch.setattr(full_status, "udf_status", fake_udf_status)
    monkeypatch.setattr(full_status, "facet_status", fake_facet_status)
    monkeypatch.setattr(
        full_status,
        "cost_baselines",
        lambda: {"embed": {"accepted": True}, "classifier": {"accepted": True}},
    )
    monkeypatch.setattr(full_status, "k8s_runtime_status", lambda: {"namespace": "chart"})
    monkeypatch.setattr(full_status, "close_client", fake_close)

    status = await collect_status()

    assert status["targets"] == {
        "namespace": "chart-notes",
        "pipeline_id": "chart-notes",
        "udf_id": "chart-classify-events",
        "full_corpus_notes": 167000,
        "embed_pipeline_cr": "chart-embed-gpu",
        "embed_compute_class": "gpu",
        "embed_image": "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-embedder-plan-20260624-dedupe2",
        "classifier_compute_class": "gpu",
        "classifier_image": "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-classifier-plan-20260624",
    }
    assert status["cost_baselines"] == {"embed": {"accepted": True}, "classifier": {"accepted": True}}
    assert status["kubernetes"] == {"namespace": "chart"}
    assert calls == [
        ("pipeline", layer, "chart-notes"),
        ("udf", layer, "chart-classify-events"),
        ("facets", layer, "chart-notes"),
        ("close", layer),
    ]


def test_cost_baseline_reads_accepted_estimate(tmp_path) -> None:
    report = tmp_path / "embed-budget.json"
    report.write_text(
        json.dumps(
            {
                "estimate": {"full_notes": 167000, "estimated_full_hours": 0.28},
                "budget": {"accepted": True},
            }
        )
        + "\n"
    )

    assert cost_baseline(str(report)) == {
        "report": str(report),
        "accepted": True,
        "estimate": {"full_notes": 167000, "estimated_full_hours": 0.28},
    }


def test_cost_baseline_reads_accepted_layer_snapshot(tmp_path) -> None:
    report = tmp_path / "embed-budget.json"
    snapshot = {
        "as_of_ms": 1782320205904,
        "window_seconds": 86400,
        "totals": {"total_usd": 12.34},
        "lines": [{"provider": "aws", "service": "compute", "basis": "invoice", "amount_usd": 12.34}],
    }
    report.write_text(
        json.dumps(
            {
                "source": "layer",
                "accepted": True,
                "layer_cost_snapshot": snapshot,
            }
        )
        + "\n"
    )

    assert cost_baseline(str(report)) == {
        "report": str(report),
        "accepted": True,
        "estimate": {},
        "source": "layer",
        "layer_cost_snapshot": snapshot,
    }


def test_cost_baseline_reports_missing_or_invalid_json(tmp_path) -> None:
    missing = tmp_path / "missing.json"
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{")

    assert cost_baseline(str(missing)) == {
        "report": str(missing),
        "accepted": False,
        "error": "missing",
    }
    invalid_result = cost_baseline(str(invalid))
    assert invalid_result["report"] == str(invalid)
    assert invalid_result["accepted"] is False
    assert invalid_result["error"].startswith("invalid json:")


def test_k8s_runtime_status_reports_pending_embed_pod_and_gpu_occupants(monkeypatch) -> None:
    def fake_kubectl_json(args):
        if args == [
            "-n",
            "chart",
            "get",
            "deployment",
            "chart-ingest-worker",
            "chart-embed-gpu-worker",
            "chart-classify-events-worker",
            "-o",
            "json",
        ]:
            return {
                "items": [
                    {
                        "metadata": {"name": "chart-embed-gpu-worker"},
                        "spec": {
                            "replicas": 1,
                            "template": {"spec": {"containers": [{"image": "ecr/embed"}]}},
                        },
                        "status": {"readyReplicas": 0},
                    }
                ]
            }
        if args == ["-n", "chart", "get", "function", "chart-classify-events", "-o", "json"]:
            return {
                "metadata": {"name": "chart-classify-events"},
                "spec": {
                    "paused": True,
                    "version": "v1",
                    "targetNamespaces": ["chart-notes"],
                    "worker": {
                        "image": "ecr/classifier",
                        "computeClass": "gpu",
                    },
                    "scaling": {
                        "mode": "autoscale",
                        "pool": "gpu",
                        "replicas": {"min": 0, "max": 2},
                    },
                },
                "status": {
                    "activeNamespaces": [],
                    "queueDepth": None,
                    "conditions": [
                        {
                            "type": "Ready",
                            "status": "True",
                            "reason": "Paused",
                            "message": "Function spec.paused=true; deployment scaled to zero",
                        }
                    ],
                },
            }
        if args == [
            "-n",
            "chart",
            "get",
            "scaledobject",
            "chart-ingest-worker",
            "chart-embed-gpu-worker",
            "-o",
            "json",
        ]:
            return {
                "items": [
                    {
                        "metadata": {"name": "chart-embed-gpu-worker"},
                        "spec": {
                            "minReplicaCount": 0,
                            "maxReplicaCount": 1,
                            "scaleTargetRef": {"name": "chart-embed-gpu-worker"},
                            "triggers": [
                                {
                                    "type": "prometheus",
                                    "authenticationRef": {"name": "chart-embed-gpu-worker"},
                                    "metadata": {
                                        "authModes": "bearer",
                                        "metricName": "layer_pipeline_pending_chart_notes",
                                        "threshold": "1000",
                                        "serverAddress": "http://layer-gateway.layer.svc.cluster.local:8080/v2/metrics",
                                    },
                                }
                            ],
                        },
                        "status": {
                            "conditions": [
                                {
                                    "type": "Ready",
                                    "status": "False",
                                    "reason": "ScaledObjectCheckFailed",
                                    "message": "bearer token=<empty> is required when bearer auth is enabled",
                                }
                            ]
                        },
                    }
                ]
            }
        if args == [
            "-n",
            "chart",
            "get",
            "triggerauthentication",
            "chart-ingest-worker",
            "chart-embed-gpu-worker",
            "-o",
            "json",
        ]:
            return {
                "items": [
                    {
                        "metadata": {"name": "chart-embed-gpu-worker"},
                        "spec": {
                            "secretTargetRef": [
                                {
                                    "parameter": "bearerToken",
                                    "name": "layer",
                                    "key": "turbopuffer-api-key",
                                }
                            ]
                        },
                        "status": {"scaledobjects": "chart-embed-gpu-worker"},
                    }
                ]
            }
        if args == ["-n", "chart", "get", "secret", "layer", "-o", "json"]:
            return {"data": {"turbopuffer-api-key": ""}}
        if args == [
            "-n",
            "chart",
            "get",
            "pods",
            "-l",
            "hevlayer.com/pipeline=chart-embed-gpu",
            "-o",
            "json",
        ]:
            return {
                "items": [
                    {
                        "metadata": {"namespace": "chart", "name": "chart-embed"},
                        "spec": {
                            "containers": [
                                {
                                    "image": "ecr/embed",
                                    "resources": {"requests": {"nvidia.com/gpu": "1"}},
                                }
                            ]
                        },
                        "status": {
                            "phase": "Pending",
                            "conditions": [
                                {
                                    "type": "PodScheduled",
                                    "status": "False",
                                    "reason": "Unschedulable",
                                }
                            ],
                        },
                    }
                ]
            }
        if args == [
            "-n",
            "chart",
            "get",
            "events",
            "--field-selector",
            "involvedObject.kind=Pod,involvedObject.name=chart-embed",
            "-o",
            "json",
        ]:
            return {"items": [{"message": "0/3 nodes are available: 1 Insufficient nvidia.com/gpu"}]}
        if args == ["get", "pods", "-A", "-o", "json"]:
            return {
                "items": [
                    {
                        "metadata": {"namespace": "hev-shop", "name": "hev-shop-embed"},
                        "spec": {
                            "nodeName": "gpu-node",
                            "containers": [
                                {
                                    "image": "ecr/hev-shop",
                                    "resources": {"limits": {"nvidia.com/gpu": "1"}},
                                }
                            ],
                        },
                        "status": {"phase": "Running"},
                    }
                ]
            }
        raise AssertionError(args)

    monkeypatch.setattr(full_status, "_kubectl_json", fake_kubectl_json)

    status = k8s_runtime_status()

    assert status["deployment_status"]["chart-embed-gpu-worker"]["replicas"] == 1
    assert status["deployment_status"]["chart-embed-gpu-worker"]["image"] == "ecr/embed"
    assert status["functions"] == {"classifier": "chart-classify-events"}
    assert status["function_status"]["chart-classify-events"]["paused"] is True
    assert status["function_status"]["chart-classify-events"]["image"] == "ecr/classifier"
    assert status["function_status"]["chart-classify-events"]["conditions"] == [
        {
            "type": "Ready",
            "status": "True",
            "reason": "Paused",
            "message": "Function spec.paused=true; deployment scaled to zero",
        }
    ]
    assert status["scaled_objects"]["chart-embed-gpu-worker"] == {
        "name": "chart-embed-gpu-worker",
        "scale_target": "chart-embed-gpu-worker",
        "min_replicas": 0,
        "max_replicas": 1,
        "triggers": [
            {
                "type": "prometheus",
                "authentication_ref": "chart-embed-gpu-worker",
                "auth_modes": "bearer",
                "metric_name": "layer_pipeline_pending_chart_notes",
                "threshold": "1000",
                "server_address": "http://layer-gateway.layer.svc.cluster.local:8080/v2/metrics",
            }
        ],
        "conditions": [
            {
                "type": "Ready",
                "status": "False",
                "reason": "ScaledObjectCheckFailed",
                "message": "bearer token=<empty> is required when bearer auth is enabled",
            }
        ],
    }
    assert status["trigger_authentications"]["chart-embed-gpu-worker"] == {
        "name": "chart-embed-gpu-worker",
        "secret_target_refs": [
            {
                "parameter": "bearerToken",
                "name": "layer",
                "key": "turbopuffer-api-key",
                "status": {
                    "secret_exists": True,
                    "key_exists": True,
                    "value_present": False,
                },
            }
        ],
        "scaled_objects": "chart-embed-gpu-worker",
    }
    assert status["embed_pods"][0]["phase"] == "Pending"
    assert status["embed_pods"][0]["gpu"] == "1"
    assert status["embed_pods"][0]["events"] == ["0/3 nodes are available: 1 Insufficient nvidia.com/gpu"]
    assert status["gpu_pods"][0]["namespace"] == "hev-shop"


@pytest.mark.anyio
async def test_collect_status_exits_before_client_setup_without_gateway_key(monkeypatch) -> None:
    monkeypatch.setattr(
        full_status,
        "Settings",
        lambda: SimpleNamespace(namespace="chart-notes", api_key=None),
    )
    monkeypatch.setattr(full_status, "make_client", lambda settings: pytest.fail("client should not be created"))

    with pytest.raises(SystemExit, match="No gateway key"):
        await collect_status()


def test_full_status_cli_writes_phase6_status_report(monkeypatch, tmp_path, capsys) -> None:
    async def fake_collect_status(*, pipeline_id=None, udf_id=full_status.DEFAULT_UDF_ID):
        return {
            "namespace": "chart-notes",
            "targets": {"pipeline_id": pipeline_id or "chart-notes", "udf_id": udf_id},
            "pipeline": {},
            "udf": {},
            "facets": {},
        }

    out = tmp_path / "reports" / "phase6-status-report.json"
    monkeypatch.setattr(full_status, "collect_status", fake_collect_status)
    monkeypatch.setattr("sys.argv", ["full_status", "--out", str(out)])

    full_status.main()

    assert '"namespace": "chart-notes"' in capsys.readouterr().out
    assert '"namespace": "chart-notes"' in out.read_text()
