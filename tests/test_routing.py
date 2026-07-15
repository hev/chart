import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from chart_common.config import EMBED_DIM
from fastapi import HTTPException
from search import app as search_app
from smoke.live import expected_route, policy_route


class FakeModel:
    def __init__(self, **values) -> None:
        self.values = values

    def model_dump(self):
        return dict(self.values)


def test_search_body_matches_gateway_auto_shape(monkeypatch) -> None:
    vector = [0.1] * EMBED_DIM
    monkeypatch.setattr(
        search_app.app,
        "state",
        SimpleNamespace(embedder=SimpleNamespace(embed_query=lambda q: vector)),
    )

    body = search_app.search_body("chest pain radiating to left arm", top_k=200)

    assert body.rank_by == [
        "text",
        "Auto",
        "chest pain radiating to left arm",
        {"vector": vector},
    ]
    assert body.top_k == 50
    assert body.include_leg_breakdown is True
    assert "id" in body.include_attributes
    assert "similar_patient_ids" in body.include_attributes


def test_search_body_rejects_wrong_vector_dimensions(monkeypatch) -> None:
    monkeypatch.setattr(
        search_app.app,
        "state",
        SimpleNamespace(embedder=SimpleNamespace(embed_query=lambda q: [0.1, 0.2])),
    )

    with pytest.raises(HTTPException) as exc:
        search_app.search_body("chest pain")

    assert exc.value.status_code == 502
    assert f"non-{EMBED_DIM}-d vector" in exc.value.detail


def test_similar_body_uses_nearest_to_id() -> None:
    body = search_app.similar_body("patient-123", top_k=0)

    assert body.nearest_to_id == ["patient-123"]
    assert body.top_k == 1
    assert body.rank_by is None


@pytest.mark.anyio
async def test_python_backend_search_query_requires_gateway_routing(monkeypatch) -> None:
    class FakeLayer:
        async def query_namespace(self, namespace, body):
            return SimpleNamespace(rows=[{"id": "patient-1"}], routing=None, hybrid=None)

    monkeypatch.setattr(
        search_app.app,
        "state",
        SimpleNamespace(settings=SimpleNamespace(namespace="chart-notes"), layer=FakeLayer()),
    )

    with pytest.raises(HTTPException) as exc:
        await search_app._run_query(SimpleNamespace(), require_routing=True)

    assert exc.value.status_code == 502
    assert exc.value.detail == "gateway response did not include a routing decision"


@pytest.mark.anyio
async def test_python_backend_similar_query_does_not_require_gateway_routing(monkeypatch) -> None:
    class FakeLayer:
        async def query_namespace(self, namespace, body):
            return SimpleNamespace(rows=[{"id": "patient-1"}], routing=None, hybrid=None)

    monkeypatch.setattr(
        search_app.app,
        "state",
        SimpleNamespace(settings=SimpleNamespace(namespace="chart-notes"), layer=FakeLayer()),
    )

    result = await search_app._run_query(SimpleNamespace(), require_routing=False)

    assert result["rows"] == [{"id": "patient-1"}]
    assert result["routing"] is None


@pytest.mark.anyio
async def test_python_backend_normalizes_model_rows_before_json_response(monkeypatch) -> None:
    class FakeLayer:
        async def query_namespace(self, namespace, body):
            return SimpleNamespace(
                rows=[FakeModel(id="patient-1", title="Case")],
                routing=FakeModel(route="hybrid_text"),
                hybrid=None,
            )

    monkeypatch.setattr(
        search_app.app,
        "state",
        SimpleNamespace(settings=SimpleNamespace(namespace="chart-notes"), layer=FakeLayer()),
    )

    result = await search_app._run_query(SimpleNamespace(), require_routing=True)

    assert result["rows"] == [{"id": "patient-1", "title": "Case"}]
    assert result["routing"] == {"route": "hybrid_text"}


@pytest.mark.anyio
async def test_python_backend_config_matches_worker_shape(monkeypatch) -> None:
    monkeypatch.setattr(
        search_app.app,
        "state",
        SimpleNamespace(
            settings=SimpleNamespace(
                namespace="chart-notes",
                gateway_url="https://aws-us-east-1.hevlayer.com/",
                li_namespace="chart-notes-li",
                li_model="answerdotai/answerai-colbert-small-v1",
            )
        ),
    )

    response = await search_app.config()

    assert json.loads(response.body) == {
        "namespace": "chart-notes",
        "gateway": "https://aws-us-east-1.hevlayer.com",
        "field": "text",
        "li_namespace": "chart-notes-li",
        "li_model": "answerdotai/answerai-colbert-small-v1",
    }


def test_search_payload_omits_unrendered_cascade_v2_maps() -> None:
    assert "event_confidence_v2" not in search_app.INCLUDE
    assert "event_spans_v2" not in search_app.INCLUDE


def test_demo_chips_match_expected_auto_routes() -> None:
    path = Path(__file__).resolve().parent.parent / "web" / "static" / "queries.json"
    examples = json.loads(path.read_text())["examples"]

    for example in examples:
        note = example["note"]
        expected = expected_route(example)
        assert expected in note
        assert example["expected_route"] == policy_route(example["text"])


def test_demo_chip_expected_routes_are_declared_not_implicit() -> None:
    path = Path(__file__).resolve().parent.parent / "web" / "static" / "queries.json"
    examples = json.loads(path.read_text())["examples"]

    assert all("expected_route" in example for example in examples)
    assert {example["expected_route"] for example in examples} == {"hybrid_text", "fused", "semantic"}


def test_healthz_returns_ok() -> None:
    # The in-cluster Deployment and the ALB target group probe /healthz; a 200
    # here means lifespan finished (Arctic model resident, gateway client up).
    response = asyncio.run(search_app.healthz())

    assert response.status_code == 200
    assert json.loads(response.body) == {"ok": True}
