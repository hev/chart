import json

import pytest

from smoke import live
from smoke.live import check_facets, check_index_shape, check_routes, check_similar


class FakeRouting:
    def __init__(self, route: str) -> None:
        self.route = route

    def model_dump(self):
        return {"route": self.route}


class FakeStrategyRouting:
    def __init__(self, strategy: str) -> None:
        self.strategy = strategy

    def model_dump(self):
        return {"strategy": self.strategy}


class FakeResponse:
    def __init__(self, *, rows=None, route=None, hybrid=None) -> None:
        self.rows = rows or []
        self.routing = FakeRouting(route) if route else None
        self.hybrid = hybrid


class FakeLayer:
    def __init__(self) -> None:
        self.bodies = []

    async def query_namespace(self, namespace, body):
        self.bodies.append(body)
        if body.nearest_to_id:
            return FakeResponse(rows=[{"id": "similar-1"}])
        text = body.rank_by[2]
        if body.include_attributes == ["id", "age_band", "gender", "similar_patient_ids"]:
            return FakeResponse(
                rows=[
                    {
                        "id": "patient-1",
                        "age_band": "adult",
                        "gender": "female",
                        "similar_patient_ids": ["patient-2"],
                    }
                ],
                route="hybrid_text",
            )
        tokens = text.split()
        route = "hybrid_text" if len(tokens) <= 2 else "semantic" if len(tokens) >= 8 else "fused"
        hybrid = {"leg_breakdown": []} if route == "fused" else None
        return FakeResponse(rows=[{"id": text}], route=route, hybrid=hybrid)


class FakeEmbedder:
    def embed_query(self, text):
        return [0.1] * 768


class FakeSettings:
    namespace = "chart-notes"
    embed_model = "fake-model"


@pytest.mark.anyio
async def test_check_index_shape_validates_vector_dim_and_core_attrs() -> None:
    result = await check_index_shape(FakeLayer(), FakeSettings(), FakeEmbedder())

    assert result == {"vector_dim": 768, "rows": 1}


def test_live_cli_rejects_non_positive_top_k_before_setup(monkeypatch) -> None:
    monkeypatch.setattr(live, "Settings", lambda: pytest.fail("Settings should not load for invalid args"))
    monkeypatch.setattr("sys.argv", ["live", "--top-k", "0"])

    with pytest.raises(SystemExit) as exc:
        live.main()

    assert exc.value.code == 2


@pytest.mark.anyio
async def test_live_run_exits_before_embedder_without_gateway_key(monkeypatch) -> None:
    monkeypatch.setattr(live, "Settings", lambda: type("Settings", (), {"api_key": None})())
    monkeypatch.setattr(live, "Embedder", lambda model: pytest.fail("embedder should not load without key"))
    monkeypatch.setattr(live, "make_client", lambda settings: pytest.fail("client should not be created without key"))

    with pytest.raises(SystemExit, match="No gateway key"):
        await live.run(top_k=1, similar_id=None, skip_facets=True, require_event_facets=False)


@pytest.mark.anyio
async def test_live_run_writes_smoke_report(monkeypatch, tmp_path, capsys) -> None:
    layer = FakeLayer()
    out = tmp_path / "reports" / "live-smoke-report.json"

    monkeypatch.setattr(
        live,
        "Settings",
        lambda: type("Settings", (), {"api_key": "key", "embed_model": "model", "namespace": "chart-notes"})(),
    )
    monkeypatch.setattr(live, "Embedder", lambda model: FakeEmbedder())
    monkeypatch.setattr(live, "make_client", lambda settings: layer)

    async def fake_close_client(layer_arg):
        return None

    monkeypatch.setattr(live, "close_client", fake_close_client)

    await live.run(top_k=3, similar_id=None, skip_facets=True, require_event_facets=False, out=out)

    assert '"index_shape"' in capsys.readouterr().out
    text = out.read_text()
    assert '"ok": true' in text
    assert '"status": "completed"' in text
    assert '"requirements"' in text
    assert '"event_facets": false' in text
    assert '"index_shape"' in text
    assert '"routes"' in text
    assert '"similar"' in text


@pytest.mark.anyio
async def test_check_index_shape_requires_populated_similar_patient_ids() -> None:
    class EmptySimilarLayer(FakeLayer):
        async def query_namespace(self, namespace, body):
            if body.include_attributes == ["id", "age_band", "gender", "similar_patient_ids"]:
                return FakeResponse(
                    rows=[
                        {
                            "id": "patient-1",
                            "age_band": "adult",
                            "gender": "female",
                            "similar_patient_ids": [],
                        }
                    ],
                    route="hybrid_text",
                )
            return await super().query_namespace(namespace, body)

    with pytest.raises(AssertionError, match="populated similar_patient_ids"):
        await check_index_shape(EmptySimilarLayer(), FakeSettings(), FakeEmbedder())


@pytest.mark.anyio
async def test_check_routes_validates_all_chip_routes() -> None:
    rows = await check_routes(FakeLayer(), FakeSettings(), FakeEmbedder(), top_k=3)

    assert {row["route"] for row in rows} == {"hybrid_text", "fused", "semantic"}
    assert all(row["rows"] == 1 for row in rows)
    assert all(row["routing"]["route"] == row["route"] for row in rows)
    assert next(row for row in rows if row["route"] == "fused")["hybrid"] == {"leg_breakdown": []}
    assert next(row for row in rows if row["route"] == "semantic")["hybrid"] is None


@pytest.mark.anyio
async def test_check_routes_accepts_gateway_strategy_alias() -> None:
    class StrategyLayer(FakeLayer):
        async def query_namespace(self, namespace, body):
            response = await super().query_namespace(namespace, body)
            if response.routing:
                response.routing = FakeStrategyRouting(response.routing.route)
            return response

    rows = await check_routes(StrategyLayer(), FakeSettings(), FakeEmbedder(), top_k=3)

    assert {row["route"] for row in rows} == {"hybrid_text", "fused", "semantic"}
    assert all("strategy" in row["routing"] for row in rows)


@pytest.mark.anyio
async def test_check_routes_requires_768_dim_query_vectors() -> None:
    class BadEmbedder:
        def embed_query(self, text):
            return [0.1, 0.2]

    with pytest.raises(AssertionError, match="768-d query vector"):
        await check_routes(FakeLayer(), FakeSettings(), BadEmbedder(), top_k=3)


@pytest.mark.anyio
async def test_check_routes_rejects_rows_without_ids() -> None:
    class MissingIdLayer(FakeLayer):
        async def query_namespace(self, namespace, body):
            self.bodies.append(body)
            if body.include_attributes == ["id", "age_band", "gender", "similar_patient_ids"]:
                return await super().query_namespace(namespace, body)
            text = body.rank_by[2]
            route = "hybrid_text" if len(text.split()) <= 2 else "semantic" if len(text.split()) >= 8 else "fused"
            return FakeResponse(rows=[{"title": "missing id"}], route=route, hybrid={"leg_breakdown": []})

    with pytest.raises(AssertionError, match="returned rows without ids"):
        await check_routes(MissingIdLayer(), FakeSettings(), FakeEmbedder(), top_k=3)


@pytest.mark.anyio
async def test_check_routes_requires_gateway_hybrid_echo_for_fused_route() -> None:
    class MissingHybridLayer(FakeLayer):
        async def query_namespace(self, namespace, body):
            response = await super().query_namespace(namespace, body)
            if body.rank_by:
                response.hybrid = None
            return response

    with pytest.raises(AssertionError, match="fused hybrid echo"):
        await check_routes(MissingHybridLayer(), FakeSettings(), FakeEmbedder(), top_k=3)


@pytest.mark.anyio
async def test_check_similar_uses_supplied_patient_id() -> None:
    layer = FakeLayer()

    result = await check_similar(layer, FakeSettings(), FakeEmbedder(), "patient-1", top_k=3)

    assert result == {"patient_id": "patient-1", "rows": 1, "neighbors": 1}
    assert layer.bodies[0].nearest_to_id == ["patient-1"]


@pytest.mark.anyio
async def test_check_similar_seed_lookup_requires_768_dim_query_vector() -> None:
    class BadEmbedder:
        def embed_query(self, text):
            return [0.1, 0.2]

    with pytest.raises(AssertionError, match="metformin 500mg.*768-d query vector"):
        await check_similar(FakeLayer(), FakeSettings(), BadEmbedder(), None, top_k=3)


@pytest.mark.anyio
async def test_check_similar_rejects_neighbor_rows_without_ids() -> None:
    class MissingNeighborIdLayer(FakeLayer):
        async def query_namespace(self, namespace, body):
            self.bodies.append(body)
            if body.nearest_to_id:
                return FakeResponse(rows=[{"title": "missing id"}])
            return await super().query_namespace(namespace, body)

    with pytest.raises(AssertionError, match="nearest_to_id returned rows without ids"):
        await check_similar(MissingNeighborIdLayer(), FakeSettings(), FakeEmbedder(), "patient-1", top_k=3)


@pytest.mark.anyio
async def test_check_similar_rejects_self_only_results() -> None:
    class SelfOnlyLayer(FakeLayer):
        async def query_namespace(self, namespace, body):
            self.bodies.append(body)
            if body.nearest_to_id:
                return FakeResponse(rows=[{"id": body.nearest_to_id[0]}])
            return await super().query_namespace(namespace, body)

    with pytest.raises(AssertionError, match="no neighbor rows"):
        await check_similar(SelfOnlyLayer(), FakeSettings(), FakeEmbedder(), "patient-1", top_k=3)


@pytest.mark.anyio
async def test_check_facets_requires_age_and_gender_snapshots(monkeypatch) -> None:
    async def fake_latest_facets(layer, namespace, *, field, limit=14):
        if field == "age_band":
            return [{"value": "adult", "count": 1}], {"sha": "abc"}
        return None, None

    monkeypatch.setattr("smoke.live.latest_facets", fake_latest_facets)

    with pytest.raises(AssertionError, match="gender"):
        await check_facets(object(), FakeSettings())


@pytest.mark.anyio
async def test_check_facets_can_require_event_snapshot(monkeypatch) -> None:
    async def fake_latest_facets(layer, namespace, *, field, limit=14):
        if field in {"age_band", "gender"}:
            return [{"value": "present", "count": 1}], {"sha": "abc"}
        return None, None

    monkeypatch.setattr("smoke.live.latest_facets", fake_latest_facets)

    with pytest.raises(AssertionError, match="events"):
        await check_facets(object(), FakeSettings(), require_events=True)


@pytest.mark.anyio
async def test_check_facets_requires_base_snapshot_sha(monkeypatch) -> None:
    async def fake_latest_facets(layer, namespace, *, field, limit=14):
        if field == "age_band":
            return [{"value": "adult", "count": 1}], {"sha": None}
        if field == "gender":
            return [{"value": "female", "count": 1}], {"sha": "gender"}
        return None, None

    monkeypatch.setattr("smoke.live.latest_facets", fake_latest_facets)

    with pytest.raises(AssertionError, match="age_band.*sha"):
        await check_facets(object(), FakeSettings())


@pytest.mark.anyio
async def test_check_facets_requires_event_snapshot_sha_when_required(monkeypatch) -> None:
    async def fake_latest_facets(layer, namespace, *, field, limit=14):
        if field in {"age_band", "gender"}:
            return [{"value": "present", "count": 1}], {"sha": field}
        if field == "events":
            return [{"value": "medication_discontinued", "count": 1}], {"sha": None}
        return None, None

    monkeypatch.setattr("smoke.live.latest_facets", fake_latest_facets)

    with pytest.raises(AssertionError, match="events.*sha"):
        await check_facets(object(), FakeSettings(), require_events=True)


@pytest.mark.anyio
async def test_check_facets_accepts_event_snapshot_when_required(monkeypatch) -> None:
    async def fake_latest_facets(layer, namespace, *, field, limit=14):
        if field in {"age_band", "gender", "events"}:
            return [{"value": "present", "count": 1}], {
                "sha": field,
                "row_count": 2000,
                "watermark_ms": 123456,
            }
        return None, None

    monkeypatch.setattr("smoke.live.latest_facets", fake_latest_facets)

    facets = await check_facets(object(), FakeSettings(), require_events=True)

    assert facets["events"] == {
        "values": 1,
        "sha": "events",
        "row_count": 2000,
        "watermark_ms": 123456,
    }


@pytest.mark.anyio
async def test_run_writes_failed_partial_report_for_late_check_failure(monkeypatch, tmp_path) -> None:
    async def fake_latest_facets(layer, namespace, *, field, limit=14):
        if field in {"age_band", "gender"}:
            return [{"value": "present", "count": 1}], {"sha": field}
        if field == "events":
            return [], {"sha": "events", "row_count": 2000, "watermark_ms": 123456}
        return None, None

    closed = False

    async def fake_close_client(layer):
        nonlocal closed
        closed = True

    monkeypatch.setattr(live, "Settings", FakeSettings)
    monkeypatch.setattr(live, "require_gateway_key", lambda settings: None)
    monkeypatch.setattr(live, "Embedder", lambda model: FakeEmbedder())
    monkeypatch.setattr(live, "make_client", lambda settings: FakeLayer())
    monkeypatch.setattr(live, "close_client", fake_close_client)
    monkeypatch.setattr(live, "latest_facets", fake_latest_facets)

    out = tmp_path / "live-smoke-report.json"
    with pytest.raises(AssertionError, match="events.*no values"):
        await live.run(
            top_k=3,
            similar_id=None,
            skip_facets=False,
            require_event_facets=True,
            out=out,
        )

    data = json.loads(out.read_text())
    assert data["ok"] is False
    assert data["status"] == "failed"
    assert data["error"] == "materialized facet snapshot for 'events' has no values"
    assert data["index_shape"] == {"vector_dim": 768, "rows": 1}
    assert data["routes"]
    assert data["similar"]["neighbors"] == 1
    assert data["facets"]["age_band"]["values"] == 1
    assert data["facets"]["gender"]["sha"] == "gender"
    assert data["facets"]["events"] == {
        "values": 0,
        "sha": "events",
        "row_count": 2000,
        "watermark_ms": 123456,
    }
    assert closed is True
