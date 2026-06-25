from __future__ import annotations

import pytest

from chart_common.gateway import latest_facets, materialize_facet_snapshots, require_gateway_key


class SnapshotRef:
    def __init__(self, sha: str) -> None:
        self.sha = sha


class FieldValue:
    def __init__(self, value: str, count: int) -> None:
        self.v = value
        self.n = count


class Field:
    def __init__(self, name: str, values: list[FieldValue]) -> None:
        self.name = name
        self.values = values


class SnapshotBody:
    def __init__(self, sha: str | None, fields: list[Field]) -> None:
        self.sha = sha
        self.fields = fields
        self.watermark_ms = 123
        self.row_count = 10


class SnapshotJob:
    def __init__(self, status: str, job_id: str = "job-1", error: str | None = None) -> None:
        self.status = status
        self.id = job_id
        self.error = error


def test_require_gateway_key_rejects_missing_or_test_double_key() -> None:
    with pytest.raises(SystemExit, match="No gateway key"):
        require_gateway_key(type("Settings", (), {"api_key": None})())

    require_gateway_key(type("Settings", (), {"api_key": "key"})())


@pytest.mark.anyio
async def test_materialize_facet_snapshots_waits_for_in_progress_states(monkeypatch) -> None:
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    class FakeLayer:
        def __init__(self) -> None:
            self.jobs = [SnapshotJob("pending"), SnapshotJob("queued"), SnapshotJob("running"), SnapshotJob("completed")]
            self.polled = []

        async def create_snapshot(self, namespace, body):
            assert body == {"field": "age_band", "source": "origin"}
            return self.jobs.pop(0)

        async def get_snapshot_job(self, namespace, job_id):
            self.polled.append((namespace, job_id))
            return self.jobs.pop(0)

    layer = FakeLayer()
    monkeypatch.setattr("chart_common.gateway.asyncio.sleep", fake_sleep)

    await materialize_facet_snapshots(layer, "chart-notes", fields=["age_band"], timeout=10)

    assert sleeps == [1.0, 1.0, 1.0]
    assert layer.polled == [("chart-notes", "job-1"), ("chart-notes", "job-1"), ("chart-notes", "job-1")]


@pytest.mark.anyio
async def test_materialize_facet_snapshots_accepts_dict_shaped_jobs(monkeypatch) -> None:
    async def fake_sleep(seconds):
        return None

    class FakeLayer:
        def __init__(self) -> None:
            self.jobs = [{"id": "job-1", "status": "running"}, {"id": "job-1", "status": "completed"}]

        async def create_snapshot(self, namespace, body):
            return self.jobs.pop(0)

        async def get_snapshot_job(self, namespace, job_id):
            return self.jobs.pop(0)

    monkeypatch.setattr("chart_common.gateway.asyncio.sleep", fake_sleep)

    await materialize_facet_snapshots(FakeLayer(), "chart-notes", fields=["events"], timeout=10)


@pytest.mark.anyio
async def test_materialize_facet_snapshots_reports_failed_jobs() -> None:
    class FakeLayer:
        async def create_snapshot(self, namespace, body):
            return SnapshotJob("failed", error="boom")

    with pytest.raises(RuntimeError, match="snapshot events failed: boom"):
        await materialize_facet_snapshots(FakeLayer(), "chart-notes", fields=["events"], timeout=10)


@pytest.mark.anyio
async def test_latest_facets_scans_recent_snapshot_history() -> None:
    class FakeLayer:
        async def list_namespace_history(self, namespace, *, limit):
            assert namespace == "chart-notes"
            assert limit == 20
            return [SnapshotRef("latest"), SnapshotRef("older")]

        async def get_namespace_snapshot(self, namespace, sha):
            if sha == "latest":
                return SnapshotBody("latest", [Field("events", [FieldValue("adverse-event", 3)])])
            return SnapshotBody(
                None,
                [
                    Field(
                        "age_band",
                        [FieldValue("adult", 7), FieldValue("child", 2), FieldValue("older-adult", 4)],
                    )
                ],
            )

    facets, provenance = await latest_facets(FakeLayer(), "chart-notes", field="age_band", limit=2)

    assert facets == [{"value": "adult", "count": 7}, {"value": "older-adult", "count": 4}]
    assert provenance == {"sha": "older", "watermark_ms": 123, "row_count": 10}


@pytest.mark.anyio
async def test_latest_facets_accepts_dict_shaped_gateway_responses() -> None:
    class FakeLayer:
        async def list_namespace_history(self, namespace, *, limit):
            return [{"sha": "latest"}, {"sha": "older"}]

        async def get_namespace_snapshot(self, namespace, sha):
            if sha == "latest":
                return {"sha": "latest", "fields": [{"name": "gender", "values": [{"v": "female", "n": 4}]}]}
            return {
                "sha": None,
                "watermark_ms": 456,
                "row_count": 20,
                "fields": [
                    {
                        "name": "age_band",
                        "values": [
                            {"v": "adult", "n": 7},
                            {"v": "child", "n": 2},
                            {"v": "older-adult", "n": 4},
                        ],
                    }
                ],
            }

    facets, provenance = await latest_facets(FakeLayer(), "chart-notes", field="age_band", limit=2)

    assert facets == [{"value": "adult", "count": 7}, {"value": "older-adult", "count": 4}]
    assert provenance == {"sha": "older", "watermark_ms": 456, "row_count": 20}


@pytest.mark.anyio
async def test_latest_facets_skips_history_entries_without_sha() -> None:
    class FakeLayer:
        async def list_namespace_history(self, namespace, *, limit):
            return [{}, {"sha": "older"}]

        async def get_namespace_snapshot(self, namespace, sha):
            assert sha == "older"
            return {
                "sha": "older",
                "watermark_ms": 789,
                "row_count": 30,
                "fields": [{"name": "events", "values": [{"v": "medication_discontinued", "n": 5}]}],
            }

    facets, provenance = await latest_facets(FakeLayer(), "chart-notes", field="events")

    assert facets == [{"value": "medication_discontinued", "count": 5}]
    assert provenance == {"sha": "older", "watermark_ms": 789, "row_count": 30}
