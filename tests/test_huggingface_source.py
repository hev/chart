import asyncio

import pytest

from source.huggingface_source import _iter_json_array_rows, _put_chunks_with_retry, _put_documents


class FakeLayer:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.ids: list[str] = []

    async def put_pipeline_document_chunks(self, pipeline_id, doc_id, body):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0)
        self.ids.append(doc_id)
        self.active -= 1


def test_put_documents_uses_bounded_concurrency() -> None:
    layer = FakeLayer()
    documents = [(f"doc-{index}", [{"id": f"doc-{index}", "text": "note"}]) for index in range(8)]

    asyncio.run(_put_documents(layer, "chart-notes", documents, concurrency=3))

    assert sorted(layer.ids) == [f"doc-{index}" for index in range(8)]
    assert layer.max_active == 3


def test_put_documents_can_run_serially() -> None:
    layer = FakeLayer()
    documents = [(f"doc-{index}", [{"id": f"doc-{index}", "text": "note"}]) for index in range(3)]

    asyncio.run(_put_documents(layer, "chart-notes", documents, concurrency=1))

    assert layer.ids == ["doc-0", "doc-1", "doc-2"]
    assert layer.max_active == 1


def test_put_chunks_retries_after_timeout(monkeypatch) -> None:
    class SlowThenOk:
        def __init__(self) -> None:
            self.calls = 0

        async def put_pipeline_document_chunks(self, pipeline_id, doc_id, body):
            self.calls += 1
            if self.calls == 1:
                await asyncio.sleep(0.05)

    layer = SlowThenOk()
    monkeypatch.setenv("CHART_HF_SOURCE_WRITE_TIMEOUT_SECONDS", "0.01")
    monkeypatch.setenv("CHART_HF_SOURCE_WRITE_RETRY_SECONDS", "0")

    asyncio.run(_put_chunks_with_retry(layer, "chart-notes", "doc-1", [{"id": "doc-1", "text": "note"}]))

    assert layer.calls == 2


def test_iter_json_array_rows_handles_split_objects() -> None:
    rows = list(_iter_json_array_rows(['[{"id": "a"', ', "x": 1}, {"id": "b"}]']))

    assert rows == [{"id": "a", "x": 1}, {"id": "b"}]


def test_iter_json_array_rows_applies_offset() -> None:
    rows = list(_iter_json_array_rows(['[{"id": "a"}, {"id": "b"}, {"id": "c"}]'], offset=1))

    assert rows == [{"id": "b"}, {"id": "c"}]


def test_iter_json_array_rows_rejects_unterminated_input() -> None:
    with pytest.raises(RuntimeError, match="unterminated JSON array"):
        list(_iter_json_array_rows(['[{"id": "a"}']))
