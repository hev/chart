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


@pytest.fixture
def source_run(monkeypatch):
    from source import huggingface_source as source

    monkeypatch.setattr(source, "Settings", lambda: None)
    monkeypatch.setattr(source, "_source_ref", lambda: {"dataset": "synthetic"})
    monkeypatch.setenv("CHART_HF_SOURCE_PAGE_SIZE", "3")
    monkeypatch.setenv("CHART_HF_SOURCE_PAGE_SLEEP_SECONDS", "0")
    monkeypatch.setenv("CHART_HF_SOURCE_WRITE_CONCURRENCY", "2")
    monkeypatch.setenv("CHART_HF_SOURCE_WRITE_ATTEMPTS", "2")
    monkeypatch.setenv("CHART_HF_SOURCE_WRITE_RETRY_SECONDS", "0")
    monkeypatch.setenv("CHART_HF_SOURCE_CHUNK_CHARS", "2000")
    monkeypatch.setenv("CHART_HF_SOURCE_CHUNK_OVERLAP_CHARS", "256")

    def execute(rows, mode, limit, start=0, fail=False):
        writes = []
        consumed = []
        closed = []
        requests = []

        class Layer:
            async def create_pipeline(self, body):
                pass

            async def put_pipeline_document_chunks(self, pipeline_id, doc_id, body):
                if fail:
                    raise RuntimeError("write failed")
                writes.append((doc_id, body))

        layer = Layer()

        async def close(client):
            assert client is layer
            closed.append(True)

        def stream(ref, *, offset):
            for index in range(offset, len(rows)):
                consumed.append(index)
                yield rows[index]

        def page(ref, *, offset, length):
            requests.append(offset)
            return [{"row": row} for row in rows[offset:offset + length]]

        monkeypatch.setattr(source, "make_client", lambda settings: layer)
        monkeypatch.setattr(source, "close_client", close)
        monkeypatch.setattr(source, "_stream_json_array", stream)
        monkeypatch.setattr(source, "_stream_rows", stream)
        monkeypatch.setattr(source, "_request_rows", page)
        monkeypatch.setenv("CHART_HF_SOURCE_MODE", mode)
        monkeypatch.setenv("CHART_HF_SOURCE_MAX_ROWS", str(limit))
        monkeypatch.setenv("CHART_HF_SOURCE_START_OFFSET", str(start))
        try:
            result = asyncio.run(source.run())
        finally:
            assert closed == [True]
        return result, writes, consumed, requests

    return execute


def synthetic_rows(count, pattern):
    return [
        {"id": f"doc-{i}", "text": "  " if pattern == "skipped" or (pattern == "mixed" and i % 2) else f"note {i}"}
        for i in range(count)
    ]


@pytest.mark.parametrize("mode", ["direct-json", "rows-api", "stream"])
@pytest.mark.parametrize("limit", [1, 3, 4, 7, 0, -1])
@pytest.mark.parametrize("pattern", ["valid", "mixed", "skipped"])
@pytest.mark.parametrize("start,count", [(0, 8), (2, 8), (0, 0), (8, 8)])
def test_run_bounds_and_offsets(source_run, mode, limit, pattern, start, count):
    rows = synthetic_rows(count, pattern)
    end = min(count, start + max(0, limit)) if limit else count
    selected = rows[start:end]
    expected = [row for row in selected if row["text"].strip()]

    result, writes, consumed, requests = source_run(rows, mode, limit, start)

    assert (result["staged"], result["skipped"], result["next_offset"]) == (
        len(expected), len(selected) - len(expected), end,
    )
    assert [doc_id for doc_id, _ in writes] == [row["id"] for row in expected]
    assert [body for _, body in writes] == [
        {"chunks": [{"id": row["id"], "text": row["text"], "metadata": {
            "source": "huggingface", "chunk_index": 0, "chunk_strategy": "fixed",
        }}]} for row in expected
    ]
    if mode != "rows-api":
        assert consumed == list(range(start, end))
    elif limit > 0 and end == start + limit:
        assert all(offset < end for offset in requests)


@pytest.mark.parametrize("mode", ["direct-json", "rows-api", "stream"])
def test_run_resume_composes_without_loss_or_duplicates(source_run, mode):
    rows = synthetic_rows(11, "mixed")
    whole, whole_writes, _, _ = source_run(rows, mode, 0, start=2)
    offset = 2
    writes = []
    staged = skipped = 0
    for limit in [1, 3, 4, 4]:
        result, batch, _, _ = source_run(rows, mode, limit, start=offset)
        offset = result["next_offset"]
        staged += result["staged"]
        skipped += result["skipped"]
        writes.extend(batch)
    assert (staged, skipped, offset) == (whole["staged"], whole["skipped"], whole["next_offset"])
    assert writes == whole_writes
    assert len({doc_id for doc_id, _ in writes}) == len(writes)


@pytest.mark.parametrize("mode", ["direct-json", "rows-api", "stream"])
def test_run_closes_client_on_write_failure(source_run, mode):
    with pytest.raises(RuntimeError, match="write failed"):
        source_run(synthetic_rows(5, "valid"), mode, 1, fail=True)
