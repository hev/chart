import asyncio

from source.huggingface_source import _put_documents


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
