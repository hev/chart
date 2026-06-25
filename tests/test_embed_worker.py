from types import SimpleNamespace

import pytest

from chart_common.config import EMBED_DIM
from indexer.embed import (
    EmbedContext,
    EmbedSettings,
    amain,
    chunk_text_and_attrs,
    normalized_chunk_attrs,
    run_once,
    vector_entries,
)


def test_chunk_text_and_attrs_preserves_chunk_metadata() -> None:
    chunks = [
        SimpleNamespace(id="doc-1#0", text=" First chunk ", metadata={"PMID": "123"}),
        SimpleNamespace(id="doc-1#1", text="", metadata={"pmid": "ignored"}),
        SimpleNamespace(id="doc-1#2", text="Second chunk", metadata=None),
    ]

    texts, attrs = chunk_text_and_attrs(chunks, "doc-1")

    assert texts == ["First chunk", "Second chunk"]
    assert attrs == [
        {
            "id": "doc-1",
            "text": "First chunk",
            "title": "",
            "pmid": "123",
            "source_url": "https://pubmed.ncbi.nlm.nih.gov/123/",
            "similar_patient_ids": [],
            "relevant_article_pmids": [],
            "chunk_id": "doc-1#0",
            "chunk_index": 0,
        },
        {
            "id": "doc-1",
            "text": "Second chunk",
            "title": "",
            "pmid": "",
            "source_url": "",
            "similar_patient_ids": [],
            "relevant_article_pmids": [],
            "chunk_id": "doc-1#2",
            "chunk_index": 2,
        },
    ]


def test_chunk_text_and_attrs_accepts_lowercase_pmid_metadata() -> None:
    texts, attrs = chunk_text_and_attrs(
        [SimpleNamespace(id="doc-2#0", text="Chunk with lowercase pmid", metadata={"pmid": "456"})],
        "doc-2",
    )

    assert texts == ["Chunk with lowercase pmid"]
    assert attrs[0]["pmid"] == "456"
    assert attrs[0]["source_url"] == "https://pubmed.ncbi.nlm.nih.gov/456/"


def test_chunk_text_and_attrs_attaches_train_qrels_similar_patient_ids() -> None:
    texts, attrs = chunk_text_and_attrs(
        [SimpleNamespace(id="patient-1#0", text="Chunk", metadata={"pmid": "456"})],
        "patient-1",
        similar_by_id={"patient-1": ["patient-2", "patient-3"]},
    )

    assert texts == ["Chunk"]
    assert attrs[0]["similar_patient_ids"] == ["patient-2", "patient-3"]


def test_normalized_chunk_attrs_matches_local_pmc_row_shape() -> None:
    attrs = normalized_chunk_attrs(
        {
            "PMID": "12345",
            "title": " Case report ",
            "age": [[72, "year"]],
            "gender": "female",
            "similar_patients": {"patient-b": 1, "patient-a": 1},
            "relevant_articles": {"999": 1, "111": 1},
        },
        doc_id="patient-1",
        text="chunk text",
    )

    assert attrs == {
        "id": "patient-1",
        "text": "chunk text",
        "title": "Case report",
        "pmid": "12345",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/12345/",
        "age_band": "older-adult",
        "gender": "female",
        "similar_patient_ids": ["patient-a", "patient-b"],
        "relevant_article_pmids": ["111", "999"],
        "age": 72,
    }


def test_vector_entries_uses_patient_id_for_single_chunk_documents() -> None:
    vector = [0.1] * EMBED_DIM
    entries = vector_entries(
        "doc-1",
        ["Only chunk"],
        [{"chunk_id": "doc-1#0"}],
        [vector],
    )

    assert [entry.id for entry in entries] == ["doc-1"]
    assert entries[0].vector == vector
    assert entries[0].attributes == {"chunk_id": "doc-1#0"}


def test_vector_entries_uses_chunk_ids_for_multi_chunk_documents() -> None:
    first = [0.1] * EMBED_DIM
    second = [0.3] * EMBED_DIM
    entries = vector_entries(
        "doc-1",
        ["First", "Second"],
        [{"chunk_id": "doc-1#0"}, {"chunk_id": "doc-1#1"}],
        [first, second],
    )

    assert [entry.id for entry in entries] == ["doc-1#0", "doc-1#1"]
    assert [entry.vector for entry in entries] == [first, second]
    assert [entry.attributes for entry in entries] == [
        {"id": "doc-1#0", "patient_uid": "doc-1", "chunk_id": "doc-1#0"},
        {"id": "doc-1#1", "patient_uid": "doc-1", "chunk_id": "doc-1#1"},
    ]


def test_vector_entries_uses_chunk_id_when_multi_chunk_document_is_embedded_in_batches() -> None:
    vector = [0.1] * EMBED_DIM
    entries = vector_entries(
        "doc-1",
        ["First"],
        [{"chunk_id": "doc-1#0"}],
        [vector],
        total_vectors=2,
    )

    assert [entry.id for entry in entries] == ["doc-1#0"]
    assert entries[0].attributes["id"] == "doc-1#0"


def test_vector_entries_normalizes_parent_id_for_multi_chunk_documents() -> None:
    first = [0.1] * EMBED_DIM
    second = [0.3] * EMBED_DIM

    entries = vector_entries(
        "doc-1",
        ["First", "Second"],
        [{"chunk_id": "doc-1", "chunk_index": 0}, {"chunk_id": "doc-1", "chunk_index": 1}],
        [first, second],
    )

    assert [entry.id for entry in entries] == ["doc-1#0", "doc-1#1"]
    assert [entry.attributes["chunk_id"] for entry in entries] == ["doc-1#0", "doc-1#1"]
    assert [entry.attributes["id"] for entry in entries] == ["doc-1#0", "doc-1#1"]
    assert [entry.attributes["patient_uid"] for entry in entries] == ["doc-1", "doc-1"]


def test_vector_entries_normalizes_repeated_chunk_ids() -> None:
    first = [0.1] * EMBED_DIM
    second = [0.3] * EMBED_DIM

    entries = vector_entries(
        "doc-1",
        ["First", "Second"],
        [{"chunk_id": "doc-1#0", "chunk_index": 0}, {"chunk_id": "doc-1#0", "chunk_index": 1}],
        [first, second],
    )

    assert [entry.id for entry in entries] == ["doc-1#0", "doc-1#1"]
    assert entries[1].attributes["chunk_id"] == "doc-1#1"
    assert entries[1].attributes["id"] == "doc-1#1"


def test_vector_entries_rejects_wrong_vector_dimensions() -> None:
    with pytest.raises(ValueError, match=f"expected {EMBED_DIM}-d vector"):
        vector_entries("doc-1", ["First"], [{"chunk_id": "doc-1#0"}], [[0.1, 0.2]])


def _ctx(layer, embedder) -> EmbedContext:
    return EmbedContext(
        settings=SimpleNamespace(namespace="chart-notes"),
        embed_settings=EmbedSettings(
            pipeline_id="chart-notes",
            namespace="chart-notes",
            worker_id="worker-1",
            claim_size=2,
            embed_batch_size=1,
            lease_seconds=900,
            heartbeat_seconds=999,
            poll_seconds=0.01,
            similar_qrels_split="train",
        ),
        layer=layer,
        embedder=embedder,
        similar_by_id={"doc-1": ["doc-2"]},
    )


def test_embed_settings_from_env_uses_pipeline_and_worker_overrides(monkeypatch) -> None:
    monkeypatch.setenv("HEVLAYER_PIPELINE_ID", "chart-full")
    monkeypatch.setenv("HEVLAYER_WORKER_ID", "worker-x")
    monkeypatch.setenv("CHART_EMBED_CLAIM_SIZE", "8")
    monkeypatch.setenv("CHART_EMBED_BATCH_SIZE", "4")
    monkeypatch.setenv("CHART_EMBED_LEASE_SECONDS", "120")
    monkeypatch.setenv("CHART_EMBED_HEARTBEAT_SECONDS", "15")
    monkeypatch.setenv("CHART_EMBED_POLL_SECONDS", "0.5")

    settings = EmbedSettings.from_env(SimpleNamespace(namespace="chart-notes"))

    assert settings.pipeline_id == "chart-full"
    assert settings.namespace == "chart-notes"
    assert settings.worker_id == "worker-x"
    assert settings.claim_size == 8
    assert settings.embed_batch_size == 4
    assert settings.lease_seconds == 120
    assert settings.heartbeat_seconds == 15
    assert settings.poll_seconds == 0.5
    assert settings.similar_qrels_split == "train"


def test_embed_settings_from_env_allows_similar_qrels_split_override(monkeypatch) -> None:
    monkeypatch.setenv("CHART_EMBED_SIMILAR_QRELS_SPLIT", "dev")

    settings = EmbedSettings.from_env(SimpleNamespace(namespace="chart-notes"))

    assert settings.similar_qrels_split == "dev"


def test_embed_settings_rejects_non_positive_env_values(monkeypatch) -> None:
    monkeypatch.setenv("CHART_EMBED_BATCH_SIZE", "0")

    with pytest.raises(ValueError, match="CHART_EMBED_BATCH_SIZE"):
        EmbedSettings.from_env(SimpleNamespace(namespace="chart-notes"))


@pytest.mark.anyio
async def test_embed_worker_exits_before_model_or_qrels_load_without_gateway_key(monkeypatch) -> None:
    monkeypatch.setattr(
        "indexer.embed.Settings",
        lambda: SimpleNamespace(api_key=None, gateway_url="https://gateway.example", http_timeout_seconds=1),
    )
    monkeypatch.setattr("indexer.embed.Embedder", lambda model: pytest.fail("embedder should not load without key"))
    monkeypatch.setattr(
        "indexer.embed.load_ppr_similar_patient_ids",
        lambda settings, split: pytest.fail("qrels should not load without key"),
    )

    with pytest.raises(SystemExit, match="No gateway key"):
        await amain(once=True)


class FakeEmbedder:
    def embed_passages(self, texts):
        return [[float(len(text))] * EMBED_DIM for text in texts]


@pytest.mark.anyio
async def test_run_once_claims_embeds_and_writes_vectors() -> None:
    class FakeLayer:
        def __init__(self) -> None:
            self.claim_body = None
            self.puts = []
            self.released = []
            self.failed = []

        async def claim_documents(self, pipeline_id, body):
            self.claim_body = body
            return SimpleNamespace(documents=["doc-1"])

        async def get_pipeline_document_chunks(self, pipeline_id, doc_id):
            assert pipeline_id == "chart-notes"
            assert doc_id == "doc-1"
            return [
                SimpleNamespace(id="doc-1#0", text="alpha", metadata={"pmid": "1"}),
                SimpleNamespace(id="doc-1#1", text="beta", metadata={"pmid": "1"}),
            ]

        async def put_pipeline_document_vectors(self, pipeline_id, doc_id, body):
            self.puts.append((pipeline_id, doc_id, body))

        async def release_documents(self, *args, **kwargs):
            self.released.append((args, kwargs))

        async def fail_documents(self, *args, **kwargs):
            self.failed.append((args, kwargs))

    layer = FakeLayer()

    complete = await run_once(_ctx(layer, FakeEmbedder()))

    assert complete == 1
    assert layer.claim_body.stage == "pending"
    assert layer.claim_body.claim_stage == "embedding"
    assert layer.claim_body.worker_id == "worker-1"
    assert layer.released == []
    assert layer.failed == []
    pipeline_id, doc_id, body = layer.puts[0]
    assert (pipeline_id, doc_id) == ("chart-notes", "doc-1")
    assert [entry.id for entry in body.vectors] == ["doc-1#0", "doc-1#1"]
    assert body.vectors[0].vector == [5.0] * EMBED_DIM
    assert body.vectors[1].vector == [4.0] * EMBED_DIM
    assert body.vectors[0].attributes["id"] == "doc-1#0"
    assert body.vectors[0].attributes["patient_uid"] == "doc-1"
    assert body.vectors[0].attributes["text"] == "alpha"
    assert body.vectors[0].attributes["similar_patient_ids"] == ["doc-2"]


@pytest.mark.anyio
async def test_run_once_fails_documents_with_no_text_chunks() -> None:
    class FakeLayer:
        async def claim_documents(self, pipeline_id, body):
            return SimpleNamespace(documents=["doc-empty"])

        async def get_pipeline_document_chunks(self, pipeline_id, doc_id):
            return [SimpleNamespace(id="doc-empty#0", text=" ", metadata={})]

        async def fail_documents(self, pipeline_id, document_ids, *, from_stage, worker_id):
            self.failed = (pipeline_id, document_ids, from_stage, worker_id)

    layer = FakeLayer()

    complete = await run_once(_ctx(layer, FakeEmbedder()))

    assert complete == 0
    assert layer.failed == ("chart-notes", ["doc-empty"], "embedding", "worker-1")


@pytest.mark.anyio
async def test_run_once_releases_documents_after_transient_embedding_failure() -> None:
    class BrokenEmbedder:
        def embed_passages(self, texts):
            raise RuntimeError("gpu unavailable")

    class FakeLayer:
        async def claim_documents(self, pipeline_id, body):
            return SimpleNamespace(documents=["doc-1"])

        async def get_pipeline_document_chunks(self, pipeline_id, doc_id):
            return [SimpleNamespace(id="doc-1#0", text="alpha", metadata={})]

        async def release_documents(self, pipeline_id, document_ids, *, from_stage, worker_id):
            self.released = (pipeline_id, document_ids, from_stage, worker_id)

    layer = FakeLayer()

    complete = await run_once(_ctx(layer, BrokenEmbedder()))

    assert complete == 0
    assert layer.released == ("chart-notes", ["doc-1"], "embedding", "worker-1")


@pytest.mark.anyio
async def test_run_once_handles_mixed_permanent_and_transient_failures() -> None:
    class BrokenEmbedder:
        def embed_passages(self, texts):
            raise RuntimeError("gpu unavailable")

    class FakeLayer:
        def __init__(self) -> None:
            self.failed = None
            self.released = None

        async def claim_documents(self, pipeline_id, body):
            return SimpleNamespace(documents=["doc-empty", "doc-transient"])

        async def get_pipeline_document_chunks(self, pipeline_id, doc_id):
            if doc_id == "doc-empty":
                return [SimpleNamespace(id="doc-empty#0", text=" ", metadata={})]
            return [SimpleNamespace(id="doc-transient#0", text="alpha", metadata={})]

        async def fail_documents(self, pipeline_id, document_ids, *, from_stage, worker_id):
            self.failed = (pipeline_id, document_ids, from_stage, worker_id)

        async def release_documents(self, pipeline_id, document_ids, *, from_stage, worker_id):
            self.released = (pipeline_id, document_ids, from_stage, worker_id)

    layer = FakeLayer()

    complete = await run_once(_ctx(layer, BrokenEmbedder()))

    assert complete == 0
    assert layer.failed == ("chart-notes", ["doc-empty"], "embedding", "worker-1")
    assert layer.released == ("chart-notes", ["doc-transient"], "embedding", "worker-1")
