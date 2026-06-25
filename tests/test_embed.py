import sys
from types import SimpleNamespace

from chart_common.config import ARCTIC_QUERY_PREFIX
from chart_common.embed import Embedder


class _Vector:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class _Encoded:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


def test_embedder_uses_fastembed_when_exact_model_is_supported(monkeypatch) -> None:
    class FakeTextEmbedding:
        def __init__(self, *, model_name):
            self.model_name = model_name

        @staticmethod
        def list_supported_models():
            return [{"model": "exact-model", "sources": {"hf": "hf-model"}}]

        def embed(self, texts):
            return [_Vector([float(len(text))]) for text in texts]

    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=FakeTextEmbedding))

    embedder = Embedder("hf-model")

    assert embedder.backend == "fastembed"
    assert embedder.embed_passages(["abc"]) == [[3.0]]
    assert embedder.embed_query("abc") == [[float(len(ARCTIC_QUERY_PREFIX + "abc"))]][0]


def test_embedder_falls_back_to_sentence_transformers_and_prefixes_queries(monkeypatch) -> None:
    seen = []

    class FakeTextEmbedding:
        @staticmethod
        def list_supported_models():
            return [{"model": "other-model", "sources": {"hf": "other-hf"}}]

    class FakeSentenceTransformer:
        def __init__(self, model_name):
            self.model_name = model_name

        def encode(self, texts, *, normalize_embeddings):
            seen.append((list(texts), normalize_embeddings))
            return _Encoded([[float(len(text))] for text in texts])

    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=FakeTextEmbedding))
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    embedder = Embedder("Snowflake/snowflake-arctic-embed-m-v1.5")

    assert embedder.backend == "sentence-transformers"
    assert embedder.embed_passages(["passage"]) == [[7.0]]
    assert embedder.embed_query("query") == [float(len(ARCTIC_QUERY_PREFIX + "query"))]
    assert seen == [
        (["passage"], True),
        ([ARCTIC_QUERY_PREFIX + "query"], True),
    ]
