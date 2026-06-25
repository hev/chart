from __future__ import annotations

from .config import ARCTIC_QUERY_PREFIX


class Embedder:
    """Snowflake Arctic Embed m v1.5 — the model ../notesearch runs.

    Arctic is asymmetric: passages embed as-is, queries get the Arctic
    instruction prefix (``ARCTIC_QUERY_PREFIX``). We apply that prefix EXPLICITLY
    on the query path rather than trusting a library helper, because it is the
    detail RFC 0076 flags as silently recall-degrading when wrong, and we want it
    visible in this code.

    The dev backend prefers fastembed when its ONNX catalog has the requested
    model. As of this pin, fastembed has older Arctic variants but not
    `snowflake-arctic-embed-m-v1.5`, so the exact notesearch model falls back to
    sentence-transformers. The prefix handling here does not change either way.
    """

    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding

        self.model_name = model_name
        supported = {m.get("model") for m in TextEmbedding.list_supported_models()}
        supported.update(m.get("sources", {}).get("hf") for m in TextEmbedding.list_supported_models())
        if model_name in supported:
            self.backend = "fastembed"
            self.model = TextEmbedding(model_name=model_name)
        else:
            from sentence_transformers import SentenceTransformer

            self.backend = "sentence-transformers"
            self.model = SentenceTransformer(model_name)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if self.backend == "fastembed":
            return [vector.tolist() for vector in self.model.embed(texts)]
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        # The asymmetric prefix — the one thing not to get wrong.
        return self._encode([ARCTIC_QUERY_PREFIX + text])[0]
