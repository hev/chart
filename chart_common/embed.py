from __future__ import annotations

from .config import ARCTIC_QUERY_PREFIX


class Embedder:
    """Snowflake Arctic Embed m v1.5 — the model ../notesearch runs.

    Arctic is asymmetric: passages embed as-is, queries get the Arctic
    instruction prefix (``ARCTIC_QUERY_PREFIX``). We apply that prefix EXPLICITLY
    on the query path rather than trusting a library helper, because it is the
    detail RFC 0076 flags as silently recall-degrading when wrong, and we want it
    visible in this code.

    The dev backend wants a CPU-friendly runtime. fastembed ships Arctic Embed
    variants; if the exact `-m-v1.5` tag is not in its catalog, swap the body of
    `_encode` for sentence-transformers (``SentenceTransformer(model_name)``),
    which is what notesearch's GPU pipeline uses. The prefix handling here does
    not change either way.
    """

    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding

        self.model_name = model_name
        self.model = TextEmbedding(model_name=model_name)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.model.embed(texts)]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        # The asymmetric prefix — the one thing not to get wrong.
        return self._encode([ARCTIC_QUERY_PREFIX + text])[0]
