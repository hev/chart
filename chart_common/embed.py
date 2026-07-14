from __future__ import annotations

import math

from .config import ARCTIC_QUERY_PREFIX, LI_QUERY_MAX_CLAUSES


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


class LateInteractionEmbedder:
    """ColBERT-style token-bag embedder for the late-interaction try-out.

    Emits one vector PER TOKEN (a "bag") instead of one per passage — the
    [][N]f32 column the Turbopuffer late-interaction beta scores with MaxSim
    (sum of per-query-token closest distances). fastembed's ONNX catalog carries
    the supported models; the default (answerai-colbert-small-v1, 96-d) is picked
    for CPU-viable indexing, not clinical-domain fit.

    Like ColBERT generally, this is asymmetric in a different way from Arctic:
    documents embed to as many vectors as they have tokens (up to the model's
    doc window), while queries are padded/truncated to a short fixed budget
    (32 tokens for the ColBERT family). Long ReCDS patient-note queries are
    therefore heavily truncated on the query side — an eval caveat, not a bug.
    """

    def __init__(self, model_name: str) -> None:
        from fastembed import LateInteractionTextEmbedding

        self.model_name = model_name
        self.model = LateInteractionTextEmbedding(model_name=model_name)

    def embed_passages(self, texts: list[str]) -> list[list[list[float]]]:
        return [bag.tolist() for bag in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[list[float]]:
        bag = next(iter(self.model.query_embed([text]))).tolist()
        return select_query_clauses(bag, LI_QUERY_MAX_CLAUSES)


def select_query_clauses(bag: list[list[float]], max_clauses: int) -> list[list[float]]:
    """Reduce a query token bag to `max_clauses` vectors for the beta clause cap.

    Turbopuffer's late-interaction beta caps query bags at 8 rank_by clauses
    (config.LI_QUERY_MAX_CLAUSES), a quarter of ColBERT's 32-token query
    encoding. The obvious reduction — mean-pooling adjacent tokens — is a trap:
    averaging unit vectors that point different ways yields a short, generic
    vector near the centroid of token space, which is near-equidistant to every
    long note, so MaxSim stops discriminating (measured: a 0.015 $dist spread
    across the whole corpus, i.e. near-random ranking).

    Instead SELECT 8 real token vectors, never averaging: start from the
    highest-norm token (ColBERT encodes token importance as vector norm) and
    greedily add the token farthest from those already chosen (farthest-point
    sampling). That keeps the most salient, most diverse query terms — which is
    what MaxSim needs — and empirically restores a wide, discriminating $dist
    spread. Still lossy vs true 32-clause MaxSim; the honest fix is a raised
    cap, tracked as tpuf beta feedback."""
    if len(bag) <= max_clauses:
        return bag
    norms = [math.sqrt(sum(x * x for x in v)) or 1.0 for v in bag]
    chosen = [max(range(len(bag)), key=lambda i: norms[i])]
    while len(chosen) < max_clauses:
        # cosine distance to the nearest already-chosen vector (higher = more
        # novel); norm-weight so a salient-but-slightly-redundant token can still
        # win over a low-information outlier.
        def novelty(i: int) -> float:
            nearest = min(
                1.0 - _cos(bag[i], bag[j], norms[i], norms[j]) for j in chosen
            )
            return nearest * norms[i]

        candidates = (i for i in range(len(bag)) if i not in chosen)
        chosen.append(max(candidates, key=novelty))
    return [bag[i] for i in sorted(chosen)]


def _cos(a: list[float], b: list[float], na: float, nb: float) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True)) / (na * nb)
