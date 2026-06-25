"""GPU embedding stage for the full chart index.

The local indexer is intentionally bounded for smoke runs. This worker is the
Phase-6 path: claim staged Pipeline documents, embed chunks with the exact Arctic
v1.5 model on a GPU-capable image, and write vectors through Layer's pipeline
vector API so the run is resumable by document id.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from hevlayer import (
    AsyncHevlayer,
    ClaimDocumentsRequest,
    HeartbeatDocumentsRequest,
    PutVectorsRequest,
    VectorEntry,
)

from chart_common.config import EMBED_DIM, Settings
from chart_common.embed import Embedder
from chart_common.records import NoteRecord
from indexer.dataset import load_ppr_similar_patient_ids

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _env_float(name: str, default: float, *, minimum: float = 0.1) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


@dataclass
class EmbedSettings:
    pipeline_id: str
    namespace: str
    worker_id: str
    claim_size: int
    embed_batch_size: int
    lease_seconds: int
    heartbeat_seconds: int
    poll_seconds: float
    similar_qrels_split: str

    @classmethod
    def from_env(cls, settings: Settings) -> "EmbedSettings":
        pipeline_id = os.environ.get("HEVLAYER_PIPELINE_ID") or settings.namespace
        return cls(
            pipeline_id=pipeline_id,
            namespace=settings.namespace,
            worker_id=os.environ.get("HEVLAYER_WORKER_ID", f"chart-embed-{os.getpid()}"),
            claim_size=_env_int("CHART_EMBED_CLAIM_SIZE", 64),
            embed_batch_size=_env_int("CHART_EMBED_BATCH_SIZE", 32),
            lease_seconds=_env_int("CHART_EMBED_LEASE_SECONDS", 900),
            heartbeat_seconds=_env_int("CHART_EMBED_HEARTBEAT_SECONDS", 60),
            poll_seconds=_env_float("CHART_EMBED_POLL_SECONDS", 2.0),
            similar_qrels_split=os.environ.get("CHART_EMBED_SIMILAR_QRELS_SPLIT", "train"),
        )


@dataclass
class EmbedContext:
    settings: Settings
    embed_settings: EmbedSettings
    layer: AsyncHevlayer
    embedder: Embedder
    similar_by_id: dict[str, list[str]]


def chunk_text_and_attrs(
    chunks: list[Any],
    doc_id: str,
    *,
    similar_by_id: dict[str, list[str]] | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    texts: list[str] = []
    attrs: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        text = str(getattr(chunk, "text", None) or "").strip()
        if not text:
            continue
        metadata = getattr(chunk, "metadata", None) or {}
        chunk_id = str(getattr(chunk, "id", None) or f"{doc_id}#{index}")
        row_attrs = normalized_chunk_attrs(
            metadata,
            doc_id=doc_id,
            text=text,
            similar_patient_ids=(similar_by_id or {}).get(doc_id),
        )
        row_attrs.setdefault("id", doc_id)
        row_attrs.setdefault("chunk_id", chunk_id)
        row_attrs.setdefault("chunk_index", index)
        row_attrs.setdefault("text", text)
        texts.append(text)
        attrs.append(row_attrs)
    return texts, attrs


def normalized_chunk_attrs(
    metadata: dict[str, Any],
    *,
    doc_id: str,
    text: str,
    similar_patient_ids: list[str] | None = None,
) -> dict[str, Any]:
    row = dict(metadata)
    row.setdefault("patient_uid", doc_id)
    row.setdefault("patient", text)
    normalized = NoteRecord.from_row(row).to_upsert()
    normalized.pop("vector", None)
    if similar_patient_ids is not None:
        normalized["similar_patient_ids"] = similar_patient_ids
    return normalized


def vector_entries(
    doc_id: str,
    texts: list[str],
    attrs: list[dict[str, Any]],
    vectors: list[list[float]],
    *,
    total_vectors: int | None = None,
) -> list[VectorEntry]:
    total_vectors = total_vectors or len(vectors)
    entries = []
    seen_ids: set[str] = set()
    for index, (attr, vector) in enumerate(zip(attrs, vectors, strict=True)):
        if len(vector) != EMBED_DIM:
            raise ValueError(f"{doc_id}: expected {EMBED_DIM}-d vector for chunk {index}, got {len(vector)}")
        if total_vectors == 1:
            vector_id = doc_id
        else:
            candidate = str(attr.get("chunk_id") or "").strip()
            if not candidate or candidate == doc_id or candidate in seen_ids:
                candidate = f"{doc_id}#{attr.get('chunk_index', index)}"
            vector_id = candidate
            attr = {**attr, "id": vector_id, "patient_uid": doc_id, "chunk_id": vector_id}
        if vector_id in seen_ids:
            raise ValueError(f"{doc_id}: duplicate vector id after normalization: {vector_id}")
        seen_ids.add(vector_id)
        entries.append(VectorEntry(id=vector_id, vector=vector, attributes=attr))
    return entries


async def run_once(ctx: EmbedContext) -> int:
    claim = await ctx.layer.claim_documents(
        ctx.embed_settings.pipeline_id,
        ClaimDocumentsRequest(
            stage="pending",
            claim_stage="embedding",
            limit=ctx.embed_settings.claim_size,
            worker_id=ctx.embed_settings.worker_id,
            lease_seconds=ctx.embed_settings.lease_seconds,
        ),
    )
    doc_ids = list(claim.documents)
    if not doc_ids:
        return 0

    active = set(doc_ids)
    heartbeat = asyncio.create_task(_heartbeat(ctx, active))
    release: list[str] = []
    fail: list[str] = []
    complete = 0
    try:
        for doc_id in doc_ids:
            try:
                chunks = await ctx.layer.get_pipeline_document_chunks(ctx.embed_settings.pipeline_id, doc_id)
                texts, attrs = chunk_text_and_attrs(
                    chunks,
                    doc_id,
                    similar_by_id=ctx.similar_by_id,
                )
                if not texts:
                    fail.append(doc_id)
                    continue
                entries: list[VectorEntry] = []
                for start in range(0, len(texts), ctx.embed_settings.embed_batch_size):
                    end = start + ctx.embed_settings.embed_batch_size
                    vectors = await asyncio.to_thread(ctx.embedder.embed_passages, texts[start:end])
                    entries.extend(
                        vector_entries(
                            doc_id,
                            texts[start:end],
                            attrs[start:end],
                            vectors,
                            total_vectors=len(texts),
                        )
                    )
                await ctx.layer.put_pipeline_document_vectors(
                    ctx.embed_settings.pipeline_id,
                    doc_id,
                    PutVectorsRequest(vectors=entries),
                )
                complete += 1
            except Exception:
                logger.exception("failed to embed document", extra={"doc_id": doc_id})
                release.append(doc_id)
            finally:
                active.discard(doc_id)
    finally:
        heartbeat.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat

    if fail:
        await ctx.layer.fail_documents(
            ctx.embed_settings.pipeline_id,
            fail,
            from_stage="embedding",
            worker_id=ctx.embed_settings.worker_id,
        )
    if release:
        await ctx.layer.release_documents(
            ctx.embed_settings.pipeline_id,
            sorted(set(release)),
            from_stage="embedding",
            worker_id=ctx.embed_settings.worker_id,
        )
    return complete


async def run_worker(ctx: EmbedContext, stop: asyncio.Event, *, once: bool = False) -> None:
    while not stop.is_set():
        embedded = await run_once(ctx)
        if once:
            return
        if embedded == 0:
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(stop.wait(), ctx.embed_settings.poll_seconds)


async def _heartbeat(ctx: EmbedContext, active: set[str]) -> None:
    while True:
        await asyncio.sleep(ctx.embed_settings.heartbeat_seconds)
        if active:
            await ctx.layer.heartbeat_documents(
                ctx.embed_settings.pipeline_id,
                HeartbeatDocumentsRequest(
                    document_ids=sorted(active),
                    stage="embedding",
                    worker_id=ctx.embed_settings.worker_id,
                ),
            )


def _install_signals(stop: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            signal.signal(sig, lambda *_: loop.call_soon_threadsafe(stop.set))


async def amain(*, once: bool = False) -> None:
    logging.basicConfig(level=logging.INFO)
    settings = Settings()
    if not settings.api_key:
        raise SystemExit(
            "No gateway key. Set LAYER_GATEWAY_API_KEY in .env — it's the upstream "
            "Turbopuffer key (1Password: layer-turbopuffer / mesh-staging)."
        )
    embed_settings = EmbedSettings.from_env(settings)
    layer = AsyncHevlayer(
        api_key=settings.api_key,
        base_url=settings.gateway_url,
        timeout=settings.http_timeout_seconds,
    )
    ctx = EmbedContext(
        settings=settings,
        embed_settings=embed_settings,
        layer=layer,
        embedder=Embedder(settings.embed_model),
        similar_by_id=load_ppr_similar_patient_ids(
            settings,
            split=embed_settings.similar_qrels_split,
        ),
    )
    stop = asyncio.Event()
    _install_signals(stop)
    try:
        await run_worker(ctx, stop, once=once)
    finally:
        await layer.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU embedding worker for chart Pipeline documents")
    parser.add_argument("--once", action="store_true", help="claim once and exit")
    args = parser.parse_args()
    asyncio.run(amain(once=args.once))


if __name__ == "__main__":
    main()
