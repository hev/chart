from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.parse
import urllib.request
from collections.abc import Iterable
from typing import Any

from datasets import load_dataset
import httpx
from hevlayer.client import HevlayerError

from chart_common.config import Settings
from chart_common.gateway import close_client, make_client


DATASETS_SERVER = "https://datasets-server.huggingface.co"
HF_DATASET_BASE = "https://huggingface.co/datasets"


def _log(message: str) -> None:
    print(message, flush=True)


def _json_env(name: str) -> dict[str, Any]:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is required")
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise SystemExit(f"{name} must be a JSON object")
    return parsed


def _source_ref() -> dict[str, Any]:
    source = _json_env("HEVLAYER_SOURCE_REF")
    if source.get("kind") != "huggingface":
        raise SystemExit(f"unsupported HEVLAYER_SOURCE_REF.kind={source.get('kind')!r}")
    return source


def _row_value(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("row")
    return value if isinstance(value, dict) else row


def _request_rows(source: dict[str, Any], *, offset: int, length: int) -> list[dict[str, Any]]:
    params = {
        "dataset": source["dataset"],
        "split": source.get("split", "train"),
        "offset": str(offset),
        "length": str(length),
    }
    if source.get("config"):
        params["config"] = source["config"]
    if source.get("revision"):
        params["revision"] = source["revision"]
    url = f"{DATASETS_SERVER}/rows?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=float(os.environ.get("CHART_HF_SOURCE_TIMEOUT_SECONDS", "60"))) as response:
        body = json.loads(response.read().decode("utf-8"))
    rows = body.get("rows") or []
    if not isinstance(rows, list):
        raise RuntimeError("datasets-server response did not include rows[]")
    return rows


def _stream_rows(source: dict[str, Any], *, offset: int) -> Iterable[dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "path": source["dataset"],
        "split": source.get("split", "train"),
        "streaming": True,
    }
    if source.get("config"):
        kwargs["name"] = source["config"]
    if source.get("revision"):
        kwargs["revision"] = source["revision"]
    stream = load_dataset(**kwargs)
    for index, row in enumerate(stream):
        if index >= offset:
            yield row


def _direct_json_url(source: dict[str, Any]) -> str:
    filename = os.environ.get("CHART_HF_SOURCE_JSON_FILE", "PMC-Patients-V2.json")
    revision = source.get("revision") or "main"
    dataset = source["dataset"].strip("/")
    return f"{HF_DATASET_BASE}/{dataset}/resolve/{revision}/{urllib.parse.quote(filename)}"


def _stream_json_array(source: dict[str, Any], *, offset: int) -> Iterable[dict[str, Any]]:
    timeout = float(os.environ.get("CHART_HF_SOURCE_TIMEOUT_SECONDS", "60"))
    url = _direct_json_url(source)
    with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as response:
        response.raise_for_status()
        yield from _iter_json_array_rows(
            (chunk.decode("utf-8") for chunk in response.iter_bytes(chunk_size=1024 * 1024)),
            offset=offset,
        )


def _iter_json_array_rows(chunks: Iterable[str], *, offset: int = 0) -> Iterable[dict[str, Any]]:
    decoder = json.JSONDecoder()
    buffer = ""
    index = 0
    closed = False
    max_buffer_bytes = int(os.environ.get("CHART_HF_SOURCE_MAX_BUFFER_BYTES", str(32 * 1024 * 1024)))
    for chunk in chunks:
        buffer += chunk
        if len(buffer.encode("utf-8")) > max_buffer_bytes:
            raise RuntimeError(f"JSON stream parser buffer exceeded {max_buffer_bytes} bytes")
        while True:
            buffer = buffer.lstrip()
            if not buffer:
                break
            if buffer[0] in "[,":
                buffer = buffer[1:]
                continue
            if buffer[0] == "]":
                closed = True
                return
            try:
                row, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                # Need more bytes for the current object.
                break
            buffer = buffer[end:]
            if isinstance(row, dict):
                if index >= offset:
                    yield row
                index += 1
    trailing = buffer.strip()
    if trailing == "]":
        closed = True
    if not closed:
        raise RuntimeError("unterminated JSON array from Hugging Face dataset file")


def _chunks_for_text(doc_id: str, text: str, attributes: dict[str, Any], chunk: dict[str, Any]) -> list[dict[str, Any]]:
    # The stock worker is tokenizer-aware. This replacement keeps the same
    # pipeline contract and uses conservative character windows so no single
    # PMC-Patients summary becomes an oversized embed request.
    size = int(os.environ.get("CHART_HF_SOURCE_CHUNK_CHARS", "2000"))
    overlap = int(os.environ.get("CHART_HF_SOURCE_CHUNK_OVERLAP_CHARS", "256"))
    if size <= 0:
        size = 2000
    if overlap < 0 or overlap >= size:
        overlap = 0
    text = text.strip()
    if not text:
        return []
    chunks: list[dict[str, Any]] = []
    start = 0
    index = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunk_id = doc_id if index == 0 and end == len(text) else f"{doc_id}#{index}"
            chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        **attributes,
                        "source": "huggingface",
                        "chunk_index": index,
                        "chunk_strategy": chunk.get("strategy", "fixed"),
                    },
                }
            )
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
        index += 1
    return chunks


def _document(row: dict[str, Any], source: dict[str, Any]) -> tuple[str, list[dict[str, Any]]] | None:
    mapping = source.get("mapping") or {}
    value = _row_value(row)
    id_field = mapping.get("id", "id")
    text_field = mapping.get("text", "text")
    raw_id = value.get(id_field)
    text = value.get(text_field)
    if raw_id is None or not text:
        return None
    doc_id = str(raw_id)
    attrs = {}
    for field in mapping.get("attributes") or []:
        out_name = "pmid" if field == "PMID" else field
        attrs[out_name] = value.get(field)
    if value.get("title") and "title" not in attrs:
        attrs["title"] = value.get("title")
    chunks = _chunks_for_text(doc_id, str(text), attrs, source.get("chunk") or {})
    return (doc_id, chunks) if chunks else None


async def _ensure_pipeline(layer: Any, *, pipeline_id: str, target_namespace: str) -> None:
    try:
        await layer.create_pipeline(
            {
                "id": pipeline_id,
                "target_namespace": target_namespace,
                "distance_metric": "cosine_distance",
            }
        )
    except HevlayerError as exc:
        if exc.status_code != 409:
            raise


async def _put_chunks_with_retry(layer: Any, pipeline_id: str, doc_id: str, chunks: list[dict[str, Any]]) -> None:
    attempts = int(os.environ.get("CHART_HF_SOURCE_WRITE_ATTEMPTS", "5"))
    delay = float(os.environ.get("CHART_HF_SOURCE_WRITE_RETRY_SECONDS", "2"))
    timeout = float(os.environ.get("CHART_HF_SOURCE_WRITE_TIMEOUT_SECONDS", "60"))
    for attempt in range(1, attempts + 1):
        try:
            await asyncio.wait_for(
                layer.put_pipeline_document_chunks(pipeline_id, doc_id, {"chunks": chunks}),
                timeout=timeout,
            )
            return
        except TimeoutError:
            if attempt == attempts:
                raise
            _log(f"transient gateway timeout staging {doc_id}; retry {attempt}/{attempts}")
            await asyncio.sleep(delay * attempt)
        except HevlayerError as exc:
            if attempt == attempts or exc.status_code not in {429, 500, 502, 503, 504}:
                raise
            _log(f"transient gateway error staging {doc_id}: status={exc.status_code}; retry {attempt}/{attempts}")
            await asyncio.sleep(delay * attempt)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            if attempt == attempts:
                raise
            _log(f"transient gateway transport error staging {doc_id}: {type(exc).__name__}; retry {attempt}/{attempts}")
            await asyncio.sleep(delay * attempt)


async def _put_documents(
    layer: Any,
    pipeline_id: str,
    documents: list[tuple[str, list[dict[str, Any]]]],
    *,
    concurrency: int,
) -> None:
    if not documents:
        return
    if concurrency <= 1:
        for doc_id, chunks in documents:
            await _put_chunks_with_retry(layer, pipeline_id, doc_id, chunks)
        return

    semaphore = asyncio.Semaphore(concurrency)

    async def put_one(doc_id: str, chunks: list[dict[str, Any]]) -> None:
        async with semaphore:
            await _put_chunks_with_retry(layer, pipeline_id, doc_id, chunks)

    await asyncio.gather(*(put_one(doc_id, chunks) for doc_id, chunks in documents))


async def run() -> dict[str, Any]:
    settings = Settings()
    source = _source_ref()
    pipeline_id = os.environ.get("HEVLAYER_PIPELINE_ID") or source.get("pipelineId") or "chart-notes"
    target_namespace = os.environ.get("HEVLAYER_TARGET_NAMESPACE") or "chart-notes"
    page_size = int(os.environ.get("CHART_HF_SOURCE_PAGE_SIZE", "100"))
    max_rows = int(os.environ.get("CHART_HF_SOURCE_MAX_ROWS", "0"))
    start_offset = int(os.environ.get("CHART_HF_SOURCE_START_OFFSET", "0"))
    sleep_seconds = float(os.environ.get("CHART_HF_SOURCE_PAGE_SLEEP_SECONDS", "0"))
    write_concurrency = int(os.environ.get("CHART_HF_SOURCE_WRITE_CONCURRENCY", "4"))
    mode = os.environ.get("CHART_HF_SOURCE_MODE", "direct-json")
    layer = make_client(settings)
    staged = 0
    skipped = 0
    offset = start_offset
    started = time.time()
    try:
        _log(f"ensuring pipeline {pipeline_id} -> {target_namespace}")
        await _ensure_pipeline(layer, pipeline_id=pipeline_id, target_namespace=target_namespace)
        _log(f"starting Hugging Face source mode={mode} dataset={source.get('dataset')} offset={start_offset}")
        if mode == "rows-api":
            while True:
                if max_rows and staged + skipped >= max_rows:
                    break
                rows = _request_rows(source, offset=offset, length=page_size)
                if not rows:
                    break
                documents: list[tuple[str, list[dict[str, Any]]]] = []
                for row in rows:
                    if max_rows and staged + skipped >= max_rows:
                        break
                    doc = _document(row, source)
                    if doc is None:
                        skipped += 1
                        continue
                    documents.append(doc)
                await _put_documents(layer, pipeline_id, documents, concurrency=write_concurrency)
                staged += len(documents)
                offset += len(rows)
                if documents:
                    _log(f"staged documents={staged} next_offset={offset}")
                if len(rows) < page_size:
                    break
                if sleep_seconds:
                    await asyncio.sleep(sleep_seconds)
        else:
            rows_iter = _stream_json_array(source, offset=start_offset) if mode == "direct-json" else _stream_rows(source, offset=start_offset)
            documents: list[tuple[str, list[dict[str, Any]]]] = []
            for row in rows_iter:
                if max_rows and staged + skipped >= max_rows:
                    break
                doc = _document(row, source)
                if doc is None:
                    skipped += 1
                    offset += 1
                    continue
                documents.append(doc)
                offset += 1
                if len(documents) >= page_size:
                    await _put_documents(layer, pipeline_id, documents, concurrency=write_concurrency)
                    staged += len(documents)
                    documents = []
                    _log(f"staged documents={staged} next_offset={offset}")
                    if sleep_seconds:
                        await asyncio.sleep(sleep_seconds)
            if documents:
                await _put_documents(layer, pipeline_id, documents, concurrency=write_concurrency)
                staged += len(documents)
                _log(f"staged documents={staged} next_offset={offset}")
    finally:
        await close_client(layer)
    return {
        "pipeline_id": pipeline_id,
        "target_namespace": target_namespace,
        "dataset": source.get("dataset"),
        "split": source.get("split", "train"),
        "start_offset": start_offset,
        "next_offset": offset,
        "mode": mode,
        "staged": staged,
        "skipped": skipped,
        "elapsed_seconds": round(time.time() - started, 3),
    }


def main() -> None:
    print(json.dumps(asyncio.run(run()), indent=2))


if __name__ == "__main__":
    main()
