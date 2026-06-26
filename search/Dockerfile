ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

# Arctic weights are baked under HF_HOME so cold starts don't pay an HF
# round-trip; the runtime then runs offline (see the OFFLINE env in the Helm
# configmap). The CPU torch wheel index keeps the ~5GB of CUDA libs out of the
# image — this service embeds Arctic on the always-on CPU node, never a GPU.
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/model-cache \
    TRANSFORMERS_CACHE=/opt/model-cache

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Pass the sibling SDK with BuildKit (matches ../shop/search/Dockerfile):
#   --build-context layer_client=../layer/clients/python
COPY --from=layer_client . /layer/clients/python

# CPU-only torch first, so sentence-transformers binds to the CPU build already
# present instead of pulling the default CUDA wheel.
RUN pip install --no-cache-dir "torch>=2.5" --index-url https://download.pytorch.org/whl/cpu

# Serving import graph only (search/app.py -> chart_common.{config,embed,gateway}):
# fastembed for the catalog check, sentence-transformers for the exact Arctic
# v1.5 fallback, hevlayer for the gateway client, fastapi/uvicorn for serving.
# `datasets` is an indexer/eval dep and is not on the serving path, so it is
# intentionally omitted to keep the image lean.
RUN pip install --no-cache-dir \
        "fastapi>=0.115" \
        "uvicorn[standard]>=0.30" \
        "sentence-transformers>=3.0" \
        "fastembed>=0.4" \
        "pydantic>=2.7" \
        "pydantic-settings>=2.3" \
    && pip install --no-cache-dir /layer/clients/python

# Bake the exact Arctic model notesearch runs. fastembed's ONNX catalog lacks
# v1.5, so Embedder falls back to sentence-transformers (chart_common/embed.py)
# — bake via the same path so the runtime cache hit is identical.
ARG EMBED_MODEL=Snowflake/snowflake-arctic-embed-m-v1.5
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBED_MODEL}'); print('baked', '${EMBED_MODEL}')"

# search/app.py resolves STATIC as ../web/static relative to itself, so the tree
# must land at /app/{search,chart_common,web}.
COPY chart_common /app/chart_common
COPY search /app/search
COPY web /app/web

# Build-time import smoke: fail the build here if the serving import graph is
# missing a dependency, rather than at the pod's readiness probe in prod.
RUN python -c "import search.app; print('search.app imports OK')"

EXPOSE 8080
CMD ["uvicorn", "search.app:app", "--host", "0.0.0.0", "--port", "8080"]
