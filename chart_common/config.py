from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Snowflake/snowflake-arctic-embed-m-v1.5 output dimensionality. Documented here
# so the schema (index) side and the query side agree without loading the model.
# This is the SAME embedding model ../notesearch runs (it slugs its namespaces
# `…-arctic`), so retrieval behavior transfers from this public twin to Trio's
# real workload. See RFC 0076 § Embedding.
EMBED_DIM = 768

# Arctic is a query/document-asymmetric model: documents embed as-is, a query is
# prefixed before embedding. Getting this wrong silently degrades semantic
# recall — RFC 0076 calls it out as a worked detail to get visibly right.
ARCTIC_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Arctic-embed-m-v1.5 max sequence length. Chunks are sized to this window; the
# model silently truncates past it (RFC 0076 § Chunking).
EMBED_MAX_TOKENS = 512

# Phase-6 target corpus size from PMC-Patients. Kept shared so status, gates, and
# cost extrapolators use the same row target.
FULL_CORPUS_NOTES = 167_000


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Gateway. The backing store is Turbopuffer (deploy/vectorstore.yaml,
    # kind=turbopuffer — the kind=search cutover was attempted but never
    # applied): the key is a Layer inbound key scoped to chart-notes, which
    # needs the mirror grant on vectorstore.turbopuffer-default (hev/layer#145).
    gateway_url: str = Field(
        default="https://aws-us-east-1.hevlayer.com",
        validation_alias=AliasChoices("LAYER_GATEWAY_URL", "HEVLAYER_BASE_URL"),
    )
    api_key: str | None = Field(
        default=None,
        validation_alias="LAYER_GATEWAY_API_KEY",
    )
    namespace: str = Field(default="chart-notes", validation_alias="CHART_NAMESPACE")
    embed_model: str = Field(
        default="Snowflake/snowflake-arctic-embed-m-v1.5",
        validation_alias="CHART_EMBED_MODEL",
    )
    # The configured Agent (deploy/agent.yaml) the agentic-search option invokes
    # via POST /v2/agents/{name}/query (docs/api/agents). The inbound key needs
    # the agent.<name> entitlement (deploy/apikey.yaml).
    agent_name: str = Field(default="chart-notes", validation_alias="CHART_AGENT_NAME")
    http_timeout_seconds: float = 60.0

    # Semantic facet counts: a semantic-routed query has no exact "match set" the
    # way a keyword query does, so the live-count scan (Scans API) counts rows
    # within a cosine-distance ball of the query vector (an `ann` radius scan),
    # not a BM25 `fts` predicate. With Arctic's asymmetric query embedding even the
    # best matches sit near ~0.5 cosine distance, so the ball must open past that;
    # 0.6 captures a focused, clearly-relevant neighborhood (a few hundred cases on
    # the demo corpus) rather than the whole index. Approximate by ANN recall.
    semantic_radius: float = Field(
        default=0.6, validation_alias="CHART_SEMANTIC_RADIUS"
    )

    # Dataset, pinned to an exact revision (RFC 0053 exact-replay discipline).
    # PMC-Patients: 167k real, de-identified, public case-report notes
    # (CC-BY-NC-SA-4.0). ReCDS qrels (eval/) must be loaded from the matching
    # revision.
    dataset_repo: str = "zhengyun21/PMC-Patients"
    dataset_revision: str = "28d8836518f86d4f1e6358ea8ec09977023e5766"
    dataset_split: str = "train"

    recds_repo: str = "zhengyun21/PMC-Patients-ReCDS"
    recds_revision: str = "a27717bb27679cf0860305997685547ca01b3dd1"
