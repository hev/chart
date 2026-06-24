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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Gateway. deriveFromStore auth: the key IS the upstream Turbopuffer key
    # (1Password: layer-turbopuffer / mesh-staging vault).
    gateway_url: str = Field(
        default="https://aws-us-east-1.hevlayer.com",
        validation_alias=AliasChoices("LAYER_GATEWAY_URL", "HEVLAYER_BASE_URL"),
    )
    api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LAYER_GATEWAY_API_KEY", "LAYER_TURBOPUFFER_KEY"),
    )
    namespace: str = Field(default="chart-notes", validation_alias="CHART_NAMESPACE")
    embed_model: str = Field(
        default="Snowflake/snowflake-arctic-embed-m-v1.5",
        validation_alias="CHART_EMBED_MODEL",
    )
    http_timeout_seconds: float = 60.0

    # Dataset, pinned to an exact revision (RFC 0053 exact-replay discipline).
    # PMC-Patients: 167k real, de-identified, public case-report notes
    # (CC-BY-NC-SA-4.0). TODO: pin an exact commit SHA before first publish; the
    # ReCDS qrels (eval/) must be loaded from the matching revision.
    dataset_repo: str = "zhengyun21/PMC-Patients"
    dataset_revision: str = "main"
    dataset_split: str = "train"
