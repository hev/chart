# `deploy/` — the in-cluster declarative twin

The CR bundle that stands chart up *declaratively*, the equivalent of the
imperative `uv run python -m indexer` + the runtime gateway client. Both spellings
round-trip through one CRD/REST schema (CLAUDE.md § Design Bias); this is the YAML
side. It is also, one-to-one, the manifest set RFC 0076 specifies.

| File | Owns | Imperative twin |
|---|---|---|
| `vectorstore.yaml` | upstream Turbopuffer connection + `deriveFromStore` inbound auth | `chart_common/gateway.py:make_client()` |
| `warehouse.yaml` | the data source identity (`huggingface`, public/no-Secret) | `chart_common/config.py` dataset pin |
| `pipeline.yaml` | staged ingestion: source → chunk → (embed) | `python -m indexer` |
| `index.yaml` | operational policy: the facet snapshots, scan fan-out, consistency | `gateway.py:materialize_facet_snapshots()` |

Two things to know:

- **No new Warehouse kind, no new gateway machinery.** The whole bundle is the
  shipped `huggingface` Warehouse (RFC 0053), `Auto`/`HybridText` routing
  (RFC 0044/0057), and the RFC 0056 chunk model. That is the RFC 0076 thesis: a
  new vertical with zero new gateway code.
- **The Trio swap is one block.** `warehouse.yaml` carries the `kind: huggingface
  → snowflake` swap that points this exact Pipeline at Trio's real `notesearch`
  notes. Nothing else in the bundle changes.

Apply order: `vectorstore` → `warehouse` → `pipeline` → `index`.
