"""Imperative ingestion for chart: load PMC-Patients → embed (Arctic) → upsert →
materialize facet snapshots. The twin of deploy/ (warehouse + pipeline + index).
Run: `uv run python -m indexer [--limit N] [--dry-run]`.
"""
