"""Shared library for the chart clinical-notes routing demo.

config   — Settings, the Arctic embed model + query prefix, the PMC-Patients pin.
embed    — Snowflake Arctic Embed m v1.5 with the explicit asymmetric query prefix.
records  — PMC-Patients row → NoteRecord (native fields; enrichment is UDF writeback).
gateway  — AsyncHevlayer client, the column schema, batch write, facet snapshots.
"""
