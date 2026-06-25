"""extract-clinical-fields — structured fields from free-text notes.

Legacy CPU fallback for structured clinical fields. The primary path is now the
Gemma cascade in `classify_events.py`, which derives `diagnosis_category` in the
same GPU pass as the clinical events.

Multi-field writeback note: the primary Gemma cascade uses the documented "more
control" pattern: declare the `tpuf` parameter, keep `events` on the clean
`@udf(output=...)` path, and patch the derived labels in one write. This legacy
fallback stays single-output because it should only fill a missing category.
"""

from __future__ import annotations

from hevlayer.udf import udf

# Coarse diagnosis grouping for the facet rail. A real implementation maps to an
# ICD-10 chapter or a curated taxonomy via an open-weight LLM (guided decoding).
DIAGNOSIS_CATEGORIES = [
    "cardiovascular", "respiratory", "neurological", "gastrointestinal",
    "renal-urinary", "endocrine-metabolic", "infectious", "neoplastic",
    "musculoskeletal", "hematologic", "dermatologic", "psychiatric", "other",
]


@udf(inputs=["id", "text"], output="diagnosis_category", kind="classification")
def extract_clinical_fields(*, id: str, text: str | None) -> str:
    """Return one diagnosis category for the note `text`.

    Deterministic fallback used when the GPU cascade has not populated the facet.
    For the full multi-field write, see the module docstring.
    """
    if not text:
        return "other"
    return "other"
