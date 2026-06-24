"""extract-clinical-fields — structured fields from free-text notes.

The transform-runtime showcase RFC 0076 keeps as the second act behind routing:
the same extraction the Augmented-Clinical-Notes overlay did with GPT-4 over the
30k LONGEST PMC-Patients notes, run as a UDF over the WHOLE corpus — and the
source of the richer facets (`chief_complaint`, `diagnosis_category`,
`body_system`) the rail wants.

Multi-field writeback note: `@udf(output=...)` writes ONE attribute. This module
ships `diagnosis_category` on that clean path (below). The full field set
(`chief_complaint` + `body_system` too) is the documented "more control" pattern —
declare the `tpuf` parameter, `patch_columns` several attributes in one write,
return None (function-crd.mdx § Workers that need more control). Kept as the
single-output form here so the scaffold stays on the documented sugar; split into
sibling single-output UDFs, or lift to the `tpuf` multi-write, when implementing.
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

    STUB. Replace with the real extractor (open-weight LLM over the taxonomy, the
    RFC 0072 digest-cascade shape). For the full multi-field write, see the module
    docstring.
    """
    if not text:
        return "other"
    return "other"
