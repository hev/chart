"""tag-specialty — map a note to a clinical specialty (the facet rail).

Legacy CPU fallback for the specialty facet. The primary path is now the Gemma
cascade in `classify_events.py`, which derives `specialty` in the same GPU pass as
the clinical events.

Deployed as a `Function` CRD pointing at the worker image that runs this module;
discovery, batching, claim leases, retries, and `patch_columns` writeback are the
gateway's, not ours (CLAUDE.md § Product Frame).
"""

from __future__ import annotations

from hevlayer.udf import udf

SPECIALTIES = [
    "cardiology", "neurology", "oncology", "gastroenterology", "pulmonology",
    "nephrology", "endocrinology", "infectious-disease", "rheumatology",
    "hematology", "dermatology", "psychiatry", "obstetrics-gynecology",
    "pediatrics", "emergency-medicine", "surgery", "other",
]


@udf(inputs=["id", "text"], output="specialty", kind="classification")
def tag_specialty(*, id: str, text: str | None) -> str:
    """Return one specialty from SPECIALTIES for the note `text`.

    Deterministic fallback used when the GPU cascade has not populated the facet.
    The decorator contract stays compatible with a later rule table or small CPU
    classifier.
    """
    if not text:
        return "other"
    return "other"
