"""tag-specialty — map a note to a clinical specialty (the facet rail).

A row-preserving Function (RFC 0004/0040), scale-to-zero. One of the three
enrichment UDFs RFC 0076 names as the (secondary) transform-runtime act behind
the routing headline — and the source of a high-signal facet PMC-Patients does
not carry natively.

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

    STUB. The real implementation is a small classifier — open-weight LLM with
    guided decoding over SPECIALTIES (the RFC 0072 cascade shape), or a rule table
    over presenting features. Kept deterministic here so the scaffold runs without
    a model. Replace the body; the decorator contract stays.
    """
    if not text:
        return "other"
    return "other"
