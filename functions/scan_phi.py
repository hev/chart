"""scan-phi — de-identification *verification* over note text.

The per-row safety transform a real `notesearch` wants: flag any residual
identifier that survived de-identification. PMC-Patients is already de-identified
(published case reports), so this should be near-empty on the public corpus — the
point is that the SAME Function runs on Trio's real notes, where it is not a
formality. RFC 0076 § Enrichment.

Its output is an OPERATIONAL attribute (`phi_flag`), not a search facet — a good
example of a UDF whose writeback is not for the rail.
"""

from __future__ import annotations

import re

from hevlayer.udf import udf

# Coarse residual-identifier signals. A production de-id verifier would use a
# trained NER model; these regexes are the cheap safety floor for the demo.
_PHI_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),            # SSN-shaped
    re.compile(r"\b\d{3}[-.)]\s?\d{3}[-.]\d{4}\b"),  # phone-shaped
    re.compile(r"\bMRN[:#]?\s*\d+\b", re.I),         # medical record number
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),     # email
]


@udf(inputs=["id", "text"], output="phi_flag", kind="classification")
def scan_phi(*, id: str, text: str | None) -> bool:
    """True if the note appears to carry a residual identifier."""
    if not text:
        return False
    return any(p.search(text) for p in _PHI_PATTERNS)
