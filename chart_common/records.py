from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# How an age band is derived from the normalized age in years. A facet, so the
# rail has stable buckets instead of 100 distinct ages.
_AGE_BANDS = [
    (0, 1, "infant"),
    (1, 12, "child"),
    (12, 18, "adolescent"),
    (18, 40, "young-adult"),
    (40, 65, "adult"),
    (65, 200, "older-adult"),
]


def _years(age: Any) -> int | None:
    """PMC-Patients `age` is a list of (value, unit) pairs, e.g. [[54.0, 'year']].
    Normalize to an integer year; return None when unparseable."""
    if not age:
        return None
    try:
        for value, unit in age:
            if unit == "year":
                return int(value)
        # Fall back to the first pair, coarsely unit-converted.
        value, unit = age[0]
        factor = {"month": 1 / 12, "week": 1 / 52, "day": 1 / 365}.get(unit, 1)
        return int(float(value) * factor)
    except (TypeError, ValueError):
        return None


def _age_band(years: int | None) -> str | None:
    if years is None:
        return None
    for lo, hi, label in _AGE_BANDS:
        if lo <= years < hi:
            return label
    return None


def _required(row: dict, *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value)
    raise KeyError(names[0])


@dataclass(slots=True)
class NoteRecord:
    """One PMC-Patients case-report note → one searchable row.

    Native fields only. The clinical facets (specialty, chief_complaint,
    diagnosis_category, body_system) and phi_flag are UDF writeback (functions/),
    not derived here — RFC 0076 § Enrichment.
    """

    id: str
    text: str
    title: str
    pmid: str
    source_url: str
    age: int | None
    age_band: str | None
    gender: str | None
    similar_patient_ids: list[str] = field(default_factory=list)
    relevant_article_pmids: list[str] = field(default_factory=list)
    vector: list[float] | None = None

    @classmethod
    def from_row(cls, row: dict) -> "NoteRecord":
        pmid = str(row.get("PMID") or row.get("pmid") or "").strip()
        years = _years(row.get("age"))
        return cls(
            id=_required(row, "patient_uid", "id").strip(),
            text=_required(row, "patient", "text"),
            title=(row.get("title") or "").strip(),
            pmid=pmid,
            # Deep-link target: the source case report on PubMed (RFC 0076 UX).
            source_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
            age=years,
            age_band=_age_band(years),
            gender=row.get("gender"),
            similar_patient_ids=sorted((row.get("similar_patients") or {}).keys()),
            relevant_article_pmids=sorted((row.get("relevant_articles") or {}).keys()),
        )

    def to_upsert(self) -> dict:
        out = {
            "id": self.id,
            "text": self.text,
            "title": self.title,
            "pmid": self.pmid,
            "source_url": self.source_url,
            "age_band": self.age_band,
            "gender": self.gender,
            "similar_patient_ids": self.similar_patient_ids,
            "relevant_article_pmids": self.relevant_article_pmids,
        }
        if self.age is not None:
            out["age"] = self.age
        if self.vector is not None:
            out["vector"] = self.vector
        return {k: v for k, v in out.items() if v is not None}
