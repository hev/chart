"""Opt-in local lexical demo contract. No models or retrieval run in Chart."""
from urllib.parse import urlsplit

INCLUDE = ["id", "title", "text", "age", "age_band", "gender", "specialty", "diagnosis_category"]
FILTERS = {
    "specialty": ["cardiology", "pulmonology", "endocrinology"],
    "age_band": ["adult", "older_adult"],
    "gender": ["female", "male"],
    "diagnosis_category": ["heart_failure", "asthma", "diabetes"],
}
CONFIG = {
    "profile": "luxir_lexical_demo",
    "notice": "Local Luxir preview · synthetic notes, no real patients · BM25 only",
    "filters": FILTERS,
    "examples": [
        {"text": "dyspnea", "note": "Lexical search; try cardiology or pulmonology filters."},
        {"text": "metformin", "note": "Lexical search of the synthetic diabetes note."},
        {"text": "wheezing", "note": "Lexical search of the synthetic asthma note."},
    ],
    "unsupported": ["Auto routing", "semantic search", "similar patients", "agentic search", "late interaction", "facet counts", "classifier writeback"],
}


def validate_target(settings):
    url = urlsplit(settings.gateway_url)
    if (url.scheme != "http" or url.hostname != "127.0.0.1" or not url.port
            or url.username or url.password or url.query or url.fragment or url.path not in ("", "/")
            or not settings.namespace.startswith("chart_demo_") or len(settings.namespace) <= len("chart_demo_")
            or not settings.namespace.isascii()
            or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in settings.namespace)):
        raise ValueError("Luxir demo requires an explicit http://127.0.0.1:PORT gateway and chart_demo_ namespace")
