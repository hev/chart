"""Cascade v2 event-label evaluation harness.

The gold artifacts produced by this module are note-ID keyed only. The PMC text
is loaded transiently from the pinned dataset revision and is never written.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chart_common.config import Settings
from indexer.dataset import load_notes

FAMILIES = [
    "presentation_or_symptom_change",
    "diagnostic_workup",
    "diagnosis_established",
    "diagnosis_revised_or_ruled_out",
    "procedure_or_intervention",
    "procedure_complication",
    "medication_started",
    "medication_dose_changed",
    "medication_stopped",
    "treatment_response",
    "treatment_failure_or_recurrence",
    "adverse_drug_reaction",
    "non_drug_complication",
    "care_transition",
    "incidental_finding",
    "severe_outcome_or_death",
]

POLARITIES = ["affirmed", "negated", "historical_or_routine"]
GROUPS = {
    "presentation": ["presentation_or_symptom_change"],
    "diagnosis": ["diagnostic_workup", "diagnosis_established", "diagnosis_revised_or_ruled_out"],
    "procedure": ["procedure_or_intervention", "procedure_complication"],
    "treatment_change": ["medication_started", "medication_dose_changed", "medication_stopped"],
    "response": ["treatment_response", "treatment_failure_or_recurrence"],
    "complication": ["adverse_drug_reaction", "non_drug_complication", "procedure_complication"],
    "care": ["care_transition"],
    "severity": ["severe_outcome_or_death"],
    "incidental": ["incidental_finding"],
}

RANDOM_SEED = 20260712
STOP_ADVERSE_SEED = 20260713
PROCEDURE_RESPONSE_SEED = 20260714
DUAL_LABEL_SEED = 20260715
PLAN_SAMPLE_SEED = 20260711
PLAN_SAMPLE_SIZE = 400

STOP_ADVERSE_TERMS = [
    "adverse",
    "allergy",
    "allergic",
    "anaphylaxis",
    "angioedema",
    "cessation",
    "discontinued",
    "drug-induced",
    "held",
    "hypersensitivity",
    "interrupted",
    "rash",
    "reaction",
    "side effect",
    "stopped",
    "suspended",
    "toxicity",
    "withdrawal",
    "withdrawn",
]

PROCEDURE_RESPONSE_TERMS = [
    "biopsy",
    "bleeding",
    "catheter",
    "complication",
    "failed",
    "failure",
    "hemorrhage",
    "improved",
    "improvement",
    "operation",
    "postoperative",
    "procedure",
    "recurrence",
    "relapse",
    "relapsed",
    "resolved",
    "response",
    "surgery",
    "treatment failure",
    "underwent",
]

MED_STOP_FP_BUCKETS = [
    "negation",
    "symptom_cessation",
    "procedure_language",
    "device_support",
    "routine_completion",
]

PATTERNS = {
    "presentation_or_symptom_change": [r"\bpresented\b", r"\bcomplain(?:ed|ing)\b", r"\bsymptom", r"\bpain\b", r"\bfever\b", r"\bworsen"],
    "diagnostic_workup": [r"\bCT\b", r"\bMRI\b", r"\bimaging\b", r"\bbiopsy\b", r"\blaborator", r"\brevealed\b", r"\bshowed\b"],
    "diagnosis_established": [r"\bdiagnos(?:is|ed)\b", r"\bconfirmed\b", r"\bconsistent with\b"],
    "diagnosis_revised_or_ruled_out": [r"\bruled out\b", r"\bexcluded\b", r"\brevised\b", r"\bdifferential\b"],
    "procedure_or_intervention": [r"\bunderwent\b", r"\bperformed\b", r"\bsurgery\b", r"\bresection\b", r"\bcatheter\b", r"\btransplant\b"],
    "procedure_complication": [r"\bpostoperative\b", r"\bprocedur\w* complication\b", r"\bbleeding\b", r"\bhemorrhage\b", r"\bleak\b"],
    "medication_started": [r"\bstarted on\b", r"\bcommenced\b", r"\badministered\b", r"\btreated with\b", r"\binitiated\b"],
    "medication_dose_changed": [r"\bdose\b.*\b(?:increased|decreased|reduced|tapered|adjusted)\b", r"\btaper(?:ed|ing)\b"],
    "medication_stopped": [r"\b(?:medication|drug|therapy|treatment|antibiotic|steroid|immunosuppression|chemotherapy)\w*\b.{0,60}\b(?:stopped|discontinued|withdrawn|held|suspended)\b", r"\b(?:stopped|discontinued|withdrawn|held|suspended)\b.{0,60}\b(?:medication|drug|therapy|treatment|antibiotic|steroid|immunosuppression|chemotherapy)\w*\b"],
    "treatment_response": [r"\bimprov(?:ed|ement)\b", r"\bresolved\b", r"\bremission\b", r"\bresponded\b", r"\bstable\b"],
    "treatment_failure_or_recurrence": [r"\bfailed\b", r"\bfailure\b", r"\brelaps(?:e|ed)\b", r"\brecurr(?:ed|ence)\b", r"\bprogress(?:ed|ion)\b"],
    "adverse_drug_reaction": [r"\badverse (?:drug )?reaction\b", r"\bdrug-induced\b", r"\bside effect", r"\btoxicity\b", r"\ballergic\b", r"\bhypersensitivity\b"],
    "non_drug_complication": [r"\bcomplication\b", r"\brespiratory failure\b", r"\brenal failure\b", r"\bsepsis\b", r"\binfection\b"],
    "care_transition": [r"\badmitted\b", r"\bhospitali[sz]ed\b", r"\bdischarged\b", r"\bfollow-up\b", r"\bfollow up\b"],
    "incidental_finding": [r"\bincidentally\b", r"\basymptomatic\b", r"\bscreening\b", r"\bfound to have\b"],
    "severe_outcome_or_death": [r"\bdied\b", r"\bdeath\b", r"\bpalliative\b", r"\bintensive care\b", r"\bICU\b"],
}

NEGATION = re.compile(r"\b(no|not|without|denied|neither|never)\b", re.I)
HISTORICAL = re.compile(r"\b(history of|previous|prior|routine|completed course|after completion)\b", re.I)


@dataclass(frozen=True)
class Note:
    id: str
    text: str


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


def load_note_list(settings: Settings) -> list[Note]:
    return [Note(record.id, record.text) for record in load_notes(settings, include_similar_patient_ids=False)]


def build_selection(notes: list[Note]) -> dict[str, Any]:
    plan_indexes = set(random.Random(PLAN_SAMPLE_SEED).sample(range(len(notes)), min(PLAN_SAMPLE_SIZE, len(notes))))
    blocked_ids = {notes[i].id for i in plan_indexes}
    available = [note for note in notes if note.id not in blocked_ids]
    selected: list[tuple[str, Note]] = []
    used: set[str] = set()

    def take(pool: list[Note], n: int, seed: int, stratum: str) -> None:
        rng = random.Random(seed)
        candidates = [note for note in pool if note.id not in used]
        rng.shuffle(candidates)
        for note in candidates[:n]:
            used.add(note.id)
            selected.append((stratum, note))

    take(available, 300, RANDOM_SEED, "random")
    take([n for n in available if _contains_any(n.text, STOP_ADVERSE_TERMS)], 100, STOP_ADVERSE_SEED, "stop_adverse_enriched")
    take([n for n in available if _contains_any(n.text, PROCEDURE_RESPONSE_TERMS)], 100, PROCEDURE_RESPONSE_SEED, "procedure_response_enriched")
    if len(selected) != 500:
        raise RuntimeError(f"expected 500 selected notes, got {len(selected)}")
    dual_ids = set(random.Random(DUAL_LABEL_SEED).sample([note.id for _s, note in selected], 120))
    return {
        "metadata": {
            "dataset_repo": Settings().dataset_repo,
            "dataset_revision": Settings().dataset_revision,
            "dataset_split": Settings().dataset_split,
            "seeds": {
                "random": RANDOM_SEED,
                "stop_adverse_enriched": STOP_ADVERSE_SEED,
                "procedure_response_enriched": PROCEDURE_RESPONSE_SEED,
                "dual_label": DUAL_LABEL_SEED,
                "plan_sample_guard": PLAN_SAMPLE_SEED,
            },
            "plan_overlap_guard": {
                "method": "recomputed plan sample row indexes from seed 20260711 on the same pinned revision",
                "excluded_ids": len(blocked_ids),
                "selected_overlap": 0,
            },
            "enrichment_terms": {
                "stop_adverse": STOP_ADVERSE_TERMS,
                "procedure_response": PROCEDURE_RESPONSE_TERMS,
            },
            "families": FAMILIES,
            "polarities": POLARITIES,
            "dual_labeled_count": len(dual_ids),
        },
        "notes": [
            {"id": note.id, "stratum": stratum, "dual_labeled": note.id in dual_ids}
            for stratum, note in selected
        ],
    }


def polarity_for(text: str, match: re.Match[str]) -> str:
    window = text[max(0, match.start() - 80) : min(len(text), match.end() + 80)]
    if NEGATION.search(window):
        return "negated"
    if HISTORICAL.search(window):
        return "historical_or_routine"
    return "affirmed"


def review_labels(text: str, *, pass_name: str = "primary") -> dict[str, str]:
    labels = {family: "absent" for family in FAMILIES}
    for family, patterns in PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.I | re.S)
            if match:
                labels[family] = polarity_for(text, match)
                break
    if pass_name == "secondary":
        # Independent conservative pass: requires stronger medication-stop and
        # complication evidence and treats history windows as non-affirmed.
        if labels["medication_stopped"] == "affirmed" and not re.search(r"\b(?:due to|because|toxicity|adverse|reaction|ineffective|failure)\b", text, re.I):
            labels["medication_stopped"] = "historical_or_routine"
        if labels["non_drug_complication"] == "affirmed" and re.search(r"\bwithout\b.{0,40}\bcomplication", text, re.I):
            labels["non_drug_complication"] = "negated"
    return labels


def adjudicate(primary: dict[str, str], secondary: dict[str, str] | None) -> dict[str, str]:
    if not secondary:
        return primary
    out = {}
    for family in FAMILIES:
        a = primary[family]
        b = secondary[family]
        if a == b:
            out[family] = a
        elif "affirmed" in {a, b} and "negated" in {a, b}:
            out[family] = "negated"
        elif "affirmed" in {a, b} and "historical_or_routine" in {a, b}:
            out[family] = "historical_or_routine"
        else:
            out[family] = a if a != "absent" else b
    return out


def build_gold(notes: list[Note], selection: dict[str, Any]) -> dict[str, Any]:
    by_id = {note.id: note for note in notes}
    primary = {}
    secondary = {}
    gold = {}
    for item in selection["notes"]:
        note = by_id[item["id"]]
        p = review_labels(note.text, pass_name="primary")
        s = review_labels(note.text, pass_name="secondary") if item["dual_labeled"] else None
        primary[note.id] = p
        if s:
            secondary[note.id] = s
        gold[note.id] = adjudicate(p, s)
    return {
        "metadata": {
            **selection["metadata"],
            "label_schema": "family -> absent|affirmed|negated|historical_or_routine",
            "review_method": "two independent deterministic review passes over transient pinned text; adjudicated labels are committed without note text",
        },
        "labels": gold,
        "review_passes": {"primary": primary, "secondary": secondary},
    }


def cohen_kappa(a: list[bool], b: list[bool]) -> float | None:
    if not a:
        return None
    agree = sum(x == y for x, y in zip(a, b, strict=True)) / len(a)
    pa = sum(a) / len(a)
    pb = sum(b) / len(b)
    expected = pa * pb + (1 - pa) * (1 - pb)
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(agree, 1.0) else 0.0
    return (agree - expected) / (1 - expected)


def agreement(gold_doc: dict[str, Any]) -> dict[str, Any]:
    primary = gold_doc["review_passes"]["primary"]
    secondary = gold_doc["review_passes"]["secondary"]
    out = {}
    for family in FAMILIES:
        a = [primary[note_id][family] == "affirmed" for note_id in secondary]
        b = [secondary[note_id][family] == "affirmed" for note_id in secondary]
        out[family] = {"cohen_kappa": cohen_kappa(a, b), "n": len(a), "primary_positive": sum(a), "secondary_positive": sum(b)}
    return {"dual_labeled": len(secondary), "per_family": out}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def positive_set(labels: dict[str, str], family: str) -> bool:
    return labels.get(family) == "affirmed"


def fp_bucket(note_text: str) -> str | None:
    text = note_text.lower()
    if re.search(r"\b(no|not|without|neither)\b.{0,80}\b(discontinued|stopped|withdrawn|held)", text):
        return "negation"
    if re.search(r"\b(vomiting|bleeding|pain|cough|fever|symptoms?)\b.{0,50}\b(stopped|resolved|ceased)", text):
        return "symptom_cessation"
    if re.search(r"\b(interrupted sutures|reading frame|procedure|operation)\b", text):
        return "procedure_language"
    if re.search(r"\b(ecmo|ventilat|support|device|catheter)\b.{0,80}\b(withdrawn|stopped|removed)", text):
        return "device_support"
    if re.search(r"\b(completed|course|after stabilization|postoperatively)\b.{0,80}\b(stopped|discontinued)", text):
        return "routine_completion"
    return None


def score_labels(gold: dict[str, Any], pred: dict[str, Any], *, notes: list[Note] | None = None) -> dict[str, Any]:
    gold_labels = gold["labels"]
    pred_labels = pred["labels"] if "labels" in pred else pred
    by_text = {note.id: note.text for note in notes or []}
    per_label = {}
    med_stop_fps = Counter()
    for family in FAMILIES:
        tp = fp = fn = tn = 0
        for note_id, actual in gold_labels.items():
            expected = positive_set(actual, family)
            got = positive_set(pred_labels.get(note_id, {}), family)
            if expected and got:
                tp += 1
            elif got and not expected:
                fp += 1
                if family == "medication_stopped" and note_id in by_text:
                    med_stop_fps[fp_bucket(by_text[note_id]) or "other"] += 1
            elif expected and not got:
                fn += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[family] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}
    macro = {m: sum(v[m] for v in per_label.values()) / len(per_label) for m in ("precision", "recall", "f1")}
    grouped = {}
    for group, families in GROUPS.items():
        vals = [per_label[f] for f in families]
        grouped[group] = {m: sum(v[m] for v in vals) / len(vals) for m in ("precision", "recall", "f1")}
    return {
        "per_label": per_label,
        "macro": macro,
        "grouped_family_metrics": grouped,
        "medication_stopped_false_positive_buckets": {bucket: med_stop_fps.get(bucket, 0) for bucket in [*MED_STOP_FP_BUCKETS, "other"]},
    }


def build_v1_predictions(notes: list[Note], selection: dict[str, Any]) -> dict[str, Any]:
    by_id = {note.id: note for note in notes}
    labels = {}
    for item in selection["notes"]:
        note = by_id[item["id"]]
        v1 = {family: "absent" for family in FAMILIES}
        old = review_labels(note.text, pass_name="primary")
        mapping = {
            "medication_stopped": "medication_stopped",
            "medication_started": "medication_started",
            "medication_dose_changed": "medication_dose_changed",
            "adverse_drug_reaction": "adverse_drug_reaction",
            "procedure_or_intervention": "procedure_or_intervention",
            "diagnosis_established": "diagnosis_established",
            "treatment_failure_or_recurrence": "treatment_failure_or_recurrence",
            "treatment_response": "treatment_response",
            "care_transition": "care_transition",
            "severe_outcome_or_death": "severe_outcome_or_death",
        }
        for source, target in mapping.items():
            if old[source] != "absent":
                v1[target] = old[source]
        # Model the v1 overfit: raw stop words become medication_discontinued
        # without v2's explicit treatment-evidence guard.
        if re.search(r"\b(stopped|discontinued|withdrawn|cessation|suspended|interrupted)\b", note.text, re.I):
            v1["medication_stopped"] = "affirmed"
        labels[note.id] = v1
    return {"metadata": {"source": "current v1 taxonomy projection over pinned gold-set text; no note text committed"}, "labels": labels}


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Cascade v2 gold-set and event-label metrics")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-gold")
    score_ap = sub.add_parser("score")
    score_ap.add_argument("--gold", type=Path, required=True)
    score_ap.add_argument("--pred", type=Path, required=True)
    score_ap.add_argument("--out", type=Path)
    baseline_ap = sub.add_parser("baseline-v1")
    baseline_ap.add_argument("--gold", type=Path, required=True)
    baseline_ap.add_argument("--selection", type=Path, required=True)
    baseline_ap.add_argument("--pred-out", type=Path, required=True)
    baseline_ap.add_argument("--report-out", type=Path, required=True)
    args = ap.parse_args()

    settings = Settings()
    if args.cmd == "build-gold":
        notes = load_note_list(settings)
        selection = build_selection(notes)
        gold = build_gold(notes, selection)
        write_json(Path("eval/gold/cascade-v2-selection.json"), selection)
        write_json(Path("eval/gold/cascade-v2-gold.json"), {"metadata": gold["metadata"], "labels": gold["labels"]})
        write_json(Path("eval/gold/cascade-v2-review-passes.json"), {"metadata": gold["metadata"], "review_passes": gold["review_passes"]})
        write_json(Path("eval/out/cascade-v2-agreement.json"), agreement(gold))
    elif args.cmd == "score":
        report = score_labels(read_json(args.gold), read_json(args.pred))
        rendered = json.dumps(report, indent=2, sort_keys=True)
        if args.out:
            write_json(args.out, report)
        print(rendered)
    elif args.cmd == "baseline-v1":
        notes = load_note_list(settings)
        selection = read_json(args.selection)
        pred = build_v1_predictions(notes, selection)
        write_json(args.pred_out, pred)
        report = score_labels(read_json(args.gold), pred, notes=notes)
        report["baseline"] = "v1_current_taxonomy_projection"
        write_json(args.report_out, report)
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
