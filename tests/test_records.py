from chart_common.records import NoteRecord


def test_note_record_shapes_pmc_patient_row() -> None:
    row = {
        "patient_uid": "patient-1",
        "patient": "A 54-year-old woman presented with dyspnea.",
        "title": " Case report ",
        "PMID": "12345",
        "age": [[54.0, "year"]],
        "gender": "female",
        "similar_patients": {"patient-2": 0.91},
        "relevant_articles": {"12345": 1},
    }

    record = NoteRecord.from_row(row)
    out = record.to_upsert()

    assert record.age == 54
    assert record.age_band == "adult"
    assert out["id"] == "patient-1"
    assert out["source_url"] == "https://pubmed.ncbi.nlm.nih.gov/12345/"
    assert out["similar_patient_ids"] == ["patient-2"]
    assert out["relevant_article_pmids"] == ["12345"]


def test_note_record_tolerates_missing_optional_fields() -> None:
    record = NoteRecord.from_row(
        {
            "patient_uid": "patient-2",
            "patient": "No demographic details were reported.",
            "age": None,
        }
    )
    out = record.to_upsert()

    assert record.age is None
    assert record.age_band is None
    assert record.gender is None
    assert "age" not in out
    assert "age_band" not in out
    assert "gender" not in out
    assert out["similar_patient_ids"] == []


def test_relationship_fields_are_sorted_for_stable_upserts() -> None:
    record = NoteRecord.from_row(
        {
            "patient_uid": "patient-3",
            "patient": "Stable relationship order.",
            "similar_patients": {"patient-z": 1, "patient-a": 1},
            "relevant_articles": {"999": 1, "111": 1},
        }
    )

    assert record.to_upsert()["similar_patient_ids"] == ["patient-a", "patient-z"]
    assert record.to_upsert()["relevant_article_pmids"] == ["111", "999"]


def test_note_record_accepts_pipeline_chunk_aliases() -> None:
    record = NoteRecord.from_row(
        {
            "id": "patient-4",
            "text": "Chunk body.",
            "pmid": "67890",
            "title": " Chunked case ",
        }
    )

    assert record.to_upsert()["id"] == "patient-4"
    assert record.to_upsert()["text"] == "Chunk body."
    assert record.to_upsert()["pmid"] == "67890"
    assert record.to_upsert()["source_url"] == "https://pubmed.ncbi.nlm.nih.gov/67890/"


def test_note_record_rejects_missing_required_id_or_text() -> None:
    try:
        NoteRecord.from_row({"patient": "Text without id."})
    except KeyError as exc:
        assert exc.args == ("patient_uid",)
    else:
        raise AssertionError("missing id should fail")

    try:
        NoteRecord.from_row({"patient_uid": "patient-5"})
    except KeyError as exc:
        assert exc.args == ("patient",)
    else:
        raise AssertionError("missing text should fail")


def test_note_record_strips_required_id_aliases() -> None:
    record = NoteRecord.from_row({"id": " patient-6 ", "text": "Chunk body."})

    assert record.id == "patient-6"


def test_age_units_convert_to_stable_bands() -> None:
    assert NoteRecord.from_row(
        {"patient_uid": "baby", "patient": "Infant.", "age": [[6, "month"]]}
    ).age_band == "infant"
    assert NoteRecord.from_row(
        {"patient_uid": "child", "patient": "Child.", "age": [[7, "year"]]}
    ).age_band == "child"
    assert NoteRecord.from_row(
        {"patient_uid": "older", "patient": "Older adult.", "age": [[72, "year"]]}
    ).age_band == "older-adult"
