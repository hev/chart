import pytest

from eval import bimodal
from eval.bimodal import (
    build_bimodal_set,
    build_short_set,
    extract_short_queries,
    perturb_typo,
    typo_fraction,
    validate_bimodal_set,
    write_beir,
    write_review_csv,
)


def test_extract_short_queries_prefers_drugs_doses_and_abbreviations() -> None:
    queries = extract_short_queries(
        "The patient with CABG was given metformin 500 mg and aspirin."
    )

    assert "aspirin" in queries
    assert "metformin" in queries
    assert "metformin 500mg" in queries
    assert "CABG" in queries


def test_perturb_typo_is_deterministic_one_edit() -> None:
    assert perturb_typo("aspirin") == "asirin"
    assert perturb_typo("AF") == "A"


def test_typo_fraction_accepts_fraction_and_rejects_out_of_range_values() -> None:
    assert typo_fraction("0.25") == 0.25
    with pytest.raises(Exception, match="<= 1"):
        typo_fraction("1.5")
    with pytest.raises(Exception, match="> 0"):
        typo_fraction("0")


def test_build_short_set_skips_relationship_qrels_for_text_only_generation(monkeypatch) -> None:
    calls = []

    def fake_load_notes(settings, **kwargs):
        calls.append(kwargs)
        return iter([type("Record", (), {"id": "patient-1", "text": "CABG and aspirin"})()])

    monkeypatch.setattr(bimodal, "load_notes", fake_load_notes)

    queries, qrels = build_short_set(object(), notes=1, typo_fraction=0.5)

    assert calls == [{"limit": 1, "include_similar_patient_ids": False}]
    assert queries[0] == {"id": "short:patient-1:0", "text": "aspirin", "kind": "short"}
    assert qrels["short:patient-1:0"] == {"patient-1": 1}


def test_build_bimodal_set_records_source_provenance(monkeypatch) -> None:
    class Settings:
        dataset_repo = "zhengyun21/PMC-Patients"
        dataset_revision = "28d8836518f86d4f1e6358ea8ec09977023e5766"
        dataset_split = "train"
        recds_repo = "zhengyun21/PMC-Patients-ReCDS"
        recds_revision = "a27717bb27679cf0860305997685547ca01b3dd1"

    def fake_build_short_set(settings, *, notes, typo_fraction):
        assert notes == 2
        assert typo_fraction == 0.5
        return ([{"id": "short:1", "text": "CABG", "kind": "short"}], {"short:1": {"patient-1": 1}})

    def fake_load_recds(task, *, settings, split):
        assert task == "ppr"
        assert split == "dev"
        return ([{"id": "q1", "text": "older patient"}], {"q1": {"patient-2": 2}})

    monkeypatch.setattr(bimodal, "build_short_set", fake_build_short_set)
    monkeypatch.setattr(bimodal, "load_recds", fake_load_recds)

    queries, qrels, meta = build_bimodal_set(
        settings=Settings(),
        short_notes=2,
        long_limit=1,
        typo_fraction=0.5,
        split="dev",
    )

    assert queries == [
        {"id": "short:1", "text": "CABG", "kind": "short"},
        {"id": "long:q1", "text": "older patient", "kind": "long", "source_id": "q1"},
    ]
    assert qrels == {"short:1": {"patient-1": 1}, "long:q1": {"patient-2": 2}}
    assert meta == {
        "short": 1,
        "long": 1,
        "typo": 0,
        "split": "dev",
        "short_notes": 2,
        "long_limit": 1,
        "typo_fraction": 0.5,
        "dataset_repo": "zhengyun21/PMC-Patients",
        "dataset_revision": "28d8836518f86d4f1e6358ea8ec09977023e5766",
        "dataset_split": "train",
        "recds_repo": "zhengyun21/PMC-Patients-ReCDS",
        "recds_revision": "a27717bb27679cf0860305997685547ca01b3dd1",
        "weak_label": "short query source note is relevant doc; long queries use ReCDS-PPR qrels",
    }


def test_write_beir_outputs_queries_and_qrels(tmp_path) -> None:
    write_beir(
        [{"id": "short:1", "text": "CABG", "kind": "short"}],
        {"short:1": {"patient-1": 1}},
        tmp_path,
    )

    assert (tmp_path / "queries.jsonl").read_text() == (
        '{"_id": "short:1", "text": "CABG", "kind": "short"}\n'
    )
    assert (tmp_path / "qrels.tsv").read_text() == (
        "query-id\tcorpus-id\tscore\nshort:1\tpatient-1\t1\n"
    )


def test_write_review_csv_samples_weak_labels_by_kind(tmp_path) -> None:
    counts = write_review_csv(
        [
            {"id": "short:1", "text": "CABG", "kind": "short"},
            {"id": "short:2", "text": "MI", "kind": "short"},
            {"id": "long:1", "text": "older patient with chest pain", "kind": "long"},
        ],
        {
            "short:1": {"patient-1": 1},
            "short:2": {"patient-2": 1},
            "long:1": {"patient-3": 2},
        },
        tmp_path / "review.csv",
        limit_per_kind=1,
    )

    assert counts == {"short": 1, "long": 1, "total": 2}
    assert (tmp_path / "review.csv").read_text() == (
        "query_id,kind,query,relevant_doc_id,weak_score,human_judgment,review_notes\n"
        "short:1,short,CABG,patient-1,1,,\n"
        "long:1,long,older patient with chest pain,patient-3,2,,\n"
    )


def test_validate_bimodal_set_rejects_empty_or_unjudged_sets() -> None:
    validate_bimodal_set(
        [
            {"id": "short:1", "text": "CABG", "kind": "short"},
            {"id": "long:1", "text": "older patient with chest pain", "kind": "long"},
        ],
        {"short:1": {"patient-1": 1}, "long:1": {"patient-2": 1}},
        {"short": 1, "long": 1},
    )

    with pytest.raises(SystemExit, match="no short"):
        validate_bimodal_set([], {}, {"short": 0, "long": 1})
    with pytest.raises(SystemExit, match="no long"):
        validate_bimodal_set([], {}, {"short": 1, "long": 0})
    with pytest.raises(SystemExit, match="queries without qrels: short:1"):
        validate_bimodal_set([{"id": "short:1", "text": "CABG"}], {}, {"short": 1, "long": 1})


def test_bimodal_cli_rejects_invalid_sizes_before_setup(monkeypatch) -> None:
    monkeypatch.setattr(bimodal, "Settings", lambda: pytest.fail("Settings should not load for invalid args"))
    monkeypatch.setattr("sys.argv", ["bimodal", "--short-notes", "0"])

    with pytest.raises(SystemExit) as exc:
        bimodal.main()

    assert exc.value.code == 2


def test_bimodal_cli_rejects_invalid_generated_set_before_writing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(bimodal, "Settings", lambda: object())
    monkeypatch.setattr(
        bimodal,
        "build_bimodal_set",
        lambda **kwargs: ([], {}, {"short": 0, "long": 1}),
    )
    monkeypatch.setattr("sys.argv", ["bimodal", "--out", str(tmp_path)])

    with pytest.raises(SystemExit, match="no short"):
        bimodal.main()

    assert not (tmp_path / "queries.jsonl").exists()
    assert not (tmp_path / "qrels.tsv").exists()
    assert not (tmp_path / "metadata.json").exists()


def test_bimodal_cli_rejects_invalid_review_limit_before_setup(monkeypatch) -> None:
    monkeypatch.setattr(bimodal, "Settings", lambda: pytest.fail("Settings should not load for invalid args"))
    monkeypatch.setattr("sys.argv", ["bimodal", "--review-limit-per-kind", "-1"])

    with pytest.raises(SystemExit) as exc:
        bimodal.main()

    assert exc.value.code == 2
