import runpy
from types import SimpleNamespace

import pytest

from chart_common.config import EMBED_DIM
from chart_common.records import NoteRecord
from indexer import dataset
from indexer import index as index_module
from indexer.index import FULL_INDEX_ENV, attach_vectors, validate_index_scope


def test_index_scope_allows_limited_live_run(monkeypatch) -> None:
    monkeypatch.delenv(FULL_INDEX_ENV, raising=False)

    validate_index_scope(limit=2000, dry_run=False)


def test_index_scope_blocks_unbounded_dry_run(monkeypatch) -> None:
    monkeypatch.delenv(FULL_INDEX_ENV, raising=False)

    with pytest.raises(SystemExit):
        validate_index_scope(limit=None, dry_run=True)


def test_index_scope_blocks_unbounded_live_cpu_run(monkeypatch) -> None:
    monkeypatch.delenv(FULL_INDEX_ENV, raising=False)

    with pytest.raises(SystemExit):
        validate_index_scope(limit=None, dry_run=False)


def test_index_scope_allows_explicit_full_cpu_override(monkeypatch) -> None:
    monkeypatch.setenv(FULL_INDEX_ENV, "1")

    validate_index_scope(limit=None, dry_run=False)


def test_indexer_cli_rejects_non_positive_limit_before_running(monkeypatch) -> None:
    monkeypatch.setattr("indexer.index.main", lambda **kwargs: pytest.fail("indexer should not run for invalid args"))
    monkeypatch.setattr("sys.argv", ["indexer", "--limit", "0"])

    with pytest.raises(SystemExit) as exc:
        runpy.run_module("indexer.__main__", run_name="__main__")

    assert exc.value.code == 2


def test_indexer_cli_writes_index_report(monkeypatch, tmp_path) -> None:
    out = tmp_path / "reports" / "slice-index-report.json"

    async def fake_run(*, limit=None, dry_run=False):
        return {
            "namespace": "chart-notes",
            "status": "completed",
            "limit": limit,
            "dry_run": dry_run,
            "indexed": 2,
            "provenance": {
                "dataset_repo": "zhengyun21/PMC-Patients",
                "dataset_revision": "28d8836518f86d4f1e6358ea8ec09977023e5766",
                "dataset_split": "train",
                "embed_model": "Snowflake/snowflake-arctic-embed-m-v1.5",
                "embed_dim": EMBED_DIM,
            },
            "schema": {
                "vector_dim": EMBED_DIM,
                "rows_with_age_band": 1,
                "rows_with_gender": 1,
                "rows_with_similar_patient_ids": 1,
            },
            "facet_snapshots_materialized": not dry_run,
        }

    monkeypatch.setattr(index_module, "run", fake_run)
    monkeypatch.setattr("sys.argv", ["indexer", "--limit", "2", "--out", str(out)])

    runpy.run_module("indexer.__main__", run_name="__main__")

    text = out.read_text()
    assert '"namespace": "chart-notes"' in text
    assert '"status": "completed"' in text
    assert '"indexed": 2' in text
    assert '"provenance"' in text
    assert '"dataset_revision": "28d8836518f86d4f1e6358ea8ec09977023e5766"' in text
    assert '"schema"' in text
    assert '"vector_dim": 768' in text
    assert '"facet_snapshots_materialized": true' in text


@pytest.mark.anyio
async def test_index_run_reports_schema_evidence(monkeypatch) -> None:
    records = [
        NoteRecord(
            id="patient-1",
            text="case text",
            title="",
            pmid="",
            source_url="",
            age=54,
            age_band="adult",
            gender="female",
            similar_patient_ids=["patient-2"],
        ),
        NoteRecord(
            id="patient-2",
            text="case text",
            title="",
            pmid="",
            source_url="",
            age=None,
            age_band=None,
            gender=None,
            similar_patient_ids=[],
        ),
    ]

    class FakeEmbedder:
        def __init__(self, model_name):
            self.model_name = model_name

        def embed_passages(self, texts):
            return [[0.1] * EMBED_DIM for _ in texts]

    monkeypatch.setattr(index_module, "Embedder", FakeEmbedder)
    monkeypatch.setattr(index_module, "load_notes", lambda settings, limit=None: iter(records[:limit]))

    report = await index_module.run(limit=2, dry_run=True)

    assert report["schema"] == {
        "vector_dim": EMBED_DIM,
        "rows_with_age_band": 1,
        "rows_with_gender": 1,
        "rows_with_similar_patient_ids": 1,
    }
    assert report["provenance"] == {
        "dataset_repo": "zhengyun21/PMC-Patients",
        "dataset_revision": "28d8836518f86d4f1e6358ea8ec09977023e5766",
        "dataset_split": "train",
        "embed_model": "Snowflake/snowflake-arctic-embed-m-v1.5",
        "embed_dim": EMBED_DIM,
    }


def _record(record_id: str) -> NoteRecord:
    return NoteRecord(
        id=record_id,
        text="case text",
        title="",
        pmid="",
        source_url="",
        age=None,
        age_band=None,
        gender=None,
    )


def test_attach_vectors_sets_valid_vectors_on_all_records() -> None:
    first = _record("patient-1")
    second = _record("patient-2")
    vectors = [[0.1] * EMBED_DIM, [0.2] * EMBED_DIM]

    attach_vectors([first, second], vectors)

    assert first.vector == vectors[0]
    assert second.vector == vectors[1]


def test_attach_vectors_rejects_vector_count_mismatch() -> None:
    with pytest.raises(ValueError, match="returned 1 vectors for 2 records"):
        attach_vectors([_record("patient-1"), _record("patient-2")], [[0.1] * EMBED_DIM])


def test_attach_vectors_rejects_wrong_vector_dimensions() -> None:
    with pytest.raises(ValueError, match=f"patient-1: expected {EMBED_DIM}-d vector"):
        attach_vectors([_record("patient-1")], [[0.1, 0.2]])


def test_load_ppr_similar_patient_ids_reads_bidirectional_train_edges(tmp_path, monkeypatch) -> None:
    qrels = tmp_path / "qrels_train.tsv"
    qrels.write_text(
        "query-id\tcorpus-id\tscore\n"
        "patient-1\tpatient-2\t1\n"
        "patient-1\tpatient-3\t1\n"
        "patient-4\tpatient-4\t1\n"
    )
    monkeypatch.setattr(dataset, "_download_qrels", lambda settings, split: qrels)

    assert dataset.load_ppr_similar_patient_ids(SimpleNamespace(), split="train") == {
        "patient-1": ["patient-2", "patient-3"],
        "patient-2": ["patient-1"],
        "patient-3": ["patient-1"],
    }


def test_load_notes_attaches_train_qrels_similar_patient_ids(monkeypatch) -> None:
    monkeypatch.setattr(dataset, "load_ppr_similar_patient_ids", lambda settings, split="train": {"patient-1": ["patient-2"]})

    def fake_load_dataset(repo, split, revision):
        return [
            {
                "patient_uid": "patient-1",
                "patient": "Case text.",
                "PMID": "123",
                "age": [[54, "year"]],
            }
        ]

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    records = list(
        dataset.load_notes(
            SimpleNamespace(dataset_repo="repo", dataset_split="train", dataset_revision="sha")
        )
    )

    assert records[0].similar_patient_ids == ["patient-2"]


def test_load_notes_can_skip_similar_patient_qrels_for_text_only_smokes(monkeypatch) -> None:
    monkeypatch.setattr(dataset, "load_ppr_similar_patient_ids", lambda settings, split="train": pytest.fail("qrels should not load"))
    monkeypatch.setattr(
        "datasets.load_dataset",
        lambda repo, split, revision: [{"patient_uid": "patient-1", "patient": "Case text."}],
    )

    records = list(
        dataset.load_notes(
            SimpleNamespace(dataset_repo="repo", dataset_split="train", dataset_revision="sha"),
            include_similar_patient_ids=False,
        )
    )

    assert records[0].similar_patient_ids == []
