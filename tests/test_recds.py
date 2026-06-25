from types import SimpleNamespace

import pytest
from hevlayer.client import HevlayerError

from chart_common.config import EMBED_DIM, Settings
from eval import recds
from eval.recds import (
    baseline_report,
    eval_summary,
    fused_dominance_report,
    load_beir_dir,
    parse_strategies,
    qrels_for_queries,
    query_body,
    report_provenance,
    run_query,
    score,
    score_by_kind,
    validate_eval_gates,
    validate_judged_queries,
    validate_qrels_cover_queries,
)


def test_recds_query_body_for_auto_supplies_vector() -> None:
    vector = [0.4] * EMBED_DIM
    body = query_body(
        "elderly woman with dyspnea",
        strategy=None,
        embedder=SimpleNamespace(embed_query=lambda text: vector),
        top_k=2000,
    )

    assert body.rank_by == ["text", "Auto", "elderly woman with dyspnea", {"vector": vector}]
    assert body.top_k == 1000
    assert body.include_attributes == ["id"]


def test_recds_query_body_for_bm25_forces_hybrid_text_route() -> None:
    vector = [0.1] * EMBED_DIM
    body = query_body(
        "CABG",
        strategy="hybrid_text",
        embedder=SimpleNamespace(embed_query=lambda text: vector),
    )

    assert body.rank_by == ["text", "Auto", "CABG", {"vector": vector, "route": "hybrid_text"}]


def test_recds_query_body_for_long_bm25_caps_lexical_text() -> None:
    vector = [0.3] * EMBED_DIM
    text = "one two three four five six seven eight nine ten eleven"
    body = query_body(
        text,
        strategy="hybrid_text",
        embedder=SimpleNamespace(embed_query=lambda value: vector),
    )

    assert body.rank_by == [
        "text",
        "Auto",
        "one two three four five six seven eight",
        {"vector": vector, "route": "hybrid_text"},
    ]


def test_recds_query_body_for_fused_caps_lexical_text_but_embeds_full_query() -> None:
    calls = []
    vector = [0.2] * EMBED_DIM
    text = "one two three four five six seven eight nine ten eleven"
    body = query_body(
        text,
        strategy="fused",
        embedder=SimpleNamespace(embed_query=lambda value: calls.append(value) or vector),
    )

    assert calls == [text]
    assert body.rank_by == [
        "text",
        "Auto",
        "one two three four five six seven eight",
        {"vector": vector, "route": "fused"},
    ]


def test_recds_query_body_rejects_wrong_vector_dimensions() -> None:
    with pytest.raises(ValueError, match=f"expected {EMBED_DIM}-d query vector"):
        query_body(
            "CABG",
            strategy=None,
            embedder=SimpleNamespace(embed_query=lambda text: [0.1, 0.2]),
        )


@pytest.mark.anyio
async def test_run_query_accepts_dict_and_object_row_ids() -> None:
    class Row:
        id = "patient-2"

    class FakeLayer:
        async def query_namespace(self, namespace, body):
            return SimpleNamespace(rows=[{"id": "patient-1"}, Row()])

    ids = await run_query(
        FakeLayer(),
        "chart-notes",
        text="CABG",
        strategy=None,
        embedder=SimpleNamespace(embed_query=lambda text: [0.1] * EMBED_DIM),
    )

    assert ids == ["patient-1", "patient-2"]


@pytest.mark.anyio
async def test_run_query_rejects_non_empty_rows_without_ids() -> None:
    class FakeLayer:
        async def query_namespace(self, namespace, body):
            return SimpleNamespace(rows=[{"title": "missing id"}])

    with pytest.raises(ValueError, match="rows without id/\\$id"):
        await run_query(
            FakeLayer(),
            "chart-notes",
            text="CABG",
            strategy=None,
            embedder=SimpleNamespace(embed_query=lambda text: [0.1] * EMBED_DIM),
        )


def test_recds_score_computes_expected_perfect_run() -> None:
    metrics = score({"q1": ["d1", "d2"]}, {"q1": {"d1": 1}})

    assert metrics["RR@10"] == 1.0
    assert metrics["nDCG@10"] == 1.0
    assert metrics["R@1000"] == 1.0


def test_recds_score_rejects_empty_qrels() -> None:
    with pytest.raises(ValueError, match="cannot score without qrels"):
        score({"q1": ["d1"]}, {"q1": {}})


def test_qrels_for_queries_scopes_to_replayed_query_ids() -> None:
    qrels = {"q1": {"patient-1": 1}, "q2": {"patient-2": 1}}

    assert qrels_for_queries(qrels, [{"id": "q2", "text": "ignored"}]) == {
        "q2": {"patient-2": 1}
    }


def test_validate_qrels_cover_queries_rejects_missing_selected_qrels() -> None:
    with pytest.raises(SystemExit, match="selected queries missing qrels: q2"):
        validate_qrels_cover_queries(
            {"q1": {"patient-1": 1}},
            [{"id": "q1", "text": "CABG"}, {"id": "q2", "text": "stroke"}],
        )


def test_score_by_kind_breaks_out_bimodal_query_shapes() -> None:
    metrics = score_by_kind(
        {
            "short:1": ["patient-1"],
            "long:1": ["patient-x", "patient-2"],
        },
        {
            "short:1": {"patient-1": 1},
            "long:1": {"patient-2": 1},
        },
        [
            {"id": "short:1", "kind": "short"},
            {"id": "long:1", "kind": "long"},
        ],
    )

    assert metrics["short"]["RR@10"] == 1.0
    assert metrics["long"]["RR@10"] == 0.5


def test_baseline_report_compares_against_published_ppr_rrf() -> None:
    report = baseline_report(
        {"RR@10": 0.30, "nDCG@10": 0.25, "R@1000": 0.90},
        task="ppr",
        baseline="rrf",
    )

    assert report["baseline_metrics"] == {
        "RR@10": 0.2776,
        "nDCG@10": 0.2412,
        "R@1000": 0.8514,
    }
    assert report["meets_or_beats"] is True
    assert report["delta"]["RR@10"] == 0.0224


def test_parse_strategies_rejects_unknown_or_empty_values() -> None:
    assert parse_strategies("auto, bm25") == ["auto", "bm25"]
    with pytest.raises(Exception, match="unknown strategy"):
        parse_strategies("auto,nope")
    with pytest.raises(Exception, match="at least one strategy"):
        parse_strategies(" , ")


def test_validate_eval_gates_requires_all_legs_for_fused_dominance() -> None:
    validate_eval_gates(strategies=["bm25", "semantic", "fused"], require_fused_dominates=True)
    with pytest.raises(SystemExit, match="missing: semantic"):
        validate_eval_gates(strategies=["bm25", "fused"], require_fused_dominates=True)


def test_validate_judged_queries_rejects_empty_post_limit_sets() -> None:
    validate_judged_queries([{"id": "q1"}], source="ReCDS ppr/dev", limit=1)
    with pytest.raises(SystemExit, match="produced no judged queries"):
        validate_judged_queries([], source="ReCDS ppr/dev", limit=500)


def test_recds_report_provenance_records_pinned_dataset_and_model() -> None:
    settings = Settings(_env_file=None)

    assert report_provenance(settings) == {
        "recds_repo": "zhengyun21/PMC-Patients-ReCDS",
        "recds_revision": "a27717bb27679cf0860305997685547ca01b3dd1",
        "embed_model": "Snowflake/snowflake-arctic-embed-m-v1.5",
        "embed_dim": EMBED_DIM,
        "namespace": "chart-notes",
    }


def test_bimodal_report_provenance_does_not_claim_recds_revision(tmp_path) -> None:
    settings = Settings(_env_file=None)

    assert report_provenance(settings, beir_dir=tmp_path) == {
        "recds_repo": None,
        "recds_revision": None,
        "embed_model": "Snowflake/snowflake-arctic-embed-m-v1.5",
        "embed_dim": EMBED_DIM,
        "namespace": "chart-notes",
    }


def test_fused_dominance_report_accepts_fused_at_or_above_legs() -> None:
    report = fused_dominance_report(
        [
            {
                "strategy": "bm25",
                "metrics": {"RR@10": 0.2, "nDCG@10": 0.3, "R@1000": 0.4},
                "queries": {"failed": 0},
            },
            {
                "strategy": "semantic",
                "metrics": {"RR@10": 0.25, "nDCG@10": 0.2, "R@1000": 0.35},
                "queries": {"failed": 0},
            },
            {
                "strategy": "fused",
                "metrics": {"RR@10": 0.25, "nDCG@10": 0.3, "R@1000": 0.4},
                "queries": {"failed": 0},
            },
        ]
    )

    assert report["accepted"] is True
    assert report["query_failures"] == []
    assert all(check["ok"] for check in report["checks"])


def test_fused_dominance_report_rejects_missing_or_lower_fused_metrics() -> None:
    report = fused_dominance_report(
        [
            {"strategy": "bm25", "metrics": {"RR@10": 0.3, "nDCG@10": 0.2, "R@1000": 0.4}},
            {"strategy": "semantic", "metrics": {"RR@10": 0.2, "nDCG@10": 0.2, "R@1000": 0.4}},
            {"strategy": "fused", "metrics": {"RR@10": 0.25, "nDCG@10": 0.2}},
        ]
    )

    assert report["accepted"] is False
    assert any(check["metric"] == "RR@10" and check["baseline"] == "bm25" for check in report["checks"])
    assert any(check["metric"] == "R@1000" and check["fused"] is None for check in report["checks"])


def test_fused_dominance_report_rejects_equal_metrics_when_required_leg_had_query_failures() -> None:
    report = fused_dominance_report(
        [
            {
                "strategy": "bm25",
                "metrics": {"RR@10": 0.0, "nDCG@10": 0.0, "R@1000": 0.0},
                "queries": {"failed": 1},
            },
            {
                "strategy": "semantic",
                "metrics": {"RR@10": 0.0, "nDCG@10": 0.0, "R@1000": 0.0},
                "queries": {"failed": 1},
            },
            {
                "strategy": "fused",
                "metrics": {"RR@10": 0.0, "nDCG@10": 0.0, "R@1000": 0.0},
                "queries": {"failed": 1},
            },
        ]
    )

    assert report["accepted"] is False
    assert report["query_failures"] == [
        {"strategy": "bm25", "failed": 1},
        {"strategy": "semantic", "failed": 1},
        {"strategy": "fused", "failed": 1},
    ]
    assert all(check["ok"] for check in report["checks"])


def test_load_beir_dir_reads_queries_qrels_and_metadata(tmp_path) -> None:
    (tmp_path / "queries.jsonl").write_text(
        '{"_id": "short:1", "text": "CABG", "kind": "short"}\n'
    )
    (tmp_path / "qrels.tsv").write_text("query-id\tcorpus-id\tscore\nshort:1\tpatient-1\t1\n")
    (tmp_path / "metadata.json").write_text('{"short": 1, "long": 0}\n')

    queries, qrels, metadata = load_beir_dir(tmp_path)

    assert queries == [{"id": "short:1", "text": "CABG", "kind": "short"}]
    assert qrels == {"short:1": {"patient-1": 1}}
    assert metadata == {"short": 1, "long": 0}


def test_eval_summary_omits_published_baseline_for_bimodal_sets() -> None:
    summary = eval_summary(
        strategy="auto",
        metrics={"RR@10": 0.5},
        queries=[{"id": "short:1", "kind": "short"}, {"id": "long:1", "kind": "long"}],
        dataset={"short": 1, "long": 1},
        top_k=25,
        query_errors=[{"query_id": "short:1", "status_code": 502, "message": "bad gateway"}],
        metrics_by_kind={"short": {"RR@10": 0.5}, "long": {"RR@10": 0.0}},
    )

    assert summary == {
        "strategy": "auto",
        "metrics": {"RR@10": 0.5},
        "queries": {
            "total": 2,
            "attempted": 2,
            "failed": 1,
            "succeeded": 1,
            "scored": 2,
            "by_kind": {"short": 1, "long": 1},
        },
        "top_k": 25,
        "metrics_by_kind": {"short": {"RR@10": 0.5}, "long": {"RR@10": 0.0}},
        "dataset": {"short": 1, "long": 1},
        "query_errors": [{"query_id": "short:1", "status_code": 502, "message": "bad gateway"}],
        "query_errors_truncated": 0,
    }


def test_eval_summary_reports_error_truncation_and_scored_coverage() -> None:
    summary = eval_summary(
        strategy="fused",
        metrics={"R@1000": 0.0},
        queries=[{"id": f"q{i}"} for i in range(7)],
        top_k=1000,
        query_errors=[
            {"query_id": f"q{i}", "status_code": 502, "message": "bad gateway"}
            for i in range(6)
        ],
    )

    assert summary["queries"] == {
        "total": 7,
        "attempted": 7,
        "failed": 6,
        "succeeded": 1,
        "scored": 7,
        "by_kind": {"unknown": 7},
    }
    assert len(summary["query_errors"]) == 5
    assert summary["query_errors_truncated"] == 1


@pytest.mark.anyio
async def test_require_no_failures_cli_exits_nonzero_on_query_error(monkeypatch, capsys) -> None:
    class FakeLayer:
        async def query_namespace(self, namespace, body):
            raise HevlayerError(502, "bad gateway")

    async def fake_close_client(layer):
        return None

    monkeypatch.setattr(recds, "Settings", lambda: SimpleNamespace(namespace="chart-notes", embed_model="model", api_key="key"))
    monkeypatch.setattr(recds, "Embedder", lambda model: SimpleNamespace(embed_query=lambda text: [0.1] * EMBED_DIM))
    monkeypatch.setattr(recds, "make_client", lambda settings: FakeLayer())
    monkeypatch.setattr(recds, "close_client", fake_close_client)
    monkeypatch.setattr(
        recds,
        "load_recds",
        lambda task, settings, split: ([{"id": "q1", "text": "CABG"}], {"q1": {"patient-1": 1}}),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["recds", "--strategies", "auto", "--limit", "1", "--require-no-failures"],
    )

    with pytest.raises(SystemExit) as exc:
        await recds.main()

    assert exc.value.code == 1
    assert '"failed": 1' in capsys.readouterr().out


@pytest.mark.anyio
async def test_recds_cli_rejects_non_positive_limits_before_setup(monkeypatch) -> None:
    monkeypatch.setattr(recds, "Settings", lambda: pytest.fail("Settings should not load for invalid args"))
    monkeypatch.setattr("sys.argv", ["recds", "--limit", "0"])

    with pytest.raises(SystemExit) as exc:
        await recds.main()

    assert exc.value.code == 2


@pytest.mark.anyio
async def test_recds_cli_rejects_empty_judged_set_before_gateway_setup(monkeypatch) -> None:
    monkeypatch.setattr(recds, "Settings", lambda: SimpleNamespace(namespace="chart-notes", embed_model="model", api_key="key"))
    monkeypatch.setattr(recds, "Embedder", lambda model: SimpleNamespace(embed_query=lambda text: [0.1] * EMBED_DIM))
    monkeypatch.setattr(recds, "make_client", lambda settings: pytest.fail("gateway client should not be created"))
    monkeypatch.setattr(recds, "load_recds", lambda task, settings, split: ([], {}))
    monkeypatch.setattr("sys.argv", ["recds", "--strategies", "auto", "--limit", "1"])

    with pytest.raises(SystemExit, match="produced no judged queries"):
        await recds.main()


@pytest.mark.anyio
async def test_recds_cli_rejects_selected_query_missing_qrels_before_gateway_setup(monkeypatch) -> None:
    monkeypatch.setattr(recds, "Settings", lambda: SimpleNamespace(namespace="chart-notes", embed_model="model", api_key="key"))
    monkeypatch.setattr(recds, "Embedder", lambda model: SimpleNamespace(embed_query=lambda text: [0.1] * EMBED_DIM))
    monkeypatch.setattr(recds, "make_client", lambda settings: pytest.fail("gateway client should not be created"))
    monkeypatch.setattr(
        recds,
        "load_recds",
        lambda task, settings, split: (
            [{"id": "q1", "text": "CABG"}, {"id": "q2", "text": "stroke"}],
            {"q1": {"patient-1": 1}},
        ),
    )
    monkeypatch.setattr("sys.argv", ["recds", "--strategies", "auto", "--limit", "2"])

    with pytest.raises(SystemExit, match="selected queries missing qrels: q2"):
        await recds.main()


@pytest.mark.anyio
async def test_recds_cli_exits_before_embedder_without_gateway_key(monkeypatch) -> None:
    monkeypatch.setattr(recds, "Settings", lambda: SimpleNamespace(namespace="chart-notes", embed_model="model", api_key=None))
    monkeypatch.setattr(recds, "Embedder", lambda model: pytest.fail("embedder should not load without key"))
    monkeypatch.setattr(recds, "make_client", lambda settings: pytest.fail("gateway client should not be created"))
    monkeypatch.setattr(
        recds,
        "load_recds",
        lambda task, settings, split: ([{"id": "q1", "text": "CABG"}], {"q1": {"patient-1": 1}}),
    )
    monkeypatch.setattr("sys.argv", ["recds", "--strategies", "auto", "--limit", "1"])

    with pytest.raises(SystemExit, match="No gateway key"):
        await recds.main()


@pytest.mark.anyio
async def test_recds_cli_scores_only_limited_query_qrels(monkeypatch) -> None:
    class FakeLayer:
        async def query_namespace(self, namespace, body):
            return SimpleNamespace(rows=[{"id": "patient-1"}])

    async def fake_close_client(layer):
        return None

    captured_qrels = []

    def fake_score(ranked, qrels):
        captured_qrels.append(qrels)
        return {"RR@10": 1.0, "nDCG@10": 1.0, "R@1000": 1.0}

    monkeypatch.setattr(recds, "Settings", lambda: SimpleNamespace(namespace="chart-notes", embed_model="model", api_key="key"))
    monkeypatch.setattr(recds, "Embedder", lambda model: SimpleNamespace(embed_query=lambda text: [0.1] * EMBED_DIM))
    monkeypatch.setattr(recds, "make_client", lambda settings: FakeLayer())
    monkeypatch.setattr(recds, "close_client", fake_close_client)
    monkeypatch.setattr(recds, "score", fake_score)
    monkeypatch.setattr(recds, "score_by_kind", lambda ranked, qrels, queries: {})
    monkeypatch.setattr(
        recds,
        "load_recds",
        lambda task, settings, split: (
            [{"id": "q1", "text": "CABG"}, {"id": "q2", "text": "stroke"}],
            {"q1": {"patient-1": 1}, "q2": {"patient-2": 1}},
        ),
    )
    monkeypatch.setattr("sys.argv", ["recds", "--strategies", "auto", "--limit", "1"])

    await recds.main()

    assert captured_qrels == [{"q1": {"patient-1": 1}}]


@pytest.mark.anyio
async def test_require_no_failures_cli_exits_zero_without_query_errors(monkeypatch, capsys) -> None:
    class FakeLayer:
        async def query_namespace(self, namespace, body):
            return SimpleNamespace(rows=[{"id": "patient-1"}])

    async def fake_close_client(layer):
        return None

    monkeypatch.setattr(recds, "Settings", lambda: SimpleNamespace(namespace="chart-notes", embed_model="model", api_key="key"))
    monkeypatch.setattr(recds, "Embedder", lambda model: SimpleNamespace(embed_query=lambda text: [0.1] * EMBED_DIM))
    monkeypatch.setattr(recds, "make_client", lambda settings: FakeLayer())
    monkeypatch.setattr(recds, "close_client", fake_close_client)
    monkeypatch.setattr(
        recds,
        "load_recds",
        lambda task, settings, split: ([{"id": "q1", "text": "CABG"}], {"q1": {"patient-1": 1}}),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["recds", "--strategies", "auto", "--limit", "1", "--require-no-failures"],
    )

    await recds.main()

    assert '"failed": 0' in capsys.readouterr().out


@pytest.mark.anyio
async def test_recds_cli_writes_aggregate_phase5_report(monkeypatch, tmp_path, capsys) -> None:
    class FakeLayer:
        async def query_namespace(self, namespace, body):
            return SimpleNamespace(rows=[{"id": "patient-1"}])

    async def fake_close_client(layer):
        return None

    out = tmp_path / "reports" / "recds-report.json"
    monkeypatch.setattr(recds, "Settings", lambda: SimpleNamespace(namespace="chart-notes", embed_model="model", api_key="key"))
    monkeypatch.setattr(recds, "Embedder", lambda model: SimpleNamespace(embed_query=lambda text: [0.1] * EMBED_DIM))
    monkeypatch.setattr(recds, "make_client", lambda settings: FakeLayer())
    monkeypatch.setattr(recds, "close_client", fake_close_client)
    monkeypatch.setattr(
        recds,
        "load_recds",
        lambda task, settings, split: ([{"id": "q1", "text": "CABG"}], {"q1": {"patient-1": 1}}),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "recds",
            "--strategies",
            "bm25,semantic,fused",
            "--limit",
            "1",
            "--require-no-failures",
            "--require-fused-dominates",
            "--out",
            str(out),
        ],
    )

    await recds.main()

    assert '"accepted": true' in capsys.readouterr().out
    text = out.read_text()
    assert '"summaries"' in text
    assert '"no_failures"' in text
    assert '"fused_dominates"' in text
    assert '"strategies": [' in text


@pytest.mark.anyio
async def test_recds_cli_reports_progress_to_stderr(monkeypatch, capsys) -> None:
    class FakeLayer:
        async def query_namespace(self, namespace, body):
            return SimpleNamespace(rows=[{"id": "patient-1"}])

    async def fake_close_client(layer):
        return None

    monkeypatch.setattr(recds, "Settings", lambda: SimpleNamespace(namespace="chart-notes", embed_model="model", api_key="key"))
    monkeypatch.setattr(recds, "Embedder", lambda model: SimpleNamespace(embed_query=lambda text: [0.1] * EMBED_DIM))
    monkeypatch.setattr(recds, "make_client", lambda settings: FakeLayer())
    monkeypatch.setattr(recds, "close_client", fake_close_client)
    monkeypatch.setattr(
        recds,
        "load_recds",
        lambda task, settings, split: (
            [{"id": "q1", "text": "CABG"}, {"id": "q2", "text": "dyspnea"}],
            {"q1": {"patient-1": 1}, "q2": {"patient-1": 1}},
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["recds", "--strategies", "auto", "--limit", "2", "--progress-every", "1"],
    )

    await recds.main()

    captured = capsys.readouterr()
    assert "recds progress strategy=auto 1/2 failed=0" in captured.err
    assert "recds progress strategy=auto 2/2 failed=0" in captured.err


@pytest.mark.anyio
async def test_require_fused_dominates_cli_exits_nonzero_when_fused_loses(monkeypatch, capsys) -> None:
    class FakeLayer:
        async def query_namespace(self, namespace, body):
            route = body.rank_by[3].get("route")
            if route == "fused":
                return SimpleNamespace(rows=[])
            return SimpleNamespace(rows=[{"id": "patient-1"}])

    async def fake_close_client(layer):
        return None

    monkeypatch.setattr(recds, "Settings", lambda: SimpleNamespace(namespace="chart-notes", embed_model="model", api_key="key"))
    monkeypatch.setattr(recds, "Embedder", lambda model: SimpleNamespace(embed_query=lambda text: [0.1] * EMBED_DIM))
    monkeypatch.setattr(recds, "make_client", lambda settings: FakeLayer())
    monkeypatch.setattr(recds, "close_client", fake_close_client)
    monkeypatch.setattr(
        recds,
        "load_recds",
        lambda task, settings, split: ([{"id": "q1", "text": "CABG"}], {"q1": {"patient-1": 1}}),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["recds", "--strategies", "bm25,semantic,fused", "--limit", "1", "--require-fused-dominates"],
    )

    with pytest.raises(SystemExit) as exc:
        await recds.main()

    assert exc.value.code == 1
    assert '"gate": "fused_dominates_legs"' in capsys.readouterr().out


@pytest.mark.anyio
async def test_require_fused_dominates_cli_exits_nonzero_on_equal_metrics_with_query_failures(monkeypatch, capsys) -> None:
    class FakeLayer:
        async def query_namespace(self, namespace, body):
            raise HevlayerError(502, "bad gateway")

    async def fake_close_client(layer):
        return None

    monkeypatch.setattr(recds, "Settings", lambda: SimpleNamespace(namespace="chart-notes", embed_model="model", api_key="key"))
    monkeypatch.setattr(recds, "Embedder", lambda model: SimpleNamespace(embed_query=lambda text: [0.1] * EMBED_DIM))
    monkeypatch.setattr(recds, "make_client", lambda settings: FakeLayer())
    monkeypatch.setattr(recds, "close_client", fake_close_client)
    monkeypatch.setattr(
        recds,
        "load_recds",
        lambda task, settings, split: ([{"id": "q1", "text": "CABG"}], {"q1": {"patient-1": 1}}),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["recds", "--strategies", "bm25,semantic,fused", "--limit", "1", "--require-fused-dominates"],
    )

    with pytest.raises(SystemExit) as exc:
        await recds.main()

    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert '"query_failures": [{"strategy": "bm25", "failed": 1}' in out
    assert '"accepted": false' in out


@pytest.mark.anyio
async def test_require_fused_dominates_cli_exits_zero_when_fused_wins(monkeypatch, capsys) -> None:
    class FakeLayer:
        async def query_namespace(self, namespace, body):
            return SimpleNamespace(rows=[{"id": "patient-1"}])

    async def fake_close_client(layer):
        return None

    monkeypatch.setattr(recds, "Settings", lambda: SimpleNamespace(namespace="chart-notes", embed_model="model", api_key="key"))
    monkeypatch.setattr(recds, "Embedder", lambda model: SimpleNamespace(embed_query=lambda text: [0.1] * EMBED_DIM))
    monkeypatch.setattr(recds, "make_client", lambda settings: FakeLayer())
    monkeypatch.setattr(recds, "close_client", fake_close_client)
    monkeypatch.setattr(
        recds,
        "load_recds",
        lambda task, settings, split: ([{"id": "q1", "text": "CABG"}], {"q1": {"patient-1": 1}}),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["recds", "--strategies", "bm25,semantic,fused", "--limit", "1", "--require-fused-dominates"],
    )

    await recds.main()

    assert '"accepted": true' in capsys.readouterr().out
