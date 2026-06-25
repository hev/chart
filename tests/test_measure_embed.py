from types import SimpleNamespace

import pytest

from chart_common.config import EMBED_DIM
from indexer import measure_embed
from indexer.measure_embed import budget_report, estimate_full_run, production_path_report


def test_embed_estimate_extrapolates_latency_and_optional_gpu_cost() -> None:
    assert estimate_full_run(notes=20, elapsed_seconds=10.0, full_notes=100, gpu_hourly_usd=3.0) == {
        "notes": 20,
        "elapsed_seconds": 10.0,
        "per_note_seconds": 0.5,
        "notes_per_second": 2.0,
        "full_notes": 100,
        "estimated_full_seconds": 50.0,
        "estimated_full_hours": 0.01,
        "gpu_hourly_usd": 3.0,
        "estimated_full_usd": 0.03,
    }


def test_embed_estimate_rejects_empty_sample() -> None:
    with pytest.raises(ValueError, match="notes must be positive"):
        estimate_full_run(notes=0, elapsed_seconds=0.0)


def test_embed_estimate_rejects_non_positive_elapsed_time() -> None:
    with pytest.raises(ValueError, match="elapsed_seconds must be positive"):
        estimate_full_run(notes=1, elapsed_seconds=0.0)


def test_embed_budget_report_accepts_estimate_under_limits() -> None:
    estimate = estimate_full_run(notes=20, elapsed_seconds=10.0, full_notes=100, gpu_hourly_usd=3.0)

    assert budget_report(estimate, max_full_hours=0.02, max_full_usd=0.05) == {
        "checks": {
            "max_full_hours": {"limit": 0.02, "actual": 0.01, "ok": True},
            "max_full_usd": {"limit": 0.05, "actual": 0.03, "ok": True},
        },
        "accepted": True,
    }


def test_embed_budget_report_rejects_estimate_over_limits_or_missing_cost() -> None:
    estimate = estimate_full_run(notes=20, elapsed_seconds=10.0, full_notes=100)

    assert budget_report(estimate, max_full_hours=0.005, max_full_usd=0.05) == {
        "checks": {
            "max_full_hours": {"limit": 0.005, "actual": 0.01, "ok": False},
            "max_full_usd": {"limit": 0.05, "actual": None, "ok": False},
        },
        "accepted": False,
    }


def test_production_path_report_documents_gpu_pipeline_path() -> None:
    assert production_path_report() == {
        "pipeline_cr": "chart-embed-gpu",
        "module": "indexer.embed",
        "compute_class": "gpu",
        "image": "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-embedder-plan-20260624-dedupe2",
        "allow_full_cpu_index": False,
    }


def test_embed_measurement_batches_sample_without_loading_model(monkeypatch) -> None:
    load_notes_kwargs = []

    def fake_load_notes(settings, **kwargs):
        load_notes_kwargs.append(kwargs)
        limit = kwargs["limit"]
        return iter(
            [
                SimpleNamespace(text="alpha"),
                SimpleNamespace(text="beta"),
                SimpleNamespace(text="gamma"),
            ][:limit]
        )

    monkeypatch.setattr(measure_embed, "load_notes", fake_load_notes)
    monkeypatch.setattr(measure_embed.time, "perf_counter", iter([100.0, 102.0]).__next__)

    seen_batches = []

    def fake_embed(texts):
        seen_batches.append(list(texts))
        return [[float(len(text))] * EMBED_DIM for text in texts]

    result = measure_embed.run(
        limit=3,
        batch_size=2,
        max_chars=4,
        gpu_hourly_usd=2.0,
        accelerator="gpu",
        embed_fn=fake_embed,
    )

    assert seen_batches == [["alph", "beta"], ["gamm"]]
    assert load_notes_kwargs == [{"limit": 3, "include_similar_patient_ids": False}]
    assert result["runtime"] == {"accelerator": "gpu"}
    assert result["sample"] == {
        "notes": 3,
        "batch_size": 2,
        "max_chars": 4,
        "vector_dim": EMBED_DIM,
        "model": "Snowflake/snowflake-arctic-embed-m-v1.5",
    }
    assert result["estimate"]["per_note_seconds"] == 0.6667
    assert result["estimate"]["estimated_full_usd"] == 61.86
    assert result["production_path"] == production_path_report()


def test_embed_measurement_rejects_wrong_vector_dimensions(monkeypatch) -> None:
    monkeypatch.setattr(
        measure_embed,
        "load_notes",
        lambda settings, **kwargs: iter([SimpleNamespace(id="patient-1", text="alpha")]),
    )

    with pytest.raises(RuntimeError, match=f"patient-1: expected {EMBED_DIM}-d embeddings"):
        measure_embed.run(limit=1, embed_fn=lambda texts: [[0.1, 0.2]])


def test_embed_measurement_rejects_per_batch_vector_count_mismatch_even_if_total_could_cancel(monkeypatch) -> None:
    monkeypatch.setattr(
        measure_embed,
        "load_notes",
        lambda settings, **kwargs: iter(
            [
                SimpleNamespace(id="patient-1", text="alpha"),
                SimpleNamespace(id="patient-2", text="beta"),
            ]
        ),
    )
    calls = 0

    def fake_embed(texts):
        nonlocal calls
        calls += 1
        if calls == 1:
            return [[0.1] * EMBED_DIM, [0.2] * EMBED_DIM]
        return []

    with pytest.raises(RuntimeError, match="returned 2 vectors for 1 notes in batch \\[patient-1\\]"):
        measure_embed.run(limit=2, batch_size=1, embed_fn=fake_embed)


def test_embed_cli_exits_nonzero_when_budget_is_rejected(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        measure_embed,
        "run",
        lambda **kwargs: {
            "sample": {"notes": 1},
            "estimate": {"estimated_full_hours": 3.0, "estimated_full_usd": 12.0},
        },
    )
    monkeypatch.setattr("sys.argv", ["measure", "--max-full-hours", "2.0", "--max-full-usd", "10.0"])

    with pytest.raises(SystemExit) as exc:
        measure_embed.main()

    assert exc.value.code == 1
    assert '"accepted": false' in capsys.readouterr().out


def test_embed_cli_writes_rejected_budget_report_before_exiting(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        measure_embed,
        "run",
        lambda **kwargs: {
            "sample": {"notes": 1},
            "estimate": {"estimated_full_hours": 3.0, "estimated_full_usd": 12.0},
        },
    )
    out = tmp_path / "reports" / "embed-budget.json"
    monkeypatch.setattr(
        "sys.argv",
        ["measure", "--max-full-hours", "2.0", "--max-full-usd", "10.0", "--out", str(out)],
    )

    with pytest.raises(SystemExit) as exc:
        measure_embed.main()

    assert exc.value.code == 1
    assert '"accepted": false' in out.read_text()


def test_embed_cli_writes_failed_report_when_measurement_runtime_fails(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CHART_GPU_DEVICE_NAME", "NVIDIA L4")
    monkeypatch.setattr(
        measure_embed,
        "run",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("embedding model failed to load")),
    )
    out = tmp_path / "reports" / "embed-budget.json"
    monkeypatch.setattr("sys.argv", ["measure", "--accelerator", "gpu", "--out", str(out)])

    with pytest.raises(RuntimeError, match="embedding model failed to load"):
        measure_embed.main()

    text = out.read_text()
    assert '"status": "failed"' in text
    assert '"error": "embedding model failed to load"' in text
    assert '"accelerator": "gpu"' in text
    assert '"gpu_device": "NVIDIA L4"' in text
    assert '"production_path"' in text


def test_embed_cli_fails_before_running_when_gpu_device_is_unverified(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.delenv("CHART_GPU_DEVICE_NAME", raising=False)
    monkeypatch.setattr(
        "chart_common.runtime.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError()),
    )
    monkeypatch.setattr(measure_embed, "run", lambda **kwargs: pytest.fail("run should not start without GPU proof"))
    out = tmp_path / "reports" / "embed-budget.json"
    monkeypatch.setattr("sys.argv", ["measure", "--accelerator", "gpu", "--out", str(out)])

    with pytest.raises(SystemExit) as exc:
        measure_embed.main()

    assert exc.value.code == 1
    stdout = capsys.readouterr().out
    assert "GPU accelerator was requested but no GPU device could be verified" in stdout
    text = out.read_text()
    assert '"status": "failed"' in text
    assert '"gpu_probe": "missing"' in text


def test_embed_cli_exits_zero_when_budget_is_accepted(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        measure_embed,
        "run",
        lambda **kwargs: {
            "sample": {"notes": 1},
            "estimate": {"estimated_full_hours": 1.0, "estimated_full_usd": 4.0},
        },
    )
    monkeypatch.setattr("sys.argv", ["measure", "--max-full-hours", "2.0", "--max-full-usd", "10.0"])

    measure_embed.main()

    assert '"accepted": true' in capsys.readouterr().out


def test_embed_cli_writes_budget_report_for_phase6_audit(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setenv("CHART_GPU_DEVICE_NAME", "NVIDIA L4")
    monkeypatch.setattr(
        measure_embed,
        "run",
        lambda **kwargs: {
            "sample": {"notes": 1},
            "estimate": {"estimated_full_hours": 1.0, "estimated_full_usd": 4.0},
        },
    )
    out = tmp_path / "reports" / "embed-budget.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "measure",
            "--accelerator",
            "gpu",
            "--max-full-hours",
            "2.0",
            "--max-full-usd",
            "10.0",
            "--out",
            str(out),
        ],
    )

    measure_embed.main()

    assert '"accepted": true' in capsys.readouterr().out
    assert '"accepted": true' in out.read_text()
    assert '"accelerator": "gpu"' in out.read_text()
    assert '"gpu_device": "NVIDIA L4"' in out.read_text()
    assert '"max_full_hours"' in out.read_text()
    assert '"max_full_usd"' in out.read_text()
    assert '"production_path"' in out.read_text()
    assert '"chart-embed-gpu"' in out.read_text()


def test_embed_cli_rejects_non_positive_numeric_inputs(monkeypatch) -> None:
    monkeypatch.setattr(
        measure_embed,
        "run",
        lambda **kwargs: pytest.fail("run should not be called for invalid args"),
    )
    monkeypatch.setattr("sys.argv", ["measure", "--limit", "0"])

    with pytest.raises(SystemExit) as exc:
        measure_embed.main()

    assert exc.value.code == 2
