from types import SimpleNamespace

import pytest

from smoke import layer_cost


def test_build_embed_layer_cost_report_shape() -> None:
    snapshot = {
        "as_of_ms": 1782320205904,
        "window_seconds": 86400,
        "totals": {"total_usd": 12.34},
        "lines": [{"basis": "invoice", "amount_usd": 12.34}],
    }

    report = layer_cost.build_report(kind="embed", snapshot=snapshot, accepted=True, signal_reviewed=False)

    assert report["source"] == "layer"
    assert report["kind"] == "embed"
    assert report["accepted"] is True
    assert report["layer_cost_snapshot"] == snapshot
    assert report["sample"]["vector_dim"] == 768
    assert report["production_path"]["pipeline_cr"] == "chart-embed-gpu"


def test_build_classifier_layer_cost_report_requires_review_signal_for_acceptance_shape() -> None:
    snapshot = {
        "as_of_ms": 1782320205904,
        "window_seconds": 86400,
        "totals": {"total_usd": 12.34},
        "lines": [{"basis": "metered", "amount_usd": 12.34}],
    }

    report = layer_cost.build_report(kind="classifier", snapshot=snapshot, accepted=True, signal_reviewed=True)

    assert report["source"] == "layer"
    assert report["kind"] == "classifier"
    assert report["accepted"] is True
    assert report["signal"]["accepted"] is True
    assert report["sample"]["med_discontinuation"] == 1
    assert report["examples"]
    assert report["writeback"]["mode"] == "tpuf.patch_columns"


@pytest.mark.anyio
async def test_fetch_layer_cost_report_closes_client(monkeypatch) -> None:
    calls = []

    class FakeLayer:
        async def get_cost_snapshot(self, *, window=None):
            calls.append(("snapshot", window))
            return SimpleNamespace(
                model_dump=lambda: {
                    "as_of_ms": 1782320205904,
                    "window_seconds": 86400,
                    "totals": {"total_usd": 12.34},
                    "lines": [{"basis": "invoice", "amount_usd": 12.34}],
                }
            )

    async def fake_close_client(layer):
        calls.append(("close", layer))

    layer = FakeLayer()
    monkeypatch.setattr(layer_cost, "Settings", lambda: SimpleNamespace(api_key="key"))
    monkeypatch.setattr(layer_cost, "make_client", lambda settings: layer)
    monkeypatch.setattr(layer_cost, "close_client", fake_close_client)

    report = await layer_cost.fetch_layer_cost_report(
        kind="embed",
        window="24h",
        accepted=True,
        signal_reviewed=False,
    )

    assert report["layer_cost_snapshot"]["totals"]["total_usd"] == 12.34
    assert calls == [("snapshot", "24h"), ("close", layer)]


def test_layer_cost_cli_requires_signal_review_for_accepted_classifier(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "layer_cost",
            "--kind",
            "classifier",
            "--accept",
            "--out",
            "eval/out/classify-events-budget.json",
        ],
    )

    with pytest.raises(SystemExit, match="--signal-reviewed is required"):
        layer_cost.main()
