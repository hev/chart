import argparse
from types import SimpleNamespace

import pytest

from smoke import facets


def test_parse_fields_accepts_configured_facets() -> None:
    assert facets._parse_fields("age_band, gender,events") == ["age_band", "gender", "events"]


def test_parse_fields_deduplicates_preserving_order() -> None:
    assert facets._parse_fields("events, age_band,events,gender,age_band") == [
        "events",
        "age_band",
        "gender",
    ]


def test_parse_fields_rejects_unknown_facets() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="unknown facet"):
        facets._parse_fields("age_band,unknown")


def test_parse_fields_rejects_empty_field_list() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="at least one"):
        facets._parse_fields(" , ")


def test_facets_cli_rejects_non_positive_timeout_before_setup(monkeypatch) -> None:
    monkeypatch.setattr(facets, "Settings", lambda: pytest.fail("Settings should not load for invalid args"))
    monkeypatch.setattr("sys.argv", ["facets", "--timeout", "0"])

    with pytest.raises(SystemExit) as exc:
        facets.main()

    assert exc.value.code == 2


def test_facets_cli_rejects_empty_fields_before_setup(monkeypatch) -> None:
    monkeypatch.setattr(facets, "Settings", lambda: pytest.fail("Settings should not load for invalid args"))
    monkeypatch.setattr("sys.argv", ["facets", "--fields", ","])

    with pytest.raises(SystemExit) as exc:
        facets.main()

    assert exc.value.code == 2


@pytest.mark.anyio
async def test_refresh_facets_materializes_requested_fields_and_closes_client(monkeypatch) -> None:
    calls = []
    layer = object()

    monkeypatch.setattr(
        facets,
        "Settings",
        lambda: SimpleNamespace(namespace="chart-notes", api_key="key"),
    )
    monkeypatch.setattr(facets, "make_client", lambda settings: layer)

    async def fake_materialize(layer_arg, namespace, *, fields, timeout):
        calls.append(("materialize", layer_arg, namespace, fields, timeout))

    async def fake_latest_facets(layer_arg, namespace, *, field, limit=14):
        calls.append(("latest", layer_arg, namespace, field, limit))
        return [{"value": field, "count": 1}], {"sha": field, "row_count": 12, "watermark_ms": 123}

    async def fake_close(layer_arg):
        calls.append(("close", layer_arg))

    monkeypatch.setattr(facets, "materialize_facet_snapshots", fake_materialize)
    monkeypatch.setattr(facets, "latest_facets", fake_latest_facets)
    monkeypatch.setattr(facets, "close_client", fake_close)

    result = await facets.refresh_facets(fields=["events"], timeout=12.5)

    assert result == {
        "namespace": "chart-notes",
        "status": "completed",
        "fields": ["events"],
        "snapshots": {"events": {"values": 1, "sha": "events", "row_count": 12, "watermark_ms": 123}},
    }
    assert calls == [
        ("materialize", layer, "chart-notes", ["events"], 12.5),
        ("latest", layer, "chart-notes", "events", 14),
        ("close", layer),
    ]


def test_facets_cli_writes_refresh_report(monkeypatch, tmp_path, capsys) -> None:
    async def fake_refresh_facets(*, fields, timeout):
        return {"namespace": "chart-notes", "status": "completed", "fields": fields, "timeout": timeout}

    out = tmp_path / "reports" / "facet-refresh-report.json"
    monkeypatch.setattr(facets, "refresh_facets", fake_refresh_facets)
    monkeypatch.setattr("sys.argv", ["facets", "--fields", "events", "--timeout", "12.5", "--out", str(out)])

    facets.main()

    assert '"fields": [' in capsys.readouterr().out
    text = out.read_text()
    assert '"namespace": "chart-notes"' in text
    assert '"status": "completed"' in text
    assert '"events"' in text
