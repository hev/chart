"""Fail-closed local profile tests. Live API assertions are scripts/luxir_demo.py."""
from types import SimpleNamespace
import json
import subprocess
from pathlib import Path

import pytest
from chart_common.config import Settings
from search import app as backend
from search import luxir_demo


def test_demo_is_opt_in():
    assert not Settings(_env_file=None).luxir_demo


@pytest.mark.parametrize('url,ns', [
    ('https://aws-us-east-1.hevlayer.com', 'chart_demo_test'),
    ('http://localhost:8080', 'chart_demo_test'),
    ('http://127.0.0.1:8080', 'chart-notes'),
    ('http://127.0.0.1:8080', 'chart_demo_'),
    ('http://127.0.0.1:8080', 'chart_demo_x/y'),
    ('http://127.0.0.1:8080/path', 'chart_demo_test'),
    ('http://user@127.0.0.1:8080', 'chart_demo_test'),
])
def test_demo_refuses_nonlocal_or_nonfixture_targets(url, ns):
    with pytest.raises(ValueError):
        luxir_demo.validate_target(SimpleNamespace(gateway_url=url, namespace=ns))


def test_valid_demo_target():
    luxir_demo.validate_target(SimpleNamespace(gateway_url='http://127.0.0.1:8080', namespace='chart_demo_test'))


@pytest.mark.anyio
async def test_demo_starts_without_loading_any_model(monkeypatch):
    monkeypatch.setenv('CHART_LUXIR_DEMO', '1')
    monkeypatch.setenv('CHART_NAMESPACE', 'chart_demo_test')
    monkeypatch.setenv('LAYER_GATEWAY_URL', 'http://127.0.0.1:8080')
    monkeypatch.setenv('LAYER_GATEWAY_API_KEY', 'local-demo')
    def forbidden(*args):
        raise AssertionError('lexical profile must not load a model')
    monkeypatch.setattr(backend, 'Embedder', forbidden)
    async with backend.lifespan(backend.app):
        assert backend.app.state.embedder is None
        assert backend.app.state.li_embedder is None


def test_worker_profile_and_python_contract_agree():
    root = Path(__file__).resolve().parent.parent
    source = (root / 'src/worker.js').read_text()
    config = source.split('const LUXIR_DEMO_CONFIG = ', 1)[1].split(';\n', 1)[0]
    assert json.loads(config) == luxir_demo.CONFIG
    include = source.split('const LUXIR_DEMO_INCLUDE = ', 1)[1].split(';\n', 1)[0]
    assert json.loads(include) == luxir_demo.INCLUDE
    # Parse the complete UI module. This is a syntax check, not browser evidence.
    ui = (root / 'web/static/index.html').read_text().split('<script type="module">', 1)[1].split('</script>', 1)[0]
    subprocess.run(['node', '--input-type=module', '--check'], input=ui, text=True, check=True)


def test_demo_note_viewer_relabels_authored_data_and_disables_unsupported_controls():
    ui = (Path(__file__).resolve().parent.parent / 'web/static/index.html').read_text()
    assert "Authored fixture attributes · no classifier" in ui
    assert "No real patient, source report or classifier output" in ui
    assert "similar.disabled = lexicalDemo" in ui
    assert "control.disabled = true" in ui
    assert "if (!lexicalDemo) refreshCounts" in ui


@pytest.mark.parametrize("field,value", [
    ("gateway_url", "http://other.chart-luxir.svc.cluster.local:8080"),
    ("gateway_url", "http://127.0.0.1:8080"),
    ("namespace", "chart-notes"),
    ("api_key", None),
    ("demo_basic_user", None),
    ("demo_basic_password", None),
])
def test_staging_requires_exact_service_namespace_and_auth(field, value):
    settings = dict(gateway_url=luxir_demo.STAGING_GATEWAY, namespace="chart_demo_staging",
                    luxir_staging=True, api_key="test-only", demo_basic_user="test-only",
                    demo_basic_password="test-only")
    luxir_demo.validate_target(SimpleNamespace(**settings))
    settings[field] = value
    with pytest.raises(ValueError):
        luxir_demo.validate_target(SimpleNamespace(**settings))


@pytest.mark.anyio
async def test_staging_auth_guards_ui_and_api_and_leaves_health_probe(monkeypatch):
    import base64
    import httpx
    for key, value in {"CHART_LUXIR_DEMO": "1", "CHART_LUXIR_STAGING": "1",
                       "CHART_NAMESPACE": "chart_demo_staging", "LAYER_GATEWAY_URL": luxir_demo.STAGING_GATEWAY,
                       "LAYER_GATEWAY_API_KEY": "test-only", "CHART_DEMO_BASIC_USER": "viewer",
                       "CHART_DEMO_BASIC_PASSWORD": "test-only"}.items():
        monkeypatch.setenv(key, value)
    async with backend.lifespan(backend.app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=backend.app), base_url="https://chart-luxir.example") as client:
            for path in ["/", "/api/config", "/api/search"]:
                response = await client.get(path)
                assert response.status_code == 401
                assert "Basic" in response.headers["www-authenticate"]
            for token in ["garbage", "Bearer test-only", "Basic !!!", "Basic " + base64.b64encode(b"viewer:wrong").decode()]:
                assert (await client.get("/api/config", headers={"Authorization": token})).status_code == 401
            assert (await client.get("/healthz")).status_code == 200
            assert (await client.get("/", auth=("viewer", "test-only"))).status_code == 200
            assert (await client.get("/api/config", auth=("viewer", "test-only"))).status_code == 200
            # No search request is made here; this test isolates the inbound gate.
