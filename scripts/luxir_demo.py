#!/usr/bin/env python3
"""Seed or verify a disposable loopback Luxir Chart demo using real HTTP."""
import time
import hashlib
import argparse
import os
import base64
import json
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from search.luxir_demo import INCLUDE, validate_target
from types import SimpleNamespace


def request(base, path, body=None, expected=200):
    headers = {"Content-Type": "application/json"}
    if base == os.environ.get("LAYER_GATEWAY_URL") and os.environ.get("LAYER_GATEWAY_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["LAYER_GATEWAY_API_KEY"]
    elif os.environ.get("CHART_LUXIR_STAGING") == "1":
        headers["Authorization"] = "Basic " + base64.b64encode((os.environ["CHART_DEMO_BASIC_USER"] + ":" + os.environ["CHART_DEMO_BASIC_PASSWORD"]).encode()).decode()
    req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as error:
        status, raw = error.code, error.read()
    value = json.loads(raw)
    print(json.dumps({"path": path, "request": body, "status": status, "response": value}), flush=True)
    assert status == expected, (path, status, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["seed", "verify"])
    parser.add_argument("--gateway", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--app")
    args = parser.parse_args()
    validate_target(SimpleNamespace(gateway_url=args.gateway, namespace=args.namespace,
        luxir_staging=os.environ.get("CHART_LUXIR_STAGING") == "1", api_key=os.environ.get("LAYER_GATEWAY_API_KEY"),
        demo_basic_user=os.environ.get("CHART_DEMO_BASIC_USER"), demo_basic_password=os.environ.get("CHART_DEMO_BASIC_PASSWORD")))
    fixture = json.loads((ROOT / "demo/luxir-notes.json").read_text())
    print(json.dumps({"fixture_sha256": hashlib.sha256((ROOT / "demo/luxir-notes.json").read_bytes()).hexdigest(), "provenance": fixture.get("provenance"), "namespace": args.namespace}))
    rows = fixture["rows"]
    ns = "/v2/namespaces/" + args.namespace
    if args.phase == "seed":
        schema = {name: {"type": "int" if name == "age" else "string"} for name in INCLUDE if name != "id"}
        schema["text"]["full_text_search"] = True
        request(args.gateway, ns, {"schema": schema, "upsert_rows": rows})
        print("PASS seed: six authored synthetic notes, no models or archives")
        return
    assert urllib.parse.urlsplit(args.app).hostname == "127.0.0.1" or (os.environ.get("CHART_LUXIR_STAGING") == "1" and args.app == "http://chart-luxir-app.chart-luxir.svc.cluster.local:8000")
    if os.environ.get("CHART_LUXIR_STAGING") == "1":
        deadline = time.monotonic() + 60
        while True:
            try:
                with urllib.request.urlopen(args.app + "/healthz", timeout=2) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("owned Chart app readiness deadline exceeded")
            time.sleep(1)
    stores = request(args.gateway, "/v2/vectorstores")
    from hevlayer.models import VectorStoreList
    typed_stores = VectorStoreList.model_validate(stores)
    assert any(store.kind == "luxir" and store.default for store in typed_stores.vectorstores)
    if os.environ.get("CHART_LUXIR_STAGING") == "1" or os.environ.get("LAYER_SECRET_CHART_LUXIR_AUTH_GATEWAY_KEY"):
        assert next(store for store in stores["vectorstores"] if store["kind"] == "luxir")["inboundAuth"]["mode"] == "keys"
    config = request(args.app, "/api/config")
    assert config["profile"] == "luxir_lexical_demo" and config["namespace"] == args.namespace
    def search(query="", filters=(), expected=()):
        path = "/api/search?" + urllib.parse.urlencode([("q", query), *[("f", f) for f in filters]])
        value = request(args.app, path)
        ids = [r["id"] for r in value["rows"]]
        assert set(ids) == set(expected), (ids, expected)
        assert value["routing"] is None and value["retrieval"] == ("BM25" if query else "ID order")
        assert all(r["text"] and r["title"] for r in value["rows"])
        return ids
    assert search(expected=[r["id"] for r in rows]) == [r["id"] for r in rows]
    search("dyspnea", expected=["case_001", "case_002", "case_004"])
    search("dyspnea", ["specialty:cardiology"], ["case_001", "case_004"])
    search("dyspnea", ["specialty:cardiology", "age_band:older_adult"], ["case_001"])
    search("dyspnea", ["specialty:cardiology", "specialty:pulmonology", "gender:female"], ["case_001", "case_002"])
    search("metformin", expected=["case_003"])
    search("unfindabletoken", expected=[])
    assert search(filters=["specialty:cardiology"], expected=["case_001", "case_004"]) == ["case_001", "case_004"]
    for path in ["/api/search?q=dyspnea&agentic=1", "/api/search?q=dyspnea&li=1", "/api/search?f=events:medication", "/api/similar/case_001", "/api/facets", "/api/facet-counts", "/api/storage"]:
        request(args.app, path, expected=422)
    # Exercise the bounded primitive directly in both directions, with projection.
    for direction, ids in [("asc", ["case_001", "case_004"]), ("desc", ["case_004", "case_001"])]:
        value = request(args.gateway, ns + "/query", {"rank_by": ["id", direction], "filters": ["specialty", "Eq", "cardiology"], "top_k": 20, "include_attributes": ["title"]})
        assert [r["id"] for r in value["rows"]] == ids
        assert all(set(r) == {"id", "title"} for r in value["rows"])
    for direction, expected_id in [("asc", "case_003"), ("desc", "case_006")]:
        value = request(args.gateway, ns + "/query", {"rank_by": ["id", direction], "filters": ["specialty", "Eq", "endocrinology"], "top_k": 1, "include_attributes": False})
        assert value["rows"] == [{"id": expected_id}], value
    for body in [{"rank_by": ["age", "asc"]}, {"rank_by": ["id", "asc"], "filters": ["age", "Eq", "72"]}, {"rank_by": ["id", "asc"], "filters": ["id", "Gt", "case_001"]}]:
        request(args.gateway, ns + "/query", body, expected=422)
    ui_headers = {}
    if os.environ.get("CHART_LUXIR_STAGING") == "1":
        ui_headers["Authorization"] = "Basic " + base64.b64encode((os.environ["CHART_DEMO_BASIC_USER"] + ":" + os.environ["CHART_DEMO_BASIC_PASSWORD"]).encode()).decode()
    with urllib.request.urlopen(urllib.request.Request(args.app + "/", headers=ui_headers), timeout=10) as response:
        assert response.status == 200 and b'id="noteview"' in response.read()
    print("PASS actual Chart API -> Layer -> native Luxir: browse, BM25, scalar Eq/In/And, projection, refusals; static UI served. Browser validation remains separate.")


if __name__ == "__main__":
    main()
