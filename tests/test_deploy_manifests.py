from pathlib import Path

import yaml

from chart_common.gateway import FACET_FIELDS, SCHEMA


ROOT = Path(__file__).resolve().parent.parent
ECR_REPOSITORY = "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh"


def _manifest(name: str) -> dict:
    return yaml.safe_load((ROOT / "deploy" / name).read_text())


def _manifest_all(name: str) -> list[dict]:
    return list(yaml.safe_load_all((ROOT / "deploy" / name).read_text()))


def test_worker_images_are_pinned_to_ecr() -> None:
    manifests = [
        _manifest("pipeline.yaml"),
        _manifest("pipeline-embed.yaml"),
        _manifest("functions-events.yaml"),
    ]

    for manifest in manifests:
        image = manifest["spec"]["worker"]["image"]
        assert image.startswith(f"{ECR_REPOSITORY}:")
        assert "ghcr.io" not in image
        assert not image.endswith(":latest")


def test_gpu_embed_pipeline_manifest_is_unpaused_and_gpu_backed() -> None:
    manifest = _manifest("pipeline-embed.yaml")

    assert manifest["kind"] == "Pipeline"
    assert manifest["metadata"]["name"] == "chart-embed-gpu"
    assert manifest["spec"]["paused"] is False
    assert manifest["spec"]["pipelineId"] == "chart-notes"
    assert manifest["spec"]["target"]["namespace"] == "chart-notes"
    assert manifest["spec"]["worker"]["computeClass"] == "gpu"
    container = manifest["spec"]["worker"]["podSpec"]["containers"][0]
    assert container["name"] == "worker"
    assert container["command"] == ["uv", "run", "python", "-m", "indexer.embed"]
    assert manifest["spec"]["scaling"]["pool"] == "gpu"
    assert manifest["spec"]["scaling"]["replicas"]["max"] == 1
    env = {item["name"]: item for item in container["env"]}
    assert env["CHART_EMBED_MODEL"]["value"] == "Snowflake/snowflake-arctic-embed-m-v1.5"
    assert env["CHART_EMBED_SIMILAR_QRELS_SPLIT"]["value"] == "train"
    assert env["LAYER_GATEWAY_API_KEY"]["valueFrom"]["secretKeyRef"] == {
        "name": "chart-gateway",
        "key": "LAYER_GATEWAY_API_KEY",
    }


def test_events_function_manifest_is_resumable() -> None:
    manifest = _manifest("functions-events.yaml")

    assert manifest["kind"] == "Function"
    assert manifest["metadata"]["name"] == "chart-classify-events"
    assert manifest["spec"]["paused"] is True
    assert manifest["spec"]["targetNamespaces"] == ["chart-notes"]
    assert manifest["spec"]["inputs"] == ["id", "text"]
    assert manifest["spec"]["filter"] == ["events", "Eq", None]
    assert set(manifest["spec"]["triggers"]) == {"discovery", "write"}
    container = manifest["spec"]["worker"]["podSpec"]["containers"][0]
    assert container["name"] == "udf"
    assert container["command"] == ["uv", "run", "python", "-m", "functions.classify_events"]
    assert manifest["spec"]["scaling"]["pool"] == "gpu"
    assert manifest["spec"]["worker"]["computeClass"] == "gpu"
    assert manifest["spec"]["scaling"]["replicas"]["max"] == 2


def test_ingest_index_and_facets_share_namespace_and_pins() -> None:
    namespace = _manifest("namespace.yaml")
    pipeline = _manifest("pipeline.yaml")
    index = _manifest("index.yaml")
    warehouse = _manifest("warehouse.yaml")
    vectorstore = _manifest("vectorstore.yaml")

    assert namespace["kind"] == "Namespace"
    assert namespace["metadata"]["name"] == "chart"
    assert pipeline["spec"]["pipelineId"] == "chart-notes"
    assert pipeline["spec"]["target"]["namespace"] == "chart-notes"
    assert pipeline["spec"]["sourceRef"]["dataset"] == "zhengyun21/PMC-Patients"
    assert pipeline["spec"]["sourceRef"]["revision"] == "28d8836518f86d4f1e6358ea8ec09977023e5766"
    assert pipeline["spec"]["sourceRef"]["mapping"]["attributes"] == [
        "patient_uid",
        "PMID",
        "title",
        "age",
        "gender",
    ]
    assert pipeline["spec"]["sourceRef"]["chunk"]["size"] == 512
    assert pipeline["spec"]["sourceRef"]["chunk"]["tokenizer"] == "snowflake-arctic-embed-m-v1.5"

    assert index["spec"]["backend"]["namespace"] == "chart-notes"
    assert index["spec"]["backend"]["storeRef"] == "turbopuffer-default"
    assert index["spec"]["snapshot"]["facetFields"] == FACET_FIELDS
    assert all(field in SCHEMA and "type" in SCHEMA[field] for field in FACET_FIELDS)
    assert warehouse["spec"]["kind"] == "huggingface"
    assert vectorstore["spec"]["kind"] == "turbopuffer"
    assert vectorstore["spec"]["inboundAuth"]["mode"] == "deriveFromStore"


def test_secret_template_documents_required_runtime_secrets() -> None:
    secrets = _manifest_all("secrets.example.yaml")

    assert [secret["metadata"]["name"] for secret in secrets] == ["chart-turbopuffer", "chart-gateway"]
    assert all(secret["kind"] == "Secret" for secret in secrets)
    assert all(secret["metadata"]["namespace"] == "chart" for secret in secrets)
    assert secrets[0]["stringData"] == {"credential": "REPLACE_WITH_TURBOPUFFER_KEY"}
    assert secrets[1]["stringData"] == {"LAYER_GATEWAY_API_KEY": "REPLACE_WITH_LAYER_GATEWAY_API_KEY"}


def test_gpu_dockerfile_builds_manifest_images() -> None:
    dockerfile = (ROOT / "deploy" / "Dockerfile.gpu").read_text()
    build_script = (ROOT / "scripts" / "build_gpu_images.sh").read_text()
    embed_manifest = _manifest("pipeline-embed.yaml")
    classifier_manifest = _manifest("functions-events.yaml")

    assert "FROM base AS embedder" in dockerfile
    assert "FROM base AS classifier" in dockerfile
    assert "python\", \"-m\", \"indexer.embed\"" in dockerfile
    assert "python\", \"-m\", \"functions.classify_events\"" in dockerfile
    assert embed_manifest["spec"]["worker"]["image"] == f"{ECR_REPOSITORY}:chart-embedder-plan-20260624-dedupe2"
    assert classifier_manifest["spec"]["worker"]["image"] == f"{ECR_REPOSITORY}:chart-classifier-plan-20260624"
    assert not embed_manifest["spec"]["worker"]["image"].endswith(":latest")
    assert not classifier_manifest["spec"]["worker"]["image"].endswith(":latest")
    assert f'default_embed_tag="{embed_manifest["spec"]["worker"]["image"]}"' in build_script
    assert f'default_classifier_tag="{classifier_manifest["spec"]["worker"]["image"]}"' in build_script
    assert 'embed_tag="${CHART_EMBED_IMAGE:-$default_embed_tag}"' in build_script
    assert 'classifier_tag="${CHART_CLASSIFIER_IMAGE:-$default_classifier_tag}"' in build_script


def test_dockerignore_keeps_gpu_context_free_of_local_secrets_and_outputs() -> None:
    dockerignore = (ROOT / ".dockerignore").read_text().splitlines()

    assert ".env" in dockerignore
    assert ".env.*" in dockerignore
    assert "!.env.example" in dockerignore
    assert "eval/out" in dockerignore
    assert "deploy/*secret*.yaml" in dockerignore
    assert "!deploy/secrets.example.yaml" in dockerignore


def test_gitignore_keeps_generated_live_reports_out_of_commits() -> None:
    gitignore = (ROOT / ".gitignore").read_text().splitlines()

    assert "eval/out/" in gitignore
