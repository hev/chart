import json
import shlex
from pathlib import Path

import pytest

from smoke import plan_audit

ROOT = Path(__file__).resolve().parent.parent


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def env_for(tmp_path):
    env = {}
    for name, (var, default) in plan_audit.DEFAULT_REPORTS.items():
        env[var] = str(tmp_path / default)
    return env


def live_route(route: str, *, rows: int = 5) -> dict:
    return {
        "route": route,
        "routing": {"route": route, "reason": "test policy"},
        "hybrid": {"leg_breakdown": [{"leg": route}]},
        "rows": rows,
    }


def full_facet_status():
    return {
        "specialty": {"values": 4, "row_count": 167000, "sha": "full-snapshot"},
        "age_band": {"values": 6, "row_count": 167000, "sha": "full-snapshot"},
        "diagnosis_category": {"values": 5, "row_count": 167000, "sha": "full-snapshot"},
        "gender": {"values": 2, "row_count": 167000, "sha": "full-snapshot"},
        "events": {"values": 4, "row_count": 167000, "sha": "full-snapshot"},
    }


def phase6_cost_baselines(
    *,
    embed_report: str = "eval/out/embed-budget.json",
    classifier_report: str = "eval/out/classify-events-budget.json",
    embed_accepted: bool = True,
    classifier_accepted: bool = True,
) -> dict:
    return {
        "embed": {
            "report": embed_report,
            "accepted": embed_accepted,
            "estimate": {
                "full_notes": 167000,
                "estimated_full_seconds": 1000.0,
                "estimated_full_hours": 0.28,
                "gpu_hourly_usd": 2.5,
                "estimated_full_usd": 0.7,
            },
        },
        "classifier": {
            "report": classifier_report,
            "accepted": classifier_accepted,
            "estimate": {
                "full_notes": 167000,
                "estimated_full_seconds": 2000.0,
                "estimated_full_hours": 0.56,
                "gpu_hourly_usd": 2.5,
                "estimated_full_usd": 1.4,
            },
        },
    }


def gpu_runtime(**overrides):
    runtime = {"accelerator": "gpu", "gpu_device": "NVIDIA L4", "gpu_probe": "nvidia-smi"}
    runtime.update(overrides)
    return runtime


def layer_cost_report(**overrides):
    report = {
        "source": "layer",
        "accepted": True,
        "layer_cost_snapshot": {
            "as_of_ms": 1782320205904,
            "window_seconds": 86400,
            "totals": {"total_usd": 12.34, "aws_usd": 10.0, "turbopuffer_usd": 2.34},
            "lines": [
                {
                    "provider": "aws",
                    "service": "compute",
                    "basis": "invoice",
                    "amount_usd": 10.0,
                }
            ],
            "caveats": [],
        },
    }
    report.update(overrides)
    return report


def phase6_targets(**overrides):
    targets = {
        "namespace": "chart-notes",
        "pipeline_id": "chart-notes",
        "udf_id": "chart-classify-events",
        "full_corpus_notes": 167000,
        "embed_pipeline_cr": "chart-embed-gpu",
        "embed_compute_class": "gpu",
        "embed_image": "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-embedder-plan-20260624-dedupe2",
        "classifier_compute_class": "gpu",
        "classifier_image": "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-classifier-plan-20260624",
    }
    targets.update(overrides)
    return targets


def embed_production_path(**overrides):
    path = {
        "pipeline_cr": "chart-embed-gpu",
        "module": "indexer.embed",
        "compute_class": "gpu",
        "image": "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-embedder-plan-20260624-dedupe2",
        "allow_full_cpu_index": False,
    }
    path.update(overrides)
    return path


def slice_provenance(**overrides):
    provenance = {
        "dataset_repo": "zhengyun21/PMC-Patients",
        "dataset_revision": "28d8836518f86d4f1e6358ea8ec09977023e5766",
        "dataset_split": "train",
        "embed_model": "Snowflake/snowflake-arctic-embed-m-v1.5",
        "embed_dim": 768,
    }
    provenance.update(overrides)
    return provenance


def recds_provenance(**overrides):
    provenance = {
        "recds_repo": "zhengyun21/PMC-Patients-ReCDS",
        "recds_revision": "a27717bb27679cf0860305997685547ca01b3dd1",
        "embed_model": "Snowflake/snowflake-arctic-embed-m-v1.5",
        "embed_dim": 768,
        "namespace": "chart-notes",
    }
    provenance.update(overrides)
    return provenance


def bimodal_dataset(**overrides):
    dataset = {
        "short": 2,
        "long": 2,
        "split": "dev",
        "dataset_repo": "zhengyun21/PMC-Patients",
        "dataset_revision": "28d8836518f86d4f1e6358ea8ec09977023e5766",
        "dataset_split": "train",
        "recds_repo": "zhengyun21/PMC-Patients-ReCDS",
        "recds_revision": "a27717bb27679cf0860305997685547ca01b3dd1",
    }
    dataset.update(overrides)
    return dataset


def fused_dominance_checks():
    return [
        {"metric": "RR@10", "baseline": "bm25", "fused": 0.31, "baseline_value": 0.3, "ok": True},
        {"metric": "RR@10", "baseline": "semantic", "fused": 0.31, "baseline_value": 0.29, "ok": True},
        {"metric": "nDCG@10", "baseline": "bm25", "fused": 0.26, "baseline_value": 0.25, "ok": True},
        {"metric": "nDCG@10", "baseline": "semantic", "fused": 0.26, "baseline_value": 0.24, "ok": True},
        {"metric": "R@1000", "baseline": "bm25", "fused": 0.9, "baseline_value": 0.88, "ok": True},
        {"metric": "R@1000", "baseline": "semantic", "fused": 0.9, "baseline_value": 0.89, "ok": True},
    ]


def bimodal_summary(strategy: str, *, failed: int = 0, published: bool = False) -> dict:
    summary = {
        "strategy": strategy,
        "metrics": {"RR@10": 0.4, "nDCG@10": 0.35, "R@1000": 0.8},
        "queries": {
            "total": 4,
            "attempted": 4,
            "failed": failed,
            "succeeded": 4 - failed,
            "scored": 4,
            "by_kind": {"short": 2, "long": 2},
        },
        "top_k": 1000,
        "metrics_by_kind": {
            "short": {"RR@10": 0.5, "nDCG@10": 0.45, "R@1000": 1.0},
            "long": {"RR@10": 0.3, "nDCG@10": 0.25, "R@1000": 0.6},
        },
        "dataset": bimodal_dataset(),
    }
    if published:
        summary["published"] = {"baseline": "rrf"}
    return summary


def test_required_plan_checks_have_operator_next_actions() -> None:
    required_checks = {name for name, spec in plan_audit.CHECKS.items() if not spec.get("optional")}

    assert set(plan_audit.NEXT_ACTIONS) == required_checks
    assert set(plan_audit.NEXT_REQUIREMENTS) == required_checks
    assert "phase5_bimodal_if_present" not in plan_audit.NEXT_ACTIONS
    assert "phase5_bimodal_if_present" not in plan_audit.NEXT_REQUIREMENTS
    for name, prerequisites in plan_audit.NEXT_PREREQUISITES.items():
        assert name in required_checks
        assert set(prerequisites).issubset(required_checks)


def test_plan_audit_next_step_prerequisites_are_acyclic() -> None:
    visiting = set()
    visited = set()

    def visit(name: str) -> None:
        assert name not in visiting, f"cycle in NEXT_PREREQUISITES at {name}"
        if name in visited:
            return
        visiting.add(name)
        for prerequisite in plan_audit.NEXT_PREREQUISITES.get(name, ()):
            visit(prerequisite)
        visiting.remove(name)
        visited.add(name)

    for name in plan_audit.NEXT_PREREQUISITES:
        visit(name)


def test_plan_audit_next_action_commands_reference_existing_entrypoints() -> None:
    def assert_entrypoint(part: str, *, command: str) -> None:
        words = shlex.split(part)
        while words and "=" in words[0] and not words[0].startswith("--"):
            words.pop(0)
        assert words, command

        if words[0].startswith("scripts/"):
            assert (ROOT / words[0]).is_file(), command
        elif words[:4] == ["uv", "run", "--extra", "classifier"]:
            assert words[4:7] == ["python", "-m", "functions.measure_classify_events"], command
        elif words[:4] == ["uv", "run", "--extra", "search"]:
            assert words[4:7] == ["python", "-m", "indexer.measure_embed"], command
        elif words[:4] == ["uv", "run", "--extra", "eval"]:
            assert words[4:7] == ["python", "-m", "eval.holdout"], command
        else:
            raise AssertionError(f"unrecognized next-action command: {command}")

    for command in plan_audit.NEXT_ACTIONS.values():
        for part in command.split(" && "):
            assert_entrypoint(part, command=command)


def test_plan_audit_reports_missing_required_artifacts(tmp_path) -> None:
    env = env_for(tmp_path)
    write(tmp_path / "eval/out/gpu-build-report.json", {"mode": "dry-run", "status": "dry-run"})

    report = plan_audit.summarize(env=env)

    assert report["complete"] is False
    assert report["checks"]["phase1_slice_index"] == {
        "ok": False,
        "optional": False,
        "reports": ["slice_index"],
        "reason": "missing report(s): slice_index",
    }
    assert report["checks"]["phase5_bimodal_if_present"]["ok"] is True
    assert report["checks"]["phase5_bimodal_if_present"]["reason"] == "optional report missing"
    assert report["checks"]["phase6_gpu_images"]["ok"] is False
    assert report["next_steps"][0] == {
        "check": "phase1_slice_index",
        "reason": "missing report(s): slice_index",
        "command": "scripts/live_slice.sh",
        "requires": ["gateway_key"],
        "ready": True,
    }
    assert {
        "check": "phase3_facet_refresh",
        "reason": "missing report(s): facet_refresh",
        "command": "scripts/refresh_facets.sh --fields age_band,gender",
        "requires": ["gateway_key"],
        "ready": True,
    } in report["next_steps"]
    live_step = next(step for step in report["next_steps"] if step["check"] == "phase2_3_live_smoke")
    assert live_step["requires"] == ["gateway_key"]
    assert live_step["ready"] is False
    assert live_step["blocked_by"] == ["phase1_slice_index", "phase3_facet_refresh"]
    unpause_step = next(step for step in report["next_steps"] if step["check"] == "phase6_unpause_embed")
    assert unpause_step["requires"] == ["kubectl_context", "k8s_secrets", "embed_cost_acceptance"]
    assert unpause_step["blocked_by"] == ["phase6_deploy_apply", "phase6_embed_cost"]
    assert all(step["check"] != "phase5_bimodal_if_present" for step in report["next_steps"])


def test_plan_audit_requirement_readiness_marks_missing_prerequisites() -> None:
    report = {
        "next_steps": [
            {
                "check": "phase4_classify_cost_signal",
                "requires": ["classifier_extra"],
                "ready": True,
            },
            {
                "check": "phase5_recds",
                "requires": ["gateway_key", "full_retrieval_corpus"],
                "ready": True,
            },
        ]
    }
    requirements = {
        "classifier_extra": {"state": "unknown"},
        "gateway_key": {"state": "unknown"},
        "full_retrieval_corpus": {"state": "missing"},
    }

    plan_audit.apply_requirement_readiness(report, requirements)

    assert report["next_steps"][0]["ready"] is True
    assert "missing_requirements" not in report["next_steps"][0]
    assert report["next_steps"][1]["ready"] is False
    assert report["next_steps"][1]["missing_requirements"] == ["full_retrieval_corpus"]


def test_plan_audit_accepts_optional_bimodal_beir_report_when_present(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/bimodal-report.json",
        {
            "task": None,
            "split": None,
            "beir_dir": "eval/out/bimodal",
            "strategies": ["auto", "semantic", "bm25", "fused"],
            "limit": 4,
            "top_k": 1000,
            "summaries": [
                bimodal_summary("auto"),
                bimodal_summary("semantic"),
                bimodal_summary("bm25"),
                bimodal_summary("fused"),
            ],
            "gates": {
                "no_failures": {"required": True, "accepted": True},
                "fused_dominates": {
                    "accepted": True,
                    "checks": fused_dominance_checks(),
                    "query_failures": [],
                },
            },
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase5_bimodal_if_present"] == {
        "ok": True,
        "optional": True,
        "reports": ["bimodal"],
    }
    assert all(step["check"] != "phase5_bimodal_if_present" for step in report["next_steps"])


def test_plan_audit_rejects_optional_bimodal_report_with_recds_shape(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/bimodal-report.json",
        {
            "task": "ppr",
            "split": "dev",
            "beir_dir": None,
            "provenance": recds_provenance(),
            "strategies": ["auto", "semantic", "bm25", "fused"],
            "limit": 4,
            "top_k": 1000,
            "summaries": [
                bimodal_summary("auto", published=True),
                bimodal_summary("semantic"),
                bimodal_summary("bm25"),
                bimodal_summary("fused"),
            ],
            "gates": {
                "no_failures": {"required": True, "accepted": True},
                "fused_dominates": {
                    "accepted": True,
                    "checks": fused_dominance_checks(),
                    "query_failures": [],
                },
            },
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase5_bimodal_if_present"] == {
        "ok": False,
        "optional": True,
        "reports": ["bimodal"],
        "reason": "report content does not satisfy gate",
        "details": [
            "task='ppr', expected null for bimodal BEIR eval",
            "split='dev', expected null for bimodal BEIR eval",
            "beir_dir must point at the bimodal BEIR directory",
            "auto summary must not include published baseline for bimodal BEIR eval",
        ],
    }
    assert all(step["check"] != "phase5_bimodal_if_present" for step in report["next_steps"])


def test_plan_audit_rejects_optional_bimodal_report_without_source_provenance(tmp_path) -> None:
    env = env_for(tmp_path)
    stale = bimodal_dataset(dataset_revision="old", recds_revision="old")
    write(
        tmp_path / "eval/out/bimodal-report.json",
        {
            "task": None,
            "split": None,
            "beir_dir": "eval/out/bimodal",
            "strategies": ["auto", "semantic", "bm25", "fused"],
            "limit": 4,
            "top_k": 1000,
            "summaries": [
                bimodal_summary("auto") | {"dataset": stale},
                bimodal_summary("semantic"),
                bimodal_summary("bm25"),
                bimodal_summary("fused"),
            ],
            "gates": {
                "no_failures": {"required": True, "accepted": True},
                "fused_dominates": {
                    "accepted": True,
                    "checks": fused_dominance_checks(),
                    "query_failures": [],
                },
            },
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase5_bimodal_if_present"]["ok"] is False
    assert (
        "auto.dataset.dataset_revision='old', expected '28d8836518f86d4f1e6358ea8ec09977023e5766'"
        in report["checks"]["phase5_bimodal_if_present"]["details"]
    )
    assert (
        "auto.dataset.recds_revision='old', expected 'a27717bb27679cf0860305997685547ca01b3dd1'"
        in report["checks"]["phase5_bimodal_if_present"]["details"]
    )
    assert all(step["check"] != "phase5_bimodal_if_present" for step in report["next_steps"])


def test_plan_audit_marks_complete_from_accepted_reports(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/slice-index-report.json",
        {
            "status": "completed",
            "limit": 2000,
            "dry_run": False,
            "indexed": 2000,
            "provenance": slice_provenance(),
            "schema": {
                "vector_dim": 768,
                "rows_with_age_band": 1800,
                "rows_with_gender": 1700,
                "rows_with_similar_patient_ids": 1200,
            },
            "facet_snapshots_materialized": True,
        },
    )
    write(
        tmp_path / "eval/out/live-smoke-base-report.json",
        {
            "ok": True,
            "status": "completed",
            "requirements": {"facets": True, "event_facets": False},
            "index_shape": {"vector_dim": 768, "rows": 20},
            "routes": [
                live_route("hybrid_text"),
                live_route("fused"),
                live_route("semantic"),
            ],
            "similar": {"neighbors": 3},
            "facets": {
                "age_band": {"values": 4, "sha": "slice-sha"},
                "gender": {"values": 2, "sha": "slice-sha"},
                "events": {"values": 3, "sha": "slice-sha"},
            },
        },
    )
    write(
        tmp_path / "eval/out/live-smoke-report.json",
        {
            "ok": True,
            "status": "completed",
            "requirements": {"facets": True, "event_facets": True},
            "index_shape": {"vector_dim": 768, "rows": 20},
            "routes": [
                live_route("hybrid_text"),
                live_route("fused"),
                live_route("semantic"),
            ],
            "similar": {"neighbors": 3},
            "facets": {
                "age_band": {"values": 4, "sha": "slice-sha"},
                "gender": {"values": 2, "sha": "slice-sha"},
                "events": {"values": 3, "sha": "slice-sha"},
            },
        },
    )
    write(
        tmp_path / "eval/out/facet-refresh-report.json",
        {
            "status": "completed",
            "fields": ["age_band", "gender", "events"],
            "snapshots": {
                "age_band": {"values": 4, "sha": "facet-sha"},
                "gender": {"values": 2, "sha": "facet-sha"},
                "events": {"values": 3, "sha": "facet-sha"},
            },
        },
    )
    write(
        tmp_path / "eval/out/embed-budget.json",
        {
            "runtime": gpu_runtime(),
            "sample": {
                "notes": 12,
                "vector_dim": 768,
                "model": "Snowflake/snowflake-arctic-embed-m-v1.5",
            },
            "production_path": embed_production_path(),
            "estimate": {
                "full_notes": 167000,
                "estimated_full_seconds": 1000.0,
                "estimated_full_hours": 0.28,
                "gpu_hourly_usd": 2.5,
                "estimated_full_usd": 0.7,
            },
            "budget": {
                "accepted": True,
                "checks": {
                    "max_full_hours": {"ok": True},
                    "max_full_usd": {"ok": True},
                },
            },
        },
    )
    write(
        tmp_path / "eval/out/classify-events-budget.json",
        {
            "runtime": gpu_runtime(),
            "sample": {"notes": 6, "med_discontinuation": 1},
            "examples": [
                {
                    "note_preview": "metformin stopped",
                    "events": [{"type": "medication_discontinued", "reason": "adverse_effect"}],
                    "labels": {"has_med_discontinuation": True},
                    "discontinuation_reason": "adverse_effect",
                }
            ],
            "writeback": {
                "mode": "tpuf.patch_columns",
                "primary_output": "events",
                "model_passes_per_note": 1,
                "patched_fields": [
                    "events",
                    "has_med_discontinuation",
                    "has_adverse_event",
                    "diagnosis_category",
                    "specialty",
                    "discontinuation_reason",
                ],
                "settles_multi_write": True,
            },
            "estimate": {
                "full_notes": 167000,
                "estimated_full_seconds": 2000.0,
                "estimated_full_hours": 0.56,
                "gpu_hourly_usd": 2.5,
                "estimated_full_usd": 1.4,
            },
            "budget": {
                "accepted": True,
                "checks": {
                    "max_full_hours": {"ok": True},
                    "max_full_usd": {"ok": True},
                },
            },
            "signal": {
                "accepted": True,
                "checks": {
                    "min_med_discontinuations": {"ok": True},
                    "min_review_examples": {"ok": True},
                },
            },
        },
    )
    write(
        tmp_path / "eval/out/holdout-report.json",
        {
            "overlap_edges": 0,
            "gate": {
                "accepted": True,
                "checks": {
                    "feature_edges_present": {"ok": True},
                    "qrel_edges_present": {"ok": True},
                    "max_overlap_edges": {"ok": True},
                },
            },
        },
    )
    write(
        tmp_path / "eval/out/recds-report.json",
        {
            "task": "ppr",
            "split": "dev",
            "beir_dir": None,
            "provenance": recds_provenance(),
            "strategies": ["auto", "semantic", "bm25", "fused"],
            "limit": 500,
            "top_k": 1000,
            "summaries": [
                {"strategy": "auto", "queries": {"failed": 0}},
                {"strategy": "semantic", "queries": {"failed": 0}},
                {"strategy": "bm25", "queries": {"failed": 0}},
                {
                    "strategy": "fused",
                    "queries": {"failed": 0},
                    "published": {
                        "baseline": "rrf",
                        "baseline_metrics": {"RR@10": 0.2776, "nDCG@10": 0.2412, "R@1000": 0.8514},
                        "delta": {"RR@10": 0.01, "nDCG@10": 0.01, "R@1000": 0.01},
                        "meets_or_beats": True,
                    },
                },
            ],
            "gates": {
                "no_failures": {"required": True, "accepted": True},
                "fused_dominates": {
                    "accepted": True,
                    "checks": fused_dominance_checks(),
                    "query_failures": [],
                },
            },
        },
    )
    write(
        tmp_path / "eval/out/gpu-build-report.json",
        {
            "mode": "push",
            "status": "completed",
            "embed_image": "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-embedder-plan-20260624-dedupe2",
            "classifier_image": "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-classifier-plan-20260624",
            "embed_command": [
                "docker",
                "buildx",
                "build",
                "--push",
                "-t",
                "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-embedder-plan-20260624-dedupe2",
                ".",
            ],
            "classifier_command": [
                "docker",
                "buildx",
                "build",
                "--push",
                "-t",
                "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-classifier-plan-20260624",
                ".",
            ],
        },
    )
    write(
        tmp_path / "eval/out/deploy-apply-report.json",
        {
            "mode": "apply",
            "status": "completed",
            "namespace": "chart",
            "kube_context": "test-context",
            "kube_context_confirmed": True,
            "classifier_cost_accepted": True,
            "classifier": "applied",
            "classifier_report": env["CHART_PHASE4_CLASSIFY_REPORT"],
            "manifests": [
                "deploy/namespace.yaml",
                "deploy/vectorstore.yaml",
                "deploy/warehouse.yaml",
                "deploy/pipeline.yaml",
                "deploy/pipeline-embed.yaml",
                "deploy/index.yaml",
                "deploy/functions-events.yaml",
            ],
            "runtime_manifests": [
                "deploy/vectorstore.yaml",
                "deploy/warehouse.yaml",
                "deploy/pipeline.yaml",
                "deploy/pipeline-embed.yaml",
                "deploy/index.yaml",
            ],
        },
    )
    write(
        tmp_path / "eval/out/phase6-unpause-report.json",
        {
            "mode": "unpause",
            "status": "unpaused",
            "namespace": "chart",
            "pipeline_cr": "chart-embed-gpu",
            "expected_pipeline_id": "chart-notes",
            "actual_pipeline_id": "chart-notes",
            "budget_report": env["CHART_PHASE6_EMBED_BUDGET_REPORT"],
        },
    )
    write(
        tmp_path / "eval/out/phase6-status-report.json",
        {
            "namespace": "chart-notes",
            "targets": phase6_targets(),
            "cost_baselines": phase6_cost_baselines(
                embed_report=env["CHART_PHASE6_EMBED_BUDGET_REPORT"],
                classifier_report=env["CHART_PHASE4_CLASSIFY_REPORT"],
            ),
            "pipeline": {"pipeline_id": "chart-notes"},
            "udf": {"udf_id": "chart-classify-events"},
            "facets": full_facet_status(),
        },
    )
    write(
        tmp_path / "eval/out/phase6-gate-report.json",
        {
            "namespace": "chart-notes",
            "targets": phase6_targets(),
            "gates": {
                "pipeline_installed": True,
                "udf_installed": True,
                "full_index_complete": True,
                "full_classify_complete": True,
                "base_facets_visible": True,
                "event_facets_visible": True,
                "full_facets_complete": True,
                "cost_baselines_accepted": True,
                "phase6_complete": True,
            },
            "failures": [],
            "status": {
                "targets": phase6_targets(),
                "pipeline": {"pipeline_id": "chart-notes"},
                "udf": {"udf_id": "chart-classify-events"},
            },
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["complete"] is True
    assert report["next_steps"] == []
    assert all(check["ok"] for check in report["checks"].values())


def test_plan_audit_requires_classify_budget_and_signal(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/classify-events-budget.json",
        {
            "sample": {"notes": 6},
            "budget": {
                "accepted": True,
                "checks": {
                    "max_full_hours": {"ok": True},
                    "max_full_usd": {"ok": True},
                },
            },
            "signal": {"accepted": False},
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase4_classify_cost_signal"] == {
        "ok": False,
        "optional": False,
        "reports": ["classify_budget"],
        "reason": "report content does not satisfy gate",
    }
    step = next(step for step in report["next_steps"] if step["check"] == "phase4_classify_cost_signal")
    assert "signal.accepted=False, expected true" in step["details"]
    assert "estimate.full_notes=None, expected 167000" in step["details"]
    assert "sample.med_discontinuation must be positive" in step["details"]
    assert "writeback.mode=None, expected 'tpuf.patch_columns'" in step["details"]


def test_plan_audit_requires_named_budget_checks(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/embed-budget.json",
        {"runtime": gpu_runtime(), "sample": {"notes": 2}, "budget": {"accepted": True}},
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase6_embed_cost"] == {
        "ok": False,
        "optional": False,
        "reports": ["embed_budget"],
        "reason": "report content does not satisfy gate",
    }
    step = next(step for step in report["next_steps"] if step["check"] == "phase6_embed_cost")
    assert "budget.checks.max_full_hours.ok=None, expected true" in step["details"]
    assert "budget.checks.max_full_usd.ok=None, expected true" in step["details"]
    assert "estimate.full_notes=None, expected 167000" in step["details"]
    assert "estimate.estimated_full_usd must be positive" in step["details"]


def test_plan_audit_requires_embed_model_and_dimension(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/embed-budget.json",
        {
            "runtime": gpu_runtime(),
            "sample": {"notes": 2, "vector_dim": 384, "model": "other"},
            "budget": {
                "accepted": True,
                "checks": {
                    "max_full_hours": {"ok": True},
                    "max_full_usd": {"ok": True},
                },
            },
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase6_embed_cost"] == {
        "ok": False,
        "optional": False,
        "reports": ["embed_budget"],
        "reason": "report content does not satisfy gate",
    }


def test_plan_audit_surfaces_embed_budget_error_report(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/embed-budget.json",
        {
            "status": "failed",
            "error": "embedding model failed to load",
            "runtime": gpu_runtime(),
            "production_path": embed_production_path(),
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase6_embed_cost"]["ok"] is False
    step = next(step for step in report["next_steps"] if step["check"] == "phase6_embed_cost")
    assert "error='embedding model failed to load'" in step["details"]


def test_plan_audit_requires_embed_budget_production_gpu_path(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/embed-budget.json",
        {
            "runtime": gpu_runtime(),
            "sample": {
                "notes": 12,
                "vector_dim": 768,
                "model": "Snowflake/snowflake-arctic-embed-m-v1.5",
            },
            "production_path": embed_production_path(
                pipeline_cr="local-indexer",
                module="indexer",
                compute_class="cpu",
                image="186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-embedder-latest",
                allow_full_cpu_index=True,
            ),
            "estimate": {
                "full_notes": 167000,
                "estimated_full_seconds": 1000.0,
                "estimated_full_hours": 0.28,
                "gpu_hourly_usd": 2.5,
                "estimated_full_usd": 0.7,
            },
            "budget": {
                "accepted": True,
                "checks": {
                    "max_full_hours": {"ok": True},
                    "max_full_usd": {"ok": True},
                },
            },
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase6_embed_cost"] == {
        "ok": False,
        "optional": False,
        "reports": ["embed_budget"],
        "reason": "report content does not satisfy gate",
    }
    step = next(step for step in report["next_steps"] if step["check"] == "phase6_embed_cost")
    assert "production_path.pipeline_cr='local-indexer', expected 'chart-embed-gpu'" in step["details"]
    assert "production_path.module='indexer', expected 'indexer.embed'" in step["details"]
    assert "production_path.compute_class='cpu', expected 'gpu'" in step["details"]
    assert any("chart-embedder-latest" in detail for detail in step["details"])
    assert "production_path.allow_full_cpu_index must be false" in step["details"]


def test_plan_audit_requires_gpu_measured_cost_reports(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/embed-budget.json",
        {
            "runtime": {"accelerator": "cpu"},
            "sample": {"notes": 2},
            "budget": {
                "accepted": True,
                "checks": {
                    "max_full_hours": {"ok": True},
                    "max_full_usd": {"ok": True},
                },
            },
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase6_embed_cost"] == {
        "ok": False,
        "optional": False,
        "reports": ["embed_budget"],
        "reason": "report content does not satisfy gate",
    }


def test_plan_audit_accepts_layer_cost_report_without_local_gpu_device(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/embed-budget.json",
        {
            "runtime": {"accelerator": "gpu"},
            "sample": {
                "notes": 12,
                "vector_dim": 768,
                "model": "Snowflake/snowflake-arctic-embed-m-v1.5",
            },
            "production_path": embed_production_path(),
            "estimate": {
                "full_notes": 167000,
                "estimated_full_seconds": 1000.0,
                "estimated_full_hours": 0.28,
                "gpu_hourly_usd": 2.5,
                "estimated_full_usd": 0.7,
            },
            "budget": {
                "accepted": True,
                "checks": {
                    "max_full_hours": {"ok": True},
                    "max_full_usd": {"ok": True},
                },
            },
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase6_embed_cost"]["ok"] is True


def test_plan_audit_accepts_layer_cost_snapshot_for_embed_gate(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/embed-budget.json",
        {
            **layer_cost_report(),
            "sample": {
                "notes": 12,
                "vector_dim": 768,
                "model": "Snowflake/snowflake-arctic-embed-m-v1.5",
            },
            "production_path": embed_production_path(),
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase6_embed_cost"]["ok"] is True


def test_plan_audit_accepts_layer_cost_snapshot_for_classifier_gate(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/classify-events-budget.json",
        {
            **layer_cost_report(),
            "sample": {"notes": 6, "med_discontinuation": 1},
            "examples": [
                {
                    "note_preview": "metformin stopped",
                    "events": [{"type": "medication_discontinued", "reason": "adverse_effect"}],
                    "labels": {"has_med_discontinuation": True},
                    "discontinuation_reason": "adverse_effect",
                }
            ],
            "writeback": {
                "mode": "tpuf.patch_columns",
                "primary_output": "events",
                "model_passes_per_note": 1,
                "patched_fields": [
                    "events",
                    "has_med_discontinuation",
                    "has_adverse_event",
                    "diagnosis_category",
                    "specialty",
                    "discontinuation_reason",
                ],
                "settles_multi_write": True,
            },
            "signal": {
                "accepted": True,
                "checks": {
                    "min_med_discontinuations": {"ok": True},
                    "min_review_examples": {"ok": True},
                },
            },
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase4_classify_cost_signal"]["ok"] is True


def test_plan_audit_surfaces_classifier_budget_error_report(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/classify-events-budget.json",
        {
            "status": "failed",
            "error": "vLLM is required for the Gemma classifier",
            "runtime": gpu_runtime(),
            "writeback": {
                "mode": "tpuf.patch_columns",
                "primary_output": "events",
                "model_passes_per_note": 1,
                "patched_fields": [
                    "events",
                    "has_med_discontinuation",
                    "has_adverse_event",
                    "diagnosis_category",
                    "specialty",
                    "discontinuation_reason",
                ],
                "settles_multi_write": True,
            },
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase4_classify_cost_signal"]["ok"] is False
    step = next(step for step in report["next_steps"] if step["check"] == "phase4_classify_cost_signal")
    assert "error='vLLM is required for the Gemma classifier'" in step["details"]


def test_plan_audit_requires_classifier_discontinuation_examples(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/classify-events-budget.json",
        {
            "runtime": gpu_runtime(),
            "sample": {"notes": 6, "med_discontinuation": 1},
            "examples": [],
            "budget": {
                "accepted": True,
                "checks": {
                    "max_full_hours": {"ok": True},
                    "max_full_usd": {"ok": True},
                },
            },
            "signal": {
                "accepted": True,
                "checks": {
                    "min_med_discontinuations": {"ok": True},
                    "min_review_examples": {"ok": True},
                },
            },
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase4_classify_cost_signal"] == {
        "ok": False,
        "optional": False,
        "reports": ["classify_budget"],
        "reason": "report content does not satisfy gate",
    }


def test_plan_audit_requires_classifier_discontinuation_reason_examples(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/classify-events-budget.json",
        {
            "runtime": gpu_runtime(),
            "sample": {"notes": 6, "med_discontinuation": 1},
            "examples": [
                {
                    "note_preview": "metformin stopped",
                    "events": [{"type": "medication_discontinued"}],
                    "labels": {"has_med_discontinuation": True},
                }
            ],
            "estimate": {
                "full_notes": 167000,
                "estimated_full_seconds": 2000.0,
                "estimated_full_hours": 0.56,
                "gpu_hourly_usd": 2.5,
                "estimated_full_usd": 1.4,
            },
            "budget": {
                "accepted": True,
                "checks": {
                    "max_full_hours": {"ok": True},
                    "max_full_usd": {"ok": True},
                },
            },
            "signal": {
                "accepted": True,
                "checks": {
                    "min_med_discontinuations": {"ok": True},
                    "min_review_examples": {"ok": True},
                },
            },
            "writeback": {
                "mode": "tpuf.patch_columns",
                "primary_output": "events",
                "model_passes_per_note": 1,
                "patched_fields": [
                    "events",
                    "has_med_discontinuation",
                    "has_adverse_event",
                    "diagnosis_category",
                    "specialty",
                    "discontinuation_reason",
                ],
                "settles_multi_write": True,
            },
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase4_classify_cost_signal"] == {
        "ok": False,
        "optional": False,
        "reports": ["classify_budget"],
        "reason": "report content does not satisfy gate",
    }
    step = next(step for step in report["next_steps"] if step["check"] == "phase4_classify_cost_signal")
    assert (
        "examples must include a medication discontinuation review example with discontinuation_reason"
        in step["details"]
    )


def test_plan_audit_requires_classifier_writeback_contract(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/classify-events-budget.json",
        {
            "runtime": gpu_runtime(),
            "sample": {"notes": 6, "med_discontinuation": 1},
            "examples": [
                {
                    "note_preview": "metformin stopped",
                    "events": [{"type": "medication_discontinued", "reason": "adverse_effect"}],
                    "labels": {"has_med_discontinuation": True},
                    "discontinuation_reason": "adverse_effect",
                }
            ],
            "estimate": {
                "full_notes": 167000,
                "estimated_full_seconds": 2000.0,
                "estimated_full_hours": 0.56,
                "gpu_hourly_usd": 2.5,
                "estimated_full_usd": 1.4,
            },
            "budget": {
                "accepted": True,
                "checks": {
                    "max_full_hours": {"ok": True},
                    "max_full_usd": {"ok": True},
                },
            },
            "signal": {
                "accepted": True,
                "checks": {
                    "min_med_discontinuations": {"ok": True},
                    "min_review_examples": {"ok": True},
                },
            },
            "writeback": {
                "mode": "single-output-only",
                "primary_output": "events",
                "model_passes_per_note": 2,
                "patched_fields": ["events"],
                "settles_multi_write": False,
            },
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase4_classify_cost_signal"] == {
        "ok": False,
        "optional": False,
        "reports": ["classify_budget"],
        "reason": "report content does not satisfy gate",
    }
    step = next(step for step in report["next_steps"] if step["check"] == "phase4_classify_cost_signal")
    assert "writeback.mode='single-output-only', expected 'tpuf.patch_columns'" in step["details"]
    assert "writeback.model_passes_per_note=2, expected 1" in step["details"]
    assert "writeback.settles_multi_write must be true" in step["details"]
    assert any("writeback.patched_fields missing: diagnosis_category" in detail for detail in step["details"])


def test_plan_audit_rejects_dry_run_slice_index_report(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/slice-index-report.json",
        {
            "status": "completed",
            "dry_run": True,
            "indexed": 2000,
            "provenance": slice_provenance(
                dataset_repo="other",
                dataset_revision="bad",
                dataset_split="dev",
                embed_model="other-model",
                embed_dim=384,
            ),
            "schema": {
                "vector_dim": 384,
                "rows_with_age_band": 0,
                "rows_with_gender": 0,
                "rows_with_similar_patient_ids": 0,
            },
            "facet_snapshots_materialized": False,
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase1_slice_index"] == {
        "ok": False,
        "optional": False,
        "reports": ["slice_index"],
        "reason": "report content does not satisfy gate",
    }
    step = next(step for step in report["next_steps"] if step["check"] == "phase1_slice_index")
    assert "dry_run=True, expected false" in step["details"]
    assert "provenance.dataset_repo='other', expected 'zhengyun21/PMC-Patients'" in step["details"]
    assert "provenance.dataset_revision='bad', expected '28d8836518f86d4f1e6358ea8ec09977023e5766'" in step["details"]
    assert "provenance.dataset_split='dev', expected 'train'" in step["details"]
    assert "provenance.embed_model='other-model', expected 'Snowflake/snowflake-arctic-embed-m-v1.5'" in step["details"]
    assert "provenance.embed_dim=384, expected 768" in step["details"]
    assert "schema.vector_dim=384, expected 768" in step["details"]
    assert "schema.rows_with_age_band must be positive" in step["details"]
    assert "schema.rows_with_gender must be positive" in step["details"]
    assert "schema.rows_with_similar_patient_ids must be positive" in step["details"]
    assert "facet_snapshots_materialized must be true" in step["details"]


def test_plan_audit_requires_two_thousand_note_slice_index_report(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/slice-index-report.json",
        {
            "status": "completed",
            "limit": 500,
            "dry_run": False,
            "indexed": 500,
            "provenance": slice_provenance(),
            "schema": {
                "vector_dim": 768,
                "rows_with_age_band": 450,
                "rows_with_gender": 430,
                "rows_with_similar_patient_ids": 300,
            },
            "facet_snapshots_materialized": True,
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase1_slice_index"] == {
        "ok": False,
        "optional": False,
        "reports": ["slice_index"],
        "reason": "report content does not satisfy gate",
    }
    step = next(step for step in report["next_steps"] if step["check"] == "phase1_slice_index")
    assert "indexed=500, expected at least 2000" in step["details"]


def test_plan_audit_requires_live_smoke_routes_facets_and_similar(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/live-smoke-base-report.json",
        {
            "ok": True,
            "status": "completed",
            "index_shape": {"vector_dim": 768, "rows": 20},
            "routes": [{"route": "semantic", "rows": 5}],
            "similar": {"neighbors": 3},
            "facets": {"age_band": {"values": 4, "sha": "slice-sha"}},
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase2_3_live_smoke"] == {
        "ok": False,
        "optional": False,
        "reports": ["live_smoke_base"],
        "reason": "report content does not satisfy gate",
    }
    step = next(step for step in report["next_steps"] if step["check"] == "phase2_3_live_smoke")
    assert "requirements.event_facets must be false" in step["details"]
    assert "missing route checks: fused, hybrid_text" in step["details"]
    assert "facet 'gender' has no values" in step["details"]


def test_plan_audit_base_live_smoke_does_not_require_event_facet(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/live-smoke-base-report.json",
        {
            "ok": True,
            "status": "completed",
            "requirements": {"facets": True, "event_facets": False},
            "index_shape": {"vector_dim": 768, "rows": 20},
            "routes": [
                live_route("hybrid_text"),
                live_route("fused"),
                live_route("semantic"),
            ],
            "similar": {"neighbors": 3},
            "facets": {
                "age_band": {"values": 4, "sha": "slice-sha"},
                "gender": {"values": 2, "sha": "slice-sha"},
            },
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase2_3_live_smoke"]["ok"] is True


def test_plan_audit_requires_phase4_event_facet_smoke_after_classifier(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/facet-refresh-report.json",
        {
            "status": "completed",
            "fields": ["age_band", "gender", "events"],
            "snapshots": {
                "age_band": {"values": 4, "sha": "facet-sha"},
                "gender": {"values": 2, "sha": "facet-sha"},
                "events": {"values": 0, "sha": "facet-sha"},
            },
        },
    )
    write(
        tmp_path / "eval/out/live-smoke-report.json",
        {
            "ok": False,
            "status": "failed",
            "requirements": {"facets": True, "event_facets": True},
            "index_shape": {"vector_dim": 768, "rows": 20},
            "routes": [live_route("hybrid_text"), live_route("fused"), live_route("semantic")],
            "similar": {"neighbors": 3},
            "facets": {
                "age_band": {"values": 4, "sha": "slice-sha"},
                "gender": {"values": 2, "sha": "slice-sha"},
                "events": {"values": 0, "sha": "slice-sha"},
            },
            "error": "materialized facet snapshot for 'events' has no values",
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase4_event_facet_smoke"] == {
        "ok": False,
        "optional": False,
        "reports": ["facet_refresh", "live_smoke"],
        "reason": "report content does not satisfy gate",
    }
    step = next(step for step in report["next_steps"] if step["check"] == "phase4_event_facet_smoke")
    assert step["blocked_by"] == ["phase4_classify_cost_signal"]
    assert step["requires"] == ["gateway_key", "classifier_extra"]
    assert step["command"] == "scripts/phase4_event_smoke.sh"
    assert "snapshot 'events' has no values" in step["details"]
    assert "facet 'events' has no values" in step["details"]


def test_plan_audit_accepts_phase4_event_facet_smoke(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/facet-refresh-report.json",
        {
            "status": "completed",
            "fields": ["age_band", "gender", "events"],
            "snapshots": {
                "age_band": {"values": 4, "sha": "facet-sha"},
                "gender": {"values": 2, "sha": "facet-sha"},
                "events": {"values": 3, "sha": "facet-sha"},
            },
        },
    )
    write(
        tmp_path / "eval/out/live-smoke-report.json",
        {
            "ok": True,
            "status": "completed",
            "requirements": {"facets": True, "event_facets": True},
            "index_shape": {"vector_dim": 768, "rows": 20},
            "routes": [live_route("hybrid_text"), live_route("fused"), live_route("semantic")],
            "similar": {"neighbors": 3},
            "facets": {
                "age_band": {"values": 4, "sha": "slice-sha"},
                "gender": {"values": 2, "sha": "slice-sha"},
                "events": {"values": 3, "sha": "slice-sha"},
            },
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase4_event_facet_smoke"]["ok"] is True


def test_plan_audit_requires_live_smoke_gateway_echoes(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/live-smoke-base-report.json",
        {
            "ok": True,
            "status": "completed",
            "requirements": {"facets": True, "event_facets": False},
            "index_shape": {"vector_dim": 768, "rows": 20},
            "routes": [
                {"route": "hybrid_text", "routing": {"route": "hybrid_text"}, "rows": 5},
                {"route": "fused", "routing": {"route": "semantic"}, "hybrid": {"leg_breakdown": []}, "rows": 5},
                {"route": "semantic", "hybrid": {"leg_breakdown": []}, "rows": 5},
            ],
            "similar": {"neighbors": 3},
            "facets": {
                "age_band": {"values": 4, "sha": "slice-sha"},
                "gender": {"values": 2, "sha": "slice-sha"},
                "events": {"values": 3, "sha": "slice-sha"},
            },
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase2_3_live_smoke"] == {
        "ok": False,
        "optional": False,
        "reports": ["live_smoke_base"],
        "reason": "report content does not satisfy gate",
    }
    step = next(step for step in report["next_steps"] if step["check"] == "phase2_3_live_smoke")
    assert "route checks missing gateway routing/fused-hybrid echo: fused, semantic" in step["details"]


def test_plan_audit_requires_strict_event_facet_smoke_mode(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/live-smoke-base-report.json",
        {
            "ok": True,
            "status": "completed",
            "requirements": {"facets": True, "event_facets": True},
            "index_shape": {"vector_dim": 768, "rows": 20},
            "routes": [
                live_route("hybrid_text"),
                live_route("fused"),
                live_route("semantic"),
            ],
            "similar": {"neighbors": 3},
            "facets": {
                "age_band": {"values": 4, "sha": "slice-sha"},
                "gender": {"values": 2, "sha": "slice-sha"},
                "events": {"values": 3, "sha": "slice-sha"},
            },
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase2_3_live_smoke"] == {
        "ok": False,
        "optional": False,
        "reports": ["live_smoke_base"],
        "reason": "report content does not satisfy gate",
    }
    step = next(step for step in report["next_steps"] if step["check"] == "phase2_3_live_smoke")
    assert step["details"] == ["requirements.event_facets must be false"]


def test_plan_audit_requires_facet_refresh_base_fields(tmp_path) -> None:
    env = env_for(tmp_path)
    write(tmp_path / "eval/out/facet-refresh-report.json", {"status": "completed", "fields": ["events"]})

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase3_facet_refresh"] == {
        "ok": False,
        "optional": False,
        "reports": ["facet_refresh"],
        "reason": "report content does not satisfy gate",
    }
    step = next(step for step in report["next_steps"] if step["check"] == "phase3_facet_refresh")
    assert "missing refreshed fields: age_band, gender" in step["details"]


def test_plan_audit_requires_base_facet_snapshot_values(tmp_path) -> None:
    env = env_for(tmp_path)
    write(tmp_path / "eval/out/facet-refresh-report.json", {"status": "completed", "fields": ["age_band", "gender"]})

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase3_facet_refresh"] == {
        "ok": False,
        "optional": False,
        "reports": ["facet_refresh"],
        "reason": "report content does not satisfy gate",
    }
    step = next(step for step in report["next_steps"] if step["check"] == "phase3_facet_refresh")
    assert "snapshot 'age_band' has no values" in step["details"]
    assert "snapshot 'gender' missing sha" in step["details"]


def test_plan_audit_requires_full_recds_gate_settings(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/recds-report.json",
        {
            "strategies": ["semantic", "bm25", "fused"],
            "limit": 25,
            "top_k": 100,
            "summaries": [
                {"strategy": "semantic", "queries": {"failed": 0}},
                {"strategy": "bm25", "queries": {"failed": 0}},
                {"strategy": "fused", "queries": {"failed": 0}},
            ],
            "gates": {"no_failures": {"required": True, "accepted": True}},
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase5_recds"] == {
        "ok": False,
        "optional": False,
        "reports": ["recds"],
        "reason": "report content does not satisfy gate",
    }
    step = next(step for step in report["next_steps"] if step["check"] == "phase5_recds")
    assert "task=None, expected 'ppr'" in step["details"]
    assert "missing strategies: auto" in step["details"]
    assert "limit=25, expected at least 500" in step["details"]


def test_plan_audit_requires_recds_ppr_not_bimodal_report(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/recds-report.json",
        {
            "task": None,
            "split": None,
            "beir_dir": "eval/out/bimodal",
            "strategies": ["auto", "semantic", "bm25", "fused"],
            "limit": 500,
            "top_k": 1000,
            "summaries": [
                {"strategy": "auto", "queries": {"failed": 0}},
                {"strategy": "semantic", "queries": {"failed": 0}},
                {"strategy": "bm25", "queries": {"failed": 0}},
                {"strategy": "fused", "queries": {"failed": 0}},
            ],
            "gates": {
                "no_failures": {"required": True, "accepted": True},
                "fused_dominates": {"accepted": True, "checks": [{"ok": True}], "query_failures": []},
            },
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase5_recds"] == {
        "ok": False,
        "optional": False,
        "reports": ["recds"],
        "reason": "report content does not satisfy gate",
    }


def test_plan_audit_requires_recds_pinned_provenance(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/recds-report.json",
        {
            "task": "ppr",
            "split": "dev",
            "beir_dir": None,
            "provenance": recds_provenance(recds_revision="bad", embed_dim=384),
            "strategies": ["auto", "semantic", "bm25", "fused"],
            "limit": 500,
            "top_k": 1000,
            "summaries": [
                {"strategy": "auto", "queries": {"failed": 0}},
                {"strategy": "semantic", "queries": {"failed": 0}},
                {"strategy": "bm25", "queries": {"failed": 0}},
                {
                    "strategy": "fused",
                    "queries": {"failed": 0},
                    "published": {
                        "baseline": "rrf",
                        "baseline_metrics": {"RR@10": 0.2776, "nDCG@10": 0.2412, "R@1000": 0.8514},
                        "delta": {"RR@10": 0.01, "nDCG@10": 0.01, "R@1000": 0.01},
                        "meets_or_beats": True,
                    },
                },
            ],
            "gates": {
                "no_failures": {"required": True, "accepted": True},
                "fused_dominates": {
                    "accepted": True,
                    "checks": fused_dominance_checks(),
                    "query_failures": [],
                },
            },
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase5_recds"] == {
        "ok": False,
        "optional": False,
        "reports": ["recds"],
        "reason": "report content does not satisfy gate",
    }
    step = next(step for step in report["next_steps"] if step["check"] == "phase5_recds")
    assert "provenance.recds_revision='bad', expected 'a27717bb27679cf0860305997685547ca01b3dd1'" in step[
        "details"
    ]
    assert "provenance.embed_dim=384, expected 768" in step["details"]


def test_plan_audit_requires_recds_fused_dominance_checks(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/recds-report.json",
        {
            "task": "ppr",
            "split": "dev",
            "beir_dir": None,
            "provenance": recds_provenance(),
            "strategies": ["auto", "semantic", "bm25", "fused"],
            "limit": 500,
            "top_k": 1000,
            "summaries": [
                {"strategy": "auto", "queries": {"failed": 0}},
                {"strategy": "semantic", "queries": {"failed": 0}},
                {"strategy": "bm25", "queries": {"failed": 0}},
                {"strategy": "fused", "queries": {"failed": 0}},
            ],
            "gates": {
                "no_failures": {"required": True, "accepted": True},
                "fused_dominates": {"accepted": True, "checks": [], "query_failures": []},
            },
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase5_recds"] == {
        "ok": False,
        "optional": False,
        "reports": ["recds"],
        "reason": "report content does not satisfy gate",
    }


def test_plan_audit_requires_complete_recds_fused_dominance_matrix(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/recds-report.json",
        {
            "task": "ppr",
            "split": "dev",
            "beir_dir": None,
            "provenance": recds_provenance(),
            "strategies": ["auto", "semantic", "bm25", "fused"],
            "limit": 500,
            "top_k": 1000,
            "summaries": [
                {"strategy": "auto", "queries": {"failed": 0}},
                {"strategy": "semantic", "queries": {"failed": 0}},
                {"strategy": "bm25", "queries": {"failed": 0}},
                {
                    "strategy": "fused",
                    "queries": {"failed": 0},
                    "published": {
                        "baseline": "rrf",
                        "baseline_metrics": {"RR@10": 0.2776, "nDCG@10": 0.2412, "R@1000": 0.8514},
                        "delta": {"RR@10": 0.01, "nDCG@10": 0.01, "R@1000": 0.01},
                        "meets_or_beats": True,
                    },
                },
            ],
            "gates": {
                "no_failures": {"required": True, "accepted": True},
                "fused_dominates": {
                    "accepted": True,
                    "checks": [
                        {"metric": "RR@10", "baseline": "bm25", "fused": 0.31, "baseline_value": 0.3, "ok": True}
                    ],
                    "query_failures": [],
                },
            },
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase5_recds"] == {
        "ok": False,
        "optional": False,
        "reports": ["recds"],
        "reason": "report content does not satisfy gate",
    }
    step = next(step for step in report["next_steps"] if step["check"] == "phase5_recds")
    assert any("gates.fused_dominates.checks missing headline comparisons" in detail for detail in step["details"])


def test_plan_audit_requires_recds_published_baseline(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/recds-report.json",
        {
            "task": "ppr",
            "split": "dev",
            "beir_dir": None,
            "provenance": recds_provenance(),
            "strategies": ["auto", "semantic", "bm25", "fused"],
            "limit": 500,
            "top_k": 1000,
            "summaries": [
                {"strategy": "auto", "queries": {"failed": 0}},
                {"strategy": "semantic", "queries": {"failed": 0}},
                {"strategy": "bm25", "queries": {"failed": 0}},
                {"strategy": "fused", "queries": {"failed": 0}},
            ],
            "gates": {
                "no_failures": {"required": True, "accepted": True},
                "fused_dominates": {"accepted": True, "checks": [{"ok": True}], "query_failures": []},
            },
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase5_recds"] == {
        "ok": False,
        "optional": False,
        "reports": ["recds"],
        "reason": "report content does not satisfy gate",
    }
    step = next(step for step in report["next_steps"] if step["check"] == "phase5_recds")
    assert "fused summary missing published baseline comparison" in step["details"]


def test_plan_audit_rejects_empty_phase6_status(tmp_path) -> None:
    env = env_for(tmp_path)
    write(tmp_path / "eval/out/phase6-status-report.json", {"pipeline": {}, "udf": {}})

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase6_runtime_status"] == {
        "ok": False,
        "optional": False,
        "reports": ["phase6_status"],
        "reason": "report content does not satisfy gate",
    }
    step = next(step for step in report["next_steps"] if step["check"] == "phase6_runtime_status")
    assert "namespace missing" in step["details"]
    assert "targets.pipeline_id missing" in step["details"]


def test_plan_audit_requires_phase6_status_targets_and_facets(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/phase6-status-report.json",
        {
            "namespace": "chart-notes",
            "targets": phase6_targets(),
            "pipeline": {"pipeline_id": "chart-notes"},
            "udf": {"udf_id": "chart-classify-events"},
            "facets": {
                "age_band": {},
                "gender": {},
            },
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase6_runtime_status"] == {
        "ok": False,
        "optional": False,
        "reports": ["phase6_status"],
        "reason": "report content does not satisfy gate",
    }


def test_plan_audit_reports_kubernetes_function_when_gateway_udf_is_missing(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/phase6-status-report.json",
        {
            "namespace": "chart-notes",
            "targets": phase6_targets(),
            "pipeline": {"pipeline_id": "chart-notes"},
            "udf": {
                "udf_id": "chart-classify-events",
                "error": {"status_code": 404, "message": "UDF 'chart-classify-events' not found"},
            },
            "facets": full_facet_status(),
            "kubernetes": {
                "function_status": {
                    "chart-classify-events": {
                        "paused": True,
                        "conditions": [
                            {
                                "type": "Ready",
                                "reason": "Paused",
                                "message": "Function spec.paused=true; deployment scaled to zero",
                            }
                        ],
                    }
                },
                "scaled_objects": {
                    "chart-embed-gpu-worker": {
                        "conditions": [
                            {
                                "type": "Ready",
                                "status": "False",
                                "reason": "ScaledObjectCheckFailed",
                                "message": "bearer token=<empty> is required when bearer auth is enabled",
                            }
                        ]
                    }
                },
                "trigger_authentications": {
                    "chart-embed-gpu-worker": {
                        "secret_target_refs": [
                            {
                                "parameter": "bearerToken",
                                "name": "layer",
                                "key": "turbopuffer-api-key",
                                "status": {
                                    "secret_exists": True,
                                    "key_exists": True,
                                    "value_present": False,
                                },
                            }
                        ]
                    }
                }
            },
        },
    )

    report = plan_audit.summarize(env=env)

    step = next(step for step in report["next_steps"] if step["check"] == "phase6_runtime_status")
    assert "udf error: {'status_code': 404, 'message': \"UDF 'chart-classify-events' not found\"}" in step["details"]
    assert (
        "kubernetes function chart-classify-events exists, paused=True, ready_reason=Paused, "
        "ready_message='Function spec.paused=true; deployment scaled to zero'"
    ) in step["details"]
    assert (
        "scaledobject chart-embed-gpu-worker not ready: ScaledObjectCheckFailed: "
        "bearer token=<empty> is required when bearer auth is enabled; "
        "bearerTokenRef=layer/turbopuffer-api-key; "
        "bearerTokenStatus=secret_exists=True,key_exists=True,value_present=False"
    ) in step["details"]


def test_plan_audit_requires_phase6_status_full_facet_snapshots(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/phase6-status-report.json",
        {
            "namespace": "chart-notes",
            "targets": phase6_targets(),
            "pipeline": {"pipeline_id": "chart-notes"},
            "udf": {"udf_id": "chart-classify-events"},
            "facets": {
                "specialty": {"values": 4, "row_count": 167000, "sha": "full-snapshot"},
                "age_band": {"values": 6, "row_count": 167000, "sha": "full-snapshot"},
                "diagnosis_category": {"values": 5, "row_count": 167000, "sha": "full-snapshot"},
                "gender": {"values": 2, "row_count": 2000, "sha": "slice-snapshot"},
                "events": {"values": 4, "row_count": 167000, "sha": None},
            },
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase6_runtime_status"] == {
        "ok": False,
        "optional": False,
        "reports": ["phase6_status"],
        "reason": "report content does not satisfy gate",
    }


def test_plan_audit_requires_phase6_status_gpu_runtime_targets(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/phase6-status-report.json",
        {
            "namespace": "chart-notes",
            "targets": phase6_targets(
                embed_pipeline_cr="chart-embed-cpu",
                embed_compute_class="cpu",
                embed_image="186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-embedder-latest",
                classifier_compute_class="cpu",
                classifier_image="186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-classifier-latest",
            ),
            "pipeline": {"pipeline_id": "chart-notes"},
            "udf": {"udf_id": "chart-classify-events"},
            "facets": full_facet_status(),
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase6_runtime_status"] == {
        "ok": False,
        "optional": False,
        "reports": ["phase6_status"],
        "reason": "report content does not satisfy gate",
    }
    step = next(step for step in report["next_steps"] if step["check"] == "phase6_runtime_status")
    assert "targets.embed_pipeline_cr='chart-embed-cpu', expected 'chart-embed-gpu'" in step["details"]
    assert "targets.embed_compute_class='cpu', expected 'gpu'" in step["details"]
    assert "targets.classifier_compute_class='cpu', expected 'gpu'" in step["details"]
    assert any("chart-embedder-latest" in detail for detail in step["details"])
    assert any("chart-classifier-latest" in detail for detail in step["details"])


def test_plan_audit_requires_phase6_status_cost_baselines(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/phase6-status-report.json",
        {
            "namespace": "chart-notes",
            "targets": phase6_targets(),
            "cost_baselines": phase6_cost_baselines(
                embed_report="eval/out/other-embed-budget.json",
                classifier_report=env["CHART_PHASE4_CLASSIFY_REPORT"],
                embed_accepted=False,
            ),
            "pipeline": {"pipeline_id": "chart-notes"},
            "udf": {"udf_id": "chart-classify-events"},
            "facets": full_facet_status(),
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase6_runtime_status"] == {
        "ok": False,
        "optional": False,
        "reports": ["phase6_status"],
        "reason": "report content does not satisfy gate",
    }
    step = next(step for step in report["next_steps"] if step["check"] == "phase6_runtime_status")
    assert "cost_baselines.embed.accepted must be true" in step["details"]
    assert f"cost_baselines.embed.report='eval/out/other-embed-budget.json', expected {env['CHART_PHASE6_EMBED_BUDGET_REPORT']!r}" in step[
        "details"
    ]


def test_plan_audit_rejects_phase6_status_report_with_error(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/phase6-status-report.json",
        {
            "namespace": "chart-notes",
            "targets": phase6_targets(),
            "cost_baselines": phase6_cost_baselines(
                embed_report=env["CHART_PHASE6_EMBED_BUDGET_REPORT"],
                classifier_report=env["CHART_PHASE4_CLASSIFY_REPORT"],
            ),
            "pipeline": {"pipeline_id": "chart-notes"},
            "udf": {"udf_id": "chart-classify-events"},
            "facets": full_facet_status(),
            "error": "stale failure",
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase6_runtime_status"]["ok"] is False
    step = next(step for step in report["next_steps"] if step["check"] == "phase6_runtime_status")
    assert "error='stale failure'" in step["details"]


def test_plan_audit_requires_pushed_gpu_images(tmp_path) -> None:
    env = env_for(tmp_path)
    write(tmp_path / "eval/out/gpu-build-report.json", {"mode": "build", "status": "completed"})

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase6_gpu_images"] == {
        "ok": False,
        "optional": False,
        "reports": ["gpu_build"],
        "reason": "report content does not satisfy gate",
    }
    step = next(step for step in report["next_steps"] if step["check"] == "phase6_gpu_images")
    assert "mode='build', expected 'push'" in step["details"]
    assert "embed_command missing --push" in step["details"]


def test_plan_audit_surfaces_gpu_build_error_details(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/gpu-build-report.json",
        {
            "mode": "push",
            "status": "failed",
            "embed_image": "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-embedder-plan-20260624-dedupe2",
            "classifier_image": "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-classifier-plan-20260624",
            "embed_command": [
                "docker",
                "buildx",
                "build",
                "--push",
                "-t",
                "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-embedder-plan-20260624-dedupe2",
                ".",
            ],
            "classifier_command": [
                "docker",
                "buildx",
                "build",
                "--push",
                "-t",
                "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-classifier-plan-20260624",
                ".",
            ],
            "error": "docker daemon is not reachable",
        },
    )

    report = plan_audit.summarize(env=env)

    step = next(step for step in report["next_steps"] if step["check"] == "phase6_gpu_images")
    assert "status='failed', expected 'completed'" in step["details"]
    assert "error='docker daemon is not reachable'" in step["details"]


def test_plan_audit_rejects_completed_gpu_build_report_with_error(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/gpu-build-report.json",
        {
            "mode": "push",
            "status": "completed",
            "embed_image": "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-embedder-plan-20260624-dedupe2",
            "classifier_image": "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-classifier-plan-20260624",
            "embed_command": [
                "docker",
                "buildx",
                "build",
                "--push",
                "-t",
                "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-embedder-plan-20260624-dedupe2",
                ".",
            ],
            "classifier_command": [
                "docker",
                "buildx",
                "build",
                "--push",
                "-t",
                "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-classifier-plan-20260624",
                ".",
            ],
            "error": "stale failure",
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase6_gpu_images"]["ok"] is False
    step = next(step for step in report["next_steps"] if step["check"] == "phase6_gpu_images")
    assert "error='stale failure'" in step["details"]


def test_plan_audit_requires_pushed_non_latest_manifest_image_commands(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/gpu-build-report.json",
        {
            "mode": "push",
            "status": "completed",
            "embed_image": "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-embedder:latest",
            "classifier_image": "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-classifier-plan-20260624",
            "embed_command": [
                "docker",
                "buildx",
                "build",
                "--push",
                "-t",
                "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-embedder:latest",
                ".",
            ],
            "classifier_command": [
                "docker",
                "buildx",
                "build",
                "-t",
                "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-classifier-plan-20260624",
                ".",
            ],
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase6_gpu_images"] == {
        "ok": False,
        "optional": False,
        "reports": ["gpu_build"],
        "reason": "report content does not satisfy gate",
    }


def test_plan_audit_accepts_pushed_non_latest_image_overrides(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/gpu-build-report.json",
        {
            "mode": "push",
            "status": "completed",
            "embed_image": "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-embedder:other-tag",
            "classifier_image": "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-classifier-plan-20260624",
            "embed_command": [
                "docker",
                "buildx",
                "build",
                "--push",
                "-t",
                "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-embedder:other-tag",
                ".",
            ],
            "classifier_command": [
                "docker",
                "buildx",
                "build",
                "--push",
                "-t",
                "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-classifier-plan-20260624",
                ".",
            ],
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase6_gpu_images"] == {"ok": True, "optional": False, "reports": ["gpu_build"]}


def test_plan_audit_accepts_pushed_embed_only_gpu_build_report(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/gpu-build-report.json",
        {
            "mode": "push",
            "status": "completed",
            "targets": ["embed"],
            "embed_image": "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-embedder:dedupe1",
            "classifier_image": "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-classifier-plan-20260624",
            "embed_command": [
                "depot",
                "build",
                "--push",
                "-t",
                "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-embedder:dedupe1",
                ".",
            ],
            "classifier_command": [
                "depot",
                "build",
                "-t",
                "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-classifier-plan-20260624",
                ".",
            ],
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase6_gpu_images"] == {"ok": True, "optional": False, "reports": ["gpu_build"]}


def test_plan_audit_requires_classifier_function_applied(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/deploy-apply-report.json",
        {"mode": "apply", "status": "completed", "classifier": "validated-skipped"},
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase6_deploy_apply"] == {
        "ok": False,
        "optional": False,
        "reports": ["deploy_apply"],
        "reason": "report content does not satisfy gate",
    }
    step = next(step for step in report["next_steps"] if step["check"] == "phase6_deploy_apply")
    assert "namespace missing" in step["details"]
    assert "kube_context missing" in step["details"]
    assert "kube_context_confirmed must be true" in step["details"]
    assert "classifier_cost_accepted must be true" in step["details"]
    assert "classifier='validated-skipped', expected 'applied'" in step["details"]


def test_plan_audit_surfaces_deploy_apply_error_details(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/deploy-apply-report.json",
        {
            "mode": "apply",
            "status": "failed",
            "namespace": "chart",
            "kube_context": "test-context",
            "kube_context_confirmed": True,
            "classifier_cost_accepted": True,
            "classifier": "pending",
            "classifier_report": "eval/out/classify-events-budget.json",
            "manifests": [],
            "runtime_manifests": [],
            "error": "missing required secret chart/chart-gateway; see deploy/secrets.example.yaml",
        },
    )

    report = plan_audit.summarize(env=env)

    step = next(step for step in report["next_steps"] if step["check"] == "phase6_deploy_apply")
    assert "status='failed', expected 'completed'" in step["details"]
    assert "error='missing required secret chart/chart-gateway; see deploy/secrets.example.yaml'" in step["details"]


def test_plan_audit_rejects_completed_deploy_apply_report_with_error(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/deploy-apply-report.json",
        {
            "mode": "apply",
            "status": "completed",
            "namespace": "chart",
            "kube_context": "test-context",
            "kube_context_confirmed": True,
            "classifier_cost_accepted": True,
            "classifier": "applied",
            "classifier_report": "eval/out/classify-events-budget.json",
            "manifests": [
                "deploy/namespace.yaml",
                "deploy/vectorstore.yaml",
                "deploy/warehouse.yaml",
                "deploy/pipeline.yaml",
                "deploy/pipeline-embed.yaml",
                "deploy/index.yaml",
                "deploy/functions-events.yaml",
            ],
            "runtime_manifests": [
                "deploy/vectorstore.yaml",
                "deploy/warehouse.yaml",
                "deploy/pipeline.yaml",
                "deploy/pipeline-embed.yaml",
                "deploy/index.yaml",
            ],
            "error": "stale failure",
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase6_deploy_apply"]["ok"] is False
    step = next(step for step in report["next_steps"] if step["check"] == "phase6_deploy_apply")
    assert "error='stale failure'" in step["details"]


def test_plan_audit_requires_full_deploy_apply_manifest_report(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/deploy-apply-report.json",
        {
            "mode": "apply",
            "status": "completed",
            "namespace": "chart",
            "kube_context": "test-context",
            "kube_context_confirmed": True,
            "classifier_cost_accepted": True,
            "classifier": "applied",
            "classifier_report": "eval/out/classify-events-budget.json",
            "manifests": ["deploy/namespace.yaml", "deploy/functions-events.yaml"],
            "runtime_manifests": ["deploy/vectorstore.yaml"],
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase6_deploy_apply"] == {
        "ok": False,
        "optional": False,
        "reports": ["deploy_apply"],
        "reason": "report content does not satisfy gate",
    }


def test_plan_audit_requires_deploy_classifier_report_to_match_audited_budget_report(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/deploy-apply-report.json",
        {
            "mode": "apply",
            "status": "completed",
            "namespace": "chart",
            "kube_context": "test-context",
            "kube_context_confirmed": True,
            "classifier_cost_accepted": True,
            "classifier": "applied",
            "classifier_report": "eval/out/other-classifier-report.json",
            "manifests": [
                "deploy/namespace.yaml",
                "deploy/vectorstore.yaml",
                "deploy/warehouse.yaml",
                "deploy/pipeline.yaml",
                "deploy/pipeline-embed.yaml",
                "deploy/index.yaml",
                "deploy/functions-events.yaml",
            ],
            "runtime_manifests": [
                "deploy/vectorstore.yaml",
                "deploy/warehouse.yaml",
                "deploy/pipeline.yaml",
                "deploy/pipeline-embed.yaml",
                "deploy/index.yaml",
            ],
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase6_deploy_apply"] == {
        "ok": False,
        "optional": False,
        "reports": ["deploy_apply"],
        "reason": "report content does not satisfy gate",
    }


def test_plan_audit_requires_phase6_unpause_embed_success(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/phase6-unpause-report.json",
        {
            "mode": "unpause",
            "status": "failed",
            "namespace": "chart",
            "pipeline_cr": "chart-embed-gpu",
            "expected_pipeline_id": "chart-notes",
            "budget_report": env["CHART_PHASE6_EMBED_BUDGET_REPORT"],
            "error": "missing required secret chart/chart-gateway",
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase6_unpause_embed"] == {
        "ok": False,
        "optional": False,
        "reports": ["phase6_unpause"],
        "reason": "report content does not satisfy gate",
    }
    step = next(step for step in report["next_steps"] if step["check"] == "phase6_unpause_embed")
    assert "status='failed', expected 'unpaused'" in step["details"]
    assert "error='missing required secret chart/chart-gateway'" in step["details"]
    assert "actual_pipeline_id=None, expected 'chart-notes'" in step["details"]


def test_plan_audit_requires_phase6_unpause_budget_report_to_match_audit(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/phase6-unpause-report.json",
        {
            "mode": "unpause",
            "status": "unpaused",
            "namespace": "chart",
            "pipeline_cr": "chart-embed-gpu",
            "expected_pipeline_id": "chart-notes",
            "actual_pipeline_id": "other",
            "budget_report": "eval/out/other-embed-budget.json",
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase6_unpause_embed"] == {
        "ok": False,
        "optional": False,
        "reports": ["phase6_unpause"],
        "reason": "report content does not satisfy gate",
    }
    step = next(step for step in report["next_steps"] if step["check"] == "phase6_unpause_embed")
    assert "actual_pipeline_id='other', expected 'chart-notes'" in step["details"]
    assert f"budget_report='eval/out/other-embed-budget.json', expected {env['CHART_PHASE6_EMBED_BUDGET_REPORT']!r}" in step[
        "details"
    ]


def test_plan_audit_rejects_unpaused_phase6_report_with_error(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/phase6-unpause-report.json",
        {
            "mode": "unpause",
            "status": "unpaused",
            "namespace": "chart",
            "pipeline_cr": "chart-embed-gpu",
            "expected_pipeline_id": "chart-notes",
            "actual_pipeline_id": "chart-notes",
            "budget_report": env["CHART_PHASE6_EMBED_BUDGET_REPORT"],
            "error": "stale failure",
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase6_unpause_embed"]["ok"] is False
    step = next(step for step in report["next_steps"] if step["check"] == "phase6_unpause_embed")
    assert "error='stale failure'" in step["details"]


def test_plan_audit_requires_full_phase6_gate_report_shape(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/phase6-status-report.json",
        {
            "namespace": "chart-notes",
            "targets": phase6_targets(),
            "pipeline": {"pipeline_id": "chart-notes"},
            "udf": {"udf_id": "chart-classify-events"},
            "facets": full_facet_status(),
        },
    )
    write(tmp_path / "eval/out/phase6-gate-report.json", {"gates": {"phase6_complete": True}})

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase6_gate_complete"] == {
        "ok": False,
        "optional": False,
        "reports": ["phase6_gate", "phase6_status"],
        "reason": "report content does not satisfy gate",
    }
    step = next(step for step in report["next_steps"] if step["check"] == "phase6_gate_complete")
    assert "namespace missing" in step["details"]
    assert "gates.pipeline_installed=None, expected true" in step["details"]


def test_plan_audit_rejects_phase6_gate_report_with_failures(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/phase6-status-report.json",
        {
            "namespace": "chart-notes",
            "targets": phase6_targets(),
            "pipeline": {"pipeline_id": "chart-notes"},
            "udf": {"udf_id": "chart-classify-events"},
            "facets": full_facet_status(),
        },
    )
    write(
        tmp_path / "eval/out/phase6-gate-report.json",
        {
            "namespace": "chart-notes",
            "targets": phase6_targets(),
            "gates": {
                "pipeline_installed": True,
                "udf_installed": True,
                "full_index_complete": True,
                "full_classify_complete": True,
                "base_facets_visible": True,
                "event_facets_visible": True,
                "full_facets_complete": False,
                "phase6_complete": True,
            },
            "failures": [{"gate": "full_facets_complete"}],
            "status": {
                "targets": phase6_targets(),
                "pipeline": {"pipeline_id": "chart-notes"},
                "udf": {"udf_id": "chart-classify-events"},
            },
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase6_gate_complete"] == {
        "ok": False,
        "optional": False,
        "reports": ["phase6_gate", "phase6_status"],
        "reason": "report content does not satisfy gate",
    }
    step = next(step for step in report["next_steps"] if step["check"] == "phase6_gate_complete")
    assert "failure full_facets_complete: gate incomplete" in step["details"]


def test_plan_audit_rejects_phase6_gate_report_with_error(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/phase6-status-report.json",
        {
            "namespace": "chart-notes",
            "targets": phase6_targets(),
            "pipeline": {"pipeline_id": "chart-notes"},
            "udf": {"udf_id": "chart-classify-events"},
            "facets": full_facet_status(),
        },
    )
    write(
        tmp_path / "eval/out/phase6-gate-report.json",
        {
            "namespace": "chart-notes",
            "targets": phase6_targets(),
            "gates": {
                "pipeline_installed": True,
                "udf_installed": True,
                "full_index_complete": True,
                "full_classify_complete": True,
                "base_facets_visible": True,
                "event_facets_visible": True,
                "full_facets_complete": True,
                "cost_baselines_accepted": True,
                "phase6_complete": True,
            },
            "failures": [],
            "status": {
                "targets": phase6_targets(),
                "pipeline": {"pipeline_id": "chart-notes"},
                "udf": {"udf_id": "chart-classify-events"},
            },
            "error": "stale failure",
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase6_gate_complete"]["ok"] is False
    step = next(step for step in report["next_steps"] if step["check"] == "phase6_gate_complete")
    assert "error='stale failure'" in step["details"]


def test_plan_audit_handles_failed_phase6_wrapper_reports(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/phase6-status-report.json",
        {"status": "failed", "error": "LAYER_GATEWAY_API_KEY is required"},
    )
    write(
        tmp_path / "eval/out/phase6-gate-report.json",
        {"status": "failed", "error": "LAYER_GATEWAY_API_KEY is required"},
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase6_gate_complete"] == {
        "ok": False,
        "optional": False,
        "reports": ["phase6_gate", "phase6_status"],
        "reason": "report content does not satisfy gate",
    }
    step = next(step for step in report["next_steps"] if step["check"] == "phase6_gate_complete")
    assert "error='LAYER_GATEWAY_API_KEY is required'" in step["details"]
    assert "targets.pipeline_id missing" in step["details"]


def test_plan_audit_rejects_phase6_gate_status_target_mismatch(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/phase6-status-report.json",
        {
            "namespace": "chart-notes",
            "targets": phase6_targets(pipeline_id="other-pipeline"),
            "pipeline": {"pipeline_id": "other-pipeline"},
            "udf": {"udf_id": "chart-classify-events"},
            "facets": full_facet_status(),
        },
    )
    write(
        tmp_path / "eval/out/phase6-gate-report.json",
        {
            "namespace": "chart-notes",
            "targets": phase6_targets(),
            "gates": {
                "pipeline_installed": True,
                "udf_installed": True,
                "full_index_complete": True,
                "full_classify_complete": True,
                "base_facets_visible": True,
                "event_facets_visible": True,
                "full_facets_complete": True,
                "cost_baselines_accepted": True,
                "phase6_complete": True,
            },
            "failures": [],
            "status": {
                "targets": phase6_targets(),
                "pipeline": {"pipeline_id": "chart-notes"},
                "udf": {"udf_id": "chart-classify-events"},
            },
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase6_gate_complete"] == {
        "ok": False,
        "optional": False,
        "reports": ["phase6_gate", "phase6_status"],
        "reason": "report content does not satisfy gate",
    }


def test_plan_audit_requires_holdout_gate_acceptance_when_present(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/holdout-report.json",
        {
            "overlap_edges": 0,
            "gate": {
                "accepted": True,
                "checks": {
                    "feature_edges_present": {"ok": True},
                    "qrel_edges_present": {"ok": False},
                    "max_overlap_edges": {"ok": True},
                },
            },
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase5_holdout"] == {
        "ok": False,
        "optional": False,
        "reports": ["holdout"],
        "reason": "report content does not satisfy gate",
    }
    step = next(step for step in report["next_steps"] if step["check"] == "phase5_holdout")
    assert "gate.checks.qrel_edges_present.ok=False, expected true" in step["details"]


def test_plan_audit_requires_structured_holdout_gate(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/holdout-report.json",
        {
            "feature_edges": 10,
            "qrel_edges": 5,
            "overlap_edges": 0,
            "accepted": True,
        },
    )

    report = plan_audit.summarize(env=env)

    assert report["checks"]["phase5_holdout"] == {
        "ok": False,
        "optional": False,
        "reports": ["holdout"],
        "reason": "report content does not satisfy gate",
    }
    step = next(step for step in report["next_steps"] if step["check"] == "phase5_holdout")
    assert step["details"] == ["gate missing"]


def test_plan_audit_requirement_statuses_report_gateway_key_without_secret(tmp_path) -> None:
    (tmp_path / ".env").write_text("LAYER_GATEWAY_API_KEY=secret-value\n")

    statuses = plan_audit.requirement_statuses(["gateway_key"], env={}, cwd=tmp_path)

    assert statuses["gateway_key"]["state"] == "present"
    assert "secret-value" not in json.dumps(statuses)


def test_plan_audit_requirement_statuses_can_probe_gateway_key_resolver(tmp_path) -> None:
    resolver = tmp_path / "scripts/lib/resolve_gateway_key.sh"
    resolver.parent.mkdir(parents=True)
    resolver.write_text(
        "resolve_gateway_key() {\n"
        "  LAYER_GATEWAY_API_KEY=resolved-secret\n"
        "  export LAYER_GATEWAY_API_KEY\n"
        "  return 0\n"
        "}\n"
    )

    statuses = plan_audit.requirement_statuses(
        ["gateway_key"],
        env={"CHART_PLAN_AUDIT_PROBE_GATEWAY_KEY": "1"},
        cwd=tmp_path,
    )

    assert statuses["gateway_key"] == {
        "state": "present",
        "description": plan_audit.REQUIREMENT_DESCRIPTIONS["gateway_key"],
        "reason": "gateway key resolved by redacted live resolver probe",
    }
    assert "resolved-secret" not in json.dumps(statuses)


def test_plan_audit_requirement_statuses_does_not_probe_gateway_key_by_default(tmp_path, monkeypatch) -> None:
    resolver = tmp_path / "scripts/lib/resolve_gateway_key.sh"
    resolver.parent.mkdir(parents=True)
    resolver.write_text("resolve_gateway_key() { return 0; }\n")
    monkeypatch.setattr(plan_audit.shutil, "which", lambda name: "/usr/bin/op" if name == "op" else None)

    statuses = plan_audit.requirement_statuses(["gateway_key"], env={}, cwd=tmp_path)

    assert statuses["gateway_key"]["state"] == "unknown"
    assert "CHART_PLAN_AUDIT_PROBE_GATEWAY_KEY=1" in statuses["gateway_key"]["reason"]


def test_plan_audit_requirement_statuses_reports_failed_gateway_key_probe(tmp_path) -> None:
    resolver = tmp_path / "scripts/lib/resolve_gateway_key.sh"
    resolver.parent.mkdir(parents=True)
    resolver.write_text("resolve_gateway_key() { return 1; }\n")

    statuses = plan_audit.requirement_statuses(
        ["gateway_key"],
        env={"CHART_PLAN_AUDIT_PROBE_GATEWAY_KEY": "1"},
        cwd=tmp_path,
    )

    assert statuses["gateway_key"]["state"] == "missing"
    assert statuses["gateway_key"]["reason"] == "redacted live resolver probe could not resolve gateway key"


def test_plan_audit_requirement_statuses_report_local_missing_tools(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(plan_audit.shutil, "which", lambda _name: None)

    statuses = plan_audit.requirement_statuses(
        ["gateway_key", "gpu", "docker_daemon", "kubectl_context", "classifier_cost_acceptance"],
        env={},
        cwd=tmp_path,
    )

    assert statuses["gateway_key"]["state"] == "missing"
    assert statuses["gpu"]["state"] == "missing"
    assert statuses["docker_daemon"]["state"] == "missing"
    assert statuses["kubectl_context"]["state"] == "missing"
    assert statuses["classifier_cost_acceptance"]["state"] == "missing"


def test_plan_audit_requirement_statuses_reports_classifier_extra_missing_off_linux(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(plan_audit.platform, "system", lambda: "Darwin")

    statuses = plan_audit.requirement_statuses(["classifier_extra"], env={}, cwd=tmp_path)

    assert statuses["classifier_extra"] == {
        "state": "missing",
        "description": plan_audit.REQUIREMENT_DESCRIPTIONS["classifier_extra"],
        "reason": "Gemma classifier runtime requires Linux/vLLM; run on a GPU Function image or Linux GPU host",
    }


def test_plan_audit_requirement_statuses_accepts_classifier_extra_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(plan_audit.platform, "system", lambda: "Darwin")

    statuses = plan_audit.requirement_statuses(
        ["classifier_extra"],
        env={"CHART_ASSUME_CLASSIFIER_EXTRA": "1"},
        cwd=tmp_path,
    )

    assert statuses["classifier_extra"]["state"] == "present"
    assert statuses["classifier_extra"]["reason"] == "CHART_ASSUME_CLASSIFIER_EXTRA=1"


def test_plan_audit_requirement_statuses_reports_classifier_extra_present_on_gpu_runtime(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(plan_audit.platform, "system", lambda: "Linux")
    monkeypatch.setattr(plan_audit, "_module_available", lambda name: name in {"vllm", "transformers"})

    statuses = plan_audit.requirement_statuses(["classifier_extra"], env={}, cwd=tmp_path)

    assert statuses["classifier_extra"]["state"] == "present"
    assert statuses["classifier_extra"]["reason"] == "vllm and transformers are importable"


def test_plan_audit_requirement_statuses_report_full_retrieval_corpus(tmp_path) -> None:
    env = env_for(tmp_path)
    status_path = tmp_path / "eval/out/phase6-status-report.json"
    write(status_path, {"pipeline": {"counts": {"indexed": 5557}}})

    statuses = plan_audit.requirement_statuses(["full_retrieval_corpus"], env=env, cwd=tmp_path)

    assert statuses["full_retrieval_corpus"]["state"] == "missing"
    assert "phase6 status: indexed rows=5557" in statuses["full_retrieval_corpus"]["reason"]

    write(status_path, {"pipeline": {"counts": {"indexed": 167000}}})
    statuses = plan_audit.requirement_statuses(["full_retrieval_corpus"], env=env, cwd=tmp_path)

    assert statuses["full_retrieval_corpus"]["state"] == "present"
    assert "phase6 status: indexed rows=167000" in statuses["full_retrieval_corpus"]["reason"]


def test_plan_audit_requirement_statuses_report_layer_autoscaling_secret_status(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/phase6-status-report.json",
        {
            "kubernetes": {
                "scaled_objects": {
                    "chart-embed-gpu-worker": {
                        "conditions": [
                            {
                                "type": "Ready",
                                "status": "False",
                                "reason": "ScaledObjectCheckFailed",
                            }
                        ]
                    }
                },
                "trigger_authentications": {
                    "chart-embed-gpu-worker": {
                        "secret_target_refs": [
                            {
                                "parameter": "bearerToken",
                                "name": "layer",
                                "key": "turbopuffer-api-key",
                                "status": {
                                    "secret_exists": False,
                                    "key_exists": False,
                                    "value_present": False,
                                },
                            }
                        ]
                    }
                },
            }
        },
    )

    statuses = plan_audit.requirement_statuses(["layer_autoscaling"], env=env, cwd=tmp_path)

    assert statuses["layer_autoscaling"] == {
        "state": "missing",
        "description": plan_audit.REQUIREMENT_DESCRIPTIONS["layer_autoscaling"],
        "reason": (
            "chart-embed-gpu-worker: ScaledObjectCheckFailed "
            "bearerTokenRef=layer/turbopuffer-api-key "
            "secret_exists=False key_exists=False value_present=False"
        ),
    }


def test_plan_audit_requirement_statuses_report_layer_autoscaling_ready(tmp_path) -> None:
    env = env_for(tmp_path)
    write(
        tmp_path / "eval/out/phase6-status-report.json",
        {
            "kubernetes": {
                "scaled_objects": {
                    "chart-embed-gpu-worker": {
                        "conditions": [{"type": "Ready", "status": "True"}],
                    },
                    "chart-ingest-worker": {
                        "conditions": [{"type": "Ready", "status": "True"}],
                    },
                }
            }
        },
    )

    statuses = plan_audit.requirement_statuses(["layer_autoscaling"], env=env, cwd=tmp_path)

    assert statuses["layer_autoscaling"]["state"] == "present"
    assert "Layer ScaledObjects are ready" in statuses["layer_autoscaling"]["reason"]


def test_plan_audit_requirement_statuses_falls_back_to_slice_index_for_full_retrieval_corpus(tmp_path) -> None:
    env = env_for(tmp_path)
    write(tmp_path / "eval/out/slice-index-report.json", {"indexed": 2000})

    statuses = plan_audit.requirement_statuses(["full_retrieval_corpus"], env=env, cwd=tmp_path)

    assert statuses["full_retrieval_corpus"]["state"] == "missing"
    assert "slice index report: indexed rows=2000" in statuses["full_retrieval_corpus"]["reason"]


def test_plan_audit_cli_writes_report_and_can_fail(monkeypatch, tmp_path, capsys) -> None:
    out = tmp_path / "audit.json"
    monkeypatch.setattr(plan_audit, "summarize", lambda: {"complete": False, "checks": {}, "reports": {}})
    monkeypatch.setattr("sys.argv", ["plan_audit", "--out", str(out), "--require-complete"])

    with pytest.raises(SystemExit) as exc:
        plan_audit.main()

    assert exc.value.code == 1
    assert '"complete": false' in capsys.readouterr().out
    assert json.loads(out.read_text()) == {"complete": False, "checks": {}, "reports": {}}


def test_plan_audit_cli_can_include_requirement_diagnostics(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        plan_audit,
        "summarize",
        lambda: {
            "complete": False,
            "checks": {},
            "reports": {},
            "next_steps": [
                {
                    "check": "phase1_slice_index",
                    "command": "scripts/live_slice.sh",
                    "ready": True,
                    "requires": ["gateway_key"],
                }
            ],
        },
    )
    monkeypatch.setattr(
        plan_audit,
        "requirement_statuses",
        lambda requirements: {"gateway_key": {"state": "missing", "description": ",".join(requirements)}},
    )
    monkeypatch.setattr("sys.argv", ["plan_audit", "--requirements"])

    plan_audit.main()

    report = json.loads(capsys.readouterr().out)
    assert report["requirements"]["gateway_key"] == {
        "state": "missing",
        "description": "gateway_key",
    }


def test_plan_audit_cli_can_print_ready_next_steps(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        plan_audit,
        "summarize",
        lambda: {
            "complete": False,
            "checks": {},
            "reports": {},
            "next_steps": [
                {"check": "phase1_slice_index", "command": "scripts/live_slice.sh", "ready": True},
                {
                    "check": "phase2_3_live_smoke",
                    "command": "CHART_REQUIRE_EVENT_FACETS=1 scripts/smoke_live.sh",
                    "ready": False,
                    "blocked_by": ["phase1_slice_index"],
                },
            ],
        },
    )
    monkeypatch.setattr("sys.argv", ["plan_audit", "--ready"])

    plan_audit.main()

    assert capsys.readouterr().out == "phase1_slice_index: scripts/live_slice.sh\n"


def test_plan_audit_cli_ready_reports_no_ready_steps(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        plan_audit,
        "summarize",
        lambda: {
            "complete": False,
            "checks": {},
            "reports": {},
            "next_steps": [
                {
                    "check": "phase2_3_live_smoke",
                    "command": "CHART_REQUIRE_EVENT_FACETS=1 scripts/smoke_live.sh",
                    "ready": False,
                    "blocked_by": ["phase1_slice_index"],
                },
            ],
        },
    )
    monkeypatch.setattr("sys.argv", ["plan_audit", "--ready"])

    plan_audit.main()

    assert capsys.readouterr().out == "no ready next steps\n"
