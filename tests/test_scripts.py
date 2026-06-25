import subprocess
import json
import os
from pathlib import Path

from smoke.plan_audit import NEXT_ACTIONS


def _script(name: str) -> str:
    return (Path(__file__).resolve().parent.parent / "scripts" / name).read_text()


def test_env_example_documents_local_gate_and_worker_controls() -> None:
    env_example = (Path(__file__).resolve().parent.parent / ".env.example").read_text()

    for name in (
        "LAYER_GATEWAY_API_KEY",
        "CHART_SLICE_LIMIT",
        "CHART_SLICE_INDEX_REPORT",
        "CHART_REQUIRE_EVENT_FACETS",
        "CHART_LIVE_SMOKE_BASE_REPORT",
        "CHART_LIVE_SMOKE_REPORT",
        "CHART_FACET_REFRESH_REPORT",
        "CHART_EVAL_HOLDOUT_MAX_OVERLAP",
        "CHART_EVAL_REQUIRE_NO_FAILURES",
        "CHART_EVAL_REQUIRE_FUSED_DOMINATES",
        "CHART_EVAL_HOLDOUT_REPORT",
        "CHART_EVAL_RECDS_REPORT",
        "CHART_EVAL_BIMODAL_REPORT",
        "CHART_PREFLIGHT_LIVE",
        "CHART_ALLOW_FULL_CPU_INDEX",
        "CHART_ACCEPT_PHASE4_CLASSIFY_COST",
        "CHART_PHASE4_CLASSIFY_REPORT",
        "CHART_ASSUME_CLASSIFIER_EXTRA",
        "CHART_LAYER_COST_WINDOW",
        "CHART_APPLY_CLASSIFIER",
        "CHART_ACCEPT_PHASE6_EMBED_COST",
        "CHART_MEASURE_ACCELERATOR",
        "CHART_EVENTS_MODEL",
        "CHART_EMBED_LEASE_SECONDS",
        "CHART_EMBED_HEARTBEAT_SECONDS",
        "CHART_EMBED_POLL_SECONDS",
        "CHART_EMBED_SIMILAR_QRELS_SPLIT",
        "HEVLAYER_PIPELINE_ID",
        "HEVLAYER_WORKER_ID",
        "CHART_EMBED_IMAGE",
        "CHART_SOURCE_IMAGE",
        "CHART_CLASSIFIER_IMAGE",
        "CHART_ECR_REPOSITORY_URL",
        "CHART_ECR_LOGIN",
        "CHART_ECR_LOGIN_REGISTRY",
        "CHART_GPU_PLATFORM",
        "CHART_GPU_BUILDER",
        "CHART_DEPOT_PROJECT_ID",
        "CHART_PRELOAD_EMBED_MODEL",
        "CHART_PRELOAD_EVENTS_MODEL",
        "CHART_GPU_BUILD_REPORT",
        "CHART_LAYER_CLIENT_CONTEXT",
        "CHART_DEPLOY_APPLY_REPORT",
        "CHART_PLAN_AUDIT_REPORT",
        "CHART_K8S_NAMESPACE",
        "CHART_K8S_CONTEXT_CONFIRM",
        "CHART_EMBED_PIPELINE_CR",
        "CHART_EMBED_PIPELINE_ID",
        "CHART_PHASE6_EMBED_BUDGET_REPORT",
        "CHART_PHASE6_UNPAUSE_REPORT",
        "CHART_PHASE6_DRAIN_REPORT",
        "CHART_PHASE6_STAGE_REPORT",
        "CHART_PHASE6_SOURCE_START_OFFSET",
        "CHART_PHASE6_SOURCE_MAX_ROWS",
        "CHART_PHASE6_SOURCE_PAGE_SIZE",
        "CHART_HF_SOURCE_WRITE_CONCURRENCY",
        "CHART_PHASE6_ALLOW_STAGE_WITH_PENDING",
        "CHART_INGEST_DEPLOYMENT",
        "CHART_LAYER_NAMESPACE",
        "CHART_LAYER_GATEWAY_POD",
        "CHART_LAYER_POSTGRES_CONTAINER",
        "CHART_PHASE6_DRAIN_WORKER_ID",
        "CHART_PHASE6_DRAIN_CLAIM_LIMIT",
        "CHART_PHASE6_STATUS_REPORT",
        "CHART_PHASE6_GATE_REPORT",
        "CHART_GATEWAY_KEY_OP_VAULT",
        "CHART_GATEWAY_KEY_OP_ITEM",
        "CHART_GATEWAY_KEY_OP_FIELD",
        "CHART_GATEWAY_KEY_OP_REF",
        "CHART_OP_TIMEOUT_SECONDS",
        "CHART_QUERY_EMBED_URL",
    ):
        assert name in env_example


def test_live_slice_script_resolves_key_without_echoing_secret() -> None:
    script = _script("live_slice.sh")

    assert "source scripts/lib/resolve_gateway_key.sh" in script
    assert "resolve_gateway_key" in script
    assert "python -m indexer --limit" in script
    assert "CHART_SLICE_INDEX_REPORT" in script
    assert "slice-index-report.json" in script
    assert '--out "$INDEX_REPORT"' in script
    assert "scripts/smoke_live.sh" in script
    assert "echo \"$LAYER_GATEWAY_API_KEY\"" not in script


def test_smoke_live_script_resolves_key_without_reindexing_or_echoing_secret() -> None:
    script = _script("smoke_live.sh")

    assert "source scripts/lib/resolve_gateway_key.sh" in script
    assert "source scripts/lib/write_failure_report.sh" in script
    assert "resolve_gateway_key" in script
    assert "python -m smoke.live" in script
    assert "CHART_REQUIRE_EVENT_FACETS" in script
    assert "--require-event-facets" in script
    assert "CHART_LIVE_SMOKE_REPORT" in script
    assert "live-smoke-report.json" in script
    assert "has_out=0" in script
    assert "python -m indexer" not in script
    assert '"$@"' in script
    assert "echo \"$LAYER_GATEWAY_API_KEY\"" not in script


def test_layer_cost_report_script_resolves_key_and_defaults_outputs() -> None:
    script = _script("layer_cost_report.sh")

    assert "source scripts/lib/resolve_gateway_key.sh" in script
    assert "source scripts/lib/write_failure_report.sh" in script
    assert "resolve_gateway_key" in script
    assert "python -m smoke.layer_cost" in script
    assert "CHART_PHASE4_CLASSIFY_REPORT" in script
    assert "CHART_PHASE6_EMBED_BUDGET_REPORT" in script
    assert "classify-events-budget.json" in script
    assert "embed-budget.json" in script
    assert "Layer cost report command failed" in script
    assert "echo \"$LAYER_GATEWAY_API_KEY\"" not in script


def test_live_wrappers_write_failure_reports_when_commands_fail(tmp_path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    uv = fake_bin / "uv"
    uv.write_text("#!/usr/bin/env bash\nexit 42\n")
    uv.chmod(0o755)
    root = Path(__file__).resolve().parent.parent
    env = {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "LAYER_GATEWAY_API_KEY": "test-key",
    }

    cases = [
        (
            ["bash", "scripts/live_slice.sh"],
            "CHART_SLICE_INDEX_REPORT",
            "slice-report.json",
            "slice index command failed",
        ),
        (
            ["bash", "scripts/smoke_live.sh"],
            "CHART_LIVE_SMOKE_REPORT",
            "smoke-report.json",
            "live smoke command failed",
        ),
        (
            ["bash", "scripts/refresh_facets.sh"],
            "CHART_FACET_REFRESH_REPORT",
            "facets-report.json",
            "facet refresh command failed",
        ),
        (
            ["bash", "scripts/eval_live.sh"],
            "CHART_EVAL_RECDS_REPORT",
            "recds-report.json",
            "ReCDS eval command failed",
        ),
        (
            ["bash", "scripts/full_status.sh"],
            "CHART_PHASE6_STATUS_REPORT",
            "phase6-status-report.json",
            "phase6 status command failed",
        ),
        (
            ["bash", "scripts/gate_report.sh"],
            "CHART_PHASE6_GATE_REPORT",
            "phase6-gate-report.json",
            "phase6 gate command failed",
        ),
    ]

    for command, env_name, filename, error in cases:
        report = tmp_path / filename
        run_env = {**env, env_name: str(report)}
        result = subprocess.run(command, cwd=root, env=run_env, check=False, capture_output=True, text=True)

        assert result.returncode == 1
        data = json.loads(report.read_text())
        assert data == {"status": "failed", "error": error}


def test_eval_live_script_runs_recds_and_optional_bimodal() -> None:
    script = _script("eval_live.sh")

    assert "source scripts/lib/resolve_gateway_key.sh" in script
    assert "source scripts/lib/write_failure_report.sh" in script
    assert "resolve_gateway_key" in script
    assert "python -m eval.recds" in script
    assert "--task ppr" in script
    assert "CHART_EVAL_TOP_K" in script
    assert "CHART_EVAL_PROGRESS_EVERY" in script
    assert "CHART_EVAL_REQUIRE_NO_FAILURES" in script
    assert "CHART_EVAL_REQUIRE_FUSED_DOMINATES" in script
    assert "CHART_EVAL_HOLDOUT_MAX_OVERLAP" in script
    assert "CHART_EVAL_HOLDOUT_SPLIT" in script
    assert "CHART_EVAL_HOLDOUT_REPORT" in script
    assert "CHART_EVAL_RECDS_REPORT" in script
    assert "CHART_EVAL_BIMODAL_REPORT" in script
    assert "python -m eval.holdout" in script
    assert "--max-overlap-edges" in script
    assert '--out "$HOLDOUT_OUT"' in script
    assert "--require-no-failures" in script
    assert "--require-fused-dominates" in script
    assert '--top-k "$TOP_K"' in script
    assert '--progress-every "$PROGRESS_EVERY"' in script
    assert '--out "$RECDS_OUT"' in script
    assert '--out "$BIMODAL_OUT"' in script
    assert "--beir-dir eval/out/bimodal" in script
    assert "echo \"$LAYER_GATEWAY_API_KEY\"" not in script


def test_eval_live_runs_local_holdout_before_requiring_gateway_key() -> None:
    script = _script("eval_live.sh")
    resolver_call = "\nif ! resolve_gateway_key; then\n"

    assert script.index("python -m eval.holdout") < script.index(resolver_call)
    assert script.index(resolver_call) < script.index("python -m eval.recds")


def test_eval_live_preserves_recds_report_written_by_failed_gate(tmp_path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    uv = fake_bin / "uv"
    uv.write_text(
        "#!/usr/bin/env bash\n"
        "out=''\n"
        "prev=''\n"
        "for arg in \"$@\"; do\n"
        "  if [[ \"$prev\" == --out ]]; then out=\"$arg\"; fi\n"
        "  prev=\"$arg\"\n"
        "done\n"
        "mkdir -p \"$(dirname \"$out\")\"\n"
        "printf '{\"task\":\"ppr\",\"gates\":{\"fused_dominates\":{\"accepted\":false}}}\\n' > \"$out\"\n"
        "exit 1\n"
    )
    uv.chmod(0o755)
    report = tmp_path / "recds-report.json"

    result = subprocess.run(
        ["bash", "scripts/eval_live.sh"],
        cwd=Path(__file__).resolve().parent.parent,
        env={
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "LAYER_GATEWAY_API_KEY": "test-key",
            "CHART_EVAL_RECDS_REPORT": str(report),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert json.loads(report.read_text()) == {
        "task": "ppr",
        "gates": {"fused_dominates": {"accepted": False}},
    }


def test_gate_report_preserves_report_written_by_failed_gate(tmp_path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    uv = fake_bin / "uv"
    uv.write_text(
        "#!/usr/bin/env bash\n"
        "out=''\n"
        "prev=''\n"
        "for arg in \"$@\"; do\n"
        "  if [[ \"$prev\" == --out ]]; then out=\"$arg\"; fi\n"
        "  prev=\"$arg\"\n"
        "done\n"
        "mkdir -p \"$(dirname \"$out\")\"\n"
        "printf '{\"gates\":{\"phase6_complete\":false},\"failures\":[{\"gate\":\"phase6_complete\"}]}\\n' > \"$out\"\n"
        "exit 1\n"
    )
    uv.chmod(0o755)
    report = tmp_path / "phase6-gate-report.json"

    result = subprocess.run(
        ["bash", "scripts/gate_report.sh"],
        cwd=Path(__file__).resolve().parent.parent,
        env={
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "LAYER_GATEWAY_API_KEY": "test-key",
            "CHART_PHASE6_GATE_REPORT": str(report),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert json.loads(report.read_text()) == {
        "gates": {"phase6_complete": False},
        "failures": [{"gate": "phase6_complete"}],
    }


def test_live_wrapper_writes_gateway_failure_report(tmp_path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    grep = fake_bin / "grep"
    grep.write_text("#!/usr/bin/env bash\nexit 1\n")
    grep.chmod(0o755)
    report = tmp_path / "smoke-report.json"
    result = subprocess.run(
        ["bash", "scripts/smoke_live.sh"],
        cwd=Path(__file__).resolve().parent.parent,
        env={
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "CHART_LIVE_SMOKE_REPORT": str(report),
            "CHART_OP_TIMEOUT_SECONDS": "0.01",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    data = json.loads(report.read_text())
    assert data == {"status": "failed", "error": "LAYER_GATEWAY_API_KEY is required"}


def test_smoke_live_zero_arg_event_facet_mode_reaches_uv(tmp_path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "uv-calls.txt"
    uv = fake_bin / "uv"
    uv.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" > "$UV_CALLS"\n'
        "exit 42\n"
    )
    uv.chmod(0o755)
    report = tmp_path / "smoke-report.json"

    result = subprocess.run(
        ["bash", "scripts/smoke_live.sh"],
        cwd=Path(__file__).resolve().parent.parent,
        env={
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "LAYER_GATEWAY_API_KEY": "test-key",
            "CHART_REQUIRE_EVENT_FACETS": "1",
            "CHART_LIVE_SMOKE_REPORT": str(report),
            "UV_CALLS": str(calls),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "unbound variable" not in result.stderr
    assert calls.read_text() == (
        f"run --extra search python -m smoke.live --require-event-facets --out {report}\n"
    )
    data = json.loads(report.read_text())
    assert data == {"status": "failed", "error": "live smoke command failed"}


def test_smoke_live_wrapper_preserves_report_written_by_failed_command(tmp_path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    uv = fake_bin / "uv"
    uv.write_text(
        "#!/usr/bin/env bash\n"
        "out=''\n"
        "while [[ $# -gt 0 ]]; do\n"
        "  if [[ \"$1\" == '--out' ]]; then out=\"$2\"; shift 2; continue; fi\n"
        "  if [[ \"$1\" == --out=* ]]; then out=\"${1#--out=}\"; shift; continue; fi\n"
        "  shift\n"
        "done\n"
        "printf '%s\\n' '{\"status\":\"failed\",\"error\":\"event facets empty\"}' > \"$out\"\n"
        "exit 42\n"
    )
    uv.chmod(0o755)
    report = tmp_path / "smoke-report.json"

    result = subprocess.run(
        ["bash", "scripts/smoke_live.sh"],
        cwd=Path(__file__).resolve().parent.parent,
        env={
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "LAYER_GATEWAY_API_KEY": "test-key",
            "CHART_LIVE_SMOKE_REPORT": str(report),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert json.loads(report.read_text()) == {"status": "failed", "error": "event facets empty"}


def test_preflight_checks_tools_key_and_local_gates() -> None:
    script = _script("preflight.sh")

    assert "need_cmd uv" in script
    assert "need_cmd bash" in script
    assert "need_cmd node" in script
    assert "need_cmd npx" in script
    assert "bash -n \"$script\"" in script
    assert "find scripts -name '*.sh' -type f | sort" in script
    assert "CHART_PREFLIGHT_DOCKER" in script
    assert "CHART_PREFLIGHT_LIVE" in script
    assert 'CHART_GPU_BUILD_REPORT="${TMPDIR:-/tmp}/chart-preflight-gpu-build-report.json"' in script
    assert "scripts/build_gpu_images.sh --dry-run" in script
    assert "source scripts/lib/resolve_gateway_key.sh" not in script
    assert "resolve_gateway_key" not in script
    assert "uv run --extra search --extra eval --extra test pytest" in script
    assert "python -m compileall chart_common indexer search functions eval smoke tests" in script
    assert "node --check src/worker.js" in script
    assert "wrangler@4.104.0 deploy --dry-run" in script
    assert "wrangler@latest" not in script
    assert 'CHART_DEPLOY_APPLY_REPORT="${TMPDIR:-/tmp}/chart-preflight-deploy-apply-report.json"' in script
    assert "scripts/deploy_apply.sh --dry-run" in script
    assert 'CHART_PLAN_AUDIT_REPORT="${TMPDIR:-/tmp}/chart-preflight-plan-audit-report.json"' in script
    assert "scripts/plan_audit.sh" in script
    assert "scripts/plan_audit.sh --requirements" in script
    assert "scripts/smoke_live.sh" in script
    assert "scripts/gate_report.sh" in script
    assert "echo \"$LAYER_GATEWAY_API_KEY\"" not in script


def test_preflight_resolves_gateway_key_only_for_optional_live_gate() -> None:
    script = _script("preflight.sh")
    live_block = script.split('if [[ "${CHART_PREFLIGHT_LIVE:-0}" == "1" ]]')[1]

    assert "resolve_gateway_key" not in script.split('if [[ "${CHART_PREFLIGHT_LIVE:-0}" == "1" ]]')[0]
    assert "resolve_gateway_key" not in live_block
    assert "scripts/smoke_live.sh" in live_block
    assert "scripts/gate_report.sh" in live_block


def test_full_status_reports_gateway_state_without_echoing_secret() -> None:
    script = _script("full_status.sh")

    assert "source scripts/lib/resolve_gateway_key.sh" in script
    assert "source scripts/lib/write_failure_report.sh" in script
    assert "resolve_gateway_key" in script
    assert "python -m smoke.full_status" in script
    assert "CHART_PHASE6_STATUS_REPORT" in script
    assert "phase6-status-report.json" in script
    assert "has_out=0" in script
    assert "echo \"$LAYER_GATEWAY_API_KEY\"" not in script


def test_gate_report_summarizes_plan_gates_without_echoing_secret() -> None:
    script = _script("gate_report.sh")

    assert "source scripts/lib/resolve_gateway_key.sh" in script
    assert "source scripts/lib/write_failure_report.sh" in script
    assert "resolve_gateway_key" in script
    assert "python -m smoke.gates" in script
    assert "CHART_PHASE6_GATE_REPORT" in script
    assert "phase6-gate-report.json" in script
    assert "has_out=0" in script
    assert '"$@"' in script
    assert "echo \"$LAYER_GATEWAY_API_KEY\"" not in script


def test_plan_audit_reads_persisted_reports_without_gateway_key() -> None:
    script = _script("plan_audit.sh")

    assert "python -m smoke.plan_audit" in script
    assert "CHART_PLAN_AUDIT_REPORT" in script
    assert "plan-audit-report.json" in script
    assert "has_out=0" in script
    assert '"$@"' in script
    assert "resolve_gateway_key" not in script
    assert "echo \"$LAYER_GATEWAY_API_KEY\"" not in script


def test_plan_audit_ready_wrapper_prints_ready_steps_without_gateway_key(tmp_path) -> None:
    report = tmp_path / "plan-audit-report.json"

    result = subprocess.run(
        ["bash", "scripts/plan_audit.sh", "--ready"],
        cwd=Path(__file__).resolve().parent.parent,
        env={
            "PATH": os.environ["PATH"],
            "CHART_PLAN_AUDIT_REPORT": str(report),
            "CHART_SLICE_INDEX_REPORT": str(tmp_path / "slice-index-report.json"),
            "CHART_LIVE_SMOKE_REPORT": str(tmp_path / "live-smoke-report.json"),
            "CHART_FACET_REFRESH_REPORT": str(tmp_path / "facet-refresh-report.json"),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "phase1_slice_index: scripts/live_slice.sh" in result.stdout
    assert "phase2_3_live_smoke" not in result.stdout
    assert report.exists()


def test_readme_documents_plan_audit_next_step_fields() -> None:
    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text()

    assert "`scripts/final_gate.sh` is the final PLAN.md audit gate" in readme
    assert "`next_steps`" in readme
    assert "`details`" in readme
    assert "`ready`" in readme
    assert "`requires`" in readme
    assert "`blocked_by`" in readme
    assert "base smoke is blocked by the slice index and" in readme
    assert "`scripts/plan_audit.sh --ready`" in readme
    assert "`scripts/plan_audit.sh --requirements`" in readme
    assert "full ReCDS retrieval corpus" in readme
    assert "Layer autoscaling" in readme
    assert "CHART_ASSUME_CLASSIFIER_EXTRA=1" in readme
    assert "preflight does not overwrite accepted gate evidence" in readme


def test_readme_live_sequence_runs_classifier_before_event_facet_smoke() -> None:
    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text()

    classifier_once = "scripts/phase4_event_smoke.sh"
    refresh_events = "scripts/refresh_facets.sh --fields age_band,gender,events"
    event_smoke = "CHART_REQUIRE_EVENT_FACETS=1 scripts/smoke_live.sh"

    assert classifier_once in readme
    assert refresh_events in readme
    assert event_smoke in readme
    assert "scripts/refresh_facets.sh --fields events" not in readme
    assert readme.index(classifier_once) < readme.index(refresh_events)
    assert readme.index(refresh_events) < readme.index(event_smoke)


def test_functions_readme_refreshes_all_audited_slice_facets() -> None:
    readme = (Path(__file__).resolve().parent.parent / "functions" / "README.md").read_text()

    assert "scripts/phase4_event_smoke.sh" in readme
    assert "uv run --extra classifier python -m functions.classify_events --once" in readme
    assert "scripts/refresh_facets.sh --fields age_band,gender,events" in readme
    assert "scripts/refresh_facets.sh --fields events" not in readme
    assert "Linux/vLLM/GPU-oriented" in readme
    assert "GPU Function image or on a Linux GPU host" in readme


def test_phase4_event_smoke_guards_classifier_runtime_and_runs_gate_sequence() -> None:
    script = _script("phase4_event_smoke.sh")

    assert "source scripts/lib/write_failure_report.sh" in script
    assert "CHART_LIVE_SMOKE_REPORT" in script
    assert "CHART_FACET_REFRESH_REPORT" in script
    assert "write_phase4_failure" in script
    assert "CHART_ASSUME_CLASSIFIER_EXTRA" in script
    assert "uname -s" in script
    assert "Gemma classifier runtime requires Linux/vLLM" in script
    assert 'for name in ("vllm", "transformers")' in script
    assert "uv run --extra classifier python -m functions.classify_events --once" in script
    assert "classifier event smoke command failed" in script
    assert "scripts/refresh_facets.sh --fields age_band,gender,events" in script
    assert "CHART_REQUIRE_EVENT_FACETS=1 scripts/smoke_live.sh" in script


def test_phase4_event_smoke_non_linux_guard_writes_failure_report(tmp_path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    uname = fake_bin / "uname"
    uname.write_text("#!/usr/bin/env bash\nprintf Darwin\n")
    uname.chmod(0o755)
    live_report = tmp_path / "live-smoke-report.json"
    facet_report = tmp_path / "facet-refresh-report.json"

    result = subprocess.run(
        ["bash", "scripts/phase4_event_smoke.sh"],
        cwd=Path(__file__).resolve().parent.parent,
        env={
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "CHART_LIVE_SMOKE_REPORT": str(live_report),
            "CHART_FACET_REFRESH_REPORT": str(facet_report),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == ""
    expected = {
        "status": "failed",
        "error": "Gemma classifier runtime requires Linux/vLLM; run on a GPU Function image or Linux GPU host",
    }
    assert json.loads(live_report.read_text()) == expected
    assert json.loads(facet_report.read_text()) == expected


def test_phase4_event_smoke_classifier_failure_writes_failure_reports(tmp_path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    uname = fake_bin / "uname"
    uname.write_text("#!/usr/bin/env bash\nprintf Linux\n")
    uname.chmod(0o755)
    uv = fake_bin / "uv"
    uv.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$*\" == *'functions.classify_events --once'* ]]; then exit 42; fi\n"
        "exit 0\n"
    )
    uv.chmod(0o755)
    live_report = tmp_path / "live-smoke-report.json"
    facet_report = tmp_path / "facet-refresh-report.json"

    result = subprocess.run(
        ["bash", "scripts/phase4_event_smoke.sh"],
        cwd=Path(__file__).resolve().parent.parent,
        env={
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "CHART_LIVE_SMOKE_REPORT": str(live_report),
            "CHART_FACET_REFRESH_REPORT": str(facet_report),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    expected = {
        "status": "failed",
        "error": "classifier event smoke command failed",
    }
    assert json.loads(live_report.read_text()) == expected
    assert json.loads(facet_report.read_text()) == expected


def test_docs_include_audit_next_action_commands() -> None:
    root = Path(__file__).resolve().parent.parent
    docs = "\n".join(
        (root / path).read_text()
        for path in ("README.md", "functions/README.md", "eval/README.md", "deploy/README.md")
    )

    for command in NEXT_ACTIONS.values():
        assert command in docs


def test_deploy_readme_documents_failed_apply_report_errors() -> None:
    readme = (Path(__file__).resolve().parent.parent / "deploy" / "README.md").read_text()

    assert "Refused `--apply` runs also persist" in readme
    assert "an `error` field" in readme
    assert "`CHART_PHASE6_UNPAUSE_REPORT`" in readme


def test_final_gate_requires_complete_plan_audit_without_gateway_key() -> None:
    script = _script("final_gate.sh")

    assert "scripts/plan_audit.sh --requirements --require-complete" in script
    assert '[[ "$arg" == "--ready" ]]' in script
    assert "use scripts/plan_audit.sh --ready" in script
    assert '"$@"' in script
    assert "resolve_gateway_key" not in script
    assert "echo \"$LAYER_GATEWAY_API_KEY\"" not in script


def test_final_gate_rejects_ready_mode() -> None:
    result = subprocess.run(
        ["bash", "scripts/final_gate.sh", "--ready"],
        cwd=Path(__file__).resolve().parent.parent,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "use scripts/plan_audit.sh --ready" in result.stderr
    assert result.stdout == ""


def test_refresh_facets_script_resolves_key_without_echoing_secret() -> None:
    script = _script("refresh_facets.sh")

    assert "source scripts/lib/resolve_gateway_key.sh" in script
    assert "resolve_gateway_key" in script
    assert "python -m smoke.facets" in script
    assert "CHART_FACET_REFRESH_REPORT" in script
    assert "facet-refresh-report.json" in script
    assert "has_out=0" in script
    assert "echo \"$LAYER_GATEWAY_API_KEY\"" not in script


def test_gateway_key_resolver_handles_op_failures_without_echoing_secret() -> None:
    script = _script("lib/resolve_gateway_key.sh")

    assert "op item get" in script
    assert '"layer-turbopuffer"' in script
    assert '"layer turbopuffer"' in script
    assert script.index('"layer turbopuffer"') < script.index('"layer-turbopuffer"')
    assert "CHART_GATEWAY_KEY_OP_ITEM" in script
    assert "CHART_GATEWAY_KEY_OP_VAULT" in script
    assert "CHART_GATEWAY_KEY_OP_FIELD" in script
    assert "CHART_GATEWAY_KEY_OP_REF" in script
    assert '"op://mesh-staging/layer turbopuffer/credential"' in script
    assert '"op://mesh-staging/layer-turbopuffer/credential"' in script
    assert "Could not read gateway key from 1Password" in script
    assert "kill -9 \"$op_pid\"" in script
    assert "Export it first or sign in with: op signin" in script
    assert "echo \"$LAYER_GATEWAY_API_KEY\"" not in script


def test_gateway_key_resolver_reads_dotenv_without_sourcing_shell(tmp_path) -> None:
    resolver = Path(__file__).resolve().parent.parent / "scripts" / "lib" / "resolve_gateway_key.sh"
    (tmp_path / ".env").write_text('LAYER_GATEWAY_API_KEY="from-dotenv" # local key\n')

    result = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{resolver}"; resolve_gateway_key; test "$LAYER_GATEWAY_API_KEY" = from-dotenv',
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_gateway_key_resolver_times_out_locked_op(tmp_path) -> None:
    resolver = Path(__file__).resolve().parent.parent / "scripts" / "lib" / "resolve_gateway_key.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    op = fake_bin / "op"
    op.write_text("#!/usr/bin/env bash\nsleep 3\n")
    op.chmod(0o755)

    result = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{resolver}"; CHART_OP_TIMEOUT_SECONDS=1 resolve_gateway_key',
        ],
        cwd=tmp_path,
        env={"PATH": f"{fake_bin}:/usr/bin:/bin"},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Timed out reading gateway key from 1Password after 1s per attempt." in result.stderr
    assert "LAYER_GATEWAY_API_KEY is required" in result.stderr


def test_gateway_key_resolver_supports_configurable_op_ref(tmp_path) -> None:
    resolver = Path(__file__).resolve().parent.parent / "scripts" / "lib" / "resolve_gateway_key.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    op = fake_bin / "op"
    op.write_text(
        "#!/usr/bin/env bash\n"
        "test \"$1\" = read\n"
        "test \"$2\" = op://mesh-staging/layer\\ turbopuffer/credential\n"
        "printf configured-key\n"
    )
    op.chmod(0o755)

    result = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{resolver}"; CHART_GATEWAY_KEY_OP_REF="op://mesh-staging/layer turbopuffer/credential" resolve_gateway_key; test "$LAYER_GATEWAY_API_KEY" = configured-key',
        ],
        cwd=tmp_path,
        env={"PATH": f"{fake_bin}:/usr/bin:/bin"},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_gateway_key_resolver_prefers_item_field_lookup(tmp_path) -> None:
    resolver = Path(__file__).resolve().parent.parent / "scripts" / "lib" / "resolve_gateway_key.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    op = fake_bin / "op"
    op.write_text(
        "#!/usr/bin/env bash\n"
        "test \"$1\" = item\n"
        "test \"$2\" = get\n"
        "test \"$3\" = 'layer turbopuffer'\n"
        "test \"$4\" = --vault\n"
        "test \"$5\" = mesh-staging\n"
        "test \"$6\" = --field\n"
        "test \"$7\" = credential\n"
        "test \"$8\" = --reveal\n"
        "printf item-key\n"
    )
    op.chmod(0o755)

    result = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{resolver}"; resolve_gateway_key; test "$LAYER_GATEWAY_API_KEY" = item-key',
        ],
        cwd=tmp_path,
        env={"PATH": f"{fake_bin}:/usr/bin:/bin"},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_gateway_key_resolver_continues_after_timed_out_candidate(tmp_path) -> None:
    resolver = Path(__file__).resolve().parent.parent / "scripts" / "lib" / "resolve_gateway_key.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    op = fake_bin / "op"
    op.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" = item && \"$3\" = 'layer turbopuffer' ]]; then sleep 3; fi\n"
        "if [[ \"$1\" = item && \"$3\" = layer-turbopuffer ]]; then printf fallback-key; exit 0; fi\n"
        "exit 1\n"
    )
    op.chmod(0o755)

    result = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{resolver}"; CHART_OP_TIMEOUT_SECONDS=1 resolve_gateway_key; test "$LAYER_GATEWAY_API_KEY" = fallback-key',
        ],
        cwd=tmp_path,
        env={"PATH": f"{fake_bin}:/usr/bin:/bin"},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_deploy_apply_defaults_to_ordered_client_dry_run() -> None:
    script = _script("deploy_apply.sh")

    assert 'MODE="client-dry-run"' in script
    assert 'kubectl_apply_manifest "$manifest" --dry-run=client --validate=false' in script
    assert 'kubectl_apply_manifest "$manifest" --dry-run=server' in script
    assert 'kubectl_apply_manifest "$manifest"' in script
    assert "render_manifest()" in script
    assert "CHART_EMBED_IMAGE" in script
    assert "CHART_CLASSIFIER_IMAGE" in script
    assert "require_secret chart-turbopuffer" in script
    assert "require_secret chart-gateway" in script
    assert "deploy/secrets.example.yaml" in script
    assert script.index("deploy/namespace.yaml") < script.index("deploy/vectorstore.yaml")
    assert script.index("deploy/vectorstore.yaml") < script.index("deploy/warehouse.yaml")
    assert script.index("deploy/warehouse.yaml") < script.index("deploy/pipeline.yaml")
    assert script.index("deploy/pipeline.yaml") < script.index("deploy/pipeline-embed.yaml")
    assert script.index("deploy/pipeline-embed.yaml") < script.index("deploy/index.yaml")
    assert script.index("deploy/index.yaml") < script.index("deploy/functions-events.yaml")
    assert "CHART_APPLY_CLASSIFIER" in script
    assert "CHART_ACCEPT_PHASE4_CLASSIFY_COST" in script
    assert "CHART_PHASE4_CLASSIFY_REPORT" in script
    assert "classify_report=\"${CHART_PHASE4_CLASSIFY_REPORT:-eval/out/classify-events-budget.json}\"" in script
    assert "CHART_DEPLOY_APPLY_REPORT" in script
    assert "deploy-apply-report.json" in script
    assert 'write_report "started"' in script
    assert 'write_report "completed"' in script
    assert "functions.measure_classify_events" in script
    assert "layer_cost_complete" in script
    assert "max_full_usd" in script
    assert "max_full_hours" in script
    assert "max_full_usd" in script
    assert "min_med_discontinuations" in script
    assert "min_review_examples" in script
    assert 'namespace="${CHART_K8S_NAMESPACE:-chart}"' in script
    assert "CHART_K8S_CONTEXT_CONFIRM" in script
    assert "kubectl config current-context" in script
    assert "require_kube_context_confirm" in script
    assert '"kube_context": kube_context or None' in script
    assert '"kube_context_confirmed": kube_context_confirmed == "true"' in script
    assert '"classifier_cost_accepted": classifier_cost_accepted == "true"' in script
    assert 'kubectl -n "$namespace" get secret "$name"' in script
    assert 'kubectl -n chart get secret "$name"' not in script
    assert 'kubectl_apply_manifest deploy/functions-events.yaml' in script
    assert 'kubectl_apply_manifest deploy/functions-events.yaml --dry-run=client --validate=false' in script
    assert "skipped deploy/functions-events.yaml" in script


def test_deploy_apply_client_dry_run_writes_report(tmp_path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    kubectl = fake_bin / "kubectl"
    kubectl.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1 $2\" == \"config current-context\" ]]; then echo test-context; exit 0; fi\n"
        "exit 0\n"
    )
    kubectl.chmod(0o755)
    report = tmp_path / "deploy-apply-report.json"

    result = subprocess.run(
        ["bash", "scripts/deploy_apply.sh", "--dry-run"],
        cwd=Path(__file__).resolve().parent.parent,
        env={
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "CHART_DEPLOY_APPLY_REPORT": str(report),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    data = json.loads(report.read_text())
    assert data["mode"] == "client-dry-run"
    assert data["status"] == "completed"
    assert data["namespace"] == "chart"
    assert data["classifier"] == "validated"
    assert data["kube_context"] is None
    assert data["kube_context_confirmed"] is False
    assert data["classifier_cost_accepted"] is False
    assert data["manifests"][0] == "deploy/namespace.yaml"
    assert "deploy/functions-events.yaml" in data["manifests"]


def test_deploy_apply_does_not_apply_classifier_in_base_runtime_loop() -> None:
    script = _script("deploy_apply.sh")
    runtime_block = script.split("runtime_manifests=(")[1].split(")")[0]

    assert "deploy/functions-events.yaml" not in runtime_block


def test_deploy_apply_apply_rejects_failed_classifier_report_checks(tmp_path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    kubectl = fake_bin / "kubectl"
    kubectl.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1 $2\" == \"config current-context\" ]]; then echo test-context; exit 0; fi\n"
        "exit 0\n"
    )
    kubectl.chmod(0o755)
    report = tmp_path / "classify-report.json"
    deploy_report = tmp_path / "deploy-apply-report.json"
    report.write_text(
        json.dumps(
            {
                "budget": {
                    "accepted": True,
                    "checks": {
                        "max_full_hours": {"ok": True},
                        "max_full_usd": {"ok": False},
                    },
                },
                "signal": {
                    "accepted": True,
                    "checks": {
                        "min_med_discontinuations": {"ok": True},
                        "min_review_examples": {"ok": True},
                    },
                },
            }
        )
        + "\n"
    )

    result = subprocess.run(
        ["bash", "scripts/deploy_apply.sh", "--apply"],
        cwd=Path(__file__).resolve().parent.parent,
        env={
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "CHART_APPLY_CLASSIFIER": "1",
            "CHART_ACCEPT_PHASE4_CLASSIFY_COST": "1",
            "CHART_PHASE4_CLASSIFY_REPORT": str(report),
            "CHART_DEPLOY_APPLY_REPORT": str(deploy_report),
            "CHART_K8S_NAMESPACE": "custom-chart",
            "CHART_K8S_CONTEXT_CONFIRM": "test-context",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "budget.checks.max_full_usd.ok=False, expected true" in result.stderr
    data = json.loads(deploy_report.read_text())
    assert data["status"] == "failed"
    assert data["mode"] == "apply"
    assert data["namespace"] == "custom-chart"
    assert data["kube_context"] == "test-context"
    assert data["kube_context_confirmed"] is True
    assert data["classifier_cost_accepted"] is True
    assert "budget.checks.max_full_usd.ok=False, expected true" in data["error"]


def test_deploy_apply_apply_requires_kube_context_confirmation(tmp_path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    kubectl = fake_bin / "kubectl"
    kubectl.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1 $2\" == \"config current-context\" ]]; then echo prod-context; exit 0; fi\n"
        "exit 0\n"
    )
    kubectl.chmod(0o755)
    report = tmp_path / "deploy-apply-report.json"
    result = subprocess.run(
        ["bash", "scripts/deploy_apply.sh", "--apply"],
        cwd=Path(__file__).resolve().parent.parent,
        env={
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "CHART_DEPLOY_APPLY_REPORT": str(report),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "without CHART_K8S_CONTEXT_CONFIRM=prod-context" in result.stderr
    data = json.loads(report.read_text())
    assert data["status"] == "failed"
    assert data["mode"] == "apply"
    assert data["kube_context"] == "prod-context"
    assert data["kube_context_confirmed"] is False
    assert data["classifier_cost_accepted"] is False
    assert data["error"] == "refusing to apply to Kubernetes context prod-context without CHART_K8S_CONTEXT_CONFIRM=prod-context"


def test_build_gpu_images_dry_runs_manifest_image_builds() -> None:
    script = _script("build_gpu_images.sh")

    assert 'MODE="dry-run"' in script
    assert "docker buildx build" in script
    assert "depot build" in script
    assert "CHART_GPU_BUILDER" in script
    assert "CHART_DEPOT_PROJECT_ID" in script
    assert "CHART_PRELOAD_EMBED_MODEL" in script
    assert "CHART_PRELOAD_EVENTS_MODEL" in script
    assert "CHART_GPU_BUILD_TARGETS" in script
    assert "--target embedder" in script
    assert "--target huggingface-source" in script
    assert "--target classifier" in script
    assert "--platform" in script
    assert "CHART_GPU_PLATFORM" in script
    assert "deploy/Dockerfile.gpu" in script
    assert "layer_client=${layer_context}" in script
    assert "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-embedder-plan-20260624-dedupe2" in script
    assert "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-huggingface-source-plan-20260624-concurrent" in script
    assert "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-classifier-plan-20260624" in script
    assert "chart-embedder:latest" not in script
    assert "chart-classifier:latest" not in script
    assert 'elif [[ "$MODE" == "build" ]]' in script
    assert "cmd_embed+=(--load)" in script
    assert "cmd_source+=(--load)" in script
    assert "cmd_classifier+=(--load)" in script
    assert "--push" in script
    assert '[[ ! -d "$layer_context" ]]' in script
    assert "CHART_LAYER_CLIENT_CONTEXT" in script
    assert "CHART_ECR_REPOSITORY_URL" in script
    assert "CHART_SOURCE_IMAGE" in script
    assert "aws ecr get-login-password" in script
    assert "CHART_GPU_BUILD_REPORT" in script
    assert "gpu-build-report.json" in script
    assert 'write_report "failed"' in script
    assert 'write_report "dry-run"' in script
    assert 'write_report "started"' in script
    assert "source_command" in script
    assert 'write_report "completed"' in script
    assert "docker info >/dev/null" in script


def test_shell_report_writers_use_atomic_rename() -> None:
    scripts = {
        "build_gpu_images.sh": "build_report",
        "deploy_apply.sh": "deploy_report",
        "phase6_unpause_embed.sh": "unpause_report",
        "lib/write_failure_report.sh": "report_path",
    }

    for name, variable in scripts.items():
        script = _script(name)
        assert f'local report_tmp="${{{variable}}}.tmp.$$"' in script
        assert '>"$report_tmp"' in script
        assert 'mv "$report_tmp" "$' + variable + '"' in script


def test_build_gpu_images_dry_run_writes_report(tmp_path) -> None:
    report = tmp_path / "gpu-build-report.json"
    result = subprocess.run(
        ["bash", "scripts/build_gpu_images.sh", "--dry-run"],
        cwd=Path(__file__).resolve().parent.parent,
        env={"PATH": "/usr/bin:/bin", "CHART_GPU_BUILD_REPORT": str(report)},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    data = json.loads(report.read_text())
    assert data["mode"] == "dry-run"
    assert data["status"] == "dry-run"
    assert data["embed_image"] == "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-embedder-plan-20260624-dedupe2"
    assert data["source_image"] == "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-huggingface-source-plan-20260624-concurrent"
    assert data["classifier_image"] == "186219257916.dkr.ecr.us-east-1.amazonaws.com/mesh:chart-classifier-plan-20260624"
    assert data["platform"] == "linux/amd64"
    assert data["builder"] == "docker"
    assert data["targets"] == ["embed", "source", "classifier"]
    assert data["preload_embed_model"] == "Snowflake/snowflake-arctic-embed-m-v1.5"
    assert data["preload_events_model"] == "google/gemma-2-9b-it"
    assert "--target" in data["embed_command"]
    assert "--target" in data["source_command"]
    assert "--target" in data["classifier_command"]
    assert "--platform" in data["embed_command"]
    assert "--platform" in data["source_command"]
    assert "--platform" in data["classifier_command"]


def test_build_gpu_images_ecr_repository_derives_chart_tags(tmp_path) -> None:
    report = tmp_path / "gpu-build-report.json"
    ecr_repo = "123456789012.dkr.ecr.us-east-1.amazonaws.com/chart-workers"
    result = subprocess.run(
        ["bash", "scripts/build_gpu_images.sh", "--dry-run"],
        cwd=Path(__file__).resolve().parent.parent,
        env={
            "PATH": "/usr/bin:/bin",
            "CHART_GPU_BUILD_REPORT": str(report),
            "CHART_ECR_REPOSITORY_URL": ecr_repo,
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    data = json.loads(report.read_text())
    assert data["embed_image"] == f"{ecr_repo}:chart-embedder-plan-20260624-dedupe2"
    assert data["source_image"] == f"{ecr_repo}:chart-huggingface-source-plan-20260624-concurrent"
    assert data["classifier_image"] == f"{ecr_repo}:chart-classifier-plan-20260624"


def test_build_gpu_images_depot_dry_run_writes_depot_commands(tmp_path) -> None:
    report = tmp_path / "gpu-build-report.json"
    result = subprocess.run(
        ["bash", "scripts/build_gpu_images.sh", "--dry-run"],
        cwd=Path(__file__).resolve().parent.parent,
        env={
            "PATH": "/usr/bin:/bin",
            "CHART_GPU_BUILD_REPORT": str(report),
            "CHART_GPU_BUILDER": "depot",
            "CHART_DEPOT_PROJECT_ID": "8zfcn2cf80",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    data = json.loads(report.read_text())
    assert data["builder"] == "depot"
    assert data["depot_project"] == "8zfcn2cf80"
    assert data["embed_command"][:4] == ["depot", "build", "--project", "8zfcn2cf80"]
    assert data["source_command"][:4] == ["depot", "build", "--project", "8zfcn2cf80"]
    assert "--push" not in data["embed_command"]
    assert "--push" not in data["source_command"]


def test_build_gpu_images_embed_only_target_skips_classifier_build(tmp_path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker = fake_bin / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == \"info\" ]]; then exit 0; fi\n"
        "if [[ \"$1\" == \"buildx\" && \"$*\" == *\"--target embedder\"* ]]; then exit 0; fi\n"
        "echo unexpected docker command: $* >&2\n"
        "exit 42\n"
    )
    docker.chmod(0o755)
    aws = fake_bin / "aws"
    aws.write_text("#!/usr/bin/env bash\nprintf password\n")
    aws.chmod(0o755)
    layer_context = tmp_path / "layer-client"
    layer_context.mkdir()
    report = tmp_path / "gpu-build-report.json"

    result = subprocess.run(
        ["bash", "scripts/build_gpu_images.sh", "--push"],
        cwd=Path(__file__).resolve().parent.parent,
        env={
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "CHART_GPU_BUILD_REPORT": str(report),
            "CHART_LAYER_CLIENT_CONTEXT": str(layer_context),
            "CHART_GPU_BUILD_TARGETS": "embed",
            "CHART_ECR_LOGIN": "0",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    data = json.loads(report.read_text())
    assert data["status"] == "completed"
    assert data["targets"] == ["embed"]
    assert "--target" in data["embed_command"]
    assert "embedder" in data["embed_command"]
    assert "--push" in data["embed_command"]
    assert "classifier" in data["classifier_command"]


def test_build_gpu_images_build_failure_writes_report(tmp_path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker = fake_bin / "docker"
    docker.write_text("#!/usr/bin/env bash\nexit 0\n")
    docker.chmod(0o755)
    report = tmp_path / "gpu-build-report.json"
    missing_context = tmp_path / "missing-layer-client"

    result = subprocess.run(
        ["bash", "scripts/build_gpu_images.sh", "--build"],
        cwd=Path(__file__).resolve().parent.parent,
        env={
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "CHART_GPU_BUILD_REPORT": str(report),
            "CHART_LAYER_CLIENT_CONTEXT": str(missing_context),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "missing Layer Python client build context" in result.stderr
    data = json.loads(report.read_text())
    assert data["mode"] == "build"
    assert data["status"] == "failed"
    assert data["layer_context"] == str(missing_context)
    assert "missing Layer Python client build context" in data["error"]


def test_build_gpu_images_command_failure_overwrites_started_report(tmp_path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker = fake_bin / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == \"info\" ]]; then exit 0; fi\n"
        "echo build failed >&2\n"
        "exit 42\n"
    )
    docker.chmod(0o755)
    layer_context = tmp_path / "layer-client"
    layer_context.mkdir()
    report = tmp_path / "gpu-build-report.json"

    result = subprocess.run(
        ["bash", "scripts/build_gpu_images.sh", "--build"],
        cwd=Path(__file__).resolve().parent.parent,
        env={
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "CHART_GPU_BUILD_REPORT": str(report),
            "CHART_LAYER_CLIENT_CONTEXT": str(layer_context),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "embed image build failed" in result.stderr
    data = json.loads(report.read_text())
    assert data["mode"] == "build"
    assert data["status"] == "failed"
    assert data["error"] == "embed image build failed"
    assert "--load" in data["embed_command"]


def test_phase6_unpause_embed_requires_explicit_yes_and_checks_live_targets() -> None:
    script = _script("phase6_unpause_embed.sh")

    assert 'MODE="status"' in script
    assert '"${1:-}" == "--yes"' in script
    assert "kubectl -n \"$namespace\" get secret chart-gateway >/dev/null" in script
    assert "kubectl -n \"$namespace\" get pipeline \"$pipeline_name\" >/dev/null" in script
    assert "expected_pipeline_id=\"${CHART_EMBED_PIPELINE_ID:-chart-notes}\"" in script
    assert "jsonpath='{.spec.pipelineId}'" in script
    assert '[[ "$actual_pipeline_id" != "$expected_pipeline_id" ]]' in script
    assert "refusing to unpause $namespace/$pipeline_name: spec.pipelineId=$actual_pipeline_id, expected $expected_pipeline_id" in script
    assert "CHART_ACCEPT_PHASE6_EMBED_COST" in script
    assert "CHART_K8S_CONTEXT_CONFIRM" in script
    assert "kubectl config current-context" in script
    assert "indexer.measure_embed" in script
    assert "budget_report=\"${CHART_PHASE6_EMBED_BUDGET_REPORT:-eval/out/embed-budget.json}\"" in script
    assert "unpause_report=\"${CHART_PHASE6_UNPAUSE_REPORT:-eval/out/phase6-unpause-report.json}\"" in script
    assert 'write_report "failed"' in script
    assert 'write_report "unpaused"' in script
    assert '[[ ! -f "$budget_report" ]]' in script
    assert "embed budget report is not accepted" in script
    assert "embed budget report has failing budget checks" in script
    assert "embed budget report must be measured on gpu accelerator" in script
    assert "max_full_hours" in script
    assert "max_full_usd" in script
    assert "patch pipeline \"$pipeline_name\" --type=merge" in script
    assert "'{\"spec\":{\"paused\":false}}'" in script
    assert "CHART_EMBED_PIPELINE_CR" in script
    assert "CHART_K8S_NAMESPACE" in script


def test_phase6_unpause_embed_refusal_writes_report(tmp_path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    kubectl = fake_bin / "kubectl"
    kubectl.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1 $2\" == \"config current-context\" ]]; then echo prod-context; exit 0; fi\n"
        "exit 0\n"
    )
    kubectl.chmod(0o755)
    report = tmp_path / "phase6-unpause-report.json"

    result = subprocess.run(
        ["bash", "scripts/phase6_unpause_embed.sh", "--yes"],
        cwd=Path(__file__).resolve().parent.parent,
        env={
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "CHART_PHASE6_UNPAUSE_REPORT": str(report),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "without CHART_K8S_CONTEXT_CONFIRM=prod-context" in result.stderr
    data = json.loads(report.read_text())
    assert data["mode"] == "unpause"
    assert data["status"] == "failed"
    assert data["namespace"] == "chart"
    assert data["pipeline_cr"] == "chart-embed-gpu"
    assert data["expected_pipeline_id"] == "chart-notes"
    assert data["error"] == (
        "refusing to unpause chart/chart-embed-gpu on Kubernetes context prod-context "
        "without CHART_K8S_CONTEXT_CONFIRM=prod-context"
    )


def test_phase6_prepare_embed_drain_pauses_source_and_checks_claimability() -> None:
    script = _script("phase6_prepare_embed_drain.sh")

    assert 'MODE="status"' in script
    assert '"${1:-}" == "--yes"' in script
    assert "source scripts/lib/resolve_gateway_key.sh" in script
    assert "resolve_gateway_key >/dev/null" in script
    assert "get secret chart-gateway" in script
    assert "base64 --decode" in script
    assert "echo \"$LAYER_GATEWAY_API_KEY\"" not in script
    assert "CHART_K8S_CONTEXT_CONFIRM" in script
    assert "kubectl config current-context" in script
    assert "kubectl -n \"$namespace\" scale deployment/\"$ingest_deployment\" --replicas=0" in script
    assert "CHART_INGEST_DEPLOYMENT" in script
    assert "CHART_LAYER_NAMESPACE" in script
    assert "CHART_LAYER_GATEWAY_POD" in script
    assert "CHART_LAYER_POSTGRES_CONTAINER" in script
    assert "CHART_PHASE6_DRAIN_REPORT" in script
    assert "pg_terminate_backend(pid)" in script
    assert "state='idle in transaction'" in script
    assert "query like 'SELECT id, stage, manifest_key FROM pipeline_segments%'" in script
    assert "claim_documents" in script
    assert "release_documents" in script
    assert "pipeline $pipeline_id has pending documents but claim check returned no documents" in script


def test_phase6_prepare_embed_drain_refusal_writes_report(tmp_path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    kubectl = fake_bin / "kubectl"
    kubectl.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1 $2\" == \"-n layer\" ]]; then echo 0; exit 0; fi\n"
        "if [[ \"$1 $2\" == \"config current-context\" ]]; then echo prod-context; exit 0; fi\n"
        "exit 0\n"
    )
    kubectl.chmod(0o755)
    report = tmp_path / "phase6-drain-report.json"

    result = subprocess.run(
        ["bash", "scripts/phase6_prepare_embed_drain.sh", "--yes"],
        cwd=Path(__file__).resolve().parent.parent,
        env={
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "CHART_PHASE6_DRAIN_REPORT": str(report),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "without CHART_K8S_CONTEXT_CONFIRM=prod-context" in result.stderr
    data = json.loads(report.read_text())
    assert data["mode"] == "prepare"
    assert data["status"] == "failed"
    assert data["namespace"] == "chart"
    assert data["pipeline_id"] == "chart-notes"
    assert data["ingest_deployment"] == "chart-ingest-worker"
    assert data["error"] == (
        "refusing to prepare embed drain on Kubernetes context prod-context "
        "without CHART_K8S_CONTEXT_CONFIRM=prod-context"
    )


def test_phase6_stage_source_window_requires_offset_and_pending_guard() -> None:
    script = _script("phase6_stage_source_window.sh")

    assert 'MODE="status"' in script
    assert '"${1:-}" == "--yes"' in script
    assert "CHART_PHASE6_SOURCE_START_OFFSET" in script
    assert "CHART_PHASE6_SOURCE_MAX_ROWS" in script
    assert "CHART_PHASE6_SOURCE_PAGE_SIZE" in script
    assert "CHART_PHASE6_STAGE_REPORT" in script
    assert "CHART_PHASE6_ALLOW_STAGE_WITH_PENDING" in script
    assert "CHART_K8S_CONTEXT_CONFIRM" in script
    assert "kubectl config current-context" in script
    assert "CHART_HF_SOURCE_START_OFFSET=\"$start_offset\"" in script
    assert "CHART_HF_SOURCE_MAX_ROWS=\"$max_rows\"" in script
    assert "CHART_HF_SOURCE_PAGE_SIZE=\"$page_size\"" in script
    assert "scale deployment/\"$ingest_deployment\" --replicas=1" in script
    assert "CHART_PHASE6_SOURCE_START_OFFSET is required for deterministic source staging" in script
    assert "drain first or set CHART_PHASE6_ALLOW_STAGE_WITH_PENDING=1" in script


def test_phase6_stage_source_window_refusal_writes_report(tmp_path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    kubectl = fake_bin / "kubectl"
    kubectl.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1 $2\" == \"-n layer\" ]]; then echo 0; exit 0; fi\n"
        "if [[ \"$1 $2\" == \"-n chart\" && \"$3\" == \"get\" ]]; then echo 0; exit 0; fi\n"
        "if [[ \"$1 $2\" == \"config current-context\" ]]; then echo prod-context; exit 0; fi\n"
        "exit 0\n"
    )
    kubectl.chmod(0o755)
    report = tmp_path / "phase6-stage-report.json"

    result = subprocess.run(
        ["bash", "scripts/phase6_stage_source_window.sh", "--yes"],
        cwd=Path(__file__).resolve().parent.parent,
        env={
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "CHART_PHASE6_STAGE_REPORT": str(report),
            "CHART_K8S_CONTEXT_CONFIRM": "prod-context",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "CHART_PHASE6_SOURCE_START_OFFSET is required" in result.stderr
    data = json.loads(report.read_text())
    assert data["mode"] == "stage"
    assert data["status"] == "failed"
    assert data["pipeline_id"] == "chart-notes"
    assert data["ingest_deployment"] == "chart-ingest-worker"
