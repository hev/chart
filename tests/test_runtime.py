from chart_common.runtime import runtime_error, runtime_report


def test_runtime_report_records_gpu_device_from_env(monkeypatch) -> None:
    monkeypatch.setenv("CHART_GPU_DEVICE_NAME", "NVIDIA L4")

    assert runtime_report("gpu") == {
        "accelerator": "gpu",
        "gpu_device": "NVIDIA L4",
        "gpu_probe": "CHART_GPU_DEVICE_NAME",
    }


def test_runtime_report_marks_missing_gpu_probe(monkeypatch) -> None:
    monkeypatch.delenv("CHART_GPU_DEVICE_NAME", raising=False)
    monkeypatch.setattr("chart_common.runtime.subprocess.run", lambda *args, **kwargs: (_ for _ in ()).throw(OSError()))

    assert runtime_report("gpu") == {"accelerator": "gpu", "gpu_probe": "missing"}


def test_runtime_error_requires_gpu_device_for_gpu_accelerator() -> None:
    assert runtime_error({"accelerator": "gpu", "gpu_probe": "missing"}) == (
        "GPU accelerator was requested but no GPU device could be verified"
    )
    assert runtime_error({"accelerator": "gpu", "gpu_device": "NVIDIA L4"}) is None
    assert runtime_error({"accelerator": "cpu"}) is None
