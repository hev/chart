from __future__ import annotations

import os
import subprocess
from typing import Any


def runtime_report(accelerator: str) -> dict[str, Any]:
    report: dict[str, Any] = {"accelerator": accelerator}
    if accelerator != "gpu":
        return report
    env_device = os.environ.get("CHART_GPU_DEVICE_NAME")
    if env_device:
        report.update({"gpu_device": env_device, "gpu_probe": "CHART_GPU_DEVICE_NAME"})
        return report
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        report["gpu_probe"] = "missing"
        return report
    device = proc.stdout.strip().splitlines()[0].strip() if proc.returncode == 0 and proc.stdout.strip() else ""
    if device:
        report.update({"gpu_device": device, "gpu_probe": "nvidia-smi"})
    else:
        report["gpu_probe"] = "missing"
    return report


def runtime_error(report: dict[str, Any]) -> str | None:
    if report.get("accelerator") == "gpu" and not report.get("gpu_device"):
        return "GPU accelerator was requested but no GPU device could be verified"
    return None
