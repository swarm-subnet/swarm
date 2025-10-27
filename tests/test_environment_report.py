from __future__ import annotations

import importlib
import json
import os
import platform
import subprocess
import sys
from typing import Any, Dict

import pytest


def _module_report(name: str, *, attr: str = "__version__") -> Dict[str, Any]:
    """Return a simple serialisable report for *name* module."""
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # pragma: no cover
        return {
            "status": "missing",
            "error": repr(exc),
        }

    version = getattr(module, attr, None)
    extras: Dict[str, Any] = {}

    if name == "pybullet":
        api_version = None
        get_api_version = getattr(module, "getAPIVersion", None)
        if callable(get_api_version):
            try:
                api_version = int(get_api_version())
            except Exception:
                api_version = "error"
        extras["api_version"] = api_version
        # Newer pybullet exposes __version__, older builds don't.
        if version is None:
            version = getattr(module, "PYBULLET_VERSION", None)

    if name == "torch":
        extras["cuda_available"] = bool(getattr(module.cuda, "is_available", lambda: False)())
        extras["cuda_device_count"] = int(getattr(module.cuda, "device_count", lambda: 0)())
        deterministic = getattr(module, "are_deterministic_algorithms_enabled", None)
        if callable(deterministic):
            try:
                extras["deterministic_algorithms"] = bool(deterministic())
            except Exception:
                extras["deterministic_algorithms"] = "error"

    if name == "gym_pybullet_drones":
        version = getattr(module, "__version__", None)
        extras["available_envs"] = sorted(
            getattr(module, "__all__", [])  # type: ignore[arg-type]
        )

    return {
        "status": "ok",
        "version": str(version) if version is not None else None,
        **({ "extras": extras } if extras else {}),
    }


def _pip_freeze(limit: int = 40) -> Dict[str, Any]:
    """Return the first *limit* lines of `pip freeze` for quick diffing."""
    try:
        output = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=30,
        )
    except Exception as exc:  # pragma: no cover
        return {
            "status": "error",
            "error": repr(exc),
        }
    lines = output.strip().splitlines()
    summary = lines[:limit]
    truncated = len(lines) > limit
    return {
        "status": "ok",
        "sample": summary,
        "total": len(lines),
        "truncated": truncated,
    }


def _collect_env_report() -> Dict[str, Any]:
    env_keys = [
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "PYTHONHASHSEED",
        "PYTORCH_DETERMINISTIC",
        "PYTORCH_ENABLE_MPS_FALLBACK",
    ]
    env_snapshot = {k: os.environ.get(k) for k in env_keys if k in os.environ}

    modules = [
        "pybullet",
        "gym_pybullet_drones",
        "numpy",
        "torch",
        "stable_baselines3",
        "gymnasium",
        "gym",
        "bittensor",
    ]

    module_reports = {name: _module_report(name) for name in modules}

    report: Dict[str, Any] = {
        "python": {
            "version": sys.version.replace("\n", " "),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "environment_variables": env_snapshot,
        "module_versions": module_reports,
        "pip_freeze": _pip_freeze(limit=50),
    }
    return report


@pytest.mark.integration
def test_environment_report():
    """Emit environment diagnostics for cross-machine reproducibility checks."""
    info = _collect_env_report()
    serialized = json.dumps(info, indent=2, sort_keys=True)
    print("=== environment report ===")
    print(serialized)
    assert info["python"]["version"], "Python version unavailable"
