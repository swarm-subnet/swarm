"""Compatibility entrypoint for the deployable agent.

The concrete baseline/final agents live under ``training_pipeline/final_agents``.
Keeping this file preserves the familiar top-level entrypoint while the
training-oriented workspace stays organized.

This loader is explicit so it still works in packaged submissions where only a
subset of repo files may be present.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_controller_class():
    here = Path(__file__).resolve().parent
    candidate = here / "final_agents" / "drone_agent.py"
    if not candidate.exists():
        raise FileNotFoundError(f"Expected final agent file at {candidate}")

    spec = importlib.util.spec_from_file_location("training_pipeline_final_agent", candidate)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {candidate}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "DroneFlightController"):
        raise AttributeError(f"{candidate} does not define DroneFlightController")
    return module.DroneFlightController


DroneFlightController = _load_controller_class()

__all__ = ["DroneFlightController"]
