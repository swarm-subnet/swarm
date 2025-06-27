"""
Shared helper utilities for the validator.
(Additions: save_flightplans + NumPy‑safe JSON handling)
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import bittensor as bt
import dataclasses
import numpy as np                      # ← NEW
from swarm.constants import SAVE_FLIGHTPLANS
from swarm.protocol import MapTask, FlightPlan, ValidationResult
from swarm.utils.logging import ColoredLogger  # keep existing import

import copy
import random
from typing import List as _List                     # avoid shadowing below


# ──────────────────────────────────────────────────────────────────────────────
# Helpers for JSON serialisation
# ──────────────────────────────────────────────────────────────────────────────
class _NumpyEncoder(json.JSONEncoder):
    """Automatically cast NumPy scalars / arrays to built‑ins."""

    def default(self, obj):  # noqa: D401
        # NumPy scalar ➜ Python scalar
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        # Small arrays ➜ list
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # pathlib.Path ➜ str
        if isinstance(obj, Path):
            return str(obj)
        # datetime ➜ ISO 8601
        if isinstance(obj, datetime):
            return obj.isoformat()

        return super().default(obj)


def _to_dict(obj):
    """
    Best‑effort conversion of dataclass / pydantic / generic object -> dict
    (Recursive conversion, incl. NumPy scalars.)
    """
    # numpy scalar / array
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # dataclass
    if dataclasses.is_dataclass(obj):
        return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}

    # pydantic (v1 or v2)
    for attr in ("dict", "model_dump"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            return {k: _to_dict(v) for k, v in fn(exclude_none=True).items()}

    # mapping
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}

    # iterable
    if isinstance(obj, (list, tuple)):
        return [_to_dict(v) for v in obj]

    # primitives
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj

    # fallback
    return str(obj)


# ──────────────────────────────────────────────────────────────────────────────
# New helper: save_flightplans (patched to be NumPy‑safe)
# ──────────────────────────────────────────────────────────────────────────────
def save_flightplans(
    task: MapTask,
    results: List[ValidationResult],
    plans: Dict[int, FlightPlan],
    max_files: int = 100,
) -> None:
    """
    Persist full FlightPlans + scores to JSON if SAVE_FLIGHTPLANS == True.

    • Creates “flightplans/” folder at repo root (if missing)
    • File name:  flightplans_<UTC‑YYYYmmdd_HHMMSS>.json
    • Keeps only the *latest* `max_files` files – older ones auto‑deleted.
    """
    if not SAVE_FLIGHTPLANS:
        return  # feature disabled – no‑op

    try:
        # ── Determine repo root (../.. from this file) & ensure folder ─────────
        root_path = Path(__file__).resolve().parents[2]   # <repo>/
        fp_dir    = root_path / "flightplans"
        fp_dir.mkdir(parents=True, exist_ok=True)

        # ── Build serializable payload ────────────────────────────────────────
        # sort “bottom ➜ top” (ascending score)
        sorted_res = sorted(results, key=lambda r: r.score)
        timestamp  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        payload = {
            "timestamp": timestamp,
            "task": _to_dict(task),
            "flightplans": [
                {
                    "uid"      : int(r.uid),
                    "score"    : float(r.score),
                    "success"  : bool(r.success),
                    "time_sec" : float(r.time_sec),
                    "energy"   : float(r.energy),
                    "plan"     : _to_dict(plans.get(r.uid)),  # full plan
                }
                for r in sorted_res
            ],
        }

        # ── Dump to file ──────────────────────────────────────────────────────
        filename = fp_dir / f"flightplans_{timestamp}.json"
        with open(filename, "w", encoding="utf‑8") as fh:
            json.dump(
                payload,
                fh,
                indent=2,
                ensure_ascii=False,
                cls=_NumpyEncoder,           # ← fixes NumPy scalars
            )
        bt.logging.info(f"[save_flightplans] Stored: {filename.relative_to(root_path)}")

        # ── Enforce retention window (keep newest `max_files`) ────────────────
        json_files = sorted(
            fp_dir.glob("flightplans_*.json"), key=lambda p: p.stat().st_mtime
        )
        if len(json_files) > max_files:
            to_delete = json_files[: len(json_files) - max_files]
            for old in to_delete:
                try:
                    old.unlink()
                    bt.logging.info(f"[save_flightplans] Removed old file: {old.name}")
                except Exception as e:
                    bt.logging.warning(
                        f"[save_flightplans] Could not delete {old}: {type(e).__name__} – {e}"
                    )

    except Exception as e:
        bt.logging.error(f"[save_flightplans] Unexpected error: {type(e).__name__} – {e}")
