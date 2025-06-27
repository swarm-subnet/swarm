"""
Shared helper utilities for the validator.
(Additions: save_flightplans)
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import bittensor as bt
import dataclasses

from swarm.constants import SAVE_FLIGHTPLANS          # bool flag
from swarm.protocol import MapTask, FlightPlan, ValidationResult
from swarm.utils.logging import ColoredLogger          

import copy
import random
from typing import List as _List                     # avoid shadowing below


# ──────────────────────────────────────────────────────────────────────────────
# New helper: save_flightplans
# ──────────────────────────────────────────────────────────────────────────────
def _to_dict(obj):
    """
    Best‑effort conversion of dataclass / pydantic / generic object -> dict
    (Recursive conversion for dataclass fields.)
    """
    # dataclass
    if dataclasses.is_dataclass(obj):
        return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}

    # pydantic (v1 or v2) – accounts for both .dict() & .model_dump()
    for attr in ("dict", "model_dump"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            return fn(exclude_none=True)

    # hasattr __dict__
    if hasattr(obj, "__dict__"):
        return {k: _to_dict(v) for k, v in obj.__dict__.items()}

    # builtin (str, int, float, list, tuple, dict, None, bool)
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}

    # Fallback – stringify
    return str(obj)


def save_flightplans(
    task: MapTask,
    results: List[ValidationResult],
    plans: Dict[int, FlightPlan],
    max_files: int = 100,
) -> None:
    """
    Persist full FlightPlans + scores to JSON if SAVE_FLIGHTPLANS == True.

    • Creates “flightplans/” folder at repo root (if missing)
    • File name:  flightplans_<YYYYmmdd_HHMMSS>.json
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
                    "uid"      : r.uid,
                    "score"    : r.score,
                    "success"  : r.success,
                    "time_sec" : r.time_sec,
                    "energy"   : r.energy,
                    "plan"     : _to_dict(plans.get(r.uid)),  # full plan
                }
                for r in sorted_res
            ],
        }

        # ── Dump to file ──────────────────────────────────────────────────────
        filename = fp_dir / f"flightplans_{timestamp}.json"
        with open(filename, "w", encoding="utf‑8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
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
