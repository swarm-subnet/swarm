"""
Shared helper utilities for the validator.
(Additions: NumPy‑safe JSON handling)
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
from swarm.protocol import MapTask, ValidationResult
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

