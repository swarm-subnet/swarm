"""Reference-time normalization for hardware-fair scoring.

Each validator measures itself against a single fixed baseline model and converts
every miner ``act()`` time into baseline-equivalent time, so the score does not
depend on how fast the validator's host is.

    speed_factor = local_p90 / owner_p90

The factor is a property of the host, not of any task. It is measured once per
worker on the committed baseline model and then applied to every miner across all
challenge families. The baseline model's family and challenge type only describe
how the timing workload is run; they do not scope where the factor is used.
"""
from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from swarm.constants import SPEED_FACTOR_MAX_ELIGIBLE, SPEED_FACTOR_MIN
from swarm.utils.hash import sha256sum

_CALIBRATION_DIR = Path(__file__).resolve().parent
_MANIFEST_PATH = _CALIBRATION_DIR / "baseline_manifest.json"
_REQUIRED_KEYS = ("calibration_version", "baseline_model", "owner_compute_p90_ms")


def load_baseline_manifest() -> dict:
    data = json.loads(_MANIFEST_PATH.read_text())
    missing = [key for key in _REQUIRED_KEYS if key not in data]
    if missing:
        raise ValueError(f"baseline manifest missing keys: {missing}")
    return data


def baseline_model_path() -> Path:
    manifest = load_baseline_manifest()
    return _CALIBRATION_DIR / manifest["baseline_model"]["artifact"]


def baseline_model_available() -> bool:
    """True if the committed baseline model is present and matches the manifest hash."""
    try:
        manifest = load_baseline_manifest()
        path = _CALIBRATION_DIR / manifest["baseline_model"]["artifact"]
        if not path.is_file():
            return False
        return sha256sum(path) == manifest["baseline_model"]["sha256"]
    except Exception:
        return False


@dataclass(frozen=True)
class SpeedFactor:
    raw: float           # local_p90 / owner_p90, unbounded
    factor: float        # value used for scoring (low-guarded, never upper-clamped)
    eligible: bool       # False -> host is too slow to score fairly and must self-exclude
    owner_p90_ms: float
    local_p90_ms: float


def percentile(values, pct: float) -> float:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return 0.0
    k = (len(ordered) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def normalize_speed_factor(
    local_p90_ms: float, *, owner_p90_ms: Optional[float] = None
) -> SpeedFactor:
    """Turn a measured local p90 act-time into a host speed factor.

    The raw ratio is used for scoring (only guarded from below against an invalid
    measurement); it is never clamped from above, so a host slower than the
    eligibility limit is flagged rather than silently treated as borderline.
    """
    if owner_p90_ms is None:
        owner_p90_ms = float(load_baseline_manifest()["owner_compute_p90_ms"])
    if owner_p90_ms <= 0:
        raise ValueError("owner_compute_p90_ms must be positive")

    raw = float(local_p90_ms) / float(owner_p90_ms)
    if not math.isfinite(raw) or raw <= 0:
        raise ValueError(f"invalid speed factor from local_p90_ms={local_p90_ms}")

    return SpeedFactor(
        raw=raw,
        factor=max(raw, SPEED_FACTOR_MIN),
        eligible=raw <= SPEED_FACTOR_MAX_ELIGIBLE,
        owner_p90_ms=float(owner_p90_ms),
        local_p90_ms=float(local_p90_ms),
    )


@dataclass
class CalibrationEntry:
    speed: SpeedFactor
    overhead_ms: float
    calibration_version: str
    computed_at: float


class CalibrationState:
    """Per-process cache of the host speed factor, keyed by worker id.

    Each Docker worker is pinned to its own cpuset, so the factor is stored per
    worker and never shared across processes.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_worker: Dict[int, CalibrationEntry] = {}

    def get(self, worker_id: int) -> Optional[CalibrationEntry]:
        with self._lock:
            return self._by_worker.get(int(worker_id))

    def set(
        self,
        worker_id: int,
        speed: SpeedFactor,
        overhead_ms: float,
        calibration_version: str,
    ) -> CalibrationEntry:
        entry = CalibrationEntry(
            speed=speed,
            overhead_ms=float(overhead_ms),
            calibration_version=str(calibration_version),
            computed_at=time.time(),
        )
        with self._lock:
            self._by_worker[int(worker_id)] = entry
        return entry

    def is_stale(self, worker_id: int, *, max_age_sec: float) -> bool:
        entry = self.get(worker_id)
        if entry is None:
            return True
        return (time.time() - entry.computed_at) > max_age_sec


CALIBRATION_STATE = CalibrationState()
