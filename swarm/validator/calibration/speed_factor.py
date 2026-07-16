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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

from swarm.constants import SPEED_FACTOR_MAX_ELIGIBLE, SPEED_FACTOR_MIN
from swarm.utils.hash import sha256sum
from swarm.model_graph import EXECUTION_PROFILE_ID, RUNNER_ABI, admit_artifact, profile_digest

_CALIBRATION_DIR = Path(__file__).resolve().parent
_MANIFEST_PATH = _CALIBRATION_DIR / "baseline_manifest.json"
_CACHE_PATH = Path(__file__).resolve().parents[2] / "state" / "calibration_cache.json"
_REQUIRED_KEYS = (
    "calibration_version", "baseline_model", "owner_compute_p90_ms",
    "interface_version", "execution_profile", "execution_profile_digest", "runner_abi",
)


def load_baseline_manifest() -> dict:
    data = json.loads(_MANIFEST_PATH.read_text())
    missing = [key for key in _REQUIRED_KEYS if key not in data]
    if missing:
        raise ValueError(f"baseline manifest missing keys: {missing}")
    if data["interface_version"] != "model_graph.v1":
        raise ValueError("baseline interface mismatch")
    if data["execution_profile"] != EXECUTION_PROFILE_ID:
        raise ValueError("baseline execution profile mismatch")
    if data["execution_profile_digest"] != profile_digest():
        raise ValueError("baseline execution profile digest mismatch")
    if data["runner_abi"] != RUNNER_ABI:
        raise ValueError("baseline runner ABI mismatch")
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
        return (
            sha256sum(path) == manifest["baseline_model"]["sha256"]
            and admit_artifact(path).accepted
        )
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
    """Cache of the host speed factor, keyed by worker id.

    Each Docker worker is pinned to its own cpuset, so the factor is stored per
    worker. The factor is a property of the host, so it is also persisted to
    disk and survives a restart until it ages out or the baseline changes.
    """

    def __init__(self, cache_path: Path = _CACHE_PATH) -> None:
        self._lock = threading.Lock()
        self._by_worker: Dict[int, CalibrationEntry] = {}
        self._cache_path = Path(cache_path)
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            payload = json.loads(self._cache_path.read_text())
            version = load_baseline_manifest()["calibration_version"]
        except Exception:
            return
        for key, row in (payload or {}).items():
            try:
                if row["calibration_version"] != version:
                    continue
                self._by_worker[int(key)] = CalibrationEntry(
                    speed=SpeedFactor(**row["speed"]),
                    overhead_ms=float(row["overhead_ms"]),
                    calibration_version=str(row["calibration_version"]),
                    computed_at=float(row["computed_at"]),
                )
            except Exception:
                continue

    def _save(self) -> None:
        payload = {
            str(worker_id): {
                "speed": asdict(entry.speed),
                "overhead_ms": entry.overhead_ms,
                "calibration_version": entry.calibration_version,
                "computed_at": entry.computed_at,
            }
            for worker_id, entry in self._by_worker.items()
        }
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.replace(self._cache_path)
        except OSError:
            pass

    def get(self, worker_id: int) -> Optional[CalibrationEntry]:
        with self._lock:
            self._load()
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
            self._load()
            self._by_worker[int(worker_id)] = entry
            self._save()
        return entry

    def is_stale(self, worker_id: int, *, max_age_sec: float) -> bool:
        entry = self.get(worker_id)
        if entry is None:
            return True
        return (time.time() - entry.computed_at) > max_age_sec


CALIBRATION_STATE = CalibrationState()
