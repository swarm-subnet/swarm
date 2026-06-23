"""Screening early-stop policy: fair loser rejection and champion-copy detection.

The decision logic is pure functions over per-seed score lists so it can be
unit-tested without the Docker/RPC stack. Screening scores are cached on disk
keyed by model hash + benchmark version, so the current champion's per-seed
scores are available for the copy check without a re-evaluation and survive
validator restarts.
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, Optional, Sequence, Tuple

from swarm.constants import (
    BENCHMARK_VERSION,
    COPY_CORR_MIN,
    COPY_MEAN_MAX,
    COPY_MIN_SEEDS,
    COPY_SD_MAX,
    SCREENING_EARLY_STOP_SIGMA_FLOOR,
)

from ._shared import STATE_DIR

_CACHE_FILE = STATE_DIR / "screening_seed_scores.json"


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _sample_sd(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def cannot_reach_bar(scores: Sequence[float], threshold: float, z: float) -> bool:
    """Return True when even an optimistic bound on the mean stays below the bar."""
    n = len(scores)
    if n == 0:
        return False
    sigma = max(_sample_sd(scores), SCREENING_EARLY_STOP_SIGMA_FLOOR)
    return _mean(scores) + z * sigma / math.sqrt(n) < threshold


def copy_metrics(
    candidate: Sequence[float], champion: Sequence[float]
) -> Tuple[float, float, float]:
    """Return (correlation, SD of gap, mean gap) for candidate vs champion per seed."""
    n = len(candidate)
    diffs = [candidate[i] - champion[i] for i in range(n)]
    mean_diff = _mean(diffs) if n else 0.0
    sd_diff = _sample_sd(diffs)
    sd_c = _sample_sd(candidate)
    sd_ch = _sample_sd(champion)
    if n < 2 or sd_c == 0.0 or sd_ch == 0.0:
        return 0.0, sd_diff, mean_diff
    mean_c = _mean(candidate)
    mean_ch = _mean(champion)
    cov = sum(
        (candidate[i] - mean_c) * (champion[i] - mean_ch) for i in range(n)
    ) / (n - 1)
    return cov / (sd_c * sd_ch), sd_diff, mean_diff


def is_blatant_copy(corr: float, sd_diff: float, mean_diff: float, n: int) -> bool:
    """Conservative gate that fires only for near-identical clones of the champion.

    The ``mean_diff <= COPY_MEAN_MAX`` (0.002) condition is well below the dynamic
    crowning floor, so a model that genuinely improves on the champion is never
    flagged — only one that tracks it within noise.
    """
    return (
        n >= COPY_MIN_SEEDS
        and corr >= COPY_CORR_MIN
        and sd_diff <= COPY_SD_MAX
        and mean_diff <= COPY_MEAN_MAX
    )


def _load_cache() -> Dict[str, dict]:
    try:
        with open(_CACHE_FILE) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_cache(cache: Dict[str, dict]) -> None:
    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _CACHE_FILE.with_suffix(f".{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(cache, f)
    tmp.replace(_CACHE_FILE)


def cache_screening_seed_scores(
    *,
    model_hash: str,
    epoch: int,
    benchmark_version: str,
    seeds: Sequence[int],
    scores: Sequence[float],
) -> None:
    """Persist a model's screening-range per-seed scores for later copy checks.

    Keyed by model hash + version so whichever model is champion already has its
    scores available with no re-evaluation. Only the current epoch is kept;
    partial coverage (a resumed run) is merged.
    """
    if not model_hash or not seeds:
        return
    cache = _load_cache()
    key = str(model_hash)
    entry = cache.get(key)
    if (
        entry is None
        or entry.get("epoch") != epoch
        or entry.get("version") != benchmark_version
    ):
        entry = {"epoch": epoch, "version": benchmark_version, "scores": {}}
    for seed, score in zip(seeds, scores):
        entry["scores"][str(int(seed))] = float(score)
    cache[key] = entry
    cache = {k: v for k, v in cache.items() if v.get("epoch") == epoch}
    _save_cache(cache)


def champion_seed_reference(
    validator, epoch: int, seeds: Sequence[int]
) -> Optional[Dict[int, float]]:
    """Per-seed scores of the current champion for ``seeds``.

    Resolved from the backend sync state. Returns None when there is no trusted
    reference (no champion, or its screening scores are not cached for this epoch
    and version) — in which case the copy check safely no-ops.
    """
    try:
        backend = getattr(validator, "backend_api", None)
        runtime = getattr(backend, "_runtime_state", {}) or {}
        current = runtime.get("current_top") or {}
        model_hash = current.get("model_hash")
        if not model_hash:
            return None
        entry = _load_cache().get(str(model_hash))
        if (
            not isinstance(entry, dict)
            or entry.get("epoch") != epoch
            or entry.get("version") != BENCHMARK_VERSION
        ):
            return None
        stored = entry.get("scores", {})
        if not isinstance(stored, dict):
            return None
        return {int(s): stored[str(int(s))] for s in seeds if str(int(s)) in stored}
    except Exception:
        return None
