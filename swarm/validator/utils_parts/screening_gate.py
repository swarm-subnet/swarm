"""Screening early-stop policy: fair loser rejection and champion-copy detection.

The decision logic is expressed as pure functions over per-seed score lists so it
can be unit-tested without the Docker/RPC evaluation stack. The champion reference
cache lets a validator compare a candidate against the champion on the same seeds.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

from swarm.constants import (
    BENCHMARK_VERSION,
    COPY_CORR_MIN,
    COPY_MEAN_MAX,
    COPY_MIN_SEEDS,
    COPY_SD_MAX,
    SCREENING_EARLY_STOP_SIGMA_FLOOR,
)


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
    """Conservative gate that fires only for near-identical clones of the champion."""
    return (
        n >= COPY_MIN_SEEDS
        and corr >= COPY_CORR_MIN
        and sd_diff <= COPY_SD_MAX
        and mean_diff <= COPY_MEAN_MAX
    )


def record_champion_seed_scores(
    validator,
    *,
    family_id: str,
    epoch: int,
    model_hash: str,
    benchmark_version: str,
    seeds: Sequence[int],
    scores: Sequence[float],
) -> None:
    """Cache champion per-seed scores from a REEVAL for later copy checks.

    Entries are replaced when the (epoch, model_hash, version) key changes and
    merged otherwise, so a resumed/partial REEVAL accumulates coverage.
    """
    cache: Dict[str, dict] = getattr(validator, "_champion_seed_scores", None) or {}
    entry = cache.get(family_id)
    if (
        entry is None
        or entry.get("epoch") != epoch
        or entry.get("model_hash") != model_hash
        or entry.get("version") != benchmark_version
    ):
        entry = {
            "epoch": epoch,
            "model_hash": model_hash,
            "version": benchmark_version,
            "scores": {},
        }
        cache[family_id] = entry
    for seed, score in zip(seeds, scores):
        entry["scores"][int(seed)] = float(score)
    validator._champion_seed_scores = cache


def champion_reference(
    validator, family_id: str, epoch: int, seeds: Sequence[int]
) -> Optional[Dict[int, float]]:
    """Champion per-seed scores for ``seeds``, or None when no trusted reference exists.

    The cache is only used when it belongs to this epoch and matches the current
    champion of ``family_id`` reported by the backend, so a stale or mid-epoch
    dethroned champion is never compared against.
    """
    cache = getattr(validator, "_champion_seed_scores", None)
    if not cache:
        return None
    entry = cache.get(family_id)
    if (
        entry is None
        or entry.get("epoch") != epoch
        or entry.get("version") != BENCHMARK_VERSION
    ):
        return None
    current = getattr(validator.backend_api, "current_top", {}) or {}
    if str(current.get("family_id") or "") != family_id:
        return None
    if not entry.get("model_hash") or entry["model_hash"] != current.get("model_hash"):
        return None
    stored = entry.get("scores", {})
    return {int(s): stored[int(s)] for s in seeds if int(s) in stored}
