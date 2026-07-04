"""King of the Hill emissions split.

Canonical formula for the V5.0.0 KotH mechanism. Both the validator and the
backend import this module, so the spec at swarm/docs/king_of_the_hill.md
and the runtime behaviour agree by construction.

For each king ``i`` in the active window:

    gain_i   = log(max(1 - prev_i, HEADROOM_EPS) / max(1 - score_i, HEADROOM_EPS))
    taper_i  = max(0, (WINDOW_SIZE - rank_i) / WINDOW_SIZE)
    share_i  = gain_i * taper_i / sum(gain_j * taper_j over the window)

Scores are clamped to [0, 1] first. Rank is derived from
``(crowned_at_epoch, lineage_id)`` descending, not list position.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional


HEADROOM_EPS: float = 0.01
WINDOW_SIZE: int = 5
RESERVED_BURN_UID: int = 0


class MalformedKingEntry(ValueError):
    """Missing or bad-typed fields on a sync king payload."""


@dataclass(frozen=True)
class KingEntry:
    """One row of the king lineage relevant to weight computation."""

    uid: int
    hotkey: str
    score: float
    prev_score: float
    crowned_at_epoch: int
    lineage_id: Optional[int] = None
    manual_override_drop: bool = False
    family_id: str = "cf_autopilot"

    @classmethod
    def from_sync_dict(cls, data: Mapping[str, Any]) -> "KingEntry":
        """Build from a /validators/sync KingSyncEntry payload."""
        try:
            uid = int(data["uid"])
            hotkey_raw = data["hotkey"]
            if hotkey_raw is None or str(hotkey_raw).strip() == "":
                raise ValueError("empty hotkey")
            hotkey = str(hotkey_raw)
            score = float(data["score"])
            prev_score = float(data["prev_score"])
            crowned_at_epoch = int(data["crowned_at_epoch"])
            lineage_id_raw = data.get("lineage_id")
            lineage_id = int(lineage_id_raw) if lineage_id_raw is not None else None
        except (KeyError, TypeError, ValueError) as exc:
            raise MalformedKingEntry(
                f"invalid king sync entry: {exc} (data={dict(data)!r})"
            ) from exc
        manual_override_drop = bool(data.get("manual_override_drop", False))
        return cls(
            uid=uid,
            hotkey=hotkey,
            score=score,
            prev_score=prev_score,
            crowned_at_epoch=crowned_at_epoch,
            lineage_id=lineage_id,
            manual_override_drop=manual_override_drop,
            family_id=str(data.get("family_id", "cf_autopilot")),
        )


def _clamp_unit(value: float) -> float:
    v = float(value)
    if not math.isfinite(v):
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def jump_delta(score: float, prev_score: float) -> float:
    """Raw absolute improvement, clamped to [0, 1] and non-negative."""
    s = _clamp_unit(float(score))
    p = _clamp_unit(float(prev_score))
    return max(0.0, s - p)


def headroom_gain(score: float, prev_score: float, eps: float = HEADROOM_EPS) -> float:
    if float(eps) <= 0.0:
        raise ValueError(f"headroom eps must be > 0, got {eps!r}")
    s = _clamp_unit(float(score))
    p = _clamp_unit(float(prev_score))
    if s <= p:
        return 0.0
    return math.log(max(1.0 - p, float(eps)) / max(1.0 - s, float(eps)))


def rank_weight(rank: int, window: int = WINDOW_SIZE) -> float:
    return max(0.0, (float(window) - float(rank)) / float(window))


def _ranks(rows: List[KingEntry]) -> List[int]:
    order = sorted(
        range(len(rows)),
        key=lambda i: (
            rows[i].crowned_at_epoch,
            rows[i].lineage_id if rows[i].lineage_id is not None else -1,
            i,
        ),
        reverse=True,
    )
    ranks = [0] * len(rows)
    for rank, idx in enumerate(order):
        ranks[idx] = rank
    return ranks


def compute_row_weights(
    kings: Iterable[KingEntry], *, eps: float = HEADROOM_EPS
) -> List[float]:
    """Per-row share for display. One float per input row, same order.

    When two rows have the same UID, each row gets its OWN per-row share
    (not the aggregated UID total). Useful for display surfaces that show
    one row per lineage entry. Use :func:`compute_weights` instead when
    you need the chain-shaped {uid: aggregated_weight} map.
    """
    rows: List[KingEntry] = list(kings)
    if not rows:
        return []

    ranks = _ranks(rows)
    adjusted_per_king: List[float] = [
        headroom_gain(king.score, king.prev_score, eps=eps) * rank_weight(ranks[i])
        for i, king in enumerate(rows)
    ]

    total = sum(adjusted_per_king)
    if total <= 0.0:
        return [0.0] * len(rows)
    return [adj / total for adj in adjusted_per_king]


def compute_weights(
    kings: Iterable[KingEntry], *, eps: float = HEADROOM_EPS
) -> Dict[int, float]:
    """Return ``{uid: weight}`` summing to 1.0 across the given kings.

    Aggregates duplicate UIDs (a chain UID may be re-occupied by a new
    hotkey that also becomes king). For per-row display, use
    :func:`compute_row_weights` instead.

    Empty input returns ``{}``. If every king has a zero adjusted jump
    (no real innovation in the window), returns ``{}`` rather than a
    degenerate division.
    """
    rows: List[KingEntry] = list(kings)
    if not rows:
        return {}

    ranks = _ranks(rows)
    adjusted_per_king: List[float] = [
        headroom_gain(king.score, king.prev_score, eps=eps) * rank_weight(ranks[i])
        for i, king in enumerate(rows)
    ]

    total = sum(adjusted_per_king)
    if total <= 0.0:
        return {}

    # Aggregate by UID. Same UID can appear twice in the lineage if the chain
    # slot was re-occupied by a different hotkey that later became king.
    weights: Dict[int, float] = {}
    for king, adj in zip(rows, adjusted_per_king):
        if adj <= 0.0:
            continue
        uid = int(king.uid)
        weights[uid] = weights.get(uid, 0.0) + adj / total
    return weights


def active_window(
    full_lineage: Iterable[KingEntry], *, window: int = WINDOW_SIZE
) -> List[KingEntry]:
    """Take the most-recent ``window`` entries, skipping admin-dropped rows.

    ``full_lineage`` must be in chronological order (oldest first). Entries
    flagged with ``manual_override_drop`` are skipped; the next-most-recent
    eligible entry is promoted into the window in their place.
    """
    if window <= 0:
        return []
    rows = list(full_lineage)
    eligible = [k for k in rows if not k.manual_override_drop]
    if len(eligible) <= window:
        return eligible
    return eligible[-window:]


# ---------------------------------------------------------------------------
# Cross-family combine (mirror of swarm-backend app/koth.py).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CombinedWeights:
    miner_raw: Dict[int, float]
    burn_extra: float
    dropped_share: float


def default_uid_validator(entry: KingEntry) -> bool:
    """Backend-side validity (no metagraph): reject the reserved burn UID."""
    return int(entry.uid) != RESERVED_BURN_UID and int(entry.uid) >= 0


def combine_family_weights(
    family_shares: Mapping[str, float],
    kings_by_family: Mapping[str, Iterable[KingEntry]],
    *,
    is_valid_uid: Callable[[KingEntry], bool],
    eps: float = HEADROOM_EPS,
) -> CombinedWeights:
    """weight(uid) = sum over families f of family_share(f) * koth_share(uid, f).

    burn_extra (families with no positive row share) and dropped_share (invalid
    rows within a paying family) are mutually exclusive per family.
    """
    miner_raw: Dict[int, float] = {}
    burn_extra = 0.0
    dropped_share = 0.0

    for family_id, share in family_shares.items():
        rows = list(kings_by_family.get(family_id, []))
        row_shares = compute_row_weights(rows, eps=eps)
        if sum(row_shares) <= 0.0:
            burn_extra += float(share)
            continue
        for entry, row_share in zip(rows, row_shares):
            if row_share <= 0.0:
                continue
            contrib = float(share) * row_share
            if is_valid_uid(entry):
                miner_raw[int(entry.uid)] = miner_raw.get(int(entry.uid), 0.0) + contrib
            else:
                dropped_share += contrib

    return CombinedWeights(miner_raw=miner_raw, burn_extra=burn_extra, dropped_share=dropped_share)
