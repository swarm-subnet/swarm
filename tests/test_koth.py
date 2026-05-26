"""Unit tests for swarm.validator.koth.

Covers the formula, the cap activation, the first-king edge case, the
Discord-example numerical match, the window helper, and degenerate inputs.
"""
from __future__ import annotations

import math

import pytest

from swarm.validator.koth import (
    HEADROOM_EPS,
    WINDOW_SIZE,
    KingEntry,
    MalformedKingEntry,
    active_window,
    compute_weights,
    headroom_adjusted,
    jump_delta,
)


def _king(uid: int, score: float, prev: float, *, epoch: int = 0) -> KingEntry:
    return KingEntry(
        uid=uid,
        hotkey=f"hk{uid}",
        score=score,
        prev_score=prev,
        crowned_at_epoch=epoch,
    )


def test_empty_lineage_returns_empty_map():
    assert compute_weights([]) == {}


def test_single_first_king_takes_100_percent():
    king = _king(uid=178, score=0.50, prev=0.0)
    weights = compute_weights([king])
    assert weights == {178: 1.0}


def test_discord_example_matches_expected_weights():
    """The Discord conversation:
       jumps +0.010, +0.001, +0.010, +0.005, +0.005
       expected weights ~[0.32, 0.03, 0.32, 0.16, 0.16]

    With low previous scores the headroom cap doesn't activate, so the
    formula collapses to raw-absolute proportions (within ~1%).
    """
    kings = [
        _king(uid=1, score=0.110, prev=0.100),  # +0.010
        _king(uid=2, score=0.111, prev=0.110),  # +0.001
        _king(uid=3, score=0.121, prev=0.111),  # +0.010
        _king(uid=4, score=0.126, prev=0.121),  # +0.005
        _king(uid=5, score=0.131, prev=0.126),  # +0.005
    ]
    weights = compute_weights(kings)
    expected = {1: 0.32, 2: 0.03, 3: 0.32, 4: 0.16, 5: 0.16}
    assert set(weights.keys()) == set(expected.keys())
    for uid, want in expected.items():
        assert math.isclose(weights[uid], want, abs_tol=0.02), (
            f"uid {uid}: got {weights[uid]:.4f}, want {want}"
        )


def test_mature_subnet_late_jump_dominates():
    """Scenario B from the plan: mature subnet (0.87→0.95)."""
    kings = [
        _king(uid=2, score=0.87, prev=0.85),
        _king(uid=3, score=0.90, prev=0.87),
        _king(uid=4, score=0.92, prev=0.90),
        _king(uid=5, score=0.95, prev=0.92),
    ]
    weights = compute_weights(kings)
    assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9)
    assert weights[5] > weights[3] > 0.0
    assert weights[5] > weights[2]


def test_miguel_question_late_jump_4x_early_same_size():
    """+0.05 absolute jump at 0.80 base vs at 0.20 base.

    With only those two kings in the window, the late one should earn
    approximately 4x what the early one does.
    """
    early = _king(uid=10, score=0.25, prev=0.20)
    late = _king(uid=11, score=0.85, prev=0.80)
    weights = compute_weights([early, late])
    ratio = weights[11] / weights[10]
    assert 3.5 < ratio < 4.5, f"expected ~4x, got {ratio:.2f}"


def test_cap_activates_near_perfect_score():
    """As prev->1.0, the eps cap prevents the multiplier from diverging.

    A +0.01 jump from 0.99 should produce adjusted = 0.01 / 0.05 = 0.20,
    NOT 0.01 / 0.01 = 1.00.
    """
    adj = headroom_adjusted(delta=0.01, prev_score=0.99)
    assert math.isclose(adj, 0.20, abs_tol=1e-9)


def test_cap_does_not_activate_below_threshold():
    """prev_score = 0.80 gives headroom 0.20, well above eps."""
    adj = headroom_adjusted(delta=0.02, prev_score=0.80)
    assert math.isclose(adj, 0.10, abs_tol=1e-9)


def test_score_clamping_to_unit_interval():
    """Out-of-range scores are clamped before subtracting."""
    # score > 1.0 clamped to 1.0
    assert jump_delta(score=1.5, prev_score=0.50) == 0.5
    # prev > 1.0 clamped to 1.0 → headroom hits eps cap
    assert jump_delta(score=0.5, prev_score=1.2) == 0.0
    # negative scores clamped to 0
    assert jump_delta(score=-0.1, prev_score=0.0) == 0.0
    # score < prev → delta is 0 (clamped, never negative)
    assert jump_delta(score=0.5, prev_score=0.6) == 0.0


def test_first_king_baseline_zero_means_jump_equals_score():
    """A king with prev=0 has jump=score (full benchmark progress)."""
    assert jump_delta(score=0.40, prev_score=0.0) == 0.40
    adj = headroom_adjusted(delta=0.40, prev_score=0.0)
    assert math.isclose(adj, 0.40, abs_tol=1e-9)


def test_zero_delta_yields_zero_weight():
    """A king with no jump shouldn't appear in the weight map at all."""
    kings = [
        _king(uid=1, score=0.50, prev=0.0),   # delta 0.50
        _king(uid=2, score=0.50, prev=0.50),  # delta 0 (re-crowned same score)
    ]
    weights = compute_weights(kings)
    assert 2 not in weights
    assert weights[1] == 1.0


def test_all_zero_deltas_returns_empty():
    """If no king in the window has positive delta, return empty (not NaN)."""
    kings = [
        _king(uid=1, score=0.50, prev=0.60),
        _king(uid=2, score=0.40, prev=0.45),
    ]
    assert compute_weights(kings) == {}


def test_weights_always_sum_to_one():
    """Invariant: sum of weights == 1.0 whenever the map is non-empty."""
    kings = [
        _king(uid=1, score=0.85, prev=0.0),
        _king(uid=2, score=0.87, prev=0.85),
        _king(uid=3, score=0.90, prev=0.87),
        _king(uid=4, score=0.92, prev=0.90),
        _king(uid=5, score=0.95, prev=0.92),
    ]
    weights = compute_weights(kings)
    assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9)


def test_active_window_smaller_than_window_returns_all():
    kings = [_king(uid=i, score=0.1 * i, prev=0.1 * (i - 1)) for i in range(1, 4)]
    assert active_window(kings) == kings


def test_active_window_exactly_window_returns_all():
    kings = [_king(uid=i, score=0.1 * i, prev=0.1 * (i - 1)) for i in range(1, WINDOW_SIZE + 1)]
    assert active_window(kings) == kings


def test_active_window_larger_than_window_takes_tail():
    """A 7-king lineage with WINDOW_SIZE=5 keeps the most-recent 5."""
    kings = [_king(uid=i, score=0.1 * i, prev=0.1 * (i - 1)) for i in range(1, 8)]
    window = active_window(kings)
    assert len(window) == WINDOW_SIZE
    assert [k.uid for k in window] == [3, 4, 5, 6, 7]


def test_active_window_zero_or_negative_returns_empty():
    kings = [_king(uid=1, score=0.5, prev=0.0)]
    assert active_window(kings, window=0) == []
    assert active_window(kings, window=-1) == []


def test_headroom_eps_is_a_module_constant():
    assert HEADROOM_EPS == 0.05


def test_window_size_is_a_module_constant():
    assert WINDOW_SIZE == 5


def test_duplicate_uid_aggregates_weight():
    """A UID can be re-occupied on chain by a different hotkey. If the same
    UID ends up in the lineage twice, the weights must aggregate, not
    overwrite.
    """
    kings = [
        _king(uid=42, score=0.30, prev=0.0),       # delta 0.30 -> adj 0.30
        _king(uid=99, score=0.40, prev=0.30),      # delta 0.10 -> adj ~0.143
        _king(uid=42, score=0.55, prev=0.40),      # delta 0.15 -> adj 0.25 — same UID as first
    ]
    weights = compute_weights(kings)
    assert set(weights.keys()) == {42, 99}
    assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9)
    # UID 42 should hold both its slices.
    assert weights[42] > weights[99]


def test_active_window_filters_manual_override_drop():
    """A dropped king is skipped; the next-most-recent eligible entry is
    promoted into the window.
    """
    lineage = [
        _king(uid=1, score=0.50, prev=0.00),
        _king(uid=2, score=0.55, prev=0.50),
        _king(uid=3, score=0.60, prev=0.55),
        KingEntry(uid=4, hotkey="hk4", score=0.65, prev_score=0.60,
                  crowned_at_epoch=4, manual_override_drop=True),  # dropped
        _king(uid=5, score=0.70, prev=0.65),
        _king(uid=6, score=0.75, prev=0.70),
        _king(uid=7, score=0.80, prev=0.75),
    ]
    window = active_window(lineage)  # WINDOW_SIZE = 5
    # UID 4 must be skipped; UIDs 1, 2, 3, 5, 6, 7 are eligible (6 total),
    # window pulls the latest 5 → 2, 3, 5, 6, 7.
    assert [k.uid for k in window] == [2, 3, 5, 6, 7]


def test_headroom_adjusted_rejects_non_positive_eps():
    """Caller bug: eps <= 0 would divide by zero near prev=1. Raise instead."""
    with pytest.raises(ValueError):
        headroom_adjusted(delta=0.01, prev_score=0.50, eps=0.0)
    with pytest.raises(ValueError):
        headroom_adjusted(delta=0.01, prev_score=0.50, eps=-0.01)


def test_from_sync_dict_full_payload():
    entry = KingEntry.from_sync_dict(
        {
            "lineage_id": 42,
            "rank": 0,
            "uid": 7,
            "hotkey": "hk7",
            "score": 0.85,
            "prev_score": 0.80,
            "crowned_at_epoch": 12,
            "manual_override_drop": False,
            "weight": 0.5,
        }
    )
    assert entry.uid == 7
    assert entry.hotkey == "hk7"
    assert entry.score == 0.85
    assert entry.prev_score == 0.80
    assert entry.crowned_at_epoch == 12
    assert entry.lineage_id == 42
    assert entry.manual_override_drop is False


def test_from_sync_dict_optional_fields_default():
    entry = KingEntry.from_sync_dict(
        {
            "uid": 1,
            "hotkey": "hk1",
            "score": 0.10,
            "prev_score": 0.0,
            "crowned_at_epoch": 1,
        }
    )
    assert entry.lineage_id is None
    assert entry.manual_override_drop is False


def test_from_sync_dict_raises_on_missing_prev_score():
    with pytest.raises(MalformedKingEntry):
        KingEntry.from_sync_dict(
            {
                "uid": 1,
                "hotkey": "hk1",
                "score": 0.10,
                "crowned_at_epoch": 1,
            }
        )


def test_from_sync_dict_raises_on_non_numeric_score():
    with pytest.raises(MalformedKingEntry):
        KingEntry.from_sync_dict(
            {
                "uid": 1,
                "hotkey": "hk1",
                "score": "not-a-number",
                "prev_score": 0.0,
                "crowned_at_epoch": 1,
            }
        )


def test_from_sync_dict_raises_on_bad_lineage_id():
    with pytest.raises(MalformedKingEntry):
        KingEntry.from_sync_dict(
            {
                "uid": 1,
                "hotkey": "hk1",
                "score": 0.10,
                "prev_score": 0.0,
                "crowned_at_epoch": 1,
                "lineage_id": "not-an-int",
            }
        )


def test_from_sync_dict_raises_on_empty_hotkey():
    with pytest.raises(MalformedKingEntry):
        KingEntry.from_sync_dict(
            {
                "uid": 1,
                "hotkey": "",
                "score": 0.10,
                "prev_score": 0.0,
                "crowned_at_epoch": 1,
            }
        )


def test_malformed_king_entry_subclasses_value_error():
    assert issubclass(MalformedKingEntry, ValueError)


def test_from_sync_dict_uid_zero_is_preserved():
    entry = KingEntry.from_sync_dict(
        {
            "uid": 0,
            "hotkey": "hk0",
            "score": 0.30,
            "prev_score": 0.20,
            "crowned_at_epoch": 5,
        }
    )
    assert entry.uid == 0
    weights = compute_weights([entry])
    assert weights == {0: 1.0}
