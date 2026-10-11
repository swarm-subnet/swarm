# The MIT License (MIT)
# Copyright © 2026 Swarm

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Per-family king window: Swarm Sentinel pays its last 10 kings, every other family its last 5."""
import math

import pytest

from swarm.domain_model import get_family_king_window, list_challenge_family_definitions
from swarm.validator import koth
from swarm.validator.utils_parts.weights import compute_koth_weights_from_sync

SENTINEL = "cf_solar_patrol"
SAR = "cf_search_and_rescue"

# Ten crownings, oldest first; each king beat the one before it. The same vector
# and the same expected shares are pinned in swarm-backend's tests/test_koth_king_window.py.
SCORES = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
TEN_KING_SHARES = [
    0.013212387251000915, 0.017429474671002095, 0.024942616516358557,
    0.03570505540490219, 0.051130981134638404, 0.0732583724008192,
    0.10503289050213253, 0.1507316905249228, 0.21661167429151099,
    0.3119448573027124,
]
FIVE_KING_SHARES = [
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.08542458591920665, 0.12247598308560993, 0.17576410485267002,
    0.2525849534354584, 0.36375037270705496,
]


def _ten_rows(family_id, uid_base=10):
    """The ten-crowning vector as sync payload rows for one family."""
    prev = [0.0] + SCORES[:-1]
    return [
        {
            "uid": uid_base + i, "hotkey": f"hk{uid_base + i}", "score": score,
            "prev_score": prev[i], "crowned_at_epoch": i + 1, "lineage_id": i + 1,
            "family_id": family_id,
        }
        for i, score in enumerate(SCORES)
    ]


def _ten_kings(family_id):
    """The ten-crowning vector as formula input for one family."""
    return [koth.KingEntry.from_sync_dict(row) for row in _ten_rows(family_id)]


def test_sentinel_window_is_ten_and_every_other_family_keeps_five():
    """The registry gives Sentinel 10 seats; families without a window, or unknown ones, keep WINDOW_SIZE."""
    assert get_family_king_window(SENTINEL) == 10
    assert get_family_king_window(SAR) == 0
    assert koth.king_window(SENTINEL) == 10
    assert koth.king_window(SAR) == koth.WINDOW_SIZE == 5
    assert koth.king_window("cf_not_registered") == koth.WINDOW_SIZE


def test_only_sentinel_sets_its_own_window():
    """Every other registered family is left on the shared window, so its payout is unchanged."""
    windows = {
        family["family_id"]: family["king_window"]
        for family in list_challenge_family_definitions()
        if "king_window" in family
    }
    assert windows == {SENTINEL: 10}


def test_family_window_reads_zero_when_the_registry_sets_none():
    """A family entry without king_window reads as 0, which the formula turns into the shared window."""
    registry = {"challenge_families": {"cf_a": {"king_window": 7}, "cf_b": {}}}
    assert get_family_king_window("cf_a", registry=registry) == 7
    assert get_family_king_window("cf_b", registry=registry) == 0


def test_sentinel_pays_all_ten_seats_on_the_ladder():
    """Ten Sentinel kings each earn the golden share, newest first, summing to the whole slice."""
    shares = koth.compute_row_weights(_ten_kings(SENTINEL))
    assert shares == pytest.approx(TEN_KING_SHARES, abs=1e-12)
    assert math.isclose(sum(shares), 1.0, abs_tol=1e-12)
    assert shares == sorted(shares)


def test_other_families_still_pay_only_five_seats():
    """The same ten rows in a five-king family pay the newest five exactly as before and nothing to the rest."""
    shares = koth.compute_row_weights(_ten_kings(SAR))
    assert shares == pytest.approx(FIVE_KING_SHARES, abs=1e-12)
    assert shares[5:] == pytest.approx(koth.compute_row_weights(_ten_kings(SAR)[5:]), abs=1e-12)


def test_validator_recomputes_ten_sentinel_seats_from_the_sync_payload():
    """From a backend sync payload the validator pays Sentinel's slice over ten kings and SAR's over five."""
    payload = {
        "family_shares": {SENTINEL: 0.65, SAR: 0.15},
        "kings_by_family": {SENTINEL: _ten_rows(SENTINEL), SAR: _ten_rows(SAR, uid_base=110)},
    }
    weights = compute_koth_weights_from_sync(payload)
    for i, share in enumerate(TEN_KING_SHARES):
        assert weights[10 + i] == pytest.approx(0.65 * share, abs=1e-12)
    paid_sar = {uid: w for uid, w in weights.items() if uid >= 110}
    assert sorted(paid_sar) == [115, 116, 117, 118, 119]
    assert sum(paid_sar.values()) == pytest.approx(0.15, abs=1e-12)
