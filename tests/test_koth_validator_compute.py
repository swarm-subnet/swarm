"""Validator-side KotH compute path (per-family payload)."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from swarm.validator import backend_api as backend_api_mod
from swarm.validator.utils_parts.weights import (
    _advisory_warn_state,
    _apply_backend_weights_to_scores,
    compute_koth_weights_from_sync,
    stamp_local_weights_on_kings,
)


@pytest.fixture(autouse=True)
def _reset_warn_state():
    _advisory_warn_state["last_log_ts"] = 0.0
    yield
    _advisory_warn_state["last_log_ts"] = 0.0


def _king_dict(uid, *, hotkey=None, score, prev_score, crowned_at_epoch=1,
               lineage_id=None, weight=0.0, manual_override_drop=False, rank=0,
               family_id="cf_autopilot"):
    return {
        "lineage_id": lineage_id if lineage_id is not None else uid + 1000,
        "rank": rank,
        "uid": uid,
        "hotkey": hotkey or f"hk{uid}",
        "score": score,
        "prev_score": prev_score,
        "weight": weight,
        "crowned_at_epoch": crowned_at_epoch,
        "manual_override_drop": manual_override_drop,
        "family_id": family_id,
    }


def _family_sync(kings_by_family, family_shares, *, weights=None, fallback=False):
    return {
        "fallback": fallback,
        "weights": weights or {},
        "family_shares": family_shares,
        "kings_by_family": kings_by_family,
    }


def test_compute_returns_weight_map_summing_to_one():
    sync = _family_sync(
        {"cf_autopilot": [_king_dict(7, score=0.85, prev_score=0.80),
                          _king_dict(5, score=0.50, prev_score=0.20)]},
        {"cf_autopilot": 1.0},
    )
    w = compute_koth_weights_from_sync(sync)
    assert pytest.approx(sum(w.values()), abs=1e-9) == 1.0
    assert w[5] > w[7]


def test_compute_ignores_advisory_weights_on_apply_path():
    sync = _family_sync(
        {"cf_autopilot": [_king_dict(7, score=0.50, prev_score=0.0)]},
        {"cf_autopilot": 1.0},
        weights={"999": 1.0},
    )
    w = compute_koth_weights_from_sync(sync)
    assert w == {7: 1.0}
    assert 999 not in w


def test_compute_drops_malformed_rows_individually():
    sync = _family_sync(
        {"cf_autopilot": [_king_dict(7, score=0.50, prev_score=0.0),
                          {"uid": 8, "hotkey": "hk8"},
                          _king_dict(9, score=0.30, prev_score=0.10)]},
        {"cf_autopilot": 1.0},
    )
    w = compute_koth_weights_from_sync(sync)
    assert 7 in w
    assert 9 in w
    assert 8 not in w


def test_compute_empty_family_returns_empty_map():
    sync = _family_sync({"cf_autopilot": []}, {"cf_autopilot": 1.0})
    assert compute_koth_weights_from_sync(sync) == {}


def test_compute_duplicate_uid_aggregates_weight():
    sync = _family_sync(
        {"cf_autopilot": [_king_dict(42, hotkey="hk-old", score=0.30, prev_score=0.0),
                          _king_dict(99, hotkey="hk99", score=0.40, prev_score=0.30),
                          _king_dict(42, hotkey="hk-new", score=0.55, prev_score=0.40)]},
        {"cf_autopilot": 1.0},
    )
    w = compute_koth_weights_from_sync(sync)
    assert 42 in w
    assert 99 in w
    assert pytest.approx(sum(w.values()), abs=1e-9) == 1.0


def test_compute_uid_zero_king_dropped():
    sync = _family_sync(
        {"cf_autopilot": [_king_dict(0, score=0.40, prev_score=0.30)]},
        {"cf_autopilot": 1.0},
    )
    # The reserved burn UID is rejected, not paid; its slice is dropped.
    assert compute_koth_weights_from_sync(sync) == {}


def test_legacy_flat_payload_refused():
    sync = {
        "fallback": False,
        "weights": {},
        "kings": [_king_dict(7, score=0.50, prev_score=0.0)],
    }
    with patch("bittensor.logging.warning") as warn_mock:
        w = compute_koth_weights_from_sync(sync)
    assert w == {}
    assert warn_mock.call_count == 1
    assert "legacy" in warn_mock.call_args[0][0].lower()


def test_advisory_divergence_skipped_on_fallback():
    sync = _family_sync(
        {"cf_autopilot": [_king_dict(7, score=0.50, prev_score=0.0)]},
        {"cf_autopilot": 1.0},
        weights={"99": 1.0},
        fallback=True,
    )
    with patch("bittensor.logging.warning") as warn_mock:
        compute_koth_weights_from_sync(sync)
    assert warn_mock.call_count == 0


def test_advisory_divergence_warns_when_live_and_diverged():
    sync = _family_sync(
        {"cf_autopilot": [_king_dict(7, score=0.50, prev_score=0.0)]},
        {"cf_autopilot": 1.0},
        weights={"99": 1.0},
    )
    with patch("bittensor.logging.warning") as warn_mock:
        compute_koth_weights_from_sync(sync)
    assert warn_mock.call_count == 1
    msg = warn_mock.call_args[0][0]
    assert "advisory divergence" in msg.lower()


def test_advisory_divergence_warns_when_local_empty_and_backend_nonempty():
    sync = _family_sync(
        {"cf_autopilot": []},
        {"cf_autopilot": 1.0},
        weights={"99": 1.0},
    )
    with patch("bittensor.logging.warning") as warn_mock:
        compute_koth_weights_from_sync(sync)
    assert warn_mock.call_count == 1


def test_advisory_divergence_silent_when_live_and_matches():
    sync = _family_sync(
        {"cf_autopilot": [_king_dict(7, score=0.50, prev_score=0.0)]},
        {"cf_autopilot": 1.0},
        weights={"7": 1.0},
    )
    with patch("bittensor.logging.warning") as warn_mock:
        compute_koth_weights_from_sync(sync)
    assert warn_mock.call_count == 0


def test_stamp_replaces_weight_with_local_per_row_share():
    payload = [
        _king_dict(7, score=0.85, prev_score=0.80, weight=0.9),
        _king_dict(5, score=0.50, prev_score=0.20, weight=0.1),
    ]
    stamped = stamp_local_weights_on_kings(payload)
    assert pytest.approx(sum(s["weight"] for s in stamped), abs=1e-9) == 1.0
    assert stamped[0]["weight"] != 0.9
    assert stamped[1]["weight"] != 0.1


def test_stamp_zeroes_weight_on_malformed_row():
    payload = [
        _king_dict(7, score=0.50, prev_score=0.0, weight=0.6),
        {"uid": 8, "hotkey": "hk8", "weight": 0.4},
    ]
    stamped = stamp_local_weights_on_kings(payload)
    assert stamped[0]["weight"] > 0
    assert stamped[1]["weight"] == 0.0
    assert stamped[1].get("local_weight_error") == "malformed"


def test_stamp_handles_non_dict_entries():
    payload = [
        _king_dict(7, score=0.50, prev_score=0.0),
        "not-a-dict",
        None,
    ]
    stamped = stamp_local_weights_on_kings(payload)
    assert len(stamped) == 3
    assert stamped[0]["weight"] > 0
    assert stamped[1] == {}
    assert stamped[2] == {}


def test_stamp_does_not_mutate_input():
    original_weight = 0.42
    payload = [_king_dict(7, score=0.50, prev_score=0.0, weight=original_weight)]
    _ = stamp_local_weights_on_kings(payload)
    assert payload[0]["weight"] == original_weight


def test_stamp_empty_input_returns_empty_list():
    assert stamp_local_weights_on_kings([]) == []


def _make_validator_self(metagraph_n=256):
    obj = SimpleNamespace()
    obj.metagraph = SimpleNamespace(n=metagraph_n)
    obj.scores = np.zeros(metagraph_n, dtype=np.float32)
    obj._scores_lock = None
    obj._mark_weights_ready_for_setting = lambda: None
    return obj


def test_compute_then_apply_routes_share_correctly():
    sync = _family_sync(
        {"cf_autopilot": [_king_dict(50, score=0.60, prev_score=0.0)]},
        {"cf_autopilot": 1.0},
    )
    w = compute_koth_weights_from_sync(sync)
    obj = _make_validator_self()
    _apply_backend_weights_to_scores(obj, w)
    assert obj.scores[50] > 0


def test_empty_computed_map_leaves_zero():
    from swarm.constants import UID_ZERO

    sync = _family_sync({"cf_autopilot": []}, {"cf_autopilot": 1.0})
    w = compute_koth_weights_from_sync(sync)
    obj = _make_validator_self()
    _apply_backend_weights_to_scores(obj, w)
    # Burning off by default: nothing on UID0 (set_weights then holds last weights).
    assert obj.scores[UID_ZERO] == pytest.approx(0.0)
    assert float(obj.scores.sum()) == pytest.approx(0.0)


def test_backend_api_runtime_state_seed_includes_last_kings(tmp_path, monkeypatch):
    fake_state = tmp_path / "runtime_state.json"
    monkeypatch.setattr(backend_api_mod, "RUNTIME_STATE_FILE", fake_state)
    state = backend_api_mod._load_runtime_state()
    assert state.get("last_kings") == []
    assert state.get("last_weights") == {}


def test_backend_api_runtime_state_missing_last_kings_defaults_safely(
    tmp_path, monkeypatch
):
    fake_state = tmp_path / "runtime_state.json"
    legacy = {
        "last_weights": {"5": 0.6},
        "reeval_queue": [],
        "assigned_tasks": [],
        "leaderboard_version": 0,
        "last_sync": 0,
        "benchmark_epoch": 0,
    }
    fake_state.write_text(json.dumps(legacy))
    monkeypatch.setattr(backend_api_mod, "RUNTIME_STATE_FILE", fake_state)

    state = backend_api_mod._load_runtime_state()
    cached_kings = state.get("last_kings", [])
    assert cached_kings == []
