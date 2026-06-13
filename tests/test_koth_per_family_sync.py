"""WS-2: validator recomputes per-family KotH weights from the sync payload."""
from swarm.validator import koth as k
from swarm.validator.utils_parts.weights import compute_koth_weights_from_sync


def _king(uid, score, prev, family_id, hotkey=None):
    return {
        "uid": uid, "hotkey": hotkey or f"hk{uid}", "score": score,
        "prev_score": prev, "crowned_at_epoch": uid, "family_id": family_id,
    }


def test_per_family_payload_combines_across_families():
    sync = {
        "family_shares": {"cf_autopilot": 0.8, "cf_search_and_rescue": 0.2},
        "kings_by_family": {
            "cf_autopilot": [_king(11, 0.4, 0.0, "cf_autopilot")],
            "cf_search_and_rescue": [_king(22, 0.5, 0.0, "cf_search_and_rescue")],
        },
    }
    w = compute_koth_weights_from_sync(sync)
    assert abs(w[11] - 0.8) < 1e-9   # sole AP king takes AP's 0.8 slice
    assert abs(w[22] - 0.2) < 1e-9   # sole SAR king takes SAR's 0.2 slice


def test_empty_family_slice_is_absent_so_it_burns():
    # SAR active but no king -> its 0.2 is not paid to anyone (burns downstream).
    sync = {
        "family_shares": {"cf_autopilot": 0.8, "cf_search_and_rescue": 0.2},
        "kings_by_family": {
            "cf_autopilot": [_king(11, 0.4, 0.0, "cf_autopilot")],
            "cf_search_and_rescue": [],
        },
    }
    w = compute_koth_weights_from_sync(sync)
    assert abs(w[11] - 0.8) < 1e-9
    assert sum(w.values()) < 1.0 + 1e-9   # 0.8 only; 0.2 missing -> burns


def test_recompute_matches_backend_combine_oracle():
    shares = {"cf_autopilot": 0.7, "cf_search_and_rescue": 0.3}
    kbf = {
        "cf_autopilot": [_king(1, 0.5, 0.0, "cf_autopilot"), _king(2, 0.6, 0.5, "cf_autopilot")],
        "cf_search_and_rescue": [_king(3, 0.4, 0.0, "cf_search_and_rescue")],
    }
    sync = {"family_shares": shares, "kings_by_family": kbf}
    w = compute_koth_weights_from_sync(sync)
    entries = {f: [k.KingEntry.from_sync_dict(r) for r in rows] for f, rows in kbf.items()}
    oracle = k.combine_family_weights(shares, entries, is_valid_uid=k.default_uid_validator).miner_raw
    assert set(w) == set(oracle)
    for uid, val in oracle.items():
        assert abs(w[uid] - val) < 1e-9


def test_falls_back_to_flat_kings_when_no_family_fields():
    sync = {"kings": [_king(5, 0.5, 0.0, "cf_autopilot"), _king(6, 0.7, 0.5, "cf_autopilot")]}
    w = compute_koth_weights_from_sync(sync)
    assert abs(sum(w.values()) - 1.0) < 1e-9   # single-family normalizes to 1.0
    assert set(w) == {5, 6}


def test_empty_family_shares_with_kings_burns_not_flat_fallback():
    # Every family score-gated to zero: backend sends family_shares={} but still
    # ships the display window. The modern per-family path must yield full burn,
    # NOT fall back to the flat `kings` list and pay those kings.
    sync = {
        "family_shares": {},
        "kings_by_family": {
            "cf_autopilot": [_king(5, 0.5, 0.0, "cf_autopilot"), _king(6, 0.7, 0.5, "cf_autopilot")],
        },
        "kings": [_king(5, 0.5, 0.0, "cf_autopilot"), _king(6, 0.7, 0.5, "cf_autopilot")],
    }
    assert compute_koth_weights_from_sync(sync) == {}


def test_reregistered_hotkey_is_dropped_to_burn():
    # uid 11's live hotkey no longer matches the king's recorded hotkey -> burn.
    class _MG:
        hotkeys = ["hk0", "DIFFERENT", "hk2"]
    sync = {
        "family_shares": {"cf_autopilot": 1.0},
        "kings_by_family": {"cf_autopilot": [_king(1, 0.4, 0.0, "cf_autopilot", hotkey="hk_old")]},
    }
    w = compute_koth_weights_from_sync(sync, metagraph=_MG())
    assert w == {}   # the only king is a rebind -> dropped -> family burns


def test_uid_validator_never_raises_on_short_hotkeys():
    class _MG:
        hotkeys = []   # shorter than any uid
    sync = {
        "family_shares": {"cf_autopilot": 1.0},
        "kings_by_family": {"cf_autopilot": [_king(3, 0.4, 0.0, "cf_autopilot")]},
    }
    # Must not raise; out-of-range uid -> burn.
    w = compute_koth_weights_from_sync(sync, metagraph=_MG())
    assert w == {}
