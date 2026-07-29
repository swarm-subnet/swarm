from __future__ import annotations

import pytest

from swarm.constants import (
    HARD_CAP_MARGIN_SEC,
    HARD_CAP_REF_SEC,
    MINER_COMPUTE_BUDGET_SEC,
    RPC_STEP_TIMEOUT_SEC,
    SPEED_FACTOR_MAX_ELIGIBLE,
    SPEED_FACTOR_MIN,
)
from swarm.validator.calibration import (
    act_hard_cap_sec,
    judge_act,
    load_baseline_manifest,
    normalize_speed_factor,
)


def test_manifest_loads_and_is_family_agnostic():
    m = load_baseline_manifest()
    assert m["calibration_version"]
    assert m["scope"] == "all_challenge_families"
    assert float(m["owner_compute_p90_ms"]) > 0
    assert len(m["baseline_model"]["sha256"]) == 64


def test_speed_factor_matches_owner_at_baseline():
    owner = load_baseline_manifest()["owner_compute_p90_ms"]
    sf = normalize_speed_factor(owner)
    assert sf.factor == pytest.approx(1.0, abs=1e-6)
    assert sf.eligible


def test_speed_factor_scales_linearly():
    sf = normalize_speed_factor(521.6, owner_p90_ms=260.8)
    assert sf.raw == pytest.approx(2.0, abs=1e-6)
    assert sf.factor == pytest.approx(2.0, abs=1e-6)
    assert sf.eligible


def test_fast_host_never_shrinks_the_guaranteed_budget():
    sf = normalize_speed_factor(130.4, owner_p90_ms=260.8)
    assert sf.raw == pytest.approx(0.5, abs=1e-6)
    assert sf.factor == pytest.approx(1.0, abs=1e-6)


def test_low_guard_only_protects_against_bad_measurement():
    sf = normalize_speed_factor(1.0, owner_p90_ms=260.8)
    assert sf.factor == pytest.approx(SPEED_FACTOR_MIN, abs=1e-9)


def test_too_slow_host_is_ineligible_not_clamped():
    local = 260.8 * (SPEED_FACTOR_MAX_ELIGIBLE + 0.6)
    sf = normalize_speed_factor(local, owner_p90_ms=260.8)
    assert sf.raw > SPEED_FACTOR_MAX_ELIGIBLE
    assert sf.factor == sf.raw  # never silently clamped down to the eligibility limit
    assert not sf.eligible


@pytest.mark.parametrize("bad", [0.0, -5.0])
def test_invalid_measurement_rejected(bad):
    with pytest.raises(ValueError):
        normalize_speed_factor(bad, owner_p90_ms=260.8)


def _hard_cap(speed_factor, overhead):
    return act_hard_cap_sec(
        speed_factor, overhead, ref_sec=HARD_CAP_REF_SEC, margin_sec=HARD_CAP_MARGIN_SEC
    )


def test_configured_normal_action_limits():
    assert RPC_STEP_TIMEOUT_SEC == pytest.approx(1.5)
    assert MINER_COMPUTE_BUDGET_SEC == pytest.approx(1.5)
    assert HARD_CAP_REF_SEC == pytest.approx(2.0)
    assert HARD_CAP_REF_SEC > MINER_COMPUTE_BUDGET_SEC


def test_action_at_compute_budget_is_accepted():
    overhead = 0.03
    v = judge_act(
        MINER_COMPUTE_BUDGET_SEC + overhead,
        overhead_sec=overhead,
        speed_factor=1.0,
        budget_sec=MINER_COMPUTE_BUDGET_SEC,
        hard_cap_sec=_hard_cap(1.0, overhead),
    )
    assert not v.strike and not v.hard_cap_hit


def test_in_budget_action_accepted():
    v = judge_act(
        0.20, overhead_sec=0.03, speed_factor=1.0,
        budget_sec=MINER_COMPUTE_BUDGET_SEC, hard_cap_sec=_hard_cap(1.0, 0.03),
    )
    assert not v.strike and not v.hard_cap_hit


def test_slow_returned_action_is_discarded():
    elapsed = MINER_COMPUTE_BUDGET_SEC * 1.2 + 0.03
    v = judge_act(
        elapsed, overhead_sec=0.03, speed_factor=1.0,
        budget_sec=MINER_COMPUTE_BUDGET_SEC, hard_cap_sec=_hard_cap(1.0, 0.03),
    )
    assert v.strike and not v.hard_cap_hit


def test_fast_host_is_judged_strictly():
    # A fast host shrinks nothing: the same compute still has to fit the budget.
    elapsed = MINER_COMPUTE_BUDGET_SEC * 0.8 * 1.1
    v = judge_act(
        elapsed, overhead_sec=0.0, speed_factor=0.8,
        budget_sec=MINER_COMPUTE_BUDGET_SEC, hard_cap_sec=_hard_cap(0.8, 0.0),
    )
    assert v.normalized_sec == pytest.approx(MINER_COMPUTE_BUDGET_SEC * 1.1, abs=1e-6)
    assert v.strike


def test_slow_host_gets_proportional_leniency():
    # 0.90s on a 2x host = 0.45s owner-equivalent -> within budget.
    v = judge_act(
        0.90, overhead_sec=0.0, speed_factor=2.0,
        budget_sec=MINER_COMPUTE_BUDGET_SEC, hard_cap_sec=_hard_cap(2.0, 0.0),
    )
    assert v.normalized_sec == pytest.approx(0.45, abs=1e-6)
    assert not v.strike


def test_hard_cap_hit_is_a_strike():
    cap = _hard_cap(1.0, 0.03)
    v = judge_act(
        cap + 0.5, overhead_sec=0.03, speed_factor=1.0,
        budget_sec=MINER_COMPUTE_BUDGET_SEC, hard_cap_sec=cap,
    )
    assert v.hard_cap_hit and v.strike


def test_same_model_same_verdict_across_hosts():
    # A model whose true owner-equivalent compute is 0.40s must pass on every host.
    owner_compute = 0.40
    for factor in (0.5, 1.0, 1.7, 2.5):
        elapsed = owner_compute * factor + 0.03  # + overhead
        v = judge_act(
            elapsed, overhead_sec=0.03, speed_factor=factor,
            budget_sec=MINER_COMPUTE_BUDGET_SEC, hard_cap_sec=_hard_cap(factor, 0.03),
        )
        assert not v.strike, f"factor {factor} disagreed"
