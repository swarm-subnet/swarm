"""Compute is priced into the seed score, and only where it was measured."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from swarm.challenge_families import evaluate_rollout
from swarm.constants import (
    COMPUTE_FULL_UNITS,
    COMPUTE_WEIGHT,
    COMPUTE_ZERO_UNITS,
    SIM_DT,
)
from swarm.protocol import FailureReason
from swarm.validator.docker.docker_evaluator_parts.cpu_meter import (
    read_cpu_seconds,
    resolve_cpu_counter,
)
from swarm.validator.reward import (
    PARTICIPATION_REWARD,
    apply_compute_multiplier,
    compute_multiplier,
)
from swarm.validator.task_gen import task_for_seed_and_type


def _task():
    return task_for_seed_and_type(
        SIM_DT, seed=101, challenge_type=2, family_id="cf_autopilot"
    )


def _rollout(**overrides):
    kwargs = dict(
        task=_task(),
        success=True,
        t=12.0,
        horizon=60.0,
        min_clearance=1.5,
        collision=False,
        failure_reason=FailureReason.NONE.value,
    )
    kwargs.update(overrides)
    return evaluate_rollout(**kwargs)


# ── the curve ────────────────────────────────────────────────────────────────

def test_cheap_models_keep_everything():
    assert compute_multiplier(COMPUTE_FULL_UNITS) == pytest.approx(1.0)
    assert compute_multiplier(COMPUTE_FULL_UNITS / 10.0) == pytest.approx(1.0)


def test_expensive_models_lose_the_whole_weight():
    floor = 1.0 - COMPUTE_WEIGHT
    assert compute_multiplier(COMPUTE_ZERO_UNITS) == pytest.approx(floor)
    assert compute_multiplier(COMPUTE_ZERO_UNITS * 100.0) == pytest.approx(floor)


def test_every_halving_is_worth_the_same():
    """The point of the log curve: a lean model still gains by getting leaner."""
    steps = [
        compute_multiplier(units) - compute_multiplier(units * 2.0)
        for units in (0.1, 0.2, 0.4, 0.8)
    ]
    for step in steps[1:]:
        assert step == pytest.approx(steps[0])
    expected = COMPUTE_WEIGHT * math.log(2.0) / math.log(
        COMPUTE_ZERO_UNITS / COMPUTE_FULL_UNITS
    )
    assert steps[0] == pytest.approx(expected)


def test_the_curve_never_rises_with_cost():
    units = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    values = [compute_multiplier(u) for u in units]
    assert values == sorted(values, reverse=True)


@pytest.mark.parametrize("bad", [None, 0.0, -1.0, float("nan"), float("inf")])
def test_unmeasurable_costs_are_never_charged(bad):
    """None is 'we could not measure', which must never be read as 'free'."""
    if bad == float("inf"):
        assert compute_multiplier(bad) == pytest.approx(1.0 - COMPUTE_WEIGHT)
    else:
        assert compute_multiplier(bad) == pytest.approx(1.0)


# ── how it is applied ────────────────────────────────────────────────────────

def test_only_the_earned_part_is_charged():
    charged = apply_compute_multiplier(0.80, COMPUTE_ZERO_UNITS)
    earned = 0.80 - PARTICIPATION_REWARD
    assert charged == pytest.approx(
        PARTICIPATION_REWARD + earned * (1.0 - COMPUTE_WEIGHT)
    )


def test_a_failed_seed_keeps_its_participation_floor():
    assert apply_compute_multiplier(PARTICIPATION_REWARD, 9.0) == PARTICIPATION_REWARD
    assert apply_compute_multiplier(0.0, 9.0) == 0.0


def test_the_charge_can_never_invert_the_score():
    for score in (0.02, 0.5, 0.9, 1.0):
        for units in (0.01, 0.5, 2.0, 50.0):
            charged = apply_compute_multiplier(score, units)
            assert PARTICIPATION_REWARD <= charged <= score


# ── the scoring path ─────────────────────────────────────────────────────────

def test_an_unmetered_seed_scores_exactly_as_before():
    plain = _rollout()
    assert _rollout(compute_units=None).score == plain.score
    assert "compute_units" not in plain.normalized_metrics


def test_a_metered_seed_is_charged_and_reports_why():
    plain = _rollout()
    metered = _rollout(compute_units=1.367)
    assert metered.score < plain.score
    assert metered.score == pytest.approx(
        apply_compute_multiplier(plain.score, 1.367)
    )
    assert metered.normalized_metrics["compute_units"] == pytest.approx(1.367)
    assert metered.normalized_metrics["final_score"] == pytest.approx(metered.score)


def test_a_failed_seed_is_not_charged_through_the_family_path():
    failed = _rollout(
        success=False,
        collision=True,
        failure_reason=FailureReason.OBSTACLE_COLLISION.value,
        compute_units=9.0,
    )
    assert failed.score == pytest.approx(PARTICIPATION_REWARD)


# ── the meter ────────────────────────────────────────────────────────────────

def test_the_meter_reads_this_process():
    """Whatever cgroup layout the host runs, the counter must be found or refused."""
    import os

    counter = resolve_cpu_counter(os.getpid())
    if counter is None:
        pytest.skip("no readable cgroup CPU counter on this host")
    first = read_cpu_seconds(counter)
    assert first is not None and first > 0.0
    for _ in range(200000):
        pass
    assert read_cpu_seconds(counter) >= first


@pytest.mark.parametrize("pid", [None, -1, 10**9])
def test_the_meter_fails_safe(pid):
    assert resolve_cpu_counter(pid) is None


def test_reading_a_missing_counter_is_none():
    assert read_cpu_seconds(None) is None
    assert read_cpu_seconds(Path("/nonexistent/cpu.stat")) is None


# ── the ceiling ──────────────────────────────────────────────────────────────

def test_the_ceiling_reason_is_the_miners_own():
    """COMPUTE_CEILING must never be mistaken for an infrastructure fault, or the
    seed would be handed back to another validator instead of scored."""
    from swarm.core.faults import INFRA_FAULT_CODES
    from swarm.validator.utils_parts.evaluation import _is_infra_failure

    assert not _is_infra_failure(FailureReason.COMPUTE_CEILING.value)
    assert FailureReason.COMPUTE_CEILING.value not in {
        code.value for code in INFRA_FAULT_CODES
    }
    assert FailureReason.COMPUTE_CEILING.value != FailureReason.SLOW_ACT_STRIKES.value


# ── the calibration handshake ────────────────────────────────────────────────

def test_the_calibration_worker_reports_both_measurements(monkeypatch):
    """The worker runs in a spawned process no test can reach through Docker, and
    a mismatch here silences scoring fleet-wide: the host would fail to calibrate
    and exclude itself. Guard the shape of the handshake directly."""
    from swarm.validator.docker.docker_evaluator_parts import batch as batch_mod

    speed = object()

    async def _fake_calibration(self, worker_id):
        return speed, 281.8

    monkeypatch.setattr(batch_mod, "_run_baseline_calibration", _fake_calibration)
    monkeypatch.setattr(
        batch_mod, "_prepared_calibration_evaluator", lambda image: object()
    )

    class _Queue:
        def __init__(self):
            self.payloads = []

        def put(self, payload):
            self.payloads.append(payload)

    queue = _Queue()
    batch_mod._host_calibration_worker_main(3, "img", queue)

    assert len(queue.payloads) == 1
    payload = queue.payloads[0]
    assert payload["worker_id"] == 3
    assert payload["speed"] is speed
    assert payload["cpu_ms_per_act"] == pytest.approx(281.8)
    assert "error" not in payload


def test_a_failed_calibration_still_unpacks(monkeypatch):
    from swarm.validator.docker.docker_evaluator_parts import batch as batch_mod

    async def _fake_calibration(self, worker_id):
        return None, None

    monkeypatch.setattr(batch_mod, "_run_baseline_calibration", _fake_calibration)
    monkeypatch.setattr(
        batch_mod, "_prepared_calibration_evaluator", lambda image: object()
    )

    class _Queue:
        def __init__(self):
            self.payloads = []

        def put(self, payload):
            self.payloads.append(payload)

    queue = _Queue()
    batch_mod._host_calibration_worker_main(0, "img", queue)
    assert queue.payloads[0]["error"]


def test_the_reference_survives_a_restart(tmp_path, monkeypatch):
    """The cache is what a validator reloads inside the 6h window. If the CPU
    figure does not round-trip, every seed it scores silently loses the term."""
    from swarm.validator.calibration.speed_factor import normalize_speed_factor
    from swarm.validator.docker.docker_evaluator_parts import batch as batch_mod

    monkeypatch.setattr(batch_mod, "_CALIBRATION_CACHE_PATH", tmp_path / "cal.json")
    speed = normalize_speed_factor(200.0)
    written = batch_mod.HostSpeedCalibration(
        speed=speed,
        worker_count=2,
        worker_speeds=(speed, speed),
        calibration_version="swarm-ref-v2",
        computed_at=__import__("time").time(),
        cpu_ms_per_act=281.8,
    )
    batch_mod._write_calibration_cache(written)
    reloaded = batch_mod._read_calibration_cache(
        worker_count=2, calibration_version="swarm-ref-v2"
    )
    assert reloaded is not None
    assert reloaded.cpu_ms_per_act == pytest.approx(281.8)


def test_the_reference_is_readable_from_a_fresh_process(tmp_path, monkeypatch):
    """Workers run in their own processes with no in-memory calibration. If the
    accessor cannot fall back to the cache there, every seed silently loses its
    compute term and the seed ceiling never arms."""
    from swarm.validator.calibration.speed_factor import normalize_speed_factor
    from swarm.validator.docker.docker_evaluator_parts import batch as batch_mod

    monkeypatch.setattr(batch_mod, "_CALIBRATION_CACHE_PATH", tmp_path / "cal.json")
    monkeypatch.setattr(batch_mod, "_HOST_SPEED_CALIBRATION", None)
    speed = normalize_speed_factor(263.3)
    version = str(batch_mod.load_baseline_manifest()["calibration_version"])
    batch_mod._write_calibration_cache(
        batch_mod.HostSpeedCalibration(
            speed=speed,
            worker_count=2,
            worker_speeds=(speed, speed),
            calibration_version=version,
            computed_at=__import__("time").time(),
            cpu_ms_per_act=222.9,
        )
    )
    assert batch_mod.current_reference_cpu_ms_per_act() == pytest.approx(222.9)


def test_the_compute_numbers_survive_the_worker_process_boundary():
    """Evaluation runs in worker processes and results come back as plain tuples.
    If the compute numbers are not packed, scores are still correct but the audit
    trail and the miner-facing report are silently empty."""
    from swarm.benchmark.engine_parts.workers import (
        _pack_validation_result,
        _unpack_validation_result,
    )
    from swarm.protocol import ValidationResult

    result = ValidationResult(
        7, True, 12.0, 0.83,
        failure_reason=FailureReason.NONE.value,
        metrics={
            "compute_units": 1.367,
            "cpu_ms_per_act": 385.1,
            "reference_cpu_ms_per_act": 281.8,
            "compute_multiplier": 0.8206,
            "act_count": 900,
            "act_wall_ms_mean": 199.8,
            "per_drone_final_score": object(),
        },
    )
    packed = _pack_validation_result(result)
    __import__("pickle").dumps(packed)
    restored = _unpack_validation_result(packed)
    assert restored.score == pytest.approx(0.83)
    assert restored.metrics["compute_units"] == pytest.approx(1.367)
    assert restored.metrics["cpu_ms_per_act"] == pytest.approx(385.1)
    assert restored.metrics["act_count"] == 900
    assert "per_drone_final_score" not in restored.metrics


def test_a_legacy_five_tuple_still_unpacks():
    from swarm.benchmark.engine_parts.workers import _unpack_validation_result

    restored = _unpack_validation_result((1, False, 0.0, 0.0, "INFRA"))
    assert restored.metrics == {}
