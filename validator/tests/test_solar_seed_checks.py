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

"""Solar Patrol's seed checks (task 24): the replacement chain, the kept verdicts, the reference pilot's route, and
whole reference flights judged on flat ground with a square fence, as the other patrol tests fly them.
"""
from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import numpy as np
import pybullet as p
import pytest
import swarm_worlds
from shapely.geometry import Point, Polygon

from swarm.challenge_families import DEFAULT_RUNTIME_FAMILY_ID, build_benchmark_tasks, require_runtime_family
from swarm.challenge_families.solar_patrol import airframe, drone_state, park, reference_pilot, seed_checks, theft, wind
from swarm.challenge_families.solar_patrol.contract import FAMILY_ID, HORIZON_S
from swarm.challenge_families.solar_patrol.seed_checks import CHECK_TRIES, SeedCheckError, Verdict, candidates
from swarm.constants import SIM_DT
from swarm.validator.utils_parts import run_task as run_task_module

_M4TD_SHIPPED = os.path.isfile(os.path.join(swarm_worlds.robots_dir(), airframe.URDF))
_APPROVE = seed_checks.approve
_PREPARE = seed_checks.prepare
_FENCE = np.array([[20.0, 60.0], [100.0, 60.0], [100.0, 140.0], [20.0, 140.0]])
_ELL = np.array([[0.0, 0.0], [160.0, 0.0], [160.0, 60.0], [60.0, 60.0], [60.0, 160.0], [0.0, 160.0]])
_THIEF_XY = (60.0, 100.0)
_DAY_SEED = next(seed for seed in range(100) if not wind.is_night(seed))
_NIGHT_SEED = next(seed for seed in range(100) if wind.is_night(seed))


class _StillMovers:
    """Movers for a park that has none."""

    body_uids = frozenset()

    def advance(self, step=None):
        """Nothing moves."""


@pytest.fixture
def kept(monkeypatch, tmp_path):
    """The real seed checks, keeping their verdicts in a folder of the test's own."""
    monkeypatch.setattr(seed_checks, "approve", _APPROVE)
    monkeypatch.setattr(seed_checks, "prepare", _PREPARE)
    monkeypatch.setenv(seed_checks.CACHE_ENV, str(tmp_path))
    return tmp_path


@pytest.fixture
def flat_park(monkeypatch):
    """Build the patrol's world as flat ground with a square fence, in still air."""
    def build(seed=0, cli=0, asset_dir=None, groups=None):
        """A flat slab standing in for the park's terrain."""
        shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[200.0, 200.0, 0.5], physicsClientId=cli)
        ground = p.createMultiBody(0, shape, -1, [60.0, 100.0, -0.5], physicsClientId=cli)
        return {"bodies": {"terrain": [ground]}, "movers": [], "asset_dir": "flat"}

    monkeypatch.setattr(park, "build_solar_map", build)
    monkeypatch.setattr(park, "build_solar_movers", lambda world, seed=0, cli=0: _StillMovers())
    monkeypatch.setattr(park, "fence_line", lambda asset_dir: _FENCE)
    monkeypatch.setattr(drone_state, "survey", lambda asset_dir: (np.zeros((0, 5)), np.zeros((0, 5))))
    monkeypatch.setattr(wind, "for_seed", lambda seed: dict(wind.STILL_AIR))


def _thief(monkeypatch, shed):
    """One thief standing at _THIEF_XY for the whole patrol, a threat from the start; inside a closed shed when shed
    is set, so no camera anywhere can see him."""
    def reset(env, ep):
        """Stand him, and the shed around him, in the world."""
        cli = env.CLIENT
        body = p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.25, 0.15, 0.85],
                                                           physicsClientId=cli), -1, [*_THIEF_XY, 0.85],
                                 physicsClientId=cli)
        if shed:
            p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[1.5, 1.5, 1.4], physicsClientId=cli),
                              -1, [*_THIEF_XY, 1.4], physicsClientId=cli)
        ep.theft = {"body": int(body)}
        ep.outcome.threats = 1

    monkeypatch.setattr(theft, "reset", reset)
    monkeypatch.setattr(theft, "advance", lambda env, ep: None)
    monkeypatch.setattr(theft, "show", lambda env, ep, view: None)
    monkeypatch.setattr(theft, "people", lambda ep, step=None: [
        {"bodies": [ep.theft["body"]], "xy": _THIEF_XY, "stage": "cable theft", "threat": True}])
    monkeypatch.setattr(theft, "bodies", lambda ep: {ep.theft["body"]: 0})


def _skip_without_m4td():
    """Whole flights need the aircraft the patrol flies."""
    if not _M4TD_SHIPPED:
        pytest.skip(f"the installed swarm-worlds has no {airframe.URDF} yet")


def _fly_with(monkeypatch, seeds_passing):
    """Stand in for the reference flights: each seed flown is kept as passing when seeds_passing says so; returns
    the list of every batch flown."""
    flown = []

    def fly(seeds, workers):
        """Keep a verdict for each seed of the batch."""
        flown.append(list(seeds))
        for seed in seeds:
            seed_checks.keep(Verdict(seed=seed, passed=seeds_passing(seed), reason="" if seeds_passing(seed) else "x"))

    monkeypatch.setattr(seed_checks, "fly", fly)
    return flown


def test_candidates_start_with_the_seed_and_are_the_same_everywhere():
    """A seed's candidates begin with the seed itself, are distinct 32-bit seeds, and never change."""
    chain = list(candidates(7))
    assert chain[0] == 7 and len(chain) == CHECK_TRIES
    assert len(set(chain)) == CHECK_TRIES and all(0 <= seed < 2 ** 32 for seed in chain)
    assert chain == list(candidates(7))
    assert chain[1] != list(candidates(8))[1]


def test_a_passing_seed_is_flown_as_itself(kept, monkeypatch):
    """A seed whose verdict passed keeps its own task."""
    flown = _fly_with(monkeypatch, lambda seed: True)
    task = build_benchmark_tasks(sim_dt=SIM_DT, seeds=[11], family_id=FAMILY_ID)[0]
    assert task.map_seed == 11 and flown == [[11]]


def test_a_failing_seed_is_flown_as_its_first_passing_candidate(kept, monkeypatch):
    """A seed that fails is replaced by the first candidate that passes, a whole patrol task of that seed."""
    chain = list(candidates(11))
    flown = _fly_with(monkeypatch, lambda seed: seed not in chain[:2])
    task = build_benchmark_tasks(sim_dt=SIM_DT, seeds=[11], family_id=FAMILY_ID)[0]
    assert task.map_seed == chain[2]
    assert task.wind_max_mps == wind.for_seed(chain[2])["wind_max_mps"]
    assert flown == [[chain[0]], [chain[1]], [chain[2]]]


def test_a_seed_is_flown_once_and_its_verdict_kept(kept, monkeypatch):
    """A second task for the same seed reads the kept verdict and flies nothing."""
    flown = _fly_with(monkeypatch, lambda seed: True)
    build_benchmark_tasks(sim_dt=SIM_DT, seeds=[11], family_id=FAMILY_ID)
    build_benchmark_tasks(sim_dt=SIM_DT, seeds=[11], family_id=FAMILY_ID)
    assert flown == [[11]]
    assert seed_checks.cached(11) == Verdict(seed=11, passed=True)


def test_verdicts_are_kept_per_benchmark_version(kept, monkeypatch):
    """A verdict kept under one benchmark version is not read under another."""
    _fly_with(monkeypatch, lambda seed: True)
    seed_checks.prepare([11], workers=1)
    assert seed_checks.cached(11) is not None
    monkeypatch.setattr(seed_checks, "BENCHMARK_VERSION", "0.0.1")
    assert seed_checks.cached(11) is None and seed_checks.replacement(11) is None


def test_prepare_follows_each_seed_only_as_far_as_it_needs(kept, monkeypatch):
    """A list is flown in batches: every seed first, then only the candidates of the seeds that failed."""
    failing = list(candidates(12))[0]
    flown = _fly_with(monkeypatch, lambda seed: seed != failing)
    assert seed_checks.prepare([11, 12, 13], workers=2) == {11: 11, 12: list(candidates(12))[1], 13: 13}
    assert flown == [[11, 12, 13], [list(candidates(12))[1]]]


def test_a_seed_with_no_passing_candidate_is_refused(kept, monkeypatch):
    """When every candidate fails the seed is refused rather than flown unchecked."""
    _fly_with(monkeypatch, lambda seed: False)
    with pytest.raises(SeedCheckError):
        seed_checks.prepare([11], workers=1)
    with pytest.raises(SeedCheckError):
        seed_checks.replacement(11)


def test_a_flight_that_leaves_no_verdict_is_an_error(kept, monkeypatch):
    """A reference flight that dies without keeping a verdict stops the preparation."""
    monkeypatch.setattr(seed_checks, "fly", lambda seeds, workers: None)
    with pytest.raises(SeedCheckError):
        seed_checks.prepare([11], workers=1)


def test_a_seed_whose_world_cannot_be_built_fails_and_others_raise(monkeypatch):
    """A seed whose own world refuses to be built fails its check; an error of the machine is raised, never kept."""
    def no_dock(seed):
        """The seed's world has no open ground for the dock."""
        raise RuntimeError(f"no open ground for the dock in seed {seed}")

    monkeypatch.setattr(reference_pilot, "check", no_dock)
    result = reference_pilot.judge(11)
    assert not result.passed and result.reason.startswith("error:no open ground")

    def disk_full(seed):
        """The machine ran out of disk."""
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(reference_pilot, "check", disk_full)
    with pytest.raises(OSError):
        reference_pilot.judge(11)


def test_lanes_are_as_far_apart_as_the_wide_camera_sees():
    """The wide camera's width on the ground from the patrol height sets the lanes, less the overlap."""
    assert reference_pilot.lane_spacing() == pytest.approx(2 * 20.0 * 0.6954 * 0.9, abs=0.05)


@pytest.mark.parametrize("spacing", [reference_pilot.lane_spacing(), 11.6])
def test_the_route_searches_the_whole_park_without_leaving_the_fence(spacing):
    """On an L-shaped park the route starts and ends at the dock, stays inside the fence, and passes within half a
    lane of every point more than the lane inset inside it."""
    dock_xy = np.array([30.0, 140.0])
    path, search_m = reference_pilot.route(_ELL, dock_xy, spacing)
    assert np.allclose(path[0], dock_xy) and np.allclose(path[-1], dock_xy)
    assert 0.0 < search_m < reference_pilot.path_length(path)
    fence = Polygon(_ELL)
    points = reference_pilot._resample(path, 0.5)
    assert all(fence.buffer(-reference_pilot.CONNECT_INSET_M + 0.01).covers(Point(xy)) for xy in points)
    inner = fence.buffer(-reference_pilot.FENCE_INSET_M)
    grid = np.array([(x, y) for x in np.arange(0.0, 160.0, 2.0) for y in np.arange(0.0, 160.0, 2.0)
                     if inner.covers(Point(x, y))])
    gaps = np.min(np.hypot(*(grid[:, None, :] - points[None, :, :]).transpose(2, 0, 1)), axis=1)
    assert gaps.max() <= spacing / 2.0 + 1.0


def test_narrow_side_is_measured_across_the_shape():
    """A 12 by 3 bar is 3 pixels across, the same bar turned on its side too, a diagonal two-pixel line about two,
    and no pixels are none."""
    bar = np.zeros((20, 20), dtype=bool)
    bar[4:16, 8:11] = True
    assert reference_pilot.narrow_px(bar) == 3.0
    assert reference_pilot.narrow_px(bar.T) == 3.0
    line = np.eye(20, dtype=bool) | np.eye(20, k=1, dtype=bool)
    assert 1.5 <= reference_pilot.narrow_px(line) <= 3.0
    assert reference_pilot.narrow_px(np.zeros((5, 5), dtype=bool)) == 0.0


def _validator_with_seeds(seeds):
    """The validator stand-in the gate reads: a seed manager serving one list for every family."""
    manager = SimpleNamespace(epoch_number=5, get_all_seeds=lambda family_id=None, epoch=None: list(seeds))
    return SimpleNamespace(seed_manager=manager)


def test_families_without_seed_checks_never_wait():
    """A family that prepares nothing builds its tasks at once, without its seed list even being read."""
    validator = SimpleNamespace(seed_manager=SimpleNamespace(epoch_number=5))
    assert run_task_module._family_seeds_prepared(validator, DEFAULT_RUNTIME_FAMILY_ID, 5)


@pytest.mark.asyncio
async def test_solar_waits_until_its_seeds_are_judged_off_the_loop(monkeypatch):
    """Until every seed of the epoch is judged the gate says wait and starts one preparation in a worker thread;
    once it is done the gate opens and remembers it."""
    runtime = require_runtime_family(FAMILY_ID)
    judged, started = set(), []

    def prepare_seeds(seeds):
        """Judge every seed, from a thread other than the event loop's."""
        started.append(list(seeds))
        judged.update(seeds)

    monkeypatch.setattr(runtime, "seeds_prepared", lambda seeds: set(seeds) <= judged)
    monkeypatch.setattr(runtime, "prepare_seeds", prepare_seeds)
    validator = _validator_with_seeds([11, 12])
    assert not run_task_module._family_seeds_prepared(validator, FAMILY_ID, 5)
    assert not run_task_module._family_seeds_prepared(validator, FAMILY_ID, 5)
    await validator._seed_preparations[(FAMILY_ID, 5)]
    assert started == [[11, 12]]
    assert run_task_module._family_seeds_prepared(validator, FAMILY_ID, 5)
    judged.clear()
    assert run_task_module._family_seeds_prepared(validator, FAMILY_ID, 5)


@pytest.mark.asyncio
async def test_a_failed_preparation_is_started_again(monkeypatch):
    """A preparation that raised is logged and started again by the next task that finds the seeds unjudged."""
    runtime = require_runtime_family(FAMILY_ID)
    calls = []

    def prepare_seeds(seeds):
        """Fail the first time."""
        calls.append(list(seeds))
        if len(calls) == 1:
            raise SeedCheckError("reference flights failed")

    monkeypatch.setattr(runtime, "seeds_prepared", lambda seeds: False)
    monkeypatch.setattr(runtime, "prepare_seeds", prepare_seeds)
    validator = _validator_with_seeds([11])
    run_task_module._family_seeds_prepared(validator, FAMILY_ID, 5)
    with pytest.raises(SeedCheckError):
        await validator._seed_preparations[(FAMILY_ID, 5)]
    run_task_module._family_seeds_prepared(validator, FAMILY_ID, 5)
    await validator._seed_preparations[(FAMILY_ID, 5)]
    assert calls == [[11], [11]]


@pytest.mark.asyncio
async def test_a_solar_task_is_left_to_another_validator_while_its_seeds_are_judged(monkeypatch):
    """A Solar Patrol task that arrives before its seeds are judged fetches no model and flies nothing."""
    runtime = require_runtime_family(FAMILY_ID)
    monkeypatch.setattr(runtime, "seeds_prepared", lambda seeds: False)
    monkeypatch.setattr(runtime, "prepare_seeds", lambda seeds: None)
    fetched = []

    async def fetch(_self, models):
        """Record a fetch that must not happen."""
        fetched.append(models)
        return {}

    monkeypatch.setattr(run_task_module, "_ensure_models_from_backend", fetch)
    validator = _validator_with_seeds([11])
    await run_task_module.run_task(
        validator, {"task_id": 1, "uid": 42, "phase": "SCREENING", "family_id": FAMILY_ID, "epoch_number": 5,
                    "model_hash": "0" * 64},
        cancel_flag=asyncio.Event(), wake_flag=asyncio.Event())
    await validator._seed_preparations[(FAMILY_ID, 5)]
    assert fetched == []


@pytest.mark.timeout(600)
@pytest.mark.parametrize("seed", [_DAY_SEED, _NIGHT_SEED])
def test_an_empty_seed_in_still_air_passes(flat_park, seed):
    """By day and by night the reference pilot searches the whole flat park, lands in the dock in time, and the seed
    passes."""
    _skip_without_m4td()
    result = reference_pilot.check(seed)
    assert result.passed, result
    assert result.end_reason == "landed" and result.landed_s < HORIZON_S
    assert result.threats == 0 and result.route_m > 0.0


@pytest.mark.timeout(600)
@pytest.mark.parametrize("seed", [_DAY_SEED, _NIGHT_SEED])
def test_a_thief_in_the_open_is_seen_and_the_seed_passes(flat_park, monkeypatch, seed):
    """A thief standing in the open shows at least PIXELS_ACROSS pixels to the passing pilot, in the colour camera by
    day and the thermal camera by night."""
    _skip_without_m4td()
    _thief(monkeypatch, shed=False)
    result = reference_pilot.check(seed)
    assert result.passed, result
    assert result.threats == 1 and result.seen == 1
    assert result.best_px[0] >= reference_pilot.PIXELS_ACROSS


@pytest.mark.timeout(600)
def test_a_thief_no_camera_can_see_fails_the_seed(flat_park, monkeypatch):
    """A thief inside a closed shed for the whole patrol shows no pixel, and the seed fails on him."""
    _skip_without_m4td()
    _thief(monkeypatch, shed=True)
    result = reference_pilot.check(_DAY_SEED)
    assert not result.passed and result.reason == "thief_unseen"
    assert result.threats == 1 and result.seen == 0 and result.best_px == [0.0]


@pytest.mark.timeout(600)
def test_a_pilot_too_slow_to_come_home_in_time_fails_the_seed(flat_park, monkeypatch):
    """A pilot held to a crawl is still searching when the patrol's time runs out, and the seed fails."""
    _skip_without_m4td()
    monkeypatch.setattr(reference_pilot, "_corner_speeds", lambda points: np.full(len(points), 0.3))
    result = reference_pilot.check(_DAY_SEED)
    assert not result.passed and result.reason == "not_landed:timeout" and result.landed_s is None


@pytest.mark.timeout(900)
def test_the_same_seed_gives_the_same_verdict(flat_park, monkeypatch):
    """Two flights of one seed come back with the same verdict, to the hundredth of a second of the landing."""
    _skip_without_m4td()
    _thief(monkeypatch, shed=False)
    first, second = reference_pilot.check(_DAY_SEED), reference_pilot.check(_DAY_SEED)
    first.cpu_s = second.cpu_s = 0.0
    assert first == second
