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

"""Solar Patrol's seed checks (task 24): the replacement chain, the kept verdicts, the reference pilot's route, how the
validator judges seeds in the background and just before they fly without ever waiting for them, and whole reference
flights judged on flat ground with a square fence, as the other patrol tests fly them.
"""
from __future__ import annotations

import asyncio
import io
import os
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pybullet as p
import pytest
import swarm_worlds
from shapely.geometry import Point, Polygon

from swarm.benchmark import engine as bench_engine
from swarm.challenge_families import build_benchmark_tasks, deferred_seed_checks, require_runtime_family
from swarm.challenge_families.solar_patrol import airframe, drone_state, park, reference_pilot, seed_checks, theft, wind
from swarm.challenge_families.solar_patrol.contract import FAMILY_ID, HORIZON_S
from swarm.challenge_families.solar_patrol.seed_checks import CHECK_TRIES, SeedCheckError, Verdict, candidates
from swarm.constants import SIM_DT
from swarm.domain_model import get_family_benchmark_seed_count
from swarm.protocol import FailureReason
from swarm.validator import forward as forward_mod
from swarm.validator.docker import docker_evaluator as de
from swarm.validator.utils_parts import heartbeat as heartbeat_module
from swarm.validator.utils_parts import run_task as run_task_module
from validator.tests.test_docker_evaluator import _ScriptedContext

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

    class Flights(seed_checks.Flights):
        """Flights that judge every seed handed to them at once, in this process."""

        def fly(self, seeds):
            """Keep a verdict for each seed of the batch."""
            if seeds:
                flown.append(list(seeds))
            for seed in seeds:
                seed_checks.keep(Verdict(seed=seed, passed=seeds_passing(seed),
                                         reason="" if seeds_passing(seed) else "x"))

    monkeypatch.setattr(seed_checks, "Flights", Flights)
    return flown


_FAKE_PILOT = '''"""A reference pilot that flies nothing: odd seeds pass, and the end reason names the process that judged it."""
import os
import sys

from swarm.challenge_families.solar_patrol.seed_checks import Verdict, keep

for line in sys.stdin:
    seed = int(line)
    keep(Verdict(seed=seed, passed=seed % 2 == 1, reason="" if seed % 2 else "even", end_reason=str(os.getpid())))
'''

_SLOW_PILOT_S = 2.0
_SLOW_PILOT = f'''"""A reference pilot that spends {_SLOW_PILOT_S} s on every seed it reads, then passes it."""
import sys
import time

from swarm.challenge_families.solar_patrol.seed_checks import Verdict, keep

for line in sys.stdin:
    time.sleep({_SLOW_PILOT_S})
    keep(Verdict(seed=int(line), passed=True))
'''


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
    """A reference flight that ends without keeping a verdict stops the preparation."""
    monkeypatch.setattr(seed_checks, "PILOT_MODULE", "this")
    with pytest.raises(SeedCheckError, match="left no verdict"):
        seed_checks.prepare([11], workers=1)


def test_a_flight_that_crashes_after_its_last_verdict_is_an_error(kept, monkeypatch, tmp_path):
    """A reference process that keeps its verdict and then exits with an error still stops the preparation."""
    (tmp_path / "crash_pilot.py").write_text(_FAKE_PILOT + "sys.exit(3)\n")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.environ.get("PYTHONPATH", "")]))
    monkeypatch.setattr(seed_checks, "PILOT_MODULE", "crash_pilot")
    with pytest.raises(SeedCheckError, match="ended with code 3"):
        seed_checks.prepare([11], workers=1)


def test_a_flight_that_ends_between_seeds_names_no_seed(kept, monkeypatch, tmp_path):
    """A reference process that ends while idle is reported as such, not as the next seed it was about to judge."""
    (tmp_path / "one_seed_pilot.py").write_text(_FAKE_PILOT.replace("for line in sys.stdin:",
                                                                    "for line in [sys.stdin.readline()]:"))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.environ.get("PYTHONPATH", "")]))
    monkeypatch.setattr(seed_checks, "PILOT_MODULE", "one_seed_pilot")
    with pytest.raises(SeedCheckError, match="ended between seeds"):
        with seed_checks.Flights(1) as flights:
            flights.fly([12])
            next(iter(flights.runs)).wait()
            flights.fly([13])


def test_a_flight_that_ends_before_reading_its_next_seed_names_no_seed(kept, monkeypatch, tmp_path):
    """A process that keeps its verdict and ends while its next seed waits unread is not blamed on that seed."""
    pilot = _FAKE_PILOT.replace("import sys\n", "import select\nimport sys\n").replace(
        "for line in sys.stdin:", "for line in [sys.stdin.readline()]:")
    (tmp_path / "unread_pilot.py").write_text(pilot + "select.select([sys.stdin], [], [])\n")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.environ.get("PYTHONPATH", "")]))
    monkeypatch.setattr(seed_checks, "PILOT_MODULE", "unread_pilot")
    with pytest.raises(SeedCheckError, match="left no verdict") as raised:
        with seed_checks.Flights(1) as flights:
            flights.fly([12])
            flights.fly([13])
    assert "seed 13" not in str(raised.value)


def test_flights_judge_seed_after_seed_in_their_own_processes(kept, monkeypatch, tmp_path):
    """Real processes: no more of them run than workers, each judges seed after seed, and every seed ends on its
    first passing candidate."""
    (tmp_path / "fake_pilot.py").write_text(_FAKE_PILOT)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.environ.get("PYTHONPATH", "")]))
    monkeypatch.setattr(seed_checks, "PILOT_MODULE", "fake_pilot")
    seeds = [11, 12, 13, 14, 15, 16]
    chains = {seed: list(candidates(seed)) for seed in seeds}
    firsts = {seed: next(c for c in chain if c % 2 == 1) for seed, chain in chains.items()}
    assert seed_checks.prepare(seeds, workers=2) == firsts
    judged = [seed_checks.cached(c) for seed, chain in chains.items() for c in chain[:chain.index(firsts[seed]) + 1]]
    assert all(judged) and len({verdict.end_reason for verdict in judged}) == 2 < len(judged)


def test_reference_flights_run_on_one_blas_thread_whatever_the_host_sets(kept, monkeypatch, tmp_path):
    """A real flight process sees one BLAS and one OpenMP thread, as in the validator image, over the host's eight."""
    (tmp_path / "threads_pilot.py").write_text(_FAKE_PILOT.replace(
        "end_reason=str(os.getpid())", "end_reason=os.environ['OPENBLAS_NUM_THREADS'] + os.environ['OMP_NUM_THREADS']"))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.environ.get("PYTHONPATH", "")]))
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "8")
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setattr(seed_checks, "PILOT_MODULE", "threads_pilot")
    seed_checks.prepare([11], workers=1)
    assert seed_checks.cached(11).end_reason == "11"


def test_the_pilot_judges_seeds_read_from_its_input(kept, monkeypatch):
    """Run with no seeds named, the pilot judges and keeps every seed it reads, one a line."""
    monkeypatch.setattr(reference_pilot, "judge", lambda seed: Verdict(seed=seed, passed=True))
    monkeypatch.setattr(sys, "stdin", io.StringIO("11\n12\n"))
    reference_pilot.main([])
    assert seed_checks.cached(11) == Verdict(seed=11, passed=True) and seed_checks.cached(12) is not None


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
    """The validator stand-in the background preparation reads: a seed manager serving one list for every family."""
    manager = SimpleNamespace(epoch_number=5, get_all_seeds=lambda family_id=None, epoch=None: list(seeds))
    return SimpleNamespace(seed_manager=manager)


def _solar_task(seed):
    """A Swarm Sentinel task as the dispatcher sees it."""
    return SimpleNamespace(family_id=FAMILY_ID, map_seed=seed, challenge_type=4, horizon=60.0)


class _Arrivals:
    """A backend client that notes when each heartbeat reaches it."""

    def __init__(self, times):
        """Note arrival times in that list."""
        self.times = times

    async def post_heartbeat(self, **kwargs):
        """Note this heartbeat's arrival."""
        self.times.append(time.monotonic())


def _dispatch(monkeypatch, tmp_path, tasks, *, workers, feeder=None, heartbeats=None):
    """Run tasks through the validator's dispatcher against workers that fly each seed at once; returns the results,
    the requests the workers were handed, and the warnings logged. Given a heartbeats list, the validator's heartbeat
    session runs on the same event loop and the time each heartbeat reaches the backend is added to it."""
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"x")
    context = _ScriptedContext(bench_engine, {index: [{"results": [(42, True, 1.0, 1.0)]}] for index in range(len(tasks))})
    warnings = []
    monkeypatch.setattr(de.parallel, "_benchmark_engine", lambda: bench_engine)
    monkeypatch.setattr(bench_engine, "_benchmark_mp_context", lambda: context)
    monkeypatch.setattr(de.parallel.bt.logging, "warning", lambda msg: warnings.append(str(msg)))

    async def run():
        """The dispatcher's run, inside a heartbeat session when one is asked for."""
        session = None
        if heartbeats is not None:
            session = heartbeat_module.HeartbeatManager(_Arrivals(heartbeats), asyncio.get_running_loop())
            session.start("evaluating", uid=42, total=len(tasks))
        try:
            return await de.parallel._run_process_parallel(
                all_tasks=list(tasks),
                task_meta=[{"group": "type4_village", "seed": task.map_seed, "index": index, "challenge_type": 4,
                            "horizon": 60.0} for index, task in enumerate(tasks)],
                batch_plan=[[index] for index in range(len(tasks))],
                uid=42,
                model_path=model_path,
                effective_workers=workers,
                phase_label="eval",
                seed_feeder=feeder,
                initial_pending=list(range(len(tasks))) if feeder else None,
            )
        finally:
            if session is not None:
                session._stop_timer()

    results = asyncio.run(run())
    return results, context.requests, warnings


def test_a_deferred_build_leaves_an_unjudged_seed_to_the_caller(kept, monkeypatch):
    """With the checks deferred a seed not judged yet is built as drawn and nothing flies; checking it then flies the
    reference pilot once, and the checked task needs no further check."""
    flown = _fly_with(monkeypatch, lambda seed: True)
    runtime = require_runtime_family(FAMILY_ID)
    with deferred_seed_checks():
        task = build_benchmark_tasks(sim_dt=SIM_DT, seeds=[11], family_id=FAMILY_ID)[0]
    assert task.map_seed == 11 and flown == []
    assert runtime.awaits_seed_check(task)
    checked = runtime.check_seed(task)
    assert checked.map_seed == 11 and flown == [[11]]
    assert not runtime.awaits_seed_check(checked)


def test_a_seed_judged_a_failure_after_its_task_was_built_never_flies(kept, monkeypatch):
    """A seed the background preparation fails after its deferred task was built is still swapped before it flies,
    without flying again, and a later deferred build starts from the replacement."""
    chain = list(candidates(11))
    flown = _fly_with(monkeypatch, lambda seed: seed != chain[0])
    runtime = require_runtime_family(FAMILY_ID)
    with deferred_seed_checks():
        drawn = build_benchmark_tasks(sim_dt=SIM_DT, seeds=[11], family_id=FAMILY_ID)[0]
    seed_checks.prepare([11], workers=1)
    assert runtime.awaits_seed_check(drawn)
    assert runtime.check_seed(drawn).map_seed == chain[1]
    assert flown == [[chain[0]], [chain[1]]]
    with deferred_seed_checks():
        rebuilt = build_benchmark_tasks(sim_dt=SIM_DT, seeds=[11], family_id=FAMILY_ID)[0]
    assert rebuilt.map_seed == chain[1] and not runtime.awaits_seed_check(rebuilt)


def test_background_flights_run_at_the_lowest_cpu_priority(kept, monkeypatch, tmp_path):
    """Real flight processes judging ahead run at niceness 19, so they only take cores the scoring leaves idle; a seed
    judged as it flies keeps the validator's own priority."""
    (tmp_path / "nice_pilot.py").write_text(_FAKE_PILOT.replace("end_reason=str(os.getpid())",
                                                                "end_reason=str(os.nice(0))"))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.environ.get("PYTHONPATH", "")]))
    monkeypatch.setattr(seed_checks, "PILOT_MODULE", "nice_pilot")
    require_runtime_family(FAMILY_ID).prepare_seeds([11], background=True)
    require_runtime_family(FAMILY_ID).check_seed(_solar_task(13))
    assert seed_checks.cached(11).end_reason == str(seed_checks.BACKGROUND_NICENESS)
    assert seed_checks.cached(13).end_reason == str(os.nice(0))


def test_the_scored_seed_count_is_read_from_the_registry():
    """Swarm Sentinel is scored on 250 seeds; a family without a count reads 0, the whole list."""
    registry = {"challenge_families": {"cf_a": {"benchmark_seed_count": 4}, "cf_b": {}}}
    assert get_family_benchmark_seed_count(FAMILY_ID) == 250
    assert get_family_benchmark_seed_count("cf_a", registry=registry) == 4
    assert get_family_benchmark_seed_count("cf_b", registry=registry) == 0


@pytest.mark.asyncio
async def test_the_scored_seeds_are_judged_once_in_the_background_highest_first(monkeypatch):
    """Of all the families only Swarm Sentinel judges ahead: its 250 scored seeds, once, in a worker thread at the
    lowest priority, highest first; once they are all judged the validator remembers it and starts nothing more."""
    runtime = require_runtime_family(FAMILY_ID)
    judged, started = set(), []

    def prepare_seeds(seeds, *, background=False):
        """Judge every seed, recording the order, the priority asked for and the thread."""
        started.append((list(seeds), background, threading.get_ident()))
        judged.update(seeds)

    monkeypatch.setattr(runtime, "seeds_prepared", lambda seeds: set(seeds) <= judged)
    monkeypatch.setattr(runtime, "prepare_seeds", prepare_seeds)
    validator = _validator_with_seeds(range(1000, 2000))
    run_task_module.start_seed_preparations(validator, 5)
    run_task_module.start_seed_preparations(validator, 5)
    assert list(validator._seed_preparations) == [(FAMILY_ID, 5)]
    await validator._seed_preparations[(FAMILY_ID, 5)]
    assert [(seeds, background) for seeds, background, _ in started] == [(list(range(1249, 999, -1)), True)]
    assert started[0][2] != threading.get_ident()
    run_task_module.start_seed_preparations(validator, 5)
    judged.clear()
    run_task_module.start_seed_preparations(validator, 5)
    assert len(started) == 1 and (FAMILY_ID, 5) in validator._seeds_prepared


@pytest.mark.asyncio
async def test_a_failed_background_preparation_waits_before_starting_again(monkeypatch):
    """A preparation that raised is not started again on every pass of the loop, only once the retry delay is over."""
    runtime = require_runtime_family(FAMILY_ID)
    calls = []

    def prepare_seeds(seeds, *, background=False):
        """Fail the first time."""
        calls.append(list(seeds))
        if len(calls) == 1:
            raise SeedCheckError("reference flights failed")

    monkeypatch.setattr(runtime, "seeds_prepared", lambda seeds: False)
    monkeypatch.setattr(runtime, "prepare_seeds", prepare_seeds)
    validator = _validator_with_seeds([11])
    run_task_module.start_seed_preparations(validator, 5)
    with pytest.raises(SeedCheckError):
        await validator._seed_preparations[(FAMILY_ID, 5)]
    await asyncio.sleep(0)
    run_task_module.start_seed_preparations(validator, 5)
    assert len(calls) == 1
    validator._seed_preparation_retry_at[(FAMILY_ID, 5)] = 0.0
    run_task_module.start_seed_preparations(validator, 5)
    await validator._seed_preparations[(FAMILY_ID, 5)]
    assert calls == [[11], [11]]


def test_a_seed_list_that_cannot_be_read_never_stops_the_validator():
    """Judging ahead is only a head start: when the seed list cannot be read nothing is started, nothing is raised,
    and it is tried again only after the retry delay."""
    validator = SimpleNamespace(seed_manager=SimpleNamespace(epoch_number=5))
    run_task_module.start_seed_preparations(validator, 5)
    assert validator._seed_preparations == {}
    assert validator._seed_preparation_retry_at[(FAMILY_ID, 5)] > time.monotonic()


@pytest.mark.asyncio
async def test_a_solar_task_starts_while_its_seeds_are_still_judged(monkeypatch):
    """A Swarm Sentinel task that arrives before the epoch's seeds are judged is taken at once: its model is fetched
    instead of the task being handed back."""
    monkeypatch.setattr(require_runtime_family(FAMILY_ID), "seeds_prepared", lambda seeds: False)
    fetched = []

    async def fetch(_self, models):
        """Record the fetch and find no model, which ends the task there."""
        fetched.append(models)
        return {}

    monkeypatch.setattr(run_task_module, "_ensure_models_from_backend", fetch)
    await run_task_module.run_task(
        _validator_with_seeds([11]),
        {"task_id": 1, "uid": 42, "phase": "SCREENING", "family_id": FAMILY_ID, "epoch_number": 5,
         "model_hash": "0" * 64},
        cancel_flag=asyncio.Event(), wake_flag=asyncio.Event())
    assert [model["uid"] for models in fetched for model in models] == [42]


@pytest.mark.asyncio
async def test_at_the_start_of_an_epoch_the_validator_takes_a_task_while_seeds_are_judged(monkeypatch):
    """The first pass of the loop in a new epoch starts judging the Sentinel seeds in the background and, without
    waiting for them, asks the backend for a task and runs it."""
    runtime = require_runtime_family(FAMILY_ID)
    release = threading.Event()
    monkeypatch.setattr(runtime, "seeds_prepared", lambda seeds: False)
    monkeypatch.setattr(runtime, "prepare_seeds", lambda seeds, background=False: release.wait(30))
    task = {"task_id": 1, "uid": 42, "phase": "SCREENING", "family_id": FAMILY_ID}
    ran = []

    async def skipped(*args, **kwargs):
        """Stand in for a step of the loop this test does not exercise."""

    async def may_score(_self):
        """The host is calibrated and its image verified."""
        return True

    async def sync():
        """A sync that changes nothing."""
        return {}

    async def next_task():
        """The backend hands out a Sentinel task."""
        return task

    async def run_task(_self, leased, **kwargs):
        """Record the task the loop runs."""
        ran.append(leased)

    monkeypatch.setattr(forward_mod, "_ensure_components", skipped)
    monkeypatch.setattr(forward_mod, "_publish_pending_epoch_seeds", skipped)
    monkeypatch.setattr(forward_mod, "_cleanup_off_loop", skipped)
    monkeypatch.setattr(forward_mod, "_host_may_score", may_score)
    monkeypatch.setattr(forward_mod, "run_task", run_task)
    validator = _validator_with_seeds([11, 12])
    validator.seed_manager.seeds_ready = lambda: True
    validator.backend_api = SimpleNamespace(sync=sync, next_task=next_task)
    validator.metagraph = None
    validator._sse_fatal = None
    validator._cancel_flag, validator._wake_flag = asyncio.Event(), asyncio.Event()
    try:
        await forward_mod._forward_iteration(validator)
        assert ran == [task]
        assert not validator._seed_preparations[(FAMILY_ID, 5)].done()
    finally:
        release.set()
        await validator._seed_preparations[(FAMILY_ID, 5)]


def test_the_dispatcher_checks_an_unjudged_seed_off_the_loop_and_flies_what_the_check_returns(monkeypatch, tmp_path):
    """A seed still unjudged when its turn comes is checked in a worker thread while a judged seed flies, and the
    worker is handed the task the check returned, here the seed's replacement."""
    runtime = require_runtime_family(FAMILY_ID)
    threads = []

    def check_seed(task):
        """Judge seed 11 slowly, away from the event loop, and swap it for seed 99."""
        threads.append(threading.get_ident())
        time.sleep(0.2)
        return _solar_task(99)

    monkeypatch.setattr(runtime, "awaits_seed_check", lambda task: task.map_seed == 11)
    monkeypatch.setattr(runtime, "check_seed", check_seed)
    results, requests, _ = _dispatch(monkeypatch, tmp_path, [_solar_task(11), _solar_task(12)], workers=2)
    assert [request.tasks[0].map_seed for request in requests] == [12, 99]
    assert len(threads) == 1 and threads[0] != threading.get_ident()
    assert all(result.success for result in results)


def test_no_more_seeds_are_checked_at_once_than_there_are_workers(monkeypatch, tmp_path):
    """Three unjudged seeds on one worker are checked one at a time, and each flies once its own check is done."""
    runtime = require_runtime_family(FAMILY_ID)
    lock, live, peak = threading.Lock(), [0], [0]

    def check_seed(task):
        """Count the checks running together while this one takes a moment."""
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(0.05)
        with lock:
            live[0] -= 1
        return task

    monkeypatch.setattr(runtime, "awaits_seed_check", lambda task: True)
    monkeypatch.setattr(runtime, "check_seed", check_seed)
    results, requests, _ = _dispatch(monkeypatch, tmp_path, [_solar_task(seed) for seed in (11, 12, 13)], workers=1)
    assert peak[0] == 1
    assert sorted(request.tasks[0].map_seed for request in requests) == [11, 12, 13]
    assert all(result.success for result in results)


@pytest.mark.parametrize("seed_flow", [False, True])
def test_a_seed_that_cannot_be_checked_is_left_unscored(monkeypatch, tmp_path, seed_flow):
    """A seed whose check fails never flies: the run carries on with the others and ends, leaving it unscored; in seed
    flow it is simply no longer held, so the backend hands it to another validator."""
    runtime = require_runtime_family(FAMILY_ID)

    def check_seed(task):
        """No candidate of the seed passed."""
        raise SeedCheckError("no candidate of seed 11 passed the seed checks")

    async def drained(_free_slots):
        """Offer nothing more and report the pool drained."""
        return [], True

    monkeypatch.setattr(runtime, "awaits_seed_check", lambda task: task.map_seed == 11)
    monkeypatch.setattr(runtime, "check_seed", check_seed)
    results, requests, warnings = _dispatch(monkeypatch, tmp_path, [_solar_task(11), _solar_task(12)], workers=1,
                                            feeder=drained if seed_flow else None)
    assert [request.tasks[0].map_seed for request in requests] == [12]
    assert results[1].success
    assert results[0] is None if seed_flow else results[0].failure_reason == FailureReason.INFRA.value
    assert any("seed check failed for UID 42" in line for line in warnings)


def test_a_deleted_check_is_flown_again_while_heartbeats_keep_going(kept, monkeypatch, tmp_path):
    """With one seed's kept verdict deleted, a real flight process judges it again before it flies, the other seed
    flies first, and the validator's heartbeats keep reaching the backend the whole time that flight runs."""
    (tmp_path / "slow_pilot.py").write_text(_SLOW_PILOT)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.environ.get("PYTHONPATH", "")]))
    monkeypatch.setattr(seed_checks, "PILOT_MODULE", "slow_pilot")
    monkeypatch.setattr(heartbeat_module, "_TIMER_INTERVAL_SECONDS", 0.05)
    for seed in (11, 12):
        seed_checks.keep(Verdict(seed=seed, passed=True))
    (seed_checks.cache_dir() / "11.json").unlink()
    runtime = require_runtime_family(FAMILY_ID)
    check_seed, spans = runtime.check_seed, []

    def timed_check_seed(task):
        """The real check, with when it started and ended noted."""
        started = time.monotonic()
        try:
            return check_seed(task)
        finally:
            spans.append((started, time.monotonic()))

    monkeypatch.setattr(runtime, "check_seed", timed_check_seed)
    heartbeats = []
    results, requests, _ = _dispatch(monkeypatch, tmp_path, [_solar_task(11), _solar_task(12)], workers=2,
                                     heartbeats=heartbeats)
    assert [request.tasks[0].map_seed for request in requests] == [12, 11]
    assert all(result.success for result in results) and seed_checks.cached(11).passed
    [(started, ended)] = spans
    assert ended - started >= _SLOW_PILOT_S
    arrivals = [started, *(t for t in heartbeats if started < t < ended), ended]
    assert max(later - earlier for earlier, later in zip(arrivals, arrivals[1:])) < _SLOW_PILOT_S / 2


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
