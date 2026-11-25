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

"""The Test Field's seed checks and reference pilot: the replacement chain, the kept verdicts under the family's own
folder, the flights run through Sentinel's machinery with the field's pilot, and what the pilot reports."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from swarm.challenge_families import build_benchmark_tasks
from swarm.challenge_families.solar_patrol import coverage, sensor_noise
from swarm.challenge_families.solar_patrol import seed_checks as sentinel_checks
from swarm.challenge_families.test_field import install, people, reference_pilot, reports, seed_checks, wind
from swarm.challenge_families.test_field.contract import ACTION_DIM, ACTION_INDEX, FAMILY_ID, REPORT_REACH_M
from swarm.challenge_families.test_field.seed_checks import FieldVerdict, candidates
from swarm.constants import SIM_DT

_APPROVE = seed_checks.approve
_PREPARE = seed_checks.prepare


@pytest.fixture
def kept(monkeypatch, tmp_path):
    """The real seed checks on an install that can fly the field, keeping their verdicts in the test's own folder."""
    monkeypatch.setattr(seed_checks, "approve", _APPROVE)
    monkeypatch.setattr(seed_checks, "prepare", _PREPARE)
    monkeypatch.setattr(install, "problems", lambda: [])
    monkeypatch.setenv(seed_checks.CACHE_ENV, str(tmp_path))
    return tmp_path


def _fly_with(monkeypatch, seeds_passing):
    """Stand in for the reference flights: each seed flown is kept as passing when seeds_passing says so; returns
    every batch flown."""
    flown = []

    class Flights(sentinel_checks.Flights):
        """Flights that judge every seed handed to them at once, in this process."""

        def fly(self, seeds):
            """Keep a verdict for each seed of the batch."""
            if seeds:
                flown.append(list(seeds))
            for seed in seeds:
                seed_checks.keep(FieldVerdict(seed=seed, passed=seeds_passing(seed), reason="" if seeds_passing(seed) else "x"))

    monkeypatch.setattr(sentinel_checks, "Flights", Flights)
    return flown


def test_default_fixture_bypasses_first_field_seed_check(monkeypatch):
    """A first-field flight keeps the requested seed without consulting its verdict cache."""
    monkeypatch.setattr(seed_checks, "replacement", lambda seed: pytest.fail("looked up a verdict"))
    task = SimpleNamespace(map_seed=13)

    assert seed_checks.approve(task) is task
    assert seed_checks.prepare([13, 14], workers=1) == {13: 13, 14: 14}


def test_the_candidates_are_the_fields_own():
    """A seed's candidates start with the seed, never change, and differ from Sentinel's chain for the same seed."""
    chain = list(candidates(7))
    assert chain[0] == 7 and chain == list(candidates(7)) and len(set(chain)) == len(chain)
    assert chain[1:] != list(sentinel_checks.candidates(7))[1:]


def test_verdicts_are_kept_apart_from_sentinels(kept):
    """The field keeps its verdicts in a folder named for the family and its version, never Sentinel's, and reads
    back what it kept."""
    assert seed_checks.cache_dir().name.startswith(f"{FAMILY_ID}-") and seed_checks.cache_dir() != sentinel_checks.cache_dir()
    verdict = FieldVerdict(seed=4, passed=True, end_reason="landed", threats=3, seen=3, min_edge_m=4.2)
    seed_checks.keep(verdict)
    assert seed_checks.cached(4) == verdict and sentinel_checks.cached(4) is None


def test_a_failing_seed_is_flown_as_its_first_passing_candidate(kept, monkeypatch):
    """A seed that fails is replaced by the first candidate that passes, a whole patrol task of that seed."""
    chain = list(candidates(11))
    flown = _fly_with(monkeypatch, lambda seed: seed not in chain[:2])
    task = build_benchmark_tasks(sim_dt=SIM_DT, seeds=[11], family_id=FAMILY_ID)[0]
    assert task.map_seed == chain[2] and task.wind_max_mps == wind.for_seed(chain[2])["wind_max_mps"]
    assert flown == [[chain[0]], [chain[1]], [chain[2]]]
    assert seed_checks.replacement(11) == chain[2]


def test_a_seed_with_no_passing_candidate_is_refused(kept, monkeypatch):
    """When every candidate fails the seed is refused rather than flown unchecked."""
    _fly_with(monkeypatch, lambda seed: False)
    with pytest.raises(sentinel_checks.SeedCheckError):
        seed_checks.prepare([11], workers=1)


_FAKE_PILOT = '''"""A field pilot that flies nothing and keeps a passing verdict for each seed it reads."""
import json
import os
import sys
from pathlib import Path

for line in sys.stdin:
    seed = int(line)
    directory = Path(os.environ["{env}"]) / "{folder}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f".{{seed}}.tmp").write_text(json.dumps({{"seed": seed, "passed": True, "min_edge_m": 3.5}}))
    (directory / f".{{seed}}.tmp").replace(directory / f"{{seed}}.json")
'''


def test_sentinels_flights_fly_the_fields_pilot_and_read_its_verdicts(kept, monkeypatch, tmp_path):
    """Sentinel's flight processes run the pilot module they are given and wait on the verdicts where the field keeps
    them."""
    module = tmp_path / "fake_field_pilot.py"
    module.write_text(_FAKE_PILOT.format(env=seed_checks.CACHE_ENV, folder=seed_checks.cache_dir().name))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), *sys.path]))
    with sentinel_checks.Flights(1, pilot="fake_field_pilot", verdict=seed_checks.cached) as flights:
        flights.fly([21])
    assert seed_checks.cached(21) == FieldVerdict(seed=21, passed=True, min_edge_m=3.5)


def test_a_report_press_carries_the_box_and_the_position_as_shares():
    """The pilot's press sets the report button, the box and the position as shares of 100 m from the dock."""
    a = np.zeros(ACTION_DIM, dtype=np.float32)
    reference_pilot.Pilot.press(a, (0.5, 0.25, 0.1, 0.2), np.array([30.0, -150.0]))
    assert a[ACTION_INDEX["report"]] == 1.0
    assert [a[ACTION_INDEX[f"report_{k}"]] for k in ("cx", "cy", "w", "h")] == pytest.approx([0.5, 0.25, 0.1, 0.2])
    assert a[ACTION_INDEX["report_east"]] == pytest.approx(30.0 / REPORT_REACH_M)
    assert a[ACTION_INDEX["report_north"]] == -1.0


def test_the_pilots_eyes_read_each_frame_once_and_never_one_too_high(monkeypatch):
    """A frame is read once, its people boxed as reports judge them, and a frame from over 24 m holds nobody."""
    objects = np.full((480, 640), -1, dtype=np.int32)
    objects[100:140, 320:336] = 9
    view = SimpleNamespace(objects=objects, sees=True, eye=(0.0, 0.0, 20.0), width=640, height=480)
    height = {"m": 20.0}
    monkeypatch.setattr(sensor_noise, "shown_view", lambda ep, image: view)
    monkeypatch.setattr(coverage, "height_above_ground", lambda env, ep, eye: height["m"])
    monkeypatch.setattr(people, "bodies", lambda ep: {9: 1})
    eyes = reference_pilot.Eyes(None, None)
    assert eyes.fresh() == {1: pytest.approx((328.0 / 640.0, 120.0 / 480.0, 16.0 / 640.0, 40.0 / 480.0))}
    assert eyes.fresh() == {}
    eyes.last, height["m"] = None, reports.MAX_REPORT_HEIGHT_M + 1.0
    assert eyes.fresh() == {}


def test_the_pilots_free_ground_keeps_off_the_edge_the_crowns_and_the_people():
    """A point is free only inside the field by EDGE_CLEAR_M, off every crown by CROWN_CLEAR_M and off every person by
    PERSON_CLEAR_M, and a path between two free points goes round a crown instead of through it."""
    square = np.array([[-30.0, -30.0], [30.0, -30.0], [30.0, 30.0], [-30.0, 30.0]])
    free = reference_pilot.Free(square, np.array([[0.0, 0.0, 4.0]]), np.array([[15.0, 15.0]]))
    assert free.holds(np.array([-25.0, -10.0]))
    assert not free.holds(np.array([30.0 - reference_pilot.EDGE_CLEAR_M + 0.6, -10.0]))
    assert not free.holds(np.array([4.0 + reference_pilot.CROWN_CLEAR_M - 0.6, 0.0]))
    assert not free.holds(np.array([15.0 + reference_pilot.PERSON_CLEAR_M - 0.6, 15.0]))
    a, b = np.array([-15.0, 0.0]), np.array([15.0, 0.0])
    assert not free.clear(a, b)
    path = free.path(a, b)
    assert path is not None and len(path) > 2 and np.allclose(path[0], a) and np.allclose(path[-1], b)
    samples = np.concatenate([np.linspace(p, q, 40) for p, q in zip(path[:-1], path[1:])])
    assert np.min(np.hypot(*samples.T)) >= 4.0 + reference_pilot.CROWN_CLEAR_M - reference_pilot.GRID_M


def test_the_sweep_lies_in_free_lanes_ten_metres_apart():
    """The sweep's runs lie on free ground along the field's long axis, LANE_M apart."""
    field = np.array([[-40.0, -15.0], [40.0, -15.0], [40.0, 15.0], [-40.0, 15.0]])
    free = reference_pilot.Free(field, np.zeros((0, 3)), np.zeros((0, 2)))
    runs = reference_pilot.lanes(free, field)
    assert len(runs) >= 2 and all(free.clear(*run) for run in runs)
    offsets = sorted({round(float(run[0][1]), 3) for run in runs})
    assert np.allclose(np.diff(offsets), reference_pilot.LANE_M)
    assert all(abs(float(run[0][1] - run[1][1])) < 1e-9 for run in runs)


# A sweep's way into a lane that doubles straight back along it, as flown on seed 76, with the drone where it stalled.
HAIRPIN = np.array([[3.4, 7.6], [7.9, 5.6], [8.0, 5.9], [-27.9, 3.4]])


def test_the_pilots_carrot_stops_at_a_sharp_corner_rather_than_folding_back_onto_the_drone():
    """Past a hairpin the carrot would land on the drone and stop it; it waits at the corner instead."""
    here = np.array([6.0, 5.7])
    carrot, _left, bend = reference_pilot._ahead(HAIRPIN, 0, here, reference_pilot.CARROT_M)
    assert bend < reference_pilot.CARROT_M
    assert np.allclose(carrot, HAIRPIN[1])
    assert float(np.hypot(*(carrot - here))) > reference_pilot.ARRIVE_M


def test_the_pilot_moves_on_past_a_corner_it_has_reached():
    """Within ARRIVE_M of a corner the pilot takes the legs after it, so the hairpin's far side carries it on."""
    assert reference_pilot._progress(HAIRPIN, 0, np.array([5.0, 7.0])) == 0
    assert reference_pilot._progress(HAIRPIN, 0, np.array([7.6, 5.6])) == 2
    assert reference_pilot._progress(HAIRPIN, 2, np.array([-27.5, 3.4])) == 2
