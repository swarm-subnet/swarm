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

"""Test Field v2's seed checks and reference pilot: the replacement chain, the kept verdicts under the family's own
folder, the flights run through Sentinel's machinery with the field's pilot, and what the pilot reports."""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pybullet as p
import pytest

from swarm.challenge_families import build_benchmark_tasks
from swarm.challenge_families.solar_patrol import camera, coverage, laser, sensor_noise, zoom
from swarm.challenge_families.solar_patrol import seed_checks as sentinel_checks
from swarm.challenge_families.solar_patrol.contract import RGB_SHAPE, THERMAL_SHAPE, ZOOM_SHAPE, new_state
from swarm.challenge_families.test_field_v2 import (
    cover,
    field_cut,
    grass,
    install,
    people,
    reference_pilot,
    reports,
    seed_checks,
    site,
    wind,
)
from swarm.challenge_families.test_field_v2.contract import ACTION_DIM, ACTION_INDEX, FAMILY_ID, REPORT_REACH_M
from swarm.challenge_families.test_field_v2.family import TestFieldV2ChallengeFamily
from swarm.challenge_families.test_field_v2.seed_checks import FieldVerdict, candidates
from swarm.constants import SIM_DT
from swarm.core.maps import andorra
from validator.tests import field_v2_stand_in
from validator.tests.test_field_v2_family import _SHIPPED

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


def test_the_candidates_are_the_fields_own():
    """A seed's candidates start with the seed, never change, and differ from Sentinel's chain for the same seed."""
    chain = list(candidates(7))
    assert chain[0] == 7 and chain == list(candidates(7)) and len(set(chain)) == len(chain)
    assert chain[1:] != list(sentinel_checks.candidates(7))[1:]


def test_verdicts_flown_before_some_seeds_were_nights_are_never_read(kept):
    """The check's version keeps its verdicts in a folder of its own, so a verdict kept under an older version is
    not read; and so does the installed worlds package, since a verdict holds only for the areas it was flown on."""
    name, version = seed_checks.cache_dir().name, seed_checks.CHECK_VERSION
    assert version == 14 and f"-c{version}-w" in name
    for other in (name.replace(f"-c{version}-", f"-c{version - 1}-"), name + ".older"):
        (kept / other).mkdir()
        (kept / other / "4.json").write_text(json.dumps(asdict(FieldVerdict(seed=4, passed=True, end_reason="landed"))))
    assert seed_checks.cached(4) is None


def test_an_area_override_has_its_own_verdict_folder(kept, monkeypatch):
    """Verdicts in an alternate root stay separate, with one folder for equivalent paths."""
    monkeypatch.delenv(andorra.ROOT_ENV, raising=False)
    packaged = seed_checks.cache_dir()
    assert packaged == kept / f"{FAMILY_ID}-1.0.0-c{seed_checks.CHECK_VERSION}-w{seed_checks.sentinel_install._installed('swarm-worlds')}"
    seed_checks.keep(FieldVerdict(seed=4, passed=True))
    override = kept / "dev" / "andorra"
    monkeypatch.setenv(andorra.ROOT_ENV, str(override))
    own = seed_checks.cache_dir()
    assert own.parent == kept and own.name.startswith(packaged.name + "-") and seed_checks.cached(4) is None
    monkeypatch.setenv(andorra.ROOT_ENV, str(override / ".." / "andorra"))
    assert seed_checks.cache_dir() == own
    monkeypatch.setenv(andorra.ROOT_ENV, str(kept / "elsewhere"))
    assert seed_checks.cache_dir() not in {packaged, own}


def test_a_check_never_rounds_a_distance_up_onto_the_rule_and_fails_a_dock_cut_off_from_its_field():
    """A pass 4.999 m from a person fails as too close; a dock whose piece of free ground holds under half of the
    field's fails as cut off; a verdict kept before that share was measured is read as it was."""
    landed = dict(seed=1, passed=False, end_reason="landed", landed_s=30.0, threats=0, seen=0)
    assert math.floor(4.999 * 100.0) / 100.0 == 4.99
    assert reference_pilot.failure(FieldVerdict(**landed, min_person_m=4.99)) == "too_close"
    assert reference_pilot.failure(FieldVerdict(**landed, min_person_m=5.0, reach_share=1.0)) == ""
    assert reference_pilot.failure(FieldVerdict(**landed, reach_share=0.49)) == "field_cut_off"
    assert reference_pilot.failure(FieldVerdict(**landed, reach_share=0.5)) == ""
    assert reference_pilot.failure(FieldVerdict(**landed)) == ""


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


def test_the_fields_check_processes_end_once_past_their_memory_bound(kept, monkeypatch):
    """The field's preparation hands Sentinel's flights its memory bound, so no check process keeps growing."""
    bounds = []

    class Flights(sentinel_checks.Flights):
        """Flights that note their bound and keep a passing verdict for each seed handed to them."""

        def fly(self, seeds):
            """Note the bound, then keep a verdict for each seed of the batch."""
            bounds.append(self.retire_rss_mb)
            for seed in seeds:
                seed_checks.keep(FieldVerdict(seed=seed, passed=True))

    monkeypatch.setattr(sentinel_checks, "Flights", Flights)
    seed_checks.prepare([11], workers=1)
    assert bounds and set(bounds) == {seed_checks.RETIRE_RSS_MB}


def test_the_field_counts_a_check_process_at_its_own_memory(monkeypatch):
    """The field sizes its pool by its own memory per check, so a 48 GB machine runs one where Sentinel runs two."""
    monkeypatch.delenv(sentinel_checks.WORKERS_ENV, raising=False)
    monkeypatch.setattr(sentinel_checks.os, "cpu_count", lambda: 12)
    monkeypatch.setattr(sentinel_checks, "_total_ram_mb", lambda: 48000.0)
    assert seed_checks.CHECK_RAM_MB > seed_checks.RETIRE_RSS_MB
    assert (seed_checks.workers(), sentinel_checks.workers()) == (1, 2)


def test_a_report_press_carries_the_box_and_the_position_as_shares():
    """The pilot's press sets the report button, the box and the position as shares of 100 m from the dock."""
    a = np.zeros(ACTION_DIM, dtype=np.float32)
    reference_pilot.Pilot.press(a, (0.5, 0.25, 0.1, 0.2), np.array([30.0, -150.0]))
    assert a[ACTION_INDEX["report"]] == 1.0
    assert [a[ACTION_INDEX[f"report_{k}"]] for k in ("cx", "cy", "w", "h")] == pytest.approx([0.5, 0.25, 0.1, 0.2])
    assert a[ACTION_INDEX["report_east"]] == pytest.approx(30.0 / REPORT_REACH_M)
    assert a[ACTION_INDEX["report_north"]] == -1.0


def _on_field_and_passing():
    """A patrol with two people on the field and a third passing outside it."""
    return SimpleNamespace(people=[people.Person(index=k, pose="standing", xy=(0.0, 0.0), ground_z=0.0, yaw=0.0,
                                                 inside=k < 2) for k in range(3)])


def test_the_pilots_eyes_read_each_frame_once_and_never_one_too_high(monkeypatch):
    """A frame is read once, its people boxed as reports judge them, the step it was taken on kept, and a frame from
    over 24 m holds nobody."""
    objects = np.full((480, 640), -1, dtype=np.int32)
    objects[100:140, 320:336] = 9
    view = SimpleNamespace(objects=objects, sees=True, eye=(0.0, 0.0, 20.0), width=640, height=480, step=40)
    height = {"m": 20.0}
    monkeypatch.setattr(sensor_noise, "shown_view", lambda ep, image: view)
    monkeypatch.setattr(coverage, "height_above_ground", lambda env, ep, eye: height["m"])
    monkeypatch.setattr(people, "bodies", lambda ep: {9: 1})
    eyes = reference_pilot.Eyes(None, _on_field_and_passing())
    assert eyes.fresh() == {1: pytest.approx((328.0 / 640.0, 120.0 / 480.0, 16.0 / 640.0, 40.0 / 480.0))}
    assert eyes.step == 40
    assert eyes.fresh() == {}
    eyes.last, height["m"] = None, reports.MAX_REPORT_HEIGHT_M + 1.0
    assert eyes.fresh() == {}


class _Frames:
    """Eyes that read the frames given in turn, each a dict of people and boxes; None is no new frame."""

    def __init__(self, frames):
        """No frame read yet."""
        self.frames, self.last, self.step = list(frames), None, 0

    def fresh(self):
        """The next frame's people, read as new, or nothing when no new frame came."""
        frame = self.frames.pop(0)
        if frame is None:
            return {}
        self.last, self.step = object(), self.step + 25
        return frame


def test_a_new_frame_that_shows_nobody_leaves_no_old_box_to_press(monkeypatch):
    """Boxes are pressed only on the frame they were read on: once a new frame shows nobody, someone left waiting
    from the frame before is not reported with that frame's box."""
    pilot = reference_pilot.Pilot.__new__(reference_pilot.Pilot)
    pilot.ep = SimpleNamespace(phase="flying")
    pilot.search = SimpleNamespace(action=lambda decision: np.zeros(ACTION_DIM, dtype=np.float32), beast=lambda n: None)
    pilot.reported, pilot.queue, pilot.pressed = set(), {}, -2
    pilot.eyes = _Frames([{0: (0.2, 0.2, 0.1, 0.1), 1: (0.6, 0.6, 0.1, 0.1)}, None, {}, None, None])
    monkeypatch.setattr(reference_pilot.Pilot, "point", lambda self, index, step=None: np.zeros(2))
    presses = [pilot.action(decision)[ACTION_INDEX["report"]] for decision in range(5)]
    assert presses == [1.0, 0.0, 0.0, 0.0, 0.0] and pilot.reported == {0}


def test_the_pilot_names_an_animal_an_animal_and_a_person_a_person(monkeypatch):
    """A subject past the people in the pilot's count is an animal: its report carries the animal class and where the
    animal stood at the frame's step, east and north of the dock; a person's carries the person class."""
    pilot = reference_pilot.Pilot.__new__(reference_pilot.Pilot)
    beast = SimpleNamespace(track=SimpleNamespace(place=lambda t: (40.0 + t, 10.0, 1.0, 0.0)))
    pilot.ep = SimpleNamespace(phase="flying", step=0, dock_position=np.array([5.0, -5.0, 2.0]))
    pilot.search = SimpleNamespace(action=lambda decision: np.zeros(ACTION_DIM, dtype=np.float32),
                                   beast=lambda n: beast if n >= 2 else None)
    pilot.reported, pilot.queue, pilot.pressed = set(), {}, -2
    pilot.eyes = _Frames([{2: (0.2, 0.2, 0.1, 0.1)}, None, {0: (0.6, 0.6, 0.1, 0.1)}, None])
    pilot.eyes.step = 75
    monkeypatch.setattr(reference_pilot.people, "reference", lambda ep, index, step=None: np.array([1.0, 2.0]))
    first = pilot.action(0)
    assert first[ACTION_INDEX["report"]] == 1.0 and first[ACTION_INDEX["report_class"]] == 1.0
    assert first[ACTION_INDEX["report_east"]] * REPORT_REACH_M == pytest.approx(40.0 + 100 * SIM_DT - 5.0)
    assert first[ACTION_INDEX["report_north"]] * REPORT_REACH_M == pytest.approx(15.0)
    pilot.action(1)
    third = pilot.action(2)
    assert third[ACTION_INDEX["report"]] == 1.0 and third[ACTION_INDEX["report_class"]] == 0.0
    assert pilot.reported == {0, 2}


def test_the_search_and_the_eyes_count_the_animals_alike_after_everyone_the_patrol_holds(monkeypatch):
    """With two people on the field, one passing outside and two animals to report, the animals take places 3 and 4
    in the pilot's count: the search plans for exactly those two there, and the eyes give a frame's box on each
    animal's body under the same place, so a report names the animal that was seen."""
    goat, dog = SimpleNamespace(name="goat"), SimpleNamespace(name="dog")
    ep = SimpleNamespace()
    monkeypatch.setattr(people, "people", lambda episode: [SimpleNamespace(inside=True), SimpleNamespace(inside=True),
                                                           SimpleNamespace(inside=False)])
    monkeypatch.setattr(reference_pilot.animals, "to_report", lambda episode: {31: goat, 17: dog})
    assert reference_pilot.herd(ep) == {3: (31, goat), 4: (17, dog)}
    search = reference_pilot.Search.__new__(reference_pilot.Search)
    search.everyone = [person for person in people.people(ep) if person.inside]
    search.beasts = {n: animal for n, (_uid, animal) in reference_pilot.herd(ep).items()}
    assert search.subjects() == [0, 1, 3, 4]
    assert [search.beast(n) for n in (0, 1, 2, 3, 4)] == [None, None, None, goat, dog]
    objects = np.full((480, 640), -1, dtype=np.int32)
    objects[100:140, 320:336], objects[300:320, 100:140] = 17, 31
    view = SimpleNamespace(objects=objects, sees=True, eye=(0.0, 0.0, 9.0), width=640, height=480, step=40)
    monkeypatch.setattr(sensor_noise, "shown_view", lambda episode, image: view)
    monkeypatch.setattr(coverage, "height_above_ground", lambda env, episode, eye: 9.0)
    monkeypatch.setattr(reports, "owners", lambda episode: {})
    seen = reference_pilot.Eyes(None, ep).fresh()
    assert set(seen) == {3, 4}
    assert seen[4] == pytest.approx((328.0 / 640.0, 120.0 / 480.0, 16.0 / 640.0, 40.0 / 480.0))


def test_an_animal_is_looked_at_from_a_ring_viewpoint_three_times_and_then_left_to_the_sweep(monkeypatch):
    """An animal is planned for where it is now and looked at from a ring viewpoint laid round that spot, at the
    cruise height; after three tries the search has no further place for it."""
    search = reference_pilot.Search.__new__(reference_pilot.Search)
    beast = SimpleNamespace(track=SimpleNamespace(place=lambda t: (40.0 + t, 10.0, 1.0, 0.0)))
    search.ep, search.everyone, search.beasts, search.tried = SimpleNamespace(step=50), [], {0: beast}, {0: 0}
    assert search.planned(0).tolist() == search.now(0).tolist() == [40.0 + 50 * SIM_DT, 10.0]
    monkeypatch.setattr(reference_pilot.Search, "viewpoint", lambda self, n, pos: np.array([33.0, 10.0]))
    for tried in range(reference_pilot.BEAST_TRIES):
        search.tried[0] = tried
        spot, height = search.goal(0, np.zeros(3))
        assert spot.tolist() == [33.0, 10.0] and height is None
    search.tried[0] = reference_pilot.BEAST_TRIES
    assert reference_pilot.BEAST_TRIES == 3 and search.goal(0, np.zeros(3)) is None


def test_a_frame_that_shows_nothing_is_read_as_new_so_no_older_box_is_pressed_on_it(monkeypatch):
    """After a frame that showed someone, a new frame that sees nothing gives the eyes nobody and still counts as
    read: the pilot's waiting boxes belong to a frame no longer shown and are dropped."""
    objects = np.full((480, 640), -1, dtype=np.int32)
    objects[100:140, 320:336] = 9
    seeing = SimpleNamespace(objects=objects, sees=True, eye=(0.0, 0.0, 9.0), width=640, height=480, step=40)
    dark = SimpleNamespace(objects=objects, sees=False, eye=(0.0, 0.0, 9.0), width=640, height=480, step=65)
    shown = {"view": seeing}
    monkeypatch.setattr(sensor_noise, "shown_view", lambda episode, image: shown["view"])
    monkeypatch.setattr(coverage, "height_above_ground", lambda env, episode, eye: 9.0)
    monkeypatch.setattr(people, "people", lambda episode: [SimpleNamespace(inside=True)])
    monkeypatch.setattr(reports, "owners", lambda episode: {9: ("person", 0)})
    monkeypatch.setattr(reference_pilot.animals, "to_report", lambda episode: {})
    eyes = reference_pilot.Eyes(None, SimpleNamespace())
    assert set(eyes.fresh()) == {0} and eyes.last is seeing
    shown["view"] = dark
    assert eyes.fresh() == {} and eyes.last is dark and eyes.step == 65


def test_the_pilot_presses_on_the_next_lit_frame_after_a_dark_frame(monkeypatch):
    """A queued box expires on a dark press decision and the next lit frame supplies its box and step."""
    first = np.full((100, 100), -1, dtype=np.int32)
    first[10:20, 10:20], first[30:40, 30:40] = 9, 10
    later = np.full_like(first, -1)
    later[60:80, 50:70] = 10
    start = camera.View("colour", (0, 0, 9), (0, 0, -1), (0, 1, 0), 100, 100, 50, True, 40, objects=first)
    dark = camera.View("colour", start.eye, start.forward, start.up, 100, 100, 50, False, 65, objects=first)
    lit = camera.View("colour", start.eye, start.forward, start.up, 100, 100, 50, True, 90, objects=later)
    ep = SimpleNamespace(phase="flying", shown=start, people=[SimpleNamespace(inside=True)] * 2, animals={})
    monkeypatch.setattr(sensor_noise, "shown_view", lambda episode, image: episode.shown)
    monkeypatch.setattr(coverage, "height_above_ground", lambda env, episode, eye: 9.0)
    monkeypatch.setattr(reports, "owners", lambda episode: {9: ("person", 0), 10: ("person", 1)})
    monkeypatch.setattr(people, "reference", lambda episode, index, step: (step, index))
    pilot = reference_pilot.Pilot.__new__(reference_pilot.Pilot)
    pilot.ep = ep
    pilot.search = SimpleNamespace(action=lambda decision: np.zeros(ACTION_DIM, dtype=np.float32), beast=lambda n: None)
    pilot.reported, pilot.queue, pilot.pressed = set(), {}, -2
    pilot.eyes = reference_pilot.Eyes(None, ep)

    assert pilot.action(0)[ACTION_INDEX["report"]] == 1.0 and pilot.reported == {0}
    assert pilot.action(1)[ACTION_INDEX["report"]] == 0.0 and set(pilot.queue) == {1}
    ep.shown = dark
    assert pilot.action(2)[ACTION_INDEX["report"]] == 0.0 and not pilot.queue
    ep.shown = lit
    action = pilot.action(3)
    assert action[ACTION_INDEX["report"]] == 1.0 and pilot.reported == {0, 1}
    assert action[ACTION_INDEX["report_east"]] * REPORT_REACH_M == pytest.approx(lit.step)
    assert action[ACTION_INDEX["report_cx"]] == pytest.approx(0.6)


def test_a_seed_whose_animal_stays_unreported_or_misreported_fails():
    """A landed flight that reported everyone still fails when an animal to report was left out, or when a report
    that named an animal failed; with every animal reported it passes."""
    landed = dict(seed=1, passed=False, end_reason="landed", landed_s=60.0, threats=2, seen=2, animals=2)
    assert reference_pilot.failure(FieldVerdict(**landed, animals_seen=1)) == "animal_unreported"
    assert reference_pilot.failure(FieldVerdict(**landed, animals_seen=2, wrong_animal_reports=1)) == "wrong_animal_report"
    assert reference_pilot.failure(FieldVerdict(**landed, animals_seen=2)) == ""
    assert reference_pilot.failure(FieldVerdict(**dict(landed, seen=1), animals_seen=1)) == "person_unreported"


def test_the_pilots_eyes_see_nobody_passing_outside_the_field_unless_told_to(monkeypatch):
    """A frame holding someone on the field and someone passing outside it gives the pilot only the first; eyes told
    to take the people outside too give both."""
    objects = np.full((480, 640), -1, dtype=np.int32)
    objects[100:140, 320:336] = 9
    objects[300:340, 100:116] = 11
    view = SimpleNamespace(objects=objects, sees=True, eye=(0.0, 0.0, 9.0), width=640, height=480, step=40)
    monkeypatch.setattr(sensor_noise, "shown_view", lambda ep, image: view)
    monkeypatch.setattr(coverage, "height_above_ground", lambda env, ep, eye: 9.0)
    monkeypatch.setattr(people, "bodies", lambda ep: {9: 1, 11: 2})
    ep = _on_field_and_passing()
    assert set(reference_pilot.Eyes(None, ep).fresh()) == {1}
    assert set(reference_pilot.Eyes(None, ep, outside=True).fresh()) == {1, 2}


def test_the_pilot_neither_visits_nor_keeps_off_someone_passing_outside(monkeypatch):
    """The search plans for the people on the field only: someone passing outside is no target and takes no free
    ground."""
    ep = SimpleNamespace(people=[people.Person(index=0, pose="standing", xy=(10.0, 0.0), ground_z=0.0, yaw=0.0),
                                 people.Person(index=1, pose="passing", xy=(-20.0, 0.0), ground_z=0.0, yaw=0.0,
                                               inside=False)],
                         fence=np.array([[-30.0, -30.0], [30.0, -30.0], [30.0, 30.0], [-30.0, 30.0]]),
                         crowns=np.zeros((0, 3)), dock_position=np.array([0.0, -20.0, 0.0]))
    search = reference_pilot.Search(None, ep, set())
    assert search.persons.tolist() == [[10.0, 0.0]]
    assert search.free.holds(np.array([-22.0, 0.0]))


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


def test_a_night_sweep_lies_in_lanes_as_far_apart_as_the_thermal_lens_sees():
    """At night the sweep's runs lie NIGHT_LANE_M apart, closer than by day."""
    field = np.array([[-40.0, -15.0], [40.0, -15.0], [40.0, 15.0], [-40.0, 15.0]])
    free = reference_pilot.Free(field, np.zeros((0, 3)), np.zeros((0, 2)))
    offsets = sorted({round(float(run[0][1]), 3) for run in reference_pilot.lanes(free, field, reference_pilot.NIGHT_LANE_M)})
    assert np.allclose(np.diff(offsets), reference_pilot.NIGHT_LANE_M) and reference_pilot.NIGHT_LANE_M < reference_pilot.LANE_M


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


_SQUARE = np.array([[-30.0, -30.0], [30.0, -30.0], [30.0, 30.0], [-30.0, 30.0]])


def _person(index, xy, ground_z=0.0, cover_kind=cover.OPEN):
    """Someone standing still on the field at a spot, 1.8 m tall, under a cover."""
    return people.Person(index=index, pose="standing", xy=xy, ground_z=ground_z, yaw=0.0, cover=cover_kind)


def _searching(sweep, people_xy=(), covers=None, grounds=None, night=False):
    """A search over a 60 m square with the drone flying over the dock and people standing at people_xy, on the
    ground heights and under the covers given (flat and open if not), sweeping after the visits or not, by day or
    under a moon."""
    covers = covers or [cover.OPEN] * len(people_xy)
    grounds = grounds or [0.0] * len(people_xy)
    ep = SimpleNamespace(people=[_person(k, xy, z, kind) for k, (xy, z, kind) in enumerate(zip(people_xy, grounds, covers))],
                         fence=_SQUARE, crowns=np.zeros((0, 3)), dock_position=np.zeros(3), phase="flying", time_s=20.0,
                         terrain_uids=frozenset())
    env = SimpleNamespace(pos=np.array([[0.0, 0.0, 10.0]]), vel=np.zeros((1, 3)), rpy=np.zeros((1, 3)), EP_LEN_SEC=200.0,
                          CLIENT=None, DRONE_IDS=(), _sun=SimpleNamespace(night=night))
    return reference_pilot.Search(env, ep, set(), sweep=sweep)


def test_the_pilot_holds_the_thermal_feed_at_night_and_the_colour_feed_by_day():
    """Under the seed's moon every action holds the thermal button and the sweep's lanes are the night's; by day the
    colour feed."""
    day = _searching(sweep=True)
    assert not day.night and day.action(10)[ACTION_INDEX["thermal"]] == 0.0
    night = _searching(sweep=True, night=True)
    assert night.night and night.action(10)[ACTION_INDEX["thermal"]] == 1.0
    night.ep.phase = "docked"
    assert night.action(0)[ACTION_INDEX["thermal"]] == 1.0
    assert len(night.runs) > len(day.runs)


def test_a_check_flies_home_straight_after_the_visits_and_a_patrol_pilot_still_sweeps():
    """With the sweep off the pilot turns home once nobody is left to visit and presses return home over the dock;
    with it on, as reference models and time flights fly it, it sweeps."""
    search = _searching(sweep=False)
    search.action(10)
    assert search.mode == "home"
    assert search.action(11)[ACTION_INDEX["return_home"]] == 1.0
    sweeping = _searching(sweep=True)
    sweeping.action(10)
    assert sweeping.mode == "sweep" and sweeping.route is not None


def test_a_check_sweeps_for_someone_its_visits_gave_up_on_and_goes_home_once_he_is_reported():
    """A person no viewpoint showed is looked for in the sweep, and the check heads home the moment he is reported."""
    search = _searching(sweep=False, people_xy=[(15.0, 15.0)])
    search.skipped.add(0)
    search.action(10)
    assert search.mode == "sweep" and search.route is not None
    search.reported.add(0)
    search.action(11)
    assert search.mode == "home"


def test_the_pilot_keeps_off_a_person_only_as_far_as_the_five_metre_rule_needs_at_its_height():
    """Across the ground the pilot keeps PERSON_CLEAR_M off someone whose head is at its height, nothing off someone
    that far under it, and in between just enough to stay PERSON_CLEAR_M from his head; so over low ground it may fly
    over him."""
    clear, cruise = reference_pilot.PERSON_CLEAR_M, reference_pilot.CRUISE_M
    assert reference_pilot.reach(cruise, cruise) == pytest.approx(clear)
    assert reference_pilot.reach(cruise - clear, cruise) == 0.0
    for head in (4.0, 6.0, 8.0):
        assert math.hypot(reference_pilot.reach(head, cruise), cruise - head) == pytest.approx(clear)
    assert _searching(sweep=False, people_xy=[(15.0, 15.0)]).free.holds(np.array([15.0, 15.0]))
    high = _searching(sweep=False, people_xy=[(15.0, 15.0)], grounds=[5.0])
    edge = reference_pilot.reach(5.0 + 1.8, cruise)
    assert not high.free.holds(np.array([15.0 + edge - 0.6, 15.0]))
    assert high.free.holds(np.array([15.0 + edge + 0.6, 15.0]))


def test_the_pilot_looks_at_someone_from_its_own_piece_of_free_ground():
    """A row of small crowns with a gap a drone may look through but not fly through cuts the free ground in two;
    the drone on the west piece looks at someone east of the row from a viewpoint on the west piece, which it can
    reach, never from the nearer ring across the row."""
    row = np.array([[0.0, float(y), 0.5] for y in range(-30, 31) if abs(y) >= 2])
    ep = SimpleNamespace(people=[_person(0, (5.0, 0.0))], fence=_SQUARE, crowns=row, dock_position=np.array([-20.0, 0.0, 0.0]))
    search = reference_pilot.Search(None, ep, set())
    drone = np.array([-20.0, 0.0])
    assert search.free.piece(drone) != search.free.piece(np.array([20.0, 0.0]))
    spot = search.viewpoint(0, np.append(drone, 9.5))
    assert spot is not None and spot[0] < 0.0
    assert search.free.piece(spot) == search.free.piece(drone) and search.free.path(drone, spot) is not None


def test_someone_under_a_tree_is_looked_at_from_his_places_in_turn_and_anyone_else_from_a_ring_first(monkeypatch):
    """Someone under a tree is looked at from his places, best first, and given up once all were tried; someone else
    from a ring viewpoint first and then from his places. A place at the cruise height is flown at it."""
    places = np.array([[0.0, -20.0, 3.0], [-20.0, 0.0, reference_pilot.CRUISE_M]])
    monkeypatch.setattr(reference_pilot.Search, "_looks", lambda self, n: places)
    search = _searching(sweep=False, people_xy=[(15.0, 15.0), (-15.0, 15.0)], covers=[cover.UNDER, cover.OPEN])
    here = np.array([0.0, 0.0, reference_pilot.CRUISE_M])
    spot, z = search.goal(0, here)
    assert np.allclose(spot, [0.0, -20.0]) and z == 3.0
    search.tried[0] += 1
    spot, z = search.goal(0, here)
    assert np.allclose(spot, [-20.0, 0.0]) and z is None
    search.tried[0] += 1
    assert search.goal(0, here) is None
    ring, z = search.goal(1, here)
    assert z is None and float(np.hypot(*(ring - [-15.0, 15.0]))) == pytest.approx(reference_pilot.STANDOFF_M[0])
    search.tried[1] += 1
    spot, z = search.goal(1, here)
    assert np.allclose(spot, [0.0, -20.0]) and z == 3.0


def test_the_places_kept_for_someone_show_him_best_first_and_measuring_stops_once_enough_show_him_well(monkeypatch):
    """Places are measured in the fairness rule's order and only while fewer than LOOKS_KEPT show LOOK_PX of him;
    those showing SEEN_PX or more are kept, most pixels first, so a place under the rule's clear count still counts
    when few show him, and a place the drone cannot hold is never measured."""
    rows = np.array([[float(k), -20.0, 3.0, 1.0] for k in range(8)])
    counts = {7: [12, 300, 5, 80, 60, 70, 90, 400], 8: [12, 300, 5, 80, 60, 20, 8, 9]}
    measured = []

    def pixels(uids, target, places):
        """His pixels from each place, as listed, noting each place measured."""
        for place in places:
            measured.append(int(place[0]))
            yield counts[uids[0]][int(place[0])]

    monkeypatch.setattr(reference_pilot.Search, "holdable", lambda self, place: int(place[0]) != 3)
    search = _searching(sweep=False, people_xy=[(0.0, -10.0), (10.0, -10.0)])
    for person, uid in zip(search.ep.people, counts):
        person.body = SimpleNamespace(bodies=[uid])
    search.sight = SimpleNamespace(places=lambda where, standing: rows, pixels=pixels)
    looks = search._looks(0)
    assert measured == [0, 1, 2, 4, 5]
    assert looks[:, 0].tolist() == [1.0, 5.0, 4.0]
    assert search._looks(0) is looks and search.measured == 5
    assert search._looks(1)[:, 0].tolist() == [1.0, 4.0, 5.0]
    assert reference_pilot.SEEN_PX <= counts[8][5] < cover.CLEAR_PX


def test_the_pilot_comes_down_straight_to_a_low_place_and_climbs_back_straight_before_it_flies_on(monkeypatch):
    """Over his low place the pilot comes straight down to its height; once he is reported it climbs straight back
    to the cruise height before it moves on, and from under it never presses return home."""
    monkeypatch.setattr(reference_pilot.Search, "_looks", lambda self, n: np.array([[0.0, -20.0, 3.0]]))
    monkeypatch.setattr(site, "ground_at", lambda *args: 0.0)
    search = _searching(sweep=False, people_xy=[(0.0, -10.0), (15.0, 15.0)], covers=[cover.UNDER, cover.OPEN])
    up, ahead, aside = (ACTION_INDEX[name] for name in ("move_up", "move_forward", "move_right"))
    search.env.pos = np.array([[0.0, -20.0, reference_pilot.CRUISE_M]])
    a = search.action(10)
    assert search.target == 0 and search.look_z == 3.0 and a[up] < 0.0 and a[ahead] == a[aside] == 0.0
    search.env.pos = np.array([[0.0, -20.0, 3.0]])
    assert search.action(11)[up] == pytest.approx(0.0)
    search.reported.add(0)
    a = search.action(12)
    assert search.target == 1 and a[up] > 0.0 and a[ahead] == a[aside] == 0.0
    search.env.pos = np.array([[0.0, -20.0, reference_pilot.CRUISE_M]])
    a = search.action(13)
    assert np.hypot(a[ahead], a[aside]) > 0.0
    search.reported.add(1)
    search.env.pos = np.array([[0.0, 0.0, 3.0]])
    for decision in (14, 15, 16):
        a = search.action(decision)
        assert search.mode == "home" and a[ACTION_INDEX["return_home"]] == 0.0 and a[up] > 0.0
    search.env.pos = np.array([[0.0, 0.0, reference_pilot.CRUISE_M]])
    assert search.action(17)[ACTION_INDEX["return_home"]] == 1.0


def test_a_low_place_needs_a_column_nothing_solid_stands_in_and_nobody_near():
    """A place under the cruise height is kept only where rays dropped round it out to COLUMN_M meet nothing solid
    within COLUMN_CLEAR_M under it, and nobody stands within PERSON_CLEAR_M across the ground; a place at the cruise
    height needs neither."""
    cli = p.connect(p.DIRECT)
    try:
        p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[40.0, 40.0, 0.5], physicsClientId=cli),
                          basePosition=[0.0, 0.0, -0.5], physicsClientId=cli)
        p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 1.0], physicsClientId=cli),
                          basePosition=[10.0, -20.0, 1.0], physicsClientId=cli)
        search = _searching(sweep=False, people_xy=[(-15.0, 15.0)])
        search.env.CLIENT = cli
        assert search.column_clear(np.array([20.0, -20.0]), 3.0)
        assert not search.column_clear(np.array([11.5, -20.0]), 3.0)
        assert search.column_clear(np.array([11.5, -20.0]), 2.0 + reference_pilot.COLUMN_CLEAR_M + 0.1)
        assert search.holdable(np.array([20.0, -20.0, 3.0]))
        assert not search.holdable(np.array([-15.0, 10.0, 3.0]))
        assert search.holdable(np.array([-15.0, 10.0, reference_pilot.CRUISE_M]))
    finally:
        p.disconnect(cli)


def test_a_seed_whose_field_cannot_be_cut_fails_as_its_own(monkeypatch):
    """A seed no area gives a field fails its check like one with no dock ground or no room, never raises."""
    def no_field(seed):
        """Fail as the cutter does when no area holds the seed's field."""
        raise field_cut.NoField("no area holds the field")

    monkeypatch.setattr(reference_pilot, "check", no_field)
    verdict = reference_pilot.judge(5)
    assert not verdict.passed and verdict.reason == "error:no area holds the field"


def test_a_flight_that_came_within_the_safe_distance_of_someone_fails_the_seed():
    """A landed flight that reported everyone with no false alarm still fails when it came within the score's safe
    distance of someone; what failed first is named."""
    landed = dict(seed=1, passed=False, end_reason="landed", landed_s=60.0, threats=2, seen=2)
    assert reference_pilot.failure(FieldVerdict(**landed)) == ""
    assert reference_pilot.failure(FieldVerdict(**landed, min_person_m=5.2)) == ""
    assert reference_pilot.failure(FieldVerdict(**landed, min_person_m=4.9)) == "too_close"
    assert reference_pilot.failure(FieldVerdict(**dict(landed, seen=1), min_person_m=4.9)) == "person_unreported"
    assert reference_pilot.failure(FieldVerdict(**dict(landed, landed_s=None, end_reason="collision"))) == "not_landed:collision"


def test_a_checks_flight_draws_the_feed_only_while_the_pilot_may_read_it_and_never_reads_the_laser(monkeypatch):
    """The family takes the feed, the zoom and the laser every step, each picture after the grass of its own view is
    shown and a laser reading that is due after the grass the wide lens shows from where the drone is; a check's
    flight, the family's own runtime with its hooks swapped, drops the laser, and the feed while the drone waits in
    the dock for its lids and once everyone on the field is validly reported."""
    calls = []
    for module in (camera, zoom, laser):
        monkeypatch.setattr(module, "update", lambda env, ep, name=module.__name__: calls.append(name.rsplit(".", 1)[-1]))
    monkeypatch.setattr(camera, "upcoming", lambda env, ep, step: f"feed view at step {step}")
    monkeypatch.setattr(camera, "aim", lambda env, ep: SimpleNamespace(eye="the drone"))
    monkeypatch.setattr(zoom, "upcoming", lambda env, ep: "zoom view")
    monkeypatch.setattr(grass, "show", lambda env, ep, view: calls.append(f"grass of the {view}"))
    monkeypatch.setattr(grass, "show_near", lambda env, ep, eye: calls.append(f"grass near {eye}"))
    ep = SimpleNamespace(outcome=SimpleNamespace(threats=2, valid_reports=1, animals=0, valid_animal_reports=0),
                         phase="taking_off", step=5, laser={"taken_step": None}, animals=None, people=None,
                         dock={"model": SimpleNamespace(opening=0.5)})
    pictures = ["grass of the feed view at step 5", "camera", "grass of the zoom view", "zoom"]
    TestFieldV2ChallengeFamily().sense(None, ep)
    assert calls == pictures + ["grass near the drone", "laser"]
    for taken, due in ((5, False), (5 - laser.READING_STEPS + 1, False), (5 - laser.READING_STEPS, True)):
        ep.laser["taken_step"] = taken
        calls.clear()
        TestFieldV2ChallengeFamily().sense(None, ep)
        assert calls == pictures + (["grass near the drone"] if due else []) + ["laser"]
    flight = reference_pilot.check_flight(TestFieldV2ChallengeFamily())
    assert type(flight) is TestFieldV2ChallengeFamily
    seen = []
    for phase, opening, valid in (("docked", 0.0, 1), ("taking_off", 0.5, 1), ("taking_off", 1.0, 1), ("flying", 1.0, 1),
                                  ("flying", 1.0, 2), ("returning", 1.0, 1)):
        ep.phase, ep.dock["model"].opening, ep.outcome.valid_reports = phase, opening, valid
        calls.clear()
        flight.sense(None, ep)
        assert "laser" not in calls and calls[-2:] == ["grass of the zoom view", "zoom"]
        assert calls[:-2] in ([], ["grass of the feed view at step 5", "camera"])
        seen.append("camera" in calls)
    assert seen == [False, False, True, True, False, True]
    # With everyone reported the feed is still drawn while an animal to report is left.
    ep.phase, ep.dock["model"].opening, ep.outcome.valid_reports, ep.outcome.animals = "flying", 1.0, 2, 1
    calls.clear()
    flight.sense(None, ep)
    assert "camera" in calls
    ep.outcome.valid_animal_reports = 1
    calls.clear()
    flight.sense(None, ep)
    assert "camera" not in calls


def test_the_patrols_first_picture_is_taken_over_the_grass_its_own_view_shows(monkeypatch):
    """The first view of a patrol has no picture yet, so its picture is taken while the view is built: the grass of
    the camera's aim is shown first. A later view shows the picture already taken and touches no grass."""
    calls = []
    monkeypatch.setattr(camera, "aim", lambda env, ep: "the first aim")
    monkeypatch.setattr(grass, "show", lambda env, ep, view: calls.append(f"grass of {view}"))
    monkeypatch.setattr(camera, "observe", lambda env, ep, state: calls.append("picture"))
    family = sys.modules[TestFieldV2ChallengeFamily.__module__]
    for module in (family.drone_state, family.wind, family.laser, family.ground_distance, family.flight_limit,
                   family.dock, zoom):
        monkeypatch.setattr(module, "observe", lambda env, ep, state: None)
    monkeypatch.setattr(family, "write_clock", lambda ep, state: None)
    monkeypatch.setattr(camera, "view", lambda ep: None)
    monkeypatch.setattr(zoom, "view", lambda ep: None)
    ep = SimpleNamespace(step=0, camera={"view": None}, frames=SimpleNamespace(rgb=None, thermal=None, zoom=None))
    TestFieldV2ChallengeFamily()._clean_view(None, ep)
    assert calls == ["grass of the first aim", "picture"]
    ep.camera["view"] = "taken"
    calls.clear()
    TestFieldV2ChallengeFamily()._clean_view(None, ep)
    assert calls == ["picture"]


def test_a_checks_flight_shows_the_frames_views_from_blank_images_without_the_stream_look(monkeypatch):
    """A check's view shows the same frames' views as the family's, from blank colour and zoom images, so the stream's
    look is never worked out; the family's own runs it on both images."""
    looks = []
    monkeypatch.setattr(sensor_noise, "stream_look", lambda frame, *args: looks.append(frame) or frame)
    ep = SimpleNamespace(seed=3, step=7)
    sensor_noise.reset(None, ep)

    def snapshot():
        """A clean snapshot whose colour and zoom images are not blank."""
        return {"step": 7, "state": new_state(), "rgb": np.ones(RGB_SHAPE, dtype=np.float32),
                "thermal": np.zeros(THERMAL_SHAPE, dtype=np.float32), "zoom": np.ones(ZOOM_SHAPE, dtype=np.float32),
                "feed_view": "feed", "zoom_view": "zoom"}

    view = reference_pilot.check_flight(TestFieldV2ChallengeFamily())._build_view(None, ep, snapshot())
    assert not looks and not view["rgb"].any() and not view["zoom"].any()
    assert ep.sensor_noise["coming"] == {"feed": "feed", "zoom": "zoom"}
    TestFieldV2ChallengeFamily()._build_view(None, ep, snapshot())
    assert len(looks) == 2


STAND_IN_SEED = 0                      # four people on the stand-in field, none walking


@pytest.mark.timeout(120)
def test_a_seed_check_on_the_stand_in_reports_everyone_with_no_sweep_no_laser_and_no_picture_after(tmp_path, monkeypatch):
    """A check flies its own runtime on the stand-in field: it reports everyone and lands, never sweeps, never reads
    the laser, and after the first picture takes one only with the lids open and someone still to be validly
    reported."""
    if not _SHIPPED:
        pytest.skip("the installed swarm-worlds has no M4TD or no intruder bodies")
    monkeypatch.setattr(install, "area_problems", lambda: [])
    field_v2_stand_in.use_area(tmp_path, monkeypatch)
    field_v2_stand_in.use_people(tmp_path / "people", monkeypatch)
    pictures, lasers, modes = [], [], []
    capture, switch = camera.capture, reference_pilot.Search._switch
    monkeypatch.setattr(camera, "capture", lambda env, ep: (pictures.append(
        ep.step == 0 or (ep.outcome.valid_reports < ep.outcome.threats and not reference_pilot.waiting_for_lids(ep))),
        capture(env, ep)))
    monkeypatch.setattr(laser, "update", lambda env, ep: lasers.append(ep.step))
    monkeypatch.setattr(reference_pilot.Search, "_switch", lambda self, mode: (modes.append(mode), switch(self, mode)))
    verdict = reference_pilot.check(STAND_IN_SEED)
    assert verdict.passed and verdict.threats > 0 and verdict.seen == verdict.threats
    assert modes == ["home"] and not lasers
    assert pictures and all(pictures)


NIGHT_STAND_IN_SEED = 16               # a night: five people on the stand-in field


@pytest.mark.timeout(120)
def test_a_night_seed_check_on_the_stand_in_reports_everyone_on_thermal_frames(tmp_path, monkeypatch):
    """Under the seed's moon the check flies the thermal feed only, reports everyone from it, lands and passes."""
    if not _SHIPPED:
        pytest.skip("the installed swarm-worlds has no M4TD or no intruder bodies")
    monkeypatch.setattr(install, "area_problems", lambda: [])
    field_v2_stand_in.use_area(tmp_path, monkeypatch)
    field_v2_stand_in.use_people(tmp_path / "people", monkeypatch)
    feeds = []
    capture = camera.capture
    monkeypatch.setattr(camera, "capture", lambda env, ep: (feeds.append((ep.phase, ep.camera["thermal"])), capture(env, ep)))
    assert site.night(NIGHT_STAND_IN_SEED)
    verdict = reference_pilot.check(NIGHT_STAND_IN_SEED)
    assert verdict.passed and verdict.night and verdict.threats == verdict.seen == 5
    assert all(thermal for phase, thermal in feeds if phase == "flying")
