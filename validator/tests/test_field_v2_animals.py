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

"""The animals near the field: the package format, which animals a seed holds, the ground they may use, plans that
keep every body on its ground and out of every other, birds that fly low over clear sky and leave the goats and dogs
as they were, boars and rats or mice that do the same, the same on every run, and bodies posed only for frames that
see them and judged as animals."""

from __future__ import annotations

import contextlib
import dataclasses
import io
import math
import os
from collections import Counter, deque
from types import SimpleNamespace

import numpy as np
import pybullet as p
import pytest

from swarm.challenge_families.solar_patrol import airframe, camera, coverage, laser, sensor_noise, zoom
from swarm.challenge_families.solar_patrol import reports as sentinel_reports
from swarm.challenge_families.solar_patrol.contract import Box
from swarm.challenge_families.test_field_v2 import animal_moves, animals, install, mix, people, reports, site, walkers
from swarm.challenge_families.test_field_v2 import animal_skins as skins
from swarm.challenge_families.test_field_v2 import task as field_task
from swarm.challenge_families.test_field_v2.contract import FieldReport
from swarm.challenge_families.test_field_v2.episode import FieldEpisode
from swarm.challenge_families.test_field_v2.family import TestFieldV2ChallengeFamily
from swarm.constants import SIM_DT
from swarm.utils.env_factory import make_env_with_initial_obs
from validator.tests import animal_stand_in, field_v2_stand_in
from validator.tests.test_field_v2_family import _SHIPPED

_HORIZON_S = 60.0


@pytest.fixture
def package(tmp_path, monkeypatch):
    """Point the animals at a generated package of a goat, three dogs, five birds, a boar, a piglet, a rat and a mouse."""
    monkeypatch.setattr(skins, "ANIMALS_DIR", animal_stand_in.write_package(tmp_path / "animals"))
    return skins.ANIMALS_DIR


def _ground(holes: bool = True) -> animal_moves.Ground:
    """A 90 m square of open ground with a closed rim, a wall across it and a pond, its middle third the field."""
    n = 180
    open_ = np.ones((n, n), dtype=bool)
    open_[:3], open_[-3:], open_[:, :3], open_[:, -3:] = False, False, False, False
    if holes:
        open_[60:66, 20:150] = False
        yy, xx = np.mgrid[:n, :n]
        open_[(yy - 120) ** 2 + (xx - 110) ** 2 < 15 ** 2] = False
    inside = np.zeros((n, n), dtype=bool)
    inside[60:120, 60:120] = True
    return animal_moves.Ground((-45.0, -45.0), 0.5, open_, inside, np.zeros((n, n)))


def _off_its_ground(ground: animal_moves.Ground, x: np.ndarray, y: np.ndarray, reach: float) -> int:
    """How many samples of a track have a point of the body's reach disc, sampled on rings and rays, off open cells."""
    a = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    rings = np.concatenate([[0.0], np.repeat([reach / 2.0, reach], 24)])
    angles = np.concatenate([[0.0], a, a])
    px = x[:, None] + rings * np.cos(angles)
    py = y[:, None] + rings * np.sin(angles)
    rows = np.floor((py - ground.origin[1]) / ground.step + 0.5).astype(int)
    cols = np.floor((px - ground.origin[0]) / ground.step + 0.5).astype(int)
    on = (rows >= 0) & (rows < ground.open.shape[0]) & (cols >= 0) & (cols < ground.open.shape[1])
    on[on] = ground.open[rows[on], cols[on]]
    return int((~on.all(axis=1)).sum())


# ---------------------------------------------------------------------- #
# the package format
# ---------------------------------------------------------------------- #
def test_a_kind_reads_its_bodies_clips_joins_and_reach(package):
    """A table reads into its kind: heat, spreads, joins, each body's poses as floats, the rest frame and the reach."""
    kind = skins.load("goat")
    body = kind.bodies["doe"]
    assert kind.name == "goat" and kind.heat_range_c == (20.0, 37.0) and kind.joins["graze"] == ("head_up", "graze_walk")
    assert body.poses.dtype == np.float32 and body.poses.shape == (11 * animal_stand_in.FRAMES, 9, 3)
    assert body.rest() == body.clips["stand"].start and body.clips["head_down"].then == "graze"
    assert body.reach_m == pytest.approx(math.hypot(0.65, 0.0), abs=1e-3)
    assert len(skins.load("dog").bodies) == 3


def test_clips_follow_the_joins_and_find_their_way_through_them(package):
    """A clip follows another only as the joins allow, and the way between two lies along them."""
    kind = skins.load("goat")
    assert skins.follows(kind, "graze", "graze") and skins.follows(kind, "stand", "walk")
    assert not skins.follows(kind, "graze", "walk")
    assert skins.route(kind, "graze", "walk") == ["graze_walk", "walk"]
    assert skins.route(kind, "graze", "look") == ["head_up", "stand", "look"]
    assert skins.route(kind, "lying", "trot") == ["get_up", "stand", "walk", "trot"]
    free = skins.Kind("x", (0.0, 1.0), 1.0, (1.0, 1.0), (1.0, 1.0), 0.2, None, {})
    assert skins.route(free, "graze", "walk") == ["walk"]


def test_a_pose_blends_between_its_frames_and_a_loop_wraps(package):
    """A fractional frame lies between its two frames; a loop wraps to its start and a clip that ends holds."""
    body = skins.load("goat").bodies["doe"]
    head = lambda clip, frame: float(skins.points(body, clip, frame)[8, 2])  # noqa: E731
    assert head("graze", 1.5) == pytest.approx((head("graze", 1.0) + head("graze", 2.0)) / 2.0, abs=1e-3)
    assert head("graze", 4.0) == pytest.approx(head("graze", 0.0))
    assert head("head_down", 9.0) == pytest.approx(head("head_down", 3.0))
    drawn = skins.drawn(body, body.poses[0], 2.0)
    assert drawn.shape == (10, 3) and np.array_equal(drawn[9], drawn[8])


def test_the_shipped_tables_hold_every_clip_the_minds_play():
    """The installed goat, dog, bird, boar and rat tables carry every clip their minds ask for, along the
    joins their minds take; a goat or dog reaches no further than its length in any pose, a bird with its wings spread
    may, and every bird the places ask for is there, its feet leaving the ground late in its take-off."""
    played = {"goat": {"walk", "trot", "graze_walk", "head_down", "graze", "head_up", "stand", "look", "lie_down",
                       "lying", "get_up"},
              "dog": {"walk", "trot", "run", "stand", "look", "sniff", "nose_down", "sniff_walk", "lie_down", "lying",
                      "get_up"},
              "bird": {"walk", "hop", "stand", "look", "peck", "take_off", "fly", "glide", "land"},
              "boar": {"walk", "trot", "graze_walk", "head_down", "graze", "head_up", "stand", "look", "lie_down",
                       "lying", "get_up"},
              "rat": {"walk", "trot", "stand", "look", "sniff", "nose_down", "sniff_walk", "groom"}}
    shipped = [kind for kind in played if skins.shipped(kind)]
    if not shipped:
        pytest.skip("the installed swarm-worlds has no animal package")
    for kind in shipped:
        for body in skins.load(kind).bodies.values():
            assert played[kind] <= set(body.clips)
            assert body.reach_m <= (body.extent_m if kind == "bird" else body.length_m)
            assert kind == "bird" or body.extent_m <= body.length_m
            assert body.source.max() < body.poses.shape[1] and body.faces.max() < len(body.source)
    for kind, bodies in (("boar", {"boar", "piglet"}), ("rat", {"mouse", "rat"})):
        if kind in shipped:
            assert set(skins.load(kind).bodies) == bodies
            assert all(skins.load(kind).bodies[name].clips["look"].loop for name in bodies)
    if "bird" in shipped:
        kind = skins.load("bird")
        assert {name for shares in animals.BIRD_PLACES.values() for name, _ in shares} == set(kind.bodies)
        for flight in (("stand", "take_off"), ("take_off", "fly"), ("fly", "glide"), ("glide", "land"), ("land", "stand")):
            assert skins.follows(kind, *flight)
        for body in kind.bodies.values():
            assert 0 < animal_moves.lift_frame(body) < body.clips["take_off"].frames - 1


# ---------------------------------------------------------------------- #
# which animals a seed holds
# ---------------------------------------------------------------------- #
def _beasts(counts: dict) -> dict:
    """The goats and dogs of a seed's counts, its birds, boars and rodents left out."""
    return {kind: n for kind, n in counts.items() if kind in ("goat", "dog")}


def test_animals_come_at_their_shares_and_never_none(package):
    """Seeds with animals hold dogs in about seven of ten, goats in about a third, always one kind and never a
    crowd: one or two dogs, two to four goats; without a herd's place dogs come instead; seeds whose mix has no
    animals hold no goat or dog."""
    drawn = [(seed, _beasts(animals.wanted(seed))) for seed in range(6000)]
    held = [counts for seed, counts in drawn if mix.present(seed).animals]
    assert all(not counts for seed, counts in drawn if not mix.present(seed).animals)
    assert all(counts for counts in held)
    assert abs(np.mean(["dog" in c for c in held]) - 0.72) < 0.03
    assert abs(np.mean(["goat" in c for c in held]) - 0.35) < 0.03
    assert all(2 <= c["goat"] <= 4 for c in held if "goat" in c) and all(1 <= c["dog"] <= 2 for c in held if "dog" in c)
    dogs = Counter(c["dog"] for c in held if "dog" in c)
    assert abs(dogs[1] / (dogs[1] + dogs[2]) - 0.70) < 0.03
    goats = Counter(c["goat"] for c in held if "goat" in c)
    assert all(abs(goats[n] / sum(goats.values()) - 1.0 / 3.0) < 0.04 for n in (2, 3, 4))
    elsewhere = [_beasts(animals.wanted(seed, herd_fits=False)) for seed in range(2000) if mix.present(seed).animals]
    assert all(set(c) == {"dog"} for c in elsewhere)


def test_a_seeds_animals_never_depend_on_what_is_installed(tmp_path, monkeypatch):
    """With no animal table installed a seed asks for the very animals it asks for with them, so two machines can
    never build different seeds; the install check is what names the missing tables, each body's files with them."""
    with_package = animal_stand_in.write_package(tmp_path / "animals")
    monkeypatch.setattr(skins, "ANIMALS_DIR", with_package)
    asked = [animals.wanted(seed) for seed in range(300)]
    listed = skins.asset_files()
    assert all(os.path.isfile(path) for path in listed) and len(listed) > 2 + 4 * 3
    monkeypatch.setattr(skins, "ANIMALS_DIR", str(tmp_path / "nothing"))
    assert [animals.wanted(seed) for seed in range(300)] == asked and any(asked)
    assert skins.asset_files() == [skins.table_path(kind) for kind in ("bird", "boar", "dog", "goat", "rat")]
    assert all(path in install.asset_files() for path in skins.asset_files())
    assert any(all(name in line for name in ("goats.json", "dogs.json", "birds.json", "boars.json", "rats.json"))
               for line in install.problems())


def test_the_places_ask_for_the_five_birds_drawn_and_a_body_left_in_a_table_is_never_read(tmp_path, monkeypatch):
    """The kinds of place name exactly the birds the skins draw; a bird table that still holds another body, its files
    gone, loads the drawn birds alone, asks the install for none of that body's files, and plans as before."""
    assert {name for shares in animals.BIRD_PLACES.values() for name, _ in shares} == skins.DRAWN["bird"]
    root = tmp_path / "animals"
    monkeypatch.setattr(skins, "ANIMALS_DIR", animal_stand_in.write_package(root))
    skins.read.cache_clear()
    planned = animals.plan(4, _ground(), {"bird": 3}, _HORIZON_S, place="valley")
    animal_stand_in.write_kind(root / "birds", "bird", [*animal_stand_in.BIRDS, "spare"])
    for leftover in (root / "birds").glob("spare*"):
        leftover.unlink()
    skins.read.cache_clear()
    listed = skins.asset_files()
    assert all(os.path.isfile(path) for path in listed) and not any("spare" in path for path in listed)
    assert set(skins.load("bird").bodies) == skins.DRAWN["bird"]
    again = animals.plan(4, _ground(), {"bird": 3}, _HORIZON_S, place="valley")
    assert [(a.body, a.coat) for a in again] == [(a.body, a.coat) for a in planned]
    assert all(np.array_equal(a.track.x, b.track.x) for a, b in zip(planned, again))
    skins.read.cache_clear()


def test_a_place_takes_a_herd_by_its_kind():
    """A herd comes to a rural place or to a field that names no kind, never to a town."""
    assert animals.herd_place(SimpleNamespace(park={"world": {}}))
    assert animals.herd_place(SimpleNamespace(park={"kind": "valley"}))
    assert not animals.herd_place(SimpleNamespace(park={"kind": "town"}))


# ---------------------------------------------------------------------- #
# plans
# ---------------------------------------------------------------------- #
def test_every_body_stays_on_its_ground_apart_and_plays_by_the_joins(package):
    """Over many seeds no animal's body leaves the open ground or comes within another's reach, no clip follows one
    the joins forbid, and nothing moves faster than its kind can run."""
    ground = _ground()
    for seed in range(12):
        planned = animals.plan(seed, ground, {"goat": 7, "dog": 3}, _HORIZON_S)
        assert planned
        reach = [a.skin().reach_m * a.size for a in planned]
        for a, r in zip(planned, reach):
            track = a.track
            assert _off_its_ground(ground, track.x, track.y, r) == 0
            fastest = max(c.speed_m_s for c in a.skin().clips.values()) * 1.1 * 1.1
            assert np.hypot(np.diff(track.x), np.diff(track.y)).max() <= fastest * animal_moves.SAMPLE_S + 1e-9
            kind = skins.load(a.kind)
            clips = [s.clip for s in track.segments]
            assert all(skins.follows(kind, x, y) for x, y in zip(clips, clips[1:]))
        for i in range(len(planned)):
            for j in range(i + 1, len(planned)):
                gap = np.hypot(planned[i].track.x - planned[j].track.x, planned[i].track.y - planned[j].track.y)
                assert gap.min() >= reach[i] + reach[j]


def test_a_herd_grazes_moves_and_rests_and_dogs_roam(package):
    """A herd's goats spend their time grazing, walking, looking and lying, near each other; the dogs travel."""
    planned = animals.plan(5, _ground(holes=False), {"goat": 5, "dog": 2}, 200.0)
    goats = [a for a in planned if a.kind == "goat"]
    seen = {s.clip for a in goats for s in a.track.segments}
    assert len(goats) == 5 and {"graze", "graze_walk", "walk", "stand"} <= seen
    middle = np.mean([[a.track.x[-1], a.track.y[-1]] for a in goats], axis=0)
    spread = np.mean([math.hypot(a.track.x[-1] - middle[0], a.track.y[-1] - middle[1]) for a in goats])
    assert spread < animal_moves.herd_radius(5) + 3.0
    assert len({a.coat for a in goats}) == 3 and len({a.body for a in planned if a.kind == "dog"}) == 2
    assert sum(np.hypot(np.diff(a.track.x), np.diff(a.track.y)).sum() for a in planned if a.kind == "dog") > 10.0


def test_at_night_the_herd_lies_close_together_all_patrol_and_the_dogs_are_the_days(package):
    """A night's herd lies the whole patrol, still and closer together than by day; the dogs are the same breeds,
    coats, sizes and paces as by day and still travel."""
    ground = _ground(holes=False)
    day = animals.plan(5, ground, {"goat": 4, "dog": 2}, 200.0)
    night = animals.plan(5, ground, {"goat": 4, "dog": 2}, 200.0, night=True)
    goats = [a for a in night if a.kind == "goat"]
    assert len(goats) == 4 and all({s.clip for s in a.track.segments} == {"lying"} for a in goats)
    assert all(np.ptp(a.track.x) == 0.0 and np.ptp(a.track.y) == 0.0 for a in goats)
    widest = max(math.hypot(a.track.x[0] - b.track.x[0], a.track.y[0] - b.track.y[0]) for a in goats for b in goats)
    assert widest <= 2.0 * animal_moves.herd_radius(4, resting=True)
    assert animal_moves.herd_radius(4, resting=True) < animal_moves.herd_radius(4)
    looks = [(a.body, a.coat, a.size, a.pace) for a in day if a.kind == "dog"]
    assert looks == [(a.body, a.coat, a.size, a.pace) for a in night if a.kind == "dog"] and len(looks) == 2
    assert sum(np.hypot(np.diff(a.track.x), np.diff(a.track.y)).sum() for a in night if a.kind == "dog") > 10.0


def test_a_plan_is_the_same_every_time_and_a_function_of_time(package):
    """The same seed plans the same animals bit for bit; a place read at a sample is the sample, and between two
    samples it lies between them; a seed differing only above 64 bits plans differently."""
    ground = _ground()
    first = animals.plan(2 ** 100 + 9, ground, {"goat": 4, "dog": 2, "bird": 3}, _HORIZON_S, place="town")
    again = animals.plan(2 ** 100 + 9, ground, {"goat": 4, "dog": 2, "bird": 3}, _HORIZON_S, place="town")
    other = animals.plan(2 ** 101 + 9, ground, {"goat": 4, "dog": 2, "bird": 3}, _HORIZON_S, place="town")
    assert len(first) == len(again) == 9 and any(a.track.up is not None for a in first)
    for a, b in zip(first, again):
        assert (a.body, a.coat, a.size, a.pace) == (b.body, b.coat, b.size, b.pace)
        for name in ("x", "y", "hx", "hy", "frame", "rate", "segment"):
            assert np.array_equal(getattr(a.track, name), getattr(b.track, name))
        assert (a.track.up is None) == (b.track.up is None) and np.array_equal(a.track.up, b.track.up)
    assert any(not np.array_equal(a.track.x, b.track.x) for a, b in zip(first, other))
    track = first[0].track
    k = 123
    assert track.place(k * animal_moves.SAMPLE_S)[:2] == pytest.approx((track.x[k], track.y[k]))
    x, _y, hx, hy = track.place((k + 0.5) * animal_moves.SAMPLE_S)
    assert min(track.x[k], track.x[k + 1]) - 1e-9 <= x <= max(track.x[k], track.x[k + 1]) + 1e-9
    assert hx * hx + hy * hy == pytest.approx(1.0)


def test_a_herd_with_no_room_leaves_the_dogs_to_come(package):
    """Where no spot has room for a herd's middle, no goat comes and dogs come in their place."""
    open_ = np.zeros((80, 40), dtype=bool)
    open_[5:75, 15:25] = True
    ground = animal_moves.Ground((0.0, 0.0), 0.5, open_, np.zeros(open_.shape, dtype=bool), np.zeros(open_.shape))
    planned = animals.plan(4, ground, {"goat": 5}, 20.0)
    assert planned and {a.kind for a in planned} == {"dog"}


# ---------------------------------------------------------------------- #
# birds
# ---------------------------------------------------------------------- #
def _sky_ground() -> animal_moves.Ground:
    """The square of open ground with a crown and a house in its sky, where birds may stand but never fly over."""
    ground = _ground()
    sky = ground.open.copy()
    yy, xx = np.mgrid[:sky.shape[0], :sky.shape[1]]
    sky[(yy - 40) ** 2 + (xx - 40) ** 2 < 12 ** 2] = False
    sky[140:160, 30:60] = False
    return animal_moves.Ground(ground.origin, ground.step, ground.open, ground.inside, ground.height, sky)


def _bird(ground: animal_moves.Ground, pace: float = 1.0) -> animal_moves.Mover:
    """A stand-in crow at the ground's middle, facing east."""
    kind = skins.load("bird")
    return animal_moves.Mover(kind=kind, body=kind.bodies["crow"], size=1.0, pace=pace, x=0.0, y=0.0, hx=1.0, hy=0.0,
                              rng=np.random.default_rng(0), ground=ground)


def _flights(up: np.ndarray) -> list:
    """Each run of samples a track spends in the air and lands within it: the first sample off the ground and the
    first back on it."""
    air = np.concatenate([[False], up > 0.0, [False]])
    edges = np.flatnonzero(air[1:] != air[:-1])
    return [(first, last) for first, last in zip(edges[::2], edges[1::2]) if last < len(up)]


def test_birds_come_in_three_day_seeds_in_ten_a_few_at_a_time_and_never_at_night(package):
    """Birds come in the day seeds whose mix has them, about three in ten, one to four at their shares where the
    seed's goats and dogs leave them the room, as many as fit where they do not, and never on a night seed, while the
    mix still gives such a seed birds."""
    drawn = [(seed, animals.wanted(seed)) for seed in range(8000)]
    nights = [seed for seed, _ in drawn if site.night(seed)]
    day = [(seed, counts) for seed, counts in drawn if not site.night(seed)]
    assert all("bird" not in animals.wanted(seed) for seed in nights) and any(mix.present(seed).birds for seed in nights)
    assert all("bird" not in counts for seed, counts in day if not mix.present(seed).birds)
    roomy = [(seed, counts) for seed, counts in day if counts.get("goat", 0) + counts.get("dog", 0) <= animals.MOST_ANIMALS - 4]
    assert all(("bird" in counts) == mix.present(seed).birds for seed, counts in roomy)
    assert abs(np.mean(["bird" in counts for _, counts in roomy]) - mix.BIRDS_SHARE) < 0.02
    held = Counter(counts["bird"] for _, counts in roomy if "bird" in counts)
    assert set(held) == {1, 2, 3, 4}
    assert all(abs(held[n] / sum(held.values()) - share) < 0.03 for n, share in animals.BIRD_COUNT)
    tight = [counts for _seed, counts in day if counts.get("goat", 0) + counts.get("dog", 0) > animals.MOST_ANIMALS - 4]
    assert tight and all(sum(counts.get(kind, 0) for kind in ("goat", "dog", "bird")) <= animals.MOST_ANIMALS for counts in tight)
    assert any("bird" in counts for counts in tight)


def test_a_place_draws_its_own_birds_and_pigeons_and_choughs_come_together():
    """Towns get town pigeons most, magpies and crows; woods and valleys wood pigeons most, magpies and crows; the
    high ground choughs and crows; a field with no kind the valley's birds; a first pigeon or chough is joined by its
    own kind at the set share, a first crow never; a wide seed keeps its flock."""
    for place, shares in animals.BIRD_PLACES.items():
        assert {name for seed in range(300) for name, _ in animals.flock(seed, place, 4)} == {name for name, _ in shares}
    town = [animals.flock(seed, "town", 1)[0][0] for seed in range(3000)]
    assert abs(town.count("pigeon") / 3000 - 0.60) < 0.03
    valley = {name for name, _ in animals.BIRD_PLACES["valley"]}
    assert {name for seed in range(300) for name, _ in animals.flock(seed, None, 3)} == valley
    flocks = [animals.flock(seed, "alpine", 4) for seed in range(2000)]
    joined = [mate for found in flocks if found[0][0] == "chough" for _, mate in found[1:]]
    assert abs(np.mean(joined) - animals.FLOCK_SHARE) < 0.03
    assert not any(mate for found in flocks if found[0][0] == "crow" for _, mate in found)
    assert animals.flock(2 ** 100 + 5, "town", 4) == animals.flock(2 ** 100 + 5, "town", 4)


def test_a_flight_climbs_flies_on_glides_down_and_lands_softly_on_its_spot(package):
    """A flight stays put until the feet leave, holds its peak a while and never passes it, never drops under a metre
    between the take-off and landing clips, comes down with no fall left and stops on its spot, plays the clips in an
    order the joins allow, and flies no faster than its clip; a short flight goes slower than a long one."""
    bird = _bird(_ground(holes=False), pace=1.1)
    kind = skins.load("bird")
    for length, peak in ((10.0, 2.0), (10.0, 3.0), (30.0, 2.0), (45.0, 5.0)):
        flight = animal_moves.Flight.plan(bird, length, 0.0, peak)
        times = np.arange(0.0, flight.end_s + 0.01, 0.01)
        up = np.array([flight.height(t) for t in times])
        gone = np.array([flight.along(t) for t in times])
        assert flight.along(flight.lift_s) == 0.0 and flight.along(flight.touch_s) == length
        assert up.max() == peak and up.min() == 0.0 and flight.glide_s - flight.top_s >= animal_moves.HOLD_S - 1e-9
        assert up[(times >= flight.rise_s) & (times < flight.land_s)].min() >= 1.0
        assert flight.height(flight.touch_s - 0.01) < 0.01 and flight.height(flight.lift_s + 0.01) < 0.2
        assert np.diff(gone).min() >= 0.0 and np.abs(np.diff(up)).max() < 0.06 and np.abs(np.diff(gone)).max() < 0.1
        clips = [flight.clip(t)[0] for t in times]
        order = [clip for n, clip in enumerate(clips) if n == 0 or clip != clips[n - 1]]
        assert order == ["take_off", "fly", "glide", "land"] and all(skins.follows(kind, a, b) for a, b in zip(order, order[1:]))
        assert flight.speed <= bird.body.clips["fly"].speed_m_s * bird.pace
    assert animal_moves.Flight.plan(bird, 10.0, 0.0, 3.0).speed < animal_moves.Flight.plan(bird, 45.0, 0.0, 3.0).speed


def test_birds_stand_under_open_sky_fly_low_over_clear_sky_and_keep_clear_of_every_animal(package):
    """Over many seeds every bird stands only on open ground under open sky, flies only over clear sky, 10 to 45 m a
    flight, peaking 2 to 5 m and never under a metre between take-off and landing, plays its clips by the joins, and
    keeps clear of every other animal while both are on the ground."""
    ground = _sky_ground()
    clear = animal_moves._eroded(ground.sky)
    flights = 0
    for seed in range(8):
        planned = animals.plan(seed, ground, {"goat": 4, "dog": 2, "bird": 4}, 2.0 * _HORIZON_S, place="alpine")
        birds = [a for a in planned if a.kind == "bird"]
        assert len(birds) == 4
        for a in birds:
            track = a.track
            up = np.zeros(len(track.x)) if track.up is None else track.up
            grounded = up == 0.0
            assert _off_its_ground(ground.for_birds(), track.x[grounded], track.y[grounded], a.skin().reach_m * a.size) == 0
            rows = np.floor((track.y[~grounded] - ground.origin[1]) / ground.step + 0.5).astype(int)
            cols = np.floor((track.x[~grounded] - ground.origin[0]) / ground.step + 0.5).astype(int)
            assert clear[rows, cols].all()
            for first, last in _flights(up):
                length = math.hypot(track.x[last] - track.x[first - 1], track.y[last] - track.y[first - 1])
                assert 10.0 <= length <= 45.0 and 2.0 <= up[first:last].max() <= 5.0
                flights += 1
            level = np.array([track.segments[n].clip in ("fly", "glide") for n in track.segment])
            assert up[level].min(initial=1.0) >= 1.0
            clips = [segment.clip for segment in track.segments]
            assert all(skins.follows(skins.load("bird"), x, y) for x, y in zip(clips, clips[1:]))
        for i, a in enumerate(planned):
            for b in planned[i + 1:]:
                if "bird" not in (a.kind, b.kind):
                    continue
                both = np.ones(len(a.track.x), dtype=bool)
                for one in (a, b):
                    if one.track.up is not None:
                        both &= one.track.up == 0.0
                gap = np.hypot(a.track.x - b.track.x, a.track.y - b.track.y)[both]
                assert gap.min(initial=np.inf) >= a.skin().reach_m * a.size + b.skin().reach_m * b.size
    assert flights > 8 * 4


def test_a_bird_never_starts_on_sky_it_cannot_fly_out_of(package):
    """Half the ground is islands of open sky a few metres across, walled apart by thin lines of closed sky: a bird may
    stand on them but no flight leaves one. Drawn as a bird's first spot often lands there; every bird starts where
    flights are open to it all the same."""
    base = _ground(holes=False)
    sky = base.open.copy()
    sky[::12, :90] = False
    sky[:, :90:12] = False
    ground = animal_moves.Ground(base.origin, base.step, base.open, base.inside, base.height, sky)
    birds_ground = ground.for_birds()
    trapped = birds_ground.level(1)[:, :84].sum() / birds_ground.level(1).sum()
    birds = [a for seed in range(20) for a in animals.plan(seed, ground, {"bird": 4}, _HORIZON_S, place="town")]
    assert trapped > 0.3 and len(birds) == 80
    assert all(birds_ground.launches(birds_ground.level_for(a.skin().reach_m * a.size), a.track.x[0], a.track.y[0])
               >= animal_moves.MIN_LAUNCHES for a in birds)


def test_birds_leave_the_goats_and_dogs_as_they_were(package):
    """A seed's goats and dogs keep their looks and every sample of their tracks when birds come, and never leave
    the ground."""
    ground = _sky_ground()
    without = animals.plan(7, ground, {"goat": 4, "dog": 2}, _HORIZON_S)
    joined = animals.plan(7, ground, {"goat": 4, "dog": 2, "bird": 4}, _HORIZON_S, place="town")
    assert [a.kind for a in joined] == [a.kind for a in without] + ["bird"] * 4
    for a, b in zip(without, joined):
        assert (a.body, a.coat, a.size, a.pace) == (b.body, b.coat, b.size, b.pace) and b.track.up is None
        for name in ("x", "y", "hx", "hy", "frame", "rate", "segment"):
            assert np.array_equal(getattr(a.track, name), getattr(b.track, name))
        assert a.track.segments == b.track.segments


def test_a_shadows_top_speed_is_correctly_rounded_so_a_take_off_gate_falls_alike_everywhere():
    """A dog's or goat's fastest step, which gates a bird's take-off, is the correctly rounded root of its squared
    length, never a hypotenuse routine that may differ in the last bit between machines,
    and this step is where it does."""
    dx, dy = 0.257662780521071, 0.1429006900440708
    path = animal_moves.Track(x=np.array([0.0, dx]), y=np.array([0.0, dy]), hx=np.ones(2), hy=np.zeros(2), frame=np.zeros(2),
                              rate=np.zeros(2), segment=np.zeros(2, dtype=np.int64), segments=[animal_moves.Segment(0.0, "walk")])
    assert animal_moves.Shadow(path, 0.6).fastest == float(np.sqrt(dx * dx + dy * dy))


def test_a_frame_poses_a_body_whose_ball_reaches_past_its_edge_even_from_a_slanted_side():
    """A body whose centre lies just off a frame's side, its ball of the margin's radius still crossing that side's
    plane, counts as seen; one just past the margin does not; and no point the old tangent test kept is dropped."""
    shot = camera.View("colour", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), 640, 480, 90.0, True, 0)
    sight = animals.Sight.of(shot)
    half_h, margin, depth = sight.half_h, 0.5, 10.0
    off_plane = lambda gap: (depth, -(half_h * depth + gap * math.sqrt(1.0 + half_h * half_h)), 0.0)  # noqa: E731
    assert sight.per_metre(off_plane(0.9 * margin), margin) > 0.0
    assert sight.per_metre(off_plane(1.1 * margin), margin) == 0.0
    rng = np.random.default_rng(0)
    for x, y, z in rng.uniform([1.0, -30.0, -30.0], [40.0, 30.0, 30.0], (2000, 3)):
        across, rise = abs(y) / x, abs(z) / x
        if across <= half_h + margin / x and rise <= sight.half_v + margin / x:
            assert sight.per_metre((x, y, z), margin) > 0.0


def test_a_landing_spot_is_taken_from_the_moment_its_flight_is_queued_and_checked_again_at_take_off(package):
    """A bird that queues a flight while it still has a clip to finish holds its landing spot at once: another bird may
    neither stand there nor fly there; a flight whose spot a dog will stand on at the touch-down is dropped at the
    take-off, and one whose spot a dog only crosses at the take-off flies."""
    ground = _ground(holes=False).for_birds()
    first, second = _bird(ground), _bird(ground)
    second.x = 3.0
    for bird in (first, second):
        bird.mind = animal_moves.Bird(np.random.default_rng(2))
        bird.mind.next_s = 0.0
        bird.clip = "look"
    first.mind.decide(first, 0, [first, second])
    spot = first.landing
    assert first.actions[0][0] == "air" and first.target is None and spot is not None
    assert not second.can_stand(*spot, [first, second]) and not animal_moves._spare(second, *spot, [first, second])
    flight = first.actions[0][4]
    still = animal_moves.Track(x=np.full(3, spot[0]), y=np.full(3, spot[1]), hx=np.ones(3), hy=np.zeros(3), frame=np.zeros(3),
                               rate=np.zeros(3), segment=np.zeros(3, dtype=np.int64), segments=[animal_moves.Segment(0.0, "stand")])
    dog = animal_moves.Shadow(still, 0.6)
    crossing = animal_moves.Track(x=np.r_[np.full(3, spot[0]), np.full(197, spot[0] + 30.0)], y=np.full(200, spot[1]), hx=np.ones(200),
                                  hy=np.zeros(200), frame=np.zeros(200), rate=np.zeros(200),
                                  segment=np.zeros(200, dtype=np.int64), segments=[animal_moves.Segment(0.0, "stand")])
    assert flight.touch_s > 1.5
    first.clip, first.action, first.actions = "stand", None, deque([("air", "take_off", *spot, flight)])
    first.prepare(0, [first, animal_moves.Shadow(crossing, 0.6)])
    assert first.action[0] == "air" and first.landing == spot
    first.clip, first.action, first.actions = "stand", None, deque([("air", "take_off", *spot, flight)])
    first.prepare(0, [first, dog])
    assert first.action[0] == "play" and first.landing is None and first.mind.fright


def test_a_bird_never_starts_where_no_flight_line_is_open_and_the_seed_holds_one_bird_fewer(package):
    """Where the only sky a bird may stand under is a square too small to fly out of, no bird starts there: the seed
    holds fewer birds than it drew rather than one that can never fly."""
    base = _ground(holes=False)
    sky = np.zeros(base.open.shape, dtype=bool)
    sky[80:96, 80:96] = True
    ground = animal_moves.Ground(base.origin, base.step, base.open, base.inside, base.height, sky)
    assert animals.plan(3, ground, {"bird": 3}, _HORIZON_S, place="town") == []


def test_a_bird_takes_off_before_a_dog_reaches_it(package):
    """A bird with no flight due flies off when a dog walking straight at it is about to come close, and is never on
    the ground within that reach of it."""
    ground = _ground(holes=False)
    samples = int(math.ceil(_HORIZON_S / animal_moves.SAMPLE_S)) + 1
    xs = np.linspace(-30.0, 30.0, samples)
    zeros = np.zeros(samples)
    path = animal_moves.Track(x=xs, y=zeros, hx=zeros + 1.0, hy=zeros, frame=zeros, rate=zeros,
                              segment=np.zeros(samples, dtype=np.int64), segments=[animal_moves.Segment(0.0, "walk")])
    dog = animal_moves.Shadow(path, 0.6)
    bird = _bird(ground.for_birds())
    bird.mind = animal_moves.Bird(np.random.default_rng(1))
    bird.mind.next_s = math.inf
    [track] = animal_moves.plan_movers([bird], _HORIZON_S, [dog])
    assert track.up is not None
    grounded = track.up == 0.0
    gap = np.hypot(track.x - xs, track.y)[grounded]
    assert gap.min() >= bird.reach + dog.reach + animal_moves.GAP_M


# ---------------------------------------------------------------------- #
# the ground, the bodies and the frames, in the engine
# ---------------------------------------------------------------------- #
@pytest.fixture
def client():
    """A bare engine client, closed after the test."""
    cli = p.connect(p.DIRECT)
    yield cli
    p.disconnect(cli)


# ---------------------------------------------------------------------- #
# boars, rats and mice
# ---------------------------------------------------------------------- #
def _others(counts: dict) -> int:
    """How many goats, dogs and birds a seed's counts hold."""
    return sum(n for kind, n in counts.items() if kind in ("goat", "dog", "bird"))


def test_boars_come_alone_in_twos_or_as_a_sow_with_piglets_and_only_to_wild_places(package):
    """Seeds whose mix has boars hold one boar, two, or a sow with two to four piglets, at their shares, in the six
    wild kinds of place and on a field that names no kind, never in a town, a village or on scree; a seed whose mix
    has none holds none, and one the sounder would crowd past six animals holds none either."""
    held = [seed for seed in range(30000) if mix.present(seed, site.night(seed)).boars]
    mixed = set(held)
    for seed in range(3000):
        counts = animals.wanted(seed, place="forest")
        if seed not in mixed:
            assert "boar" not in counts
        elif "boar" not in counts:
            assert _others(counts) + len(animals.sounder(seed)) > animals.MOST_ANIMALS
    roomy = [seed for seed in held
             if _others(animals.wanted(seed, place="forest")) + len(animals.sounder(seed)) <= animals.MOST_ANIMALS][:300]
    for place in ("alpine", "forest", "lake", "river", "ski", "valley", None):
        assert all(animals.wanted(seed, place=place)["boar"] == len(animals.sounder(seed)) for seed in roomy), place
    for place in ("town", "village", "scree"):
        assert all("boar" not in animals.wanted(seed, place=place) for seed in roomy), place
    groups = [animals.sounder(seed) for seed in held]
    assert all(group[0] == "boar" for group in groups)
    kinds = Counter("one" if group == ["boar"] else "two" if group == ["boar", "boar"] else "sow" for group in groups)
    for name, share in (("one", 0.45), ("two", 0.20), ("sow", 0.35)):
        assert abs(kinds[name] / len(groups) - share) < 0.04
    sows = [group for group in groups if "piglet" in group]
    assert all(group[1:] == ["piglet"] * (len(group) - 1) and 2 <= len(group) - 1 <= 4 for group in sows)
    assert {len(group) - 1 for group in sows} == {2, 3, 4}


def test_rats_come_to_towns_villages_and_rivers_and_mice_anywhere_else(package):
    """Seeds whose mix has rodents hold one to three rats by people or water and one or two mice elsewhere, at their
    shares; the kind of place changes which, never whether, unless they would crowd the seed past six animals."""
    held = [seed for seed in range(30000) if mix.present(seed, site.night(seed)).rodents]
    mixed = set(held)
    assert all("rat" not in animals.wanted(seed, place=place) for seed in range(3000) if seed not in mixed for place in ("town", "lake"))
    for place, body in (("town", "rat"), ("village", "rat"), ("river", "rat"), ("forest", "mouse"), ("alpine", "mouse"),
                        ("lake", "mouse"), ("scree", "mouse"), (None, "mouse")):
        for seed in held[:200]:
            bodies, counts = animals.rodents(seed, place), animals.wanted(seed, place=place)
            assert set(bodies) == {body}
            if "rat" in counts:
                assert counts["rat"] == len(bodies)
            else:
                assert _others(counts) + counts.get("boar", 0) + len(bodies) > animals.MOST_ANIMALS
    rats = Counter(len(animals.rodents(seed, "town")) for seed in held)
    mice = Counter(len(animals.rodents(seed, "forest")) for seed in held)
    for count, share in ((1, 0.50), (2, 0.30), (3, 0.20)):
        assert abs(rats[count] / len(held) - share) < 0.04
    for count, share in ((1, 0.65), (2, 0.35)):
        assert abs(mice[count] / len(held) - share) < 0.04
    assert sum("rat" in animals.wanted(seed, place="lake") for seed in held) > 0.85 * len(held)


def test_boars_and_rodents_never_crowd_a_seed_and_never_change_its_goats_dogs_and_birds(package):
    """Whatever the place, a seed holds at most six animals, a sounder comes whole or not at all, some crowded seeds
    do go without, and the goats, dogs and birds are those of a place with neither."""
    refused = 0
    for seed in range(40000):
        river, town = animals.wanted(seed, place="river"), animals.wanted(seed, place="town")
        assert {k: v for k, v in river.items() if k not in ("boar", "rat")} == {k: v for k, v in town.items() if k not in ("boar", "rat")}
        assert sum(river.values()) <= animals.MOST_ANIMALS == 6 and sum(town.values()) <= animals.MOST_ANIMALS
        if "boar" in river:
            assert river["boar"] == len(animals.sounder(seed))
        present = mix.present(seed, site.night(seed))
        refused += (present.boars and "boar" not in river) or (present.rodents and "rat" not in river)
    assert 0 < refused < 4000


def test_every_animal_on_the_ground_lives_on_one_side_of_the_edge(package):
    """Goats, dogs, boars, piglets and rats each live a whole patrol on the side of the field's edge they start on:
    on the field 3 m and more inside it and out from under the crowns, off it 6 m and more outside; over many seeds
    both sides are lived on."""
    ground = _sky_ground()
    on_field = ground.side(True, animals.INSIDE_CLEAR_M).for_birds()
    off_field = ground.side(False, animals.OUTSIDE_CLEAR_M)
    sides = Counter()
    for seed in range(12):
        planned = animals.plan(seed, ground, {"goat": 3, "dog": 2, "boar": 1, "rat": 1}, _HORIZON_S, place="river")
        for animal in planned:
            sides[animal.on_field] += 1
            home = on_field if animal.on_field else off_field
            for t in np.arange(0.0, _HORIZON_S, 1.0):
                x, y, _hx, _hy = animal.track.place(float(t))
                assert home.free(0, x, y), (seed, animal.kind, animal.on_field, t)
                assert ground.is_inside(x, y) == animal.on_field
    assert sides[True] > 10 and sides[False] > 10


def _past_the_edge_m(ground: animal_moves.Ground, x: float, y: float) -> float:
    """How far a point lies from the edge of the test ground's square field, measured on the square itself: above
    zero inside it, below zero outside."""
    cells = np.flatnonzero(ground.inside.any(axis=0))
    low = ground.origin[0] + (cells[0] - 0.5) * ground.step
    high = ground.origin[0] + (cells[-1] + 0.5) * ground.step
    out_x, out_y = max(low - x, x - high), max(low - y, y - high)
    return -max(out_x, out_y) if max(out_x, out_y) <= 0.0 else -math.hypot(max(out_x, 0.0), max(out_y, 0.0))


def test_an_animal_keeps_its_whole_body_three_metres_inside_the_edge_or_six_outside_it(package):
    """Measured on the field's own square, not with the planner's grids: ten times a second of a patrol, every goat,
    dog, boar, piglet and rat on the field is its reach and 3 m or more inside the edge, and every one off it its
    reach and 6 m or more outside."""
    ground = _ground(holes=False)
    for seed in range(6):
        for animal in animals.plan(seed, ground, {"goat": 3, "dog": 2, "boar": 1, "rat": 1}, _HORIZON_S, place="river"):
            reach = skins.load(animal.kind).bodies[animal.body].reach_m * animal.size
            clear = animals.INSIDE_CLEAR_M if animal.on_field else animals.OUTSIDE_CLEAR_M
            for t in np.arange(0.0, _HORIZON_S, 0.1):
                x, y, _hx, _hy = animal.track.place(float(t))
                inside = _past_the_edge_m(ground, x, y)
                assert (inside if animal.on_field else -inside) >= clear + reach, (seed, animal.kind, animal.body, t)
    assert (animals.INSIDE_CLEAR_M, animals.OUTSIDE_CLEAR_M) == (3.0, 6.0)


def test_piglets_take_no_room_and_live_on_their_sows_side(package):
    """A sow with piglets asks the field for one place, her own: with room for one she may live on it with all her
    piglets and is the only animal to report; with room for none they all live off it."""
    ground, seed = _sky_ground(), _sow_seed()
    for room in (1, 0):
        planned = animals.plan(seed, ground, {"boar": 1}, _HORIZON_S, place="river", room=room)
        sows = [animal for animal in planned if animal.body == "boar"]
        piglets = [animal for animal in planned if animal.body == "piglet"]
        assert len(sows) == 1 and len(piglets) >= 2 and not any(piglet.to_report for piglet in piglets)
        assert all(piglet.on_field == sows[0].on_field for piglet in piglets)
        assert sum(animal.to_report for animal in planned) == int(sows[0].on_field) <= room


def _striped_ground() -> animal_moves.Ground:
    """Open ground in strips 3 m wide: wide enough for a dog, too narrow for a herd to find its room."""
    ground = _ground(holes=False)
    strips = np.zeros_like(ground.open)
    for col in range(8, ground.open.shape[1] - 8, 20):
        strips[8:-8, col:col + 6] = True
    return animal_moves.Ground(ground.origin, ground.step, strips, ground.inside, ground.height)


def test_a_planned_seed_never_holds_more_than_six_animals_even_when_dogs_come_for_a_herd(package):
    """What a seed asks for and what the planner then stands never pass six animals, on open ground and on ground
    where the herd finds no room and dogs come in its place; that case does come up."""
    stood_in = 0
    for ground in (_sky_ground(), _striped_ground()):
        for seed in range(60):
            counts = animals.wanted(seed, place="river")
            if not counts:
                continue
            planned = animals.plan(seed, ground, counts, _HORIZON_S, night=site.night(seed), place="river")
            assert len(planned) <= animals.MOST_ANIMALS == 6, (seed, counts)
            stood_in += bool(counts.get("goat")) and not counts.get("dog") and any(animal.kind == "dog" for animal in planned)
    assert stood_in > 0


def test_the_field_takes_only_as_many_animals_to_report_as_it_has_room_for(package):
    """With room for every animal the seeds put some groups on the field; with room for none, every goat, dog and
    grown boar lives off it, while a rat, which is not asked for, may still live on it. Only goats, dogs and grown
    boars on the field are animals to report."""
    ground = _sky_ground()
    asked, small = 0, 0
    for seed in range(12):
        counts = {"goat": 3, "dog": 2, "boar": 1, "rat": 1}
        roomy = animals.plan(seed, ground, counts, _HORIZON_S, place="river")
        full = animals.plan(seed, ground, counts, _HORIZON_S, place="river", room=0)
        asked += sum(animal.to_report for animal in roomy)
        assert not any(animal.to_report for animal in full)
        assert all(not animal.on_field for animal in full if animal.asked)
        small += sum(animal.on_field for animal in full)
        for animal in roomy:
            assert animal.asked == (animal.kind in ("goat", "dog") or (animal.kind, animal.body) == ("boar", "boar"))
            assert animal.to_report == (animal.asked and animal.on_field)
        one = animals.plan(seed, ground, counts, _HORIZON_S, place="river", room=1)
        assert sum(animal.to_report for animal in one) <= 1
    assert asked > 12 and small > 0


def _sow_seed() -> int:
    """A seed whose boars are a sow with piglets."""
    return next(seed for seed in range(400) if "piglet" in animals.sounder(seed))


def test_boars_and_rodents_keep_clear_of_every_animal_and_a_seed_without_them_plans_as_before(package):
    """With boars and rats a seed's goats and dogs keep their looks and their starts, and every animal on the ground
    keeps out of every other's reach all patrol, on its own ground, playing only by its joins; asking for none of
    them plans the seed exactly as a call that never heard of them."""
    ground = _sky_ground()
    seed = _sow_seed()
    without = animals.plan(seed, ground, {"goat": 4, "dog": 2}, _HORIZON_S)
    joined = animals.plan(seed, ground, {"goat": 4, "dog": 2, "boar": 1, "rat": 1, "bird": 3}, _HORIZON_S, place="river")
    boars, rats = animals.sounder(seed), animals.rodents(seed, "river")
    assert [a.kind for a in joined] == [a.kind for a in without] + ["boar"] * len(boars) + ["rat"] * len(rats) + ["bird"] * 3
    assert [a.body for a in joined if a.kind in ("boar", "rat")] == boars + rats
    for a, b in zip(without, joined):
        assert (a.body, a.coat, a.size, a.pace) == (b.body, b.coat, b.size, b.pace)
        assert (a.track.x[0], a.track.y[0]) == (b.track.x[0], b.track.y[0])
    again = animals.plan(seed, ground, {"goat": 4, "dog": 2, "boar": 0, "rat": 0}, _HORIZON_S, place="river")
    for a, b in zip(without, again):
        for name in ("x", "y", "hx", "hy", "frame", "rate", "segment"):
            assert np.array_equal(getattr(a.track, name), getattr(b.track, name))
        assert a.track.segments == b.track.segments
    reach = [a.skin().reach_m * a.size for a in joined]
    for a, r in zip(joined, reach):
        if a.kind in ("boar", "rat"):
            assert a.track.up is None and _off_its_ground(ground, a.track.x, a.track.y, r) == 0
            clips = [s.clip for s in a.track.segments]
            assert all(skins.follows(skins.load(a.kind), x, y) for x, y in zip(clips, clips[1:]))
    for i, a in enumerate(joined):
        for j, b in enumerate(joined[:i]):
            down = np.ones(len(a.track.x), dtype=bool)
            for track in (a.track, b.track):
                down &= track.up is None or track.up == 0.0
            gap = np.hypot(a.track.x - b.track.x, a.track.y - b.track.y)
            assert gap[down].min() >= reach[i] + reach[j]


def test_piglets_keep_by_the_sow_boars_are_up_at_night_and_a_rodent_stays_near_home(package):
    """At night a sow and her piglets are up, rooting and walking, the piglets near her to the end; a rat keeps
    within its range of where it started, scurries, rears up and washes, and never lies down."""
    ground = _ground(holes=False)
    seed = _sow_seed()
    planned = animals.plan(seed, ground, {"boar": 1, "rat": 1}, 300.0, night=True, place="town")
    boars = [a for a in planned if a.kind == "boar"]
    rats = [a for a in planned if a.kind == "rat"]
    assert [a.body for a in boars] == animals.sounder(seed) and boars[0].body == "boar"
    seen = {s.clip for a in boars for s in a.track.segments}
    assert {"graze", "walk", "stand"} <= seen
    for boar in boars:
        lying = np.array([boar.track.segments[k].clip == "lying" for k in boar.track.segment])
        assert lying.mean() <= 0.5
    sow = boars[0].track
    for piglet in boars[1:]:
        apart = np.hypot(piglet.track.x - sow.x, piglet.track.y - sow.y)
        assert apart.max() <= 2.0 * animal_moves.herd_radius(len(boars)) + 3.0
    assert rats and {a.body for a in rats} == {"rat"}
    played = set()
    for rat in rats:
        away = np.hypot(rat.track.x - rat.track.x[0], rat.track.y - rat.track.y[0])
        # Every spot it runs to lies in its range; the arc it turns through on the way may swing a step past the edge.
        assert away.max() <= animal_moves.RODENT_HOME_M + 1.0
        played |= {s.clip for s in rat.track.segments}
    for other in range(seed + 1, seed + 4):
        played |= {s.clip for a in animals.plan(other, ground, {"rat": 1}, 300.0, place="town") for s in a.track.segments}
    assert played <= set(skins.load("rat").bodies["rat"].clips) and not played & {"lying", "lie_down", "get_up"}
    assert {"trot", "look", "groom", "sniff", "sniff_walk"} <= played
    assert sum(np.hypot(np.diff(a.track.x), np.diff(a.track.y)).sum() for a in rats) > 2.0


def test_piglets_never_come_without_their_sow(package, monkeypatch):
    """Where the sow finds no room to stand, her piglets stay away with her, though they would fit."""
    seed = _sow_seed()
    stands = animal_moves.Mover.can_stand
    monkeypatch.setattr(animal_moves.Mover, "can_stand", lambda self, x, y, others: self.body.name != "boar" and stands(self, x, y, others))
    assert [a.kind for a in animals.plan(seed, _ground(holes=False), {"boar": 1}, _HORIZON_S, place="forest")] == []
    monkeypatch.setattr(animal_moves.Mover, "can_stand", stands)
    assert [a.body for a in animals.plan(seed, _ground(holes=False), {"boar": 1}, _HORIZON_S, place="forest")] == animals.sounder(seed)


def test_the_install_check_names_a_body_its_table_lacks_and_a_file_that_is_gone(tmp_path, monkeypatch):
    """A boar table with no piglet in it, and a rat whose pose file is gone, are both named by the install check
    before any seed asks for them."""
    root = tmp_path / "animals"
    monkeypatch.setattr(skins, "ANIMALS_DIR", animal_stand_in.write_package(root))
    assert all(os.path.isfile(path) for path in skins.asset_files())
    animal_stand_in.write_kind(root / "boars", "boar", ["boar"])
    os.remove(root / "rats" / "rat_poses.npz")
    missing = [path for path in skins.asset_files() if not os.path.isfile(path)]
    assert missing == [str(root / "boars" / "piglet in boars.json"), str(root / "rats" / "rat_poses.npz")]
    assert any(all(os.path.basename(path) in line for path in missing) for line in install.problems())


def test_boars_and_rats_stand_on_the_terrain_itself_and_a_report_on_one_is_an_animal(client, monkeypatch, package):
    """A rat, a few centimetres tall, and a boar or its piglet are drawn on the terrain a ray finds under them where
    the grid's blend lies off it, a dog on the grid as before; each body is named for what it is, so a box on a boar
    or a rat is an animal."""
    ep = _episode(client, monkeypatch)
    monkeypatch.setattr(animals, "wanted", lambda seed, herd_fits=True, place=None: {"dog": 1, "boar": 1, "rat": 1})
    env = SimpleNamespace(CLIENT=client, EP_LEN_SEC=_HORIZON_S)
    animals.reset(env, ep)
    kinds = [a.kind for a in ep.animals.members]
    assert kinds == ["dog"] + ["boar"] * len(animals.sounder(ep.seed)) + ["rat"] * len(animals.rodents(ep.seed))
    assert set(animals.bodies(ep).values()) == {"dog", "boar", "rat"} and len(animals.bodies(ep)) == len(kinds)
    ep.animals.ground.height = ep.animals.ground.height + 0.3
    animals.place(env, ep, 40)
    dog, boar, rat = ep.animals.members[0], ep.animals.members[1], ep.animals.members[-1]
    assert rat.drawn_at[2] == pytest.approx(0.0, abs=1e-6) and boar.drawn_at[2] == pytest.approx(0.0, abs=1e-6)
    assert dog.drawn_at[2] == pytest.approx(0.3, abs=1e-6)
    assert all(p.getCollisionShapeData(a.uid, -1, physicsClientId=client) == () for a in ep.animals.members)


def _slab(cli: int, half: list, centre: list) -> int:
    """A solid box body."""
    shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=cli)
    return p.createMultiBody(0, shape, -1, centre, physicsClientId=cli)


def _episode(cli: int, monkeypatch) -> FieldEpisode:
    """A 20 m square field on a flat slab with a stone off its edge, a person standing on it, a walker, a trunk and
    a house listed by the survey, and the dock in a corner."""
    ground = _slab(cli, [100.0, 100.0, 0.5], [0.0, 0.0, -0.5])
    _slab(cli, [1.0, 1.0, 1.0], [20.0, 0.0, 1.0])
    ep = FieldEpisode(seed=3)
    ep.fence = np.array([[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]])
    ep.terrain_uids = frozenset({ground})
    ep.dock_position = np.array([-8.0, -8.0, 0.0])
    ep.park = {"world": {}}
    ep.people = [people.Person(index=0, pose="standing", xy=(3.0, 3.0), ground_z=0.0, yaw=0.0),
                 people.Person(index=1, pose="walking", xy=(-4.0, 5.0), ground_z=0.0, yaw=0.0, walk=SimpleNamespace(patch=walkers.PATCH_M))]
    monkeypatch.setattr(site, "passable_outlines", lambda episode: np.array([[0.0, -20.0, 3.0]]))
    monkeypatch.setattr(site, "trees", lambda episode: np.array([[0.0, -20.0, 3.0], [15.0, 15.0, 2.5]]))
    monkeypatch.setattr(site, "buildings", lambda episode: np.array([[-25.0, 0.0, 6.0, 4.0, 30.0]]))
    return ep


def test_a_due_laser_uses_the_same_animal_pose_with_or_without_a_zoom(client, monkeypatch, package):
    """A dog leaving the beam before a due reading clears it even without an intervening zoom."""
    ep = FieldEpisode(seed=3, park={}, camera={})
    shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[20.0, 20.0, 0.1], physicsClientId=client)
    ground = p.createMultiBody(0.0, -1, shape, [0.0, 0.0, -0.1], physicsClientId=client)
    env = SimpleNamespace(CLIENT=client, _solar=ep)
    monkeypatch.setattr(airframe, "pose", lambda *args: None)
    monkeypatch.setattr(airframe, "laser_pose", lambda env, tilt: (np.array([0.0, 0.0, 9.0]), np.array([0.0, 0.0, -1.0])))
    monkeypatch.setattr(camera, "aim", lambda *args: SimpleNamespace(eye=(12.0, 0.0, 9.0)))
    for sensor in (camera, zoom):
        monkeypatch.setattr(sensor, "upcoming", lambda *args: None)
        monkeypatch.setattr(sensor, "update", lambda *args: None)
    dog = animals.Animal(kind="dog", body="husky", coat="", size=1.0, pace=1.0, track=SimpleNamespace(
        place=lambda t: (0.0 if t == 0 else 3.0, 0.0, 1.0, 0.0),
        lift=lambda t: 0.0,
    ))
    dog_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.3], physicsClientId=client)
    dog.uid = p.createMultiBody(0.0, -1, dog_shape, physicsClientId=client)
    ep.animals = animals.Animals([dog], _ground(holes=False))
    monkeypatch.setattr(animals, "shape", lambda animal, t: np.zeros((1, 3)))
    monkeypatch.setattr(p, "resetMeshData", lambda *args, **kwargs: None)
    runtime = TestFieldV2ChallengeFamily()
    readings = []
    for zoomed in (False, True):
        animals.place(env, ep, 0)
        assert laser.read(env, 0.0)["body"] == dog.uid
        if zoomed:
            ep.step = laser.READING_STEPS - 2
            with monkeypatch.context() as patch:
                shot = _view([0.0, 0.0, 9.0], [1.5, 0.0, 0.0])
                patch.setattr(zoom, "upcoming", lambda *args: shot)
                animals.advance(env, ep)
        assert dog.drawn_at[0] == (3.0 if zoomed else 0.0)
        ep.step = laser.READING_STEPS
        laser.reset(env, ep)
        runtime.sense(env, ep)
        readings.append((ep.laser["body"], ep.laser["range_m"], ep.laser["status"]))
    assert readings[0] == readings[1] and readings[0][0] == ground


def test_the_ground_keeps_off_the_dock_people_stones_trunks_and_houses_inside_the_band(client, monkeypatch):
    """Rays find the open ground in the band round the field, off the dock, each person's ground, a stone, a listed
    trunk and a listed house; the field's own cells are marked."""
    ep = _episode(client, monkeypatch)
    ground = animals.open_ground(client, ep)
    rows, cols = np.nonzero(ground.open)
    x, y = ground.origin[0] + cols * ground.step, ground.origin[1] + rows * ground.step
    assert ground.open.any() and ground.inside[ground.cell(0.0, 0.0)] and not ground.inside[ground.cell(15.0, 0.0)]
    outside = np.maximum(np.abs(x) - 10.0, 0.0) ** 2 + np.maximum(np.abs(y) - 10.0, 0.0) ** 2
    assert outside.max() <= animals.BAND_M ** 2 + 1e-6
    assert np.hypot(x + 8.0, y + 8.0).min() >= animals.DOCK_CLEAR_M
    assert np.hypot(x - 3.0, y - 3.0).min() >= animals.PERSON_CLEAR_M
    assert np.hypot(x + 4.0, y - 5.0).min() >= animals.PERSON_CLEAR_M + walkers.PATCH_M
    assert not ((np.abs(x - 20.0) < 1.0) & (np.abs(y) < 1.0)).any()
    assert np.hypot(x, y + 20.0).min() >= animals.TRUNK_CLEAR_M
    assert np.hypot(x + 25.0, y).min() >= math.hypot(6.0, 4.0) / 2.0 + animals.HOUSE_CLEAR_M
    assert not ground.sky[ground.cell(15.0, 15.0)] and not ground.sky[ground.cell(15.0, 13.0)]
    assert not ground.sky[ground.cell(3.0, 3.0 + animals.PERSON_CLEAR_M - 0.5)] and ground.sky[ground.cell(5.0, -5.0)]
    ep.park["open_at"] = lambda gx, gy: gx > 0.0
    assert (ground.origin[0] + np.nonzero(animals.open_ground(client, ep).open)[1] * ground.step).min() > 0.0


def _view(eye, target, width=320, height=240) -> camera.View:
    """A colour camera from a point at a target."""
    eye, target = np.asarray(eye, dtype=float), np.asarray(target, dtype=float)
    forward = (target - eye) / np.linalg.norm(target - eye)
    side = np.cross(forward, [0.0, 0.0, 1.0])
    up = np.cross(side / np.linalg.norm(side), forward)
    return camera.View("colour", tuple(eye), tuple(forward), tuple(up), width, height, 50.0, True, 0)


def test_bodies_are_visual_only_and_posed_only_for_frames_that_see_them_move(client, monkeypatch, package):
    """Every animal gets one body with no collision shape, mapped to its kind; a frame that sees an animal rewrites
    its mesh when its pose moved, a frame that sees nothing moves and rewrites none."""
    ep = _episode(client, monkeypatch)
    monkeypatch.setattr(animals, "wanted", lambda seed, herd_fits=True, place=None: {"goat": 4, "dog": 2})
    env = SimpleNamespace(CLIENT=client, EP_LEN_SEC=_HORIZON_S)
    animals.reset(env, ep)
    uids = [a.uid for a in ep.animals.members]
    assert len(uids) == 6 and animals.bodies(ep) == {a.uid: a.kind for a in ep.animals.members}
    assert all(p.getCollisionShapeData(uid, -1, physicsClientId=client) == () for uid in uids)
    x, y, _ = animals.positions(ep, 0)[0]
    assert p.rayTest([x, y, 50.0], [x, y, -50.0], physicsClientId=client)[0][0] not in uids
    rewrites = []
    real = p.resetMeshData
    monkeypatch.setattr(animals.p, "resetMeshData", lambda uid, *a, **k: (rewrites.append(uid), real(uid, *a, **k)))
    first = ep.animals.members[0]
    step = next(k for k in range(1, 3000) if np.abs(animals.shape(first, k * SIM_DT) - first.shown).max() > 0.04)
    away = _view([500.0, 500.0, 10.0], [600.0, 500.0, 10.0])
    stood = p.getBasePositionAndOrientation(first.uid, physicsClientId=client)
    animals.place(env, ep, step, [away])
    assert rewrites == [] and p.getBasePositionAndOrientation(first.uid, physicsClientId=client) == stood
    x, y, _ = animals.positions(ep, step)[0]
    looking = _view([x - 6.0, y - 6.0, 4.0], [x, y, 0.3])
    animals.place(env, ep, step, [looking])
    animals.place(env, ep, step, [looking])
    assert uids[0] in rewrites and rewrites.count(uids[0]) == 1


def test_a_seed_on_the_stand_in_field_puts_animals_a_report_calls_animal(tmp_path, monkeypatch, package):
    """On the stand-in field a seed with animals builds them clear of the people; a report boxed on an animal in a
    drawn frame is judged an animal and counted apart."""
    if not _SHIPPED:
        pytest.skip("the installed swarm-worlds has no M4TD or no intruder bodies")
    monkeypatch.setattr(install, "area_problems", lambda: [])
    field_v2_stand_in.use_area(tmp_path, monkeypatch)
    field_v2_stand_in.use_people(tmp_path / "people", monkeypatch)
    monkeypatch.setattr(animals, "wanted", lambda seed, herd_fits=True, place=None: {"goat": 3, "dog": 1})
    with contextlib.redirect_stdout(io.StringIO()):
        env, _obs = make_env_with_initial_obs(field_task.test_field_task(seed=11, sim_dt=SIM_DT))
    try:
        ep = env._solar
        assert len(ep.animals.members) == 4
        first = ep.animals.members[0]
        x, y, _ = animals.positions(ep, 0)[0]
        for person in people.people(ep):
            assert math.hypot(x - person.xy[0], y - person.xy[1]) >= animals.PERSON_CLEAR_M
        z = float(ep.animals.ground.height_at(np.array([[x, y]]))[0])
        view = _view([x - 5.0, y - 5.0, z + 4.0], [x, y, z + 0.3], 640, 480)
        view = dataclasses.replace(view, objects=np.asarray(camera.colour_frame(env, view)[1]))
        boxes = sentinel_reports.object_boxes(view, reports.candidates(ep))
        left, top, right, bottom = boxes[("animal", first.uid)]
        monkeypatch.setattr(sensor_noise, "shown_view", lambda episode, image: view)
        monkeypatch.setattr(coverage, "height_above_ground", lambda env_, episode, eye: 5.0)
        box = Box(cx=(left + right) / 1280.0, cy=(top + bottom) / 960.0, w=(right - left) / 640.0, h=(bottom - top) / 480.0)
        reports.submit(env, ep, FieldReport("person", "feed", box, east_m=0.0, north_m=0.0))
        assert ep.reports["log"][-1][2] == "animal" and ep.outcome.animal_reports == 1
    finally:
        env.close()



def test_a_report_boxed_on_a_boar_or_on_a_rat_is_judged_an_animal(tmp_path, monkeypatch, package):
    """On the stand-in field a report whose box lies on a boar, and one whose box lies on a rat, are each judged an
    animal and counted apart."""
    if not _SHIPPED:
        pytest.skip("the installed swarm-worlds has no M4TD or no intruder bodies")
    monkeypatch.setattr(install, "area_problems", lambda: [])
    field_v2_stand_in.use_area(tmp_path, monkeypatch)
    field_v2_stand_in.use_people(tmp_path / "people", monkeypatch)
    monkeypatch.setattr(animals, "wanted", lambda seed, herd_fits=True, place=None: {"boar": 1, "rat": 1})
    monkeypatch.setattr(animals, "rodents", lambda seed, place=None: ["rat"])
    with contextlib.redirect_stdout(io.StringIO()):
        env, _obs = make_env_with_initial_obs(field_task.test_field_task(seed=11, sim_dt=SIM_DT))
    try:
        ep = env._solar
        assert [a.kind for a in ep.animals.members][0] == "boar" and ep.animals.members[-1].body == "rat"
        monkeypatch.setattr(coverage, "height_above_ground", lambda env_, episode, eye: 5.0)
        for count, (animal, back) in enumerate(((ep.animals.members[0], 5.0), (ep.animals.members[-1], 1.5)), start=1):
            x, y, z = animal.drawn_at
            view = _view([x - back, y - back, z + 0.8 * back], [x, y, z + 0.1], 640, 480)
            view = dataclasses.replace(view, objects=np.asarray(camera.colour_frame(env, view)[1]))
            left, top, right, bottom = sentinel_reports.object_boxes(view, reports.candidates(ep))[("animal", animal.uid)]
            monkeypatch.setattr(sensor_noise, "shown_view", lambda episode, image, view=view: view)
            box = Box(cx=(left + right) / 1280.0, cy=(top + bottom) / 960.0, w=(right - left) / 640.0, h=(bottom - top) / 480.0)
            reports.submit(env, ep, FieldReport("person", "feed", box, east_m=0.0, north_m=0.0))
            assert ep.reports["log"][-1][2] == "animal" and ep.outcome.animal_reports == count
    finally:
        env.close()


def test_a_frame_asks_after_a_bird_as_wide_as_it_reaches_with_its_wings_spread(client, monkeypatch, package):
    """Whether a frame sees a bird is asked of a ball as wide as the bird reaches in any pose, wider than its length,
    while a goat or dog is asked of its length."""
    ep = _episode(client, monkeypatch)
    monkeypatch.setattr(animals, "wanted", lambda seed, herd_fits=True, place=None: {"dog": 1, "bird": 1})
    env = SimpleNamespace(CLIENT=client, EP_LEN_SEC=_HORIZON_S)
    animals.reset(env, ep)
    dog, bird = ep.animals.members
    assert bird.kind == "bird" and bird.skin().extent_m > bird.skin().length_m
    asked = {}
    real = animals.Sight.per_metre

    def recording(sight, point, margin):
        """The sight's own answer, the margin asked kept by the point's place."""
        asked[round(float(point[0]), 6)] = margin
        return real(sight, point, margin)

    monkeypatch.setattr(animals.Sight, "per_metre", recording)
    animals.place(env, ep, 20, [_view([0.0, 0.0, 30.0], [0.0, 0.1, 0.0])])
    reaches = {round(float(a.drawn_at[0]), 6): a for a in (dog, bird)}
    assert set(asked) >= set(reaches)
    assert asked[round(float(bird.drawn_at[0]), 6)] == pytest.approx(bird.skin().extent_m * bird.size)
    assert asked[round(float(dog.drawn_at[0]), 6)] == pytest.approx(dog.skin().length_m * dog.size)


def test_a_bird_stands_on_the_terrain_itself_where_the_grid_blends_it_off(client, monkeypatch, package):
    """Where the ground grid's blend lies off the terrain, a bird is drawn on the terrain a ray finds under it, a dog
    on the grid as before."""
    ep = _episode(client, monkeypatch)
    monkeypatch.setattr(animals, "wanted", lambda seed, herd_fits=True, place=None: {"dog": 1, "bird": 1})
    env = SimpleNamespace(CLIENT=client, EP_LEN_SEC=_HORIZON_S)
    animals.reset(env, ep)
    ep.animals.ground.height = ep.animals.ground.height + 0.3
    step = next(k for k in range(0, 3000, 5) if ep.animals.members[1].track.lift(k * SIM_DT) == 0.0)
    animals.place(env, ep, step)
    dog, bird = ep.animals.members
    assert bird.kind == "bird" and bird.drawn_at[2] == pytest.approx(0.0, abs=1e-6)
    assert dog.drawn_at[2] == pytest.approx(0.3, abs=1e-6)


def test_a_bird_is_held_up_by_as_much_as_its_pose_reaches_under_it(client, monkeypatch, package):
    """A pose that reaches under its own ground, as wings do at the bottom of a beat, lifts the bird by that much and
    no more than its shape needs; a pose that stands on its ground lifts it by nothing."""
    ep = _episode(client, monkeypatch)
    monkeypatch.setattr(animals, "wanted", lambda seed, herd_fits=True, place=None: {"bird": 1})
    env = SimpleNamespace(CLIENT=client, EP_LEN_SEC=_HORIZON_S)
    animals.reset(env, ep)
    (bird,) = ep.animals.members
    step = next(k for k in range(0, 3000, 5) if bird.track.lift(k * SIM_DT) == 0.0)
    animals.place(env, ep, step)
    assert bird.drawn_at[2] == pytest.approx(0.0, abs=1e-6)
    lows = bird.skin().lows
    kept = lows.copy()
    try:
        ground = ep.animals.ground
        ground.height = ground.height + 0.05 * np.arange(ground.height.shape[1])[None, :]
        animals.place(env, ep, step)
        _at, level = p.getBasePositionAndOrientation(bird.uid, physicsClientId=client)
        lows[:] = np.float32(-0.05)
        animals.place(env, ep, step)
        _at, lifted = p.getBasePositionAndOrientation(bird.uid, physicsClientId=client)
        assert animals.sink(bird, step * SIM_DT) == pytest.approx(0.05 * bird.size, abs=1e-6)
        assert bird.drawn_at[2] == pytest.approx(0.05 * bird.size, abs=1e-6)
        # Held up for its wings, it still lies along the slope it stands on.
        assert lifted == level and level != (0.0, 0.0, 0.0, 1.0)
    finally:
        lows[:] = kept
    for k in range(0, 3000, 7):
        deepest = -float(animals.shape(bird, k * SIM_DT)[:, 2].min())
        assert animals.sink(bird, k * SIM_DT) >= deepest - 1e-6


def test_a_bird_in_the_air_is_drawn_at_its_height_and_a_report_on_it_is_an_animal(tmp_path, monkeypatch, package):
    """On the stand-in field a seed's birds fly; one in the air is drawn over the ground by its height, is named a
    bird, and a report boxed on it in a drawn frame is judged an animal and counted apart."""
    if not _SHIPPED:
        pytest.skip("the installed swarm-worlds has no M4TD or no intruder bodies")
    monkeypatch.setattr(install, "area_problems", lambda: [])
    field_v2_stand_in.use_area(tmp_path, monkeypatch)
    field_v2_stand_in.use_people(tmp_path / "people", monkeypatch)
    monkeypatch.setattr(animals, "wanted", lambda seed, herd_fits=True, place=None: {"bird": 4})
    with contextlib.redirect_stdout(io.StringIO()):
        env, _obs = make_env_with_initial_obs(field_task.test_field_task(seed=11, sim_dt=SIM_DT))
    try:
        ep = env._solar
        bird = next(a for a in ep.animals.members if a.track.up is not None)
        step = int(round(int(np.argmax(bird.track.up)) * animal_moves.SAMPLE_S / SIM_DT))
        height = bird.track.lift(step * SIM_DT)
        animals.place(env, ep, step)
        (x, y, z), _turn = p.getBasePositionAndOrientation(bird.uid, physicsClientId=env.CLIENT)
        assert height >= 2.0 and z - site.ground_at(env.CLIENT, ep.terrain_uids, x, y) == pytest.approx(height)
        assert animals.bodies(ep)[bird.uid] == "bird" and reports.candidates(ep)[bird.uid] == ("animal", bird.uid)
        view = _view([x - 4.0, y - 4.0, z + 1.5], [x, y, z], 640, 480)
        view = dataclasses.replace(view, objects=np.asarray(camera.colour_frame(env, view)[1]))
        left, top, right, bottom = sentinel_reports.object_boxes(view, reports.candidates(ep))[("animal", bird.uid)]
        monkeypatch.setattr(sensor_noise, "shown_view", lambda episode, image: view)
        monkeypatch.setattr(coverage, "height_above_ground", lambda env_, episode, eye: 5.0)
        box = Box(cx=(left + right) / 1280.0, cy=(top + bottom) / 960.0, w=(right - left) / 640.0, h=(bottom - top) / 480.0)
        reports.submit(env, ep, FieldReport("person", "feed", box, east_m=0.0, north_m=0.0))
        assert ep.reports["log"][-1][2] == "animal" and ep.outcome.animal_reports == 1
    finally:
        env.close()
