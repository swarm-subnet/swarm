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

"""Solar Patrol's dock (task 9): where each seed stands it, and the take-off, return home and landing it flies.

The spot is checked on small worlds built here; the flights stand the family on flat ground with a square fence, as
the skeleton's tests do, and fly the M4TD that ships in swarm-worlds.
"""
from __future__ import annotations

import contextlib
import io
import math
import os

import numpy as np
import pybullet as p
import pytest
import swarm_worlds

from swarm.challenge_families import build_benchmark_tasks
from swarm.challenge_families.solar_patrol import airframe, dock, drone_state, park
from swarm.challenge_families.solar_patrol.contract import (
    ACTION_DIM,
    ACTION_INDEX,
    FAMILY_ID,
    FLIGHT_PHASES,
    HORIZON_S,
    PATROL_HEIGHT_M,
    STATE_SLICES,
    decode_action,
)
from swarm.challenge_families.solar_patrol.episode import SolarEpisode
from swarm.challenge_families.solar_patrol.family import SolarPatrolChallengeFamily
from swarm.constants import SIM_DT
from swarm.core.maps.solar.builder import SOLAR_ASSET_DIR, build_solar_map, solar_manifest, solar_shifts
from swarm.utils.env_factory import make_env_with_initial_obs
from validator.tests.test_solar_patrol_family import blank_camera  # noqa: F401

pytestmark = pytest.mark.usefixtures("blank_camera")
_M4TD_SHIPPED = os.path.isfile(os.path.join(swarm_worlds.robots_dir(), airframe.URDF))
_SOLAR_ASSETS = os.environ.get("SOLAR_ASSET_DIR", SOLAR_ASSET_DIR)
_FENCE = np.array([[0.0, 40.0], [120.0, 40.0], [120.0, 160.0], [0.0, 160.0]])
_SQUARE = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])
_NONE = np.zeros((0, 3))


class _StillMovers:
    """Movers for a park that has none."""

    body_uids = frozenset()

    def advance(self, step=None):
        """Nothing moves."""


@pytest.fixture
def flat_park(monkeypatch):
    """Build the patrol's world as flat ground with a square fence."""
    def build(seed=0, cli=0, asset_dir=None, groups=None):
        """A flat slab standing in for the park's terrain."""
        shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[200.0, 200.0, 0.5], physicsClientId=cli)
        ground = p.createMultiBody(0, shape, -1, [60.0, 100.0, -0.5], physicsClientId=cli)
        return {"bodies": {"terrain": [ground]}, "movers": [], "asset_dir": "flat"}

    monkeypatch.setattr(park, "build_solar_map", build)
    monkeypatch.setattr(park, "build_solar_movers", lambda world, seed=0, cli=0: _StillMovers())
    monkeypatch.setattr(park, "fence_line", lambda asset_dir: _FENCE)
    monkeypatch.setattr(drone_state, "survey", lambda asset_dir: (np.zeros((0, 5)), np.zeros((0, 5))))


@pytest.fixture
def world():
    """A bare physics client, closed after the test."""
    cli = p.connect(p.DIRECT)
    yield cli
    p.disconnect(cli)


def _box(cli, centre, half, pitch=0.0):
    """A static box, tilted about the north axis by pitch radians."""
    shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=list(half), physicsClientId=cli)
    orn = p.getQuaternionFromEuler([0.0, pitch, 0.0])
    return int(p.createMultiBody(0, shape, -1, list(centre), orn, physicsClientId=cli))


def _ground(cli):
    """Flat ground under the whole square park; returns its body as the terrain set."""
    return frozenset({_box(cli, [50.0, 50.0, -0.5], [150.0, 150.0, 0.5])})


def _spots(cli, terrain, passable=_NONE, seeds=range(24), fence=_SQUARE):
    """The dock spot each seed draws in this world."""
    return np.array([dock.find_site(cli, seed, fence, terrain, passable) for seed in seeds])


def _action(**values):
    """An action vector at rest, with the named fields set."""
    a = np.zeros(ACTION_DIM, dtype=np.float32)
    for name, value in values.items():
        a[ACTION_INDEX[name]] = value
    return a


def _fly(seed, pilot, max_decisions=4000):
    """Fly one patrol on flat ground; pilot maps (decision, phase, env) to an action. Returns the per-decision log,
    the episode, the last step's info and the environment's light."""
    if not _M4TD_SHIPPED:
        pytest.skip(f"the installed swarm-worlds has no {airframe.URDF} yet")
    task = build_benchmark_tasks(sim_dt=SIM_DT, seeds=[seed], family_id=FAMILY_ID)[0]
    with contextlib.redirect_stdout(io.StringIO()):
        env, obs = make_env_with_initial_obs(task)
    log, info = [], {}
    try:
        for i in range(max_decisions):
            phase = FLIGHT_PHASES[int(obs["state"][STATE_SLICES["flight_phase"]][0])]
            action = pilot(i, phase, env)
            obs, _r, terminated, truncated, info = env.step(action[None, :])
            log.append({"i": i, "t": env._time_alive, "phase": env._solar.phase, "pos": np.array(env.pos[0]),
                        "vel": np.array(env.vel[0]), "yaw": float(p.getEulerFromQuaternion(env.quat[0])[2]),
                        "action": action})
            if terminated or truncated:
                break
        return log, env._solar, info, getattr(env, "_sun", None)
    finally:
        env.close()


def _height(env):
    """The drone's height above the dock."""
    return float(env.pos[0][2] - env._solar.dock_position[2])


def _out_and_home(height, distance, speed=1.0, hover_until=None, cancel_after=None):
    """A pilot that takes off, settles at a height, flies a distance towards the middle of the fence at a share of
    full speed, brakes, then presses return home; with cancel_after it cancels that many decisions later and
    presses again once hovering."""
    state = {"stage": "up", "from": None, "wait": 0}

    def pilot(i, phase, env):
        """The next action for this decision."""
        if i == 0:
            return _action(take_off=1.0)
        if phase != "flying" and state["stage"] in ("up", "out"):
            return _action()
        hold = float(np.clip(height - _height(env), -1.0, 1.0))
        if state["stage"] == "up":
            if abs(height - _height(env)) < 0.2:
                state["stage"], state["from"] = "out", np.array(env.pos[0][:2])
            return _action(move_up=hold)
        if state["stage"] == "out":
            if np.hypot(*(env.pos[0][:2] - state["from"])) >= distance:
                state["stage"] = "brake"
            ux, uy = (_FENCE.mean(axis=0) - state["from"]) / np.linalg.norm(_FENCE.mean(axis=0) - state["from"])
            c, s = math.cos(float(env.rpy[0, 2])), math.sin(float(env.rpy[0, 2]))
            return _action(move_forward=speed * (ux * c + uy * s), move_right=speed * (ux * s - uy * c), move_up=hold)
        if state["stage"] == "brake":
            state["wait"] += 1
            if state["wait"] >= 40 and (hover_until is None or env._time_alive >= hover_until):
                state["stage"], state["wait"] = "home", 0
                return _action(return_home=1.0)
            return _action(move_up=hold)
        if state["stage"] == "home" and cancel_after is not None:
            state["wait"] += 1
            if state["wait"] >= cancel_after:
                state["stage"], state["wait"] = "hover", 0
                return _action(cancel_return=1.0)
        if state["stage"] == "hover":
            state["wait"] += 1
            if state["wait"] >= 40:
                state["stage"] = "again"
                return _action(return_home=1.0)
        return _action()

    return pilot


def _rows(log, phase):
    """The log rows flown in one phase."""
    return [r for r in log if r["phase"] == phase]


def _speed(row):
    """Horizontal speed of one log row."""
    return float(math.hypot(*row["vel"][:2]))


# ---------------------------------------------------------------- where the dock stands


def test_the_dock_stands_inside_the_fence_by_djis_take_off_margin(world):
    """Every spot lies at least 10 m inside the fence, so at least 10 m inside a flight limit drawn outside it."""
    spots = _spots(world, _ground(world))
    margin = np.minimum.reduce([spots[:, 0], 100.0 - spots[:, 0], spots[:, 1], 100.0 - spots[:, 1]])
    assert margin.min() >= dock.FENCE_MARGIN_M


def test_the_dock_keeps_clear_of_panel_rows(world):
    """Across rows of 4 m deep tables 9 m apart, split by 5 m gaps as the park's are, every spot keeps 3.4 m from
    the nearest table, so none sits in an aisle or a gap: all stand beyond the ends of the rows."""
    terrain = _ground(world)
    rows = [(x, y) for y in range(12, 90, 9) for x in (30.0, 55.0)]
    for x, y in rows:
        _box(world, [x, y, 1.25], [10.0, 2.0, 1.25])
    spots = _spots(world, terrain)
    for x, y in rows:
        gap = np.hypot(np.maximum(np.abs(spots[:, 0] - x) - 10.0, 0.0), np.maximum(np.abs(spots[:, 1] - y) - 2.0, 0.0))
        assert gap.min() >= dock.CLEAR_M
    assert np.all((spots[:, 0] < 20.0 - dock.CLEAR_M) | (spots[:, 0] > 65.0 + dock.CLEAR_M) |
                  (spots[:, 1] > 86.0 + dock.CLEAR_M))


def test_the_dock_finds_a_trunk_thinner_than_its_coarse_look(world):
    """Trunks 0.13 m in radius on an 8 m grid never stand within 3.4 m of a spot."""
    terrain = _ground(world)
    trunk = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.13, height=4.0, physicsClientId=world)
    trunks = np.array([(x, y) for x in np.arange(4.0, 100.0, 8.0) for y in np.arange(4.0, 100.0, 8.0)])
    for x, y in trunks:
        p.createMultiBody(0, trunk, -1, [x, y, 2.0], physicsClientId=world)
    spots = _spots(world, terrain)
    nearest = np.min(np.hypot(*(spots[:, None, :] - trunks[None, :, :]).transpose(2, 0, 1)), axis=1)
    assert nearest.min() >= dock.CLEAR_M


def test_a_crown_the_drone_could_fly_through_still_keeps_the_dock_away(world):
    """A passable outline, like the olive's crown, is kept clear by its radius and the dock's clearance."""
    spots = _spots(world, _ground(world), passable=np.array([[50.0, 50.0, 25.0]]))
    assert np.hypot(spots[:, 0] - 50.0, spots[:, 1] - 50.0).min() >= 25.0 + dock.CLEAR_M


@pytest.mark.parametrize("tilt_deg, on_the_slope", [(30.0, False), (15.0, True)])
def test_ground_steeper_than_the_base_can_level_is_turned_away(world, tilt_deg, on_the_slope):
    """With the east half of the park tilted, 30 degrees keeps every spot on the flat half, and 15 degrees does not."""
    tilt = math.radians(tilt_deg)
    flat = _box(world, [0.0, 50.0, -0.5], [50.0, 150.0, 0.5])
    slope = _box(world, [50.0 + 100.0 * math.cos(tilt), 50.0, -100.0 * math.sin(tilt) - 0.5 * math.cos(tilt)],
                 [100.0, 150.0, 0.5], pitch=tilt)
    spots = _spots(world, frozenset({flat, slope}), seeds=range(40))
    assert (spots[:, 0] > 50.0 + dock.INSTALL_M).any() == on_the_slope


def test_each_seed_has_its_own_spot_and_keeps_it(world):
    """The same seed stands the dock in the same place; different seeds spread it over the park."""
    terrain = _ground(world)
    spots = _spots(world, terrain)
    assert np.array_equal(spots, _spots(world, terrain))
    assert len({tuple(np.round(s, 3)) for s in spots}) == len(spots)
    assert spots.std(axis=0).min() > 10.0


def test_a_park_with_no_room_says_so(world):
    """When nothing inside the fence is open, the seed fails loudly instead of standing the dock anywhere."""
    with pytest.raises(RuntimeError, match="no open ground"):
        dock.find_site(world, 0, _SQUARE, _ground(world), np.array([[50.0, 50.0, 80.0]]))


@pytest.mark.skipif(not os.path.exists(os.path.join(_SOLAR_ASSETS, "manifest.json")),
                    reason=f"solar map not built at {_SOLAR_ASSETS}")
def test_the_map_lists_the_olive_crown_as_passable(world):
    """The park's only piece without a collision shape, the olive's crown, is listed where this seed stands the olive,
    with its 3.3 m radius at this seed's size."""
    built = build_solar_map(seed=0, cli=world, asset_dir=_SOLAR_ASSETS, groups=("park",))
    olive = next(shift for index, shift in solar_shifts(0, _SOLAR_ASSETS).items()
                 if solar_manifest(_SOLAR_ASSETS)["placements"][index]["item"] == "olive_leaves_0")
    crowns = np.array(built["passable"])
    assert len(crowns) == 2
    assert np.allclose(crowns[:, :2], olive["position"][:2], atol=0.3)
    assert np.allclose(crowns[:, 2], 3.3 * olive["scale"][0], atol=0.1)


# ---------------------------------------------------------------- the dock's buttons


def _episode(phase):
    """A patrol in a given flight phase, with a dock that has not been asked for anything."""
    ep = SolarEpisode(seed=0, phase=phase)
    ep.dock = {"return_z": 0.0, "at_height": False}
    return ep


@pytest.mark.parametrize("phase", ["docked", "taking_off", "flying", "landed"])
def test_cancel_only_stops_a_return(phase):
    """Cancel does nothing unless the drone is on its way home or landing."""
    ep = _episode(phase)
    dock.command(None, ep, decode_action(_action(cancel_return=1.0), None))
    assert ep.phase == phase


@pytest.mark.parametrize("phase", ["docked", "taking_off", "returning", "landing", "landed"])
def test_return_home_is_taken_only_in_flight(phase):
    """Return home is a button of the patrol in flight: before take-off, during it, and on the way home it is ignored."""
    ep = _episode(phase)
    dock.command(None, ep, decode_action(_action(return_home=1.0), None))
    assert ep.phase == phase and not ep.outcome.returned_by_model


# ---------------------------------------------------------------- the flights


@pytest.mark.timeout(300)
def test_take_off_opens_the_lids_then_climbs_straight_to_20_m(flat_park):
    """One command flies the whole take-off: the model's moves and turns are ignored, the drone rises straight up
    and hands over at 20 m, never above it, about 11 s after the press."""
    def pilot(i, phase, env):
        """Press take-off, then push forward and turn on every decision."""
        return _action(take_off=1.0) if i == 0 else _action(move_forward=1.0, turn=1.0, move_up=1.0)

    log, episode, _info, _sun = _fly(21, pilot, max_decisions=400)
    rising = [r for r in log if r["phase"] == "taking_off"]
    start = episode.dock_position
    handed = next(r for r in log if r["phase"] == "flying")
    assert max(np.hypot(*(r["pos"][:2] - start[:2])) for r in rising) < 0.3
    assert max(abs(r["yaw"] - log[0]["yaw"]) for r in rising) < math.radians(2.0)
    assert max(r["pos"][2] - start[2] for r in rising) <= PATROL_HEIGHT_M + 0.5
    assert handed["pos"][2] - start[2] == pytest.approx(PATROL_HEIGHT_M, abs=dock.ARRIVE_M)
    assert 9.0 <= handed["t"] <= 14.0


@pytest.mark.timeout(300)
@pytest.mark.parametrize("height", [14.0, 26.0])
def test_a_far_return_goes_to_20_m_first_then_flies_straight_home(flat_park, height):
    """Pressed 30 m out, the drone first climbs or descends to 20 m where it is, then flies a straight line at 20 m
    to the dock and lands in it; the dock itself never takes it past 22 m."""
    log, episode, info, _sun = _fly(22, _out_and_home(height, 30.0))
    back = _rows(log, "returning")
    start = back[0]["pos"]
    level = next(k for k, r in enumerate(back) if abs(r["pos"][2] - episode.dock_position[2] - PATROL_HEIGHT_M) < dock.ARRIVE_M)
    assert max(np.hypot(*(r["pos"][:2] - start[:2])) for r in back[:level]) < 1.0
    home = (episode.dock_position[:2] - start[:2]) / np.linalg.norm(episode.dock_position[:2] - start[:2])
    off_line = [abs(home[0] * (r["pos"][1] - start[1]) - home[1] * (r["pos"][0] - start[0])) for r in back[level:]]
    assert max(off_line) < 1.0
    assert max(abs(r["pos"][2] - episode.dock_position[2] - PATROL_HEIGHT_M) for r in back[level:]) < 1.0
    flown = [r["pos"][2] - episode.dock_position[2] for r in back[level:] + _rows(log, "landing")]
    assert max(flown) <= 22.0
    assert episode.outcome.end_reason == "landed" and info["success"] is True


@pytest.mark.timeout(300)
def test_a_near_return_keeps_a_height_above_20_m(flat_park):
    """Pressed 3 m from the dock at 25 m, the drone does not come down to 20 m first: it crosses over at 25 m, then
    lands."""
    log, episode, info, _sun = _fly(23, _out_and_home(25.0, 3.0, speed=0.2))
    back = _rows(log, "returning")
    assert np.hypot(*(back[0]["pos"][:2] - episode.dock_position[:2])) <= dock.NEAR_M
    assert min(r["pos"][2] - episode.dock_position[2] for r in back) > 24.0
    assert episode.outcome.end_reason == "landed" and info["success"] is True


@pytest.mark.timeout(300)
def test_a_near_return_from_low_climbs_to_20_m_first(flat_park):
    """Pressed 3 m from the dock at 14 m, the drone climbs to 20 m where it is before it moves across."""
    log, episode, info, _sun = _fly(24, _out_and_home(14.0, 3.0, speed=0.2))
    back = _rows(log, "returning")
    start = back[0]["pos"]
    below = [r for r in back if r["pos"][2] - episode.dock_position[2] < PATROL_HEIGHT_M - dock.ARRIVE_M]
    assert below and max(np.hypot(*(r["pos"][:2] - start[:2])) for r in below) < 0.5
    assert episode.outcome.end_reason == "landed" and info["success"] is True


@pytest.mark.timeout(300)
def test_cancel_hovers_and_hands_back_then_a_second_return_still_lands(flat_park):
    """Cancel stops the return where it is: the model flies again from a hover, and pressing return home later
    still lands in the dock and earns the landing."""
    log, episode, info, _sun = _fly(25, _out_and_home(20.0, 25.0, cancel_after=20))
    hover = [r for r in log if r["phase"] == "flying" and r["i"] > _rows(log, "returning")[0]["i"]]
    assert hover and _speed(hover[-1]) < 0.3
    assert episode.outcome.returned_by_model and episode.outcome.end_reason == "landed"
    assert info["success"] is True


@pytest.mark.timeout(600)
def test_a_return_still_under_way_when_time_runs_out_earns_no_landing(flat_park):
    """Return home pressed 30 m out a few seconds before 390 s: the clock ends the patrol before the landing."""
    log, episode, info, _sun = _fly(26, _out_and_home(20.0, 30.0, hover_until=HORIZON_S - 5.0))
    assert _rows(log, "returning")
    assert episode.outcome.end_reason == "timeout"
    assert episode.outcome.returned_by_model and not episode.outcome.landed_in_dock
    assert info["success"] is False


@pytest.mark.timeout(300)
def test_take_off_and_landing_fly_the_same_at_night(flat_park, monkeypatch):
    """Under the moon the dock's flights do not change: take-off, the return and the landing land in the dock."""
    monkeypatch.setattr(SolarPatrolChallengeFamily, "seeded_sun", True)
    monkeypatch.setattr(SolarPatrolChallengeFamily, "night_share", 1.0)
    log, episode, info, sun = _fly(27, _out_and_home(20.0, 30.0))
    assert sun is not None and sun.night
    assert episode.outcome.end_reason == "landed" and info["success"] is True


@pytest.mark.timeout(300)
@pytest.mark.parametrize("seed", [20, 22, 23])
def test_a_strong_wind_landing_enters_the_dock_centred(flat_park, seed):
    """In a strong seeded wind the dock still lowers the drone into its body within 3 cm of the pad centre, the room
    the Dock 3 leaves the M4TD at rest, and lands it."""
    log, episode, info, _sun = _fly(seed, _out_and_home(20.0, 30.0))
    inside = [r for r in _rows(log, "landing") if r["pos"][2] - episode.dock_position[2] < 0.25]
    assert inside
    assert max(np.hypot(*(r["pos"][:2] - episode.dock_position[:2])) for r in inside) < 0.03
    assert episode.outcome.end_reason == "landed" and info["success"] is True
