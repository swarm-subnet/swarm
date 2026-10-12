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
from types import SimpleNamespace

import numpy as np
import pybullet as p
import pytest
import swarm_worlds

from swarm.challenge_families import build_benchmark_tasks
from swarm.challenge_families.solar_patrol import airframe, camera, dock, dock3, drone_state, laser, park, zoom
from swarm.challenge_families.solar_patrol.contract import (
    ACTION_DIM,
    ACTION_INDEX,
    FAMILY_ID,
    FLIGHT_PHASES,
    HORIZON_S,
    MAX_YAW_RATE_DEG_S,
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
                        "action": action, "lids": env._solar.dock["model"].opening, "info": info})
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


def _nose_off(row, home):
    """How far the drone's nose points away from the home point, in radians."""
    east, north = home - row["pos"][:2]
    return abs((math.atan2(north, east) - row["yaw"] + math.pi) % (2.0 * math.pi) - math.pi)


def _off_line(rows, start, home):
    """The furthest the rows stray sideways from the straight line between start and home."""
    line = (home - start[:2]) / np.linalg.norm(home - start[:2])
    return max(abs(line[0] * (r["pos"][1] - start[1]) - line[1] * (r["pos"][0] - start[0])) for r in rows)


def _counted(monkeypatch, steps, module, name):
    """Replace a module's function with one that notes the control step of every call before making it."""
    real = getattr(module, name)

    def counted(env, *args, **kwargs):
        """Note the step, then do what the real one does."""
        steps.append(env._solar.step)
        return real(env, *args, **kwargs)

    monkeypatch.setattr(module, name, counted)


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
    """One command flies the whole take-off: the drone waits on the pad until the lids are open, the model's moves
    and turns are ignored, it rises straight up and hands over at 20 m, never above it, about 11 s after the press,
    and the lids shut behind it."""
    def pilot(i, phase, env):
        """Press take-off, then push forward and turn on every decision."""
        return _action(take_off=1.0) if i == 0 else _action(move_forward=1.0, turn=1.0, move_up=1.0)

    log, episode, _info, _sun = _fly(21, pilot, max_decisions=400)
    rising = [r for r in log if r["phase"] == "taking_off"]
    start = episode.dock_position
    handed = next(r for r in log if r["phase"] == "flying")
    assert all(r["pos"][2] - start[2] < 0.05 for r in rising if r["lids"] < 1.0)
    assert max(np.hypot(*(r["pos"][:2] - start[:2])) for r in rising) < 0.3
    assert max(abs(r["yaw"] - log[0]["yaw"]) for r in rising) < math.radians(2.0)
    assert max(r["pos"][2] - start[2] for r in rising) <= PATROL_HEIGHT_M + 0.5
    assert handed["pos"][2] - start[2] == pytest.approx(PATROL_HEIGHT_M, abs=dock.ARRIVE_M)
    assert 9.0 <= handed["t"] <= 14.0
    shut = [r for r in log if r["t"] > handed["t"] + dock3.LID_TRAVEL_S + 0.1]
    assert shut and all(r["lids"] == 0.0 for r in shut)


@pytest.mark.timeout(300)
def test_a_return_from_5_to_50_m_turns_to_face_home_then_flies_straight_at_its_own_height(flat_park):
    """Pressed 30 m out at 10 m facing away, the drone turns where it is to face the dock, no faster than the model's
    own turn limit, then flies a straight line home at 10 m without climbing first, and lands."""
    log, episode, info, _sun = _fly(13, _out_and_home(10.0, 30.0))
    back = _rows(log, "returning")
    start, home = back[0]["pos"], episode.dock_position[:2]
    assert 30.0 < np.hypot(*(start[:2] - home)) < dock.RTH_FAR_M
    assert _nose_off(back[0], home) > math.radians(90.0)
    faced = next(k for k, r in enumerate(back) if _nose_off(r, home) < math.radians(dock.FACED_DEG))
    assert max(np.hypot(*(r["pos"][:2] - start[:2])) for r in back[:faced]) < 1.0
    turned = np.abs(np.diff(np.unwrap([r["yaw"] for r in back[:faced + 1]])))
    assert turned.max() <= math.radians(MAX_YAW_RATE_DEG_S) * 0.1 * 1.1
    assert max(abs(r["pos"][2] - start[2]) for r in back) < 1.0
    assert _off_line(back[faced:], start, home) < 1.0
    assert episode.outcome.end_reason == "landed" and info["success"] is True


@pytest.mark.timeout(300)
def test_a_return_from_over_50_m_below_20_m_turns_then_climbs_to_20_m_first(flat_park):
    """Pressed 60 m out at 10 m, the drone turns to face the dock, then climbs to 20 m where it is, then flies a
    straight line home at 20 m and lands; the dock itself never takes it past 22 m."""
    log, episode, info, _sun = _fly(21, _out_and_home(10.0, 60.0))
    back = _rows(log, "returning")
    start, home = back[0]["pos"], episode.dock_position[:2]
    assert np.hypot(*(start[:2] - home)) > dock.RTH_FAR_M
    faced = next(k for k, r in enumerate(back) if _nose_off(r, home) < math.radians(dock.FACED_DEG))
    level = next(k for k, r in enumerate(back) if abs(r["pos"][2] - episode.dock_position[2] - PATROL_HEIGHT_M) < dock.ARRIVE_M)
    assert 0 < faced < level
    assert max(abs(r["pos"][2] - start[2]) for r in back[:faced]) < 0.5
    assert max(np.hypot(*(r["pos"][:2] - start[:2])) for r in back[:level]) < 1.0
    assert max(abs(r["pos"][2] - episode.dock_position[2] - PATROL_HEIGHT_M) for r in back[level:]) < 1.0
    assert _off_line(back[level:], start, home) < 1.5
    assert max(r["pos"][2] - episode.dock_position[2] for r in back + _rows(log, "landing")) <= 22.0
    assert episode.outcome.end_reason == "landed" and info["success"] is True


@pytest.mark.timeout(300)
def test_a_return_from_over_50_m_above_20_m_keeps_its_height(flat_park):
    """Pressed 60 m out at 25 m, the drone does not come down to 20 m first: it turns, flies a straight line home at
    25 m, then lands."""
    log, episode, info, _sun = _fly(28, _out_and_home(25.0, 60.0))
    back = _rows(log, "returning")
    start, home = back[0]["pos"], episode.dock_position[:2]
    assert np.hypot(*(start[:2] - home)) > dock.RTH_FAR_M
    assert max(abs(r["pos"][2] - start[2]) for r in back) < 1.0
    faced = next(k for k, r in enumerate(back) if _nose_off(r, home) < math.radians(dock.FACED_DEG))
    assert _off_line(back[faced:], start, home) < 1.5
    assert episode.outcome.end_reason == "landed" and info["success"] is True


@pytest.mark.timeout(300)
def test_a_near_return_keeps_a_height_above_20_m(flat_park):
    """Pressed 3 m from the dock at 25 m, the drone does not come down to 20 m first: it crosses over at 25 m, then
    lands."""
    log, episode, info, _sun = _fly(23, _out_and_home(25.0, 3.0, speed=0.2))
    back = _rows(log, "returning")
    assert np.hypot(*(back[0]["pos"][:2] - episode.dock_position[:2])) <= dock.RTH_NEAR_M
    assert min(r["pos"][2] - episode.dock_position[2] for r in back) > 24.0
    assert episode.outcome.end_reason == "landed" and info["success"] is True


@pytest.mark.timeout(300)
def test_a_return_under_5_m_crosses_at_its_own_height_without_turning(flat_park):
    """Pressed 3 m from the dock at 14 m, the drone neither climbs nor turns: it crosses over at 14 m, then lands."""
    log, episode, info, _sun = _fly(24, _out_and_home(14.0, 3.0, speed=0.2))
    back = _rows(log, "returning")
    start = back[0]
    assert np.hypot(*(start["pos"][:2] - episode.dock_position[:2])) < dock.RTH_NEAR_M
    assert max(abs(r["pos"][2] - start["pos"][2]) for r in back) < 1.0
    assert _nose_off(start, episode.dock_position[:2]) > math.radians(15.0)
    assert max(abs(r["yaw"] - start["yaw"]) for r in back) < math.radians(dock.FACED_DEG)
    assert episode.outcome.end_reason == "landed" and info["success"] is True


@pytest.mark.timeout(300)
def test_a_landing_waits_above_the_dock_until_the_lids_are_open(flat_park):
    """Pressed right over the pad 1.5 m above it, the return reaches the dock before its lids are open: the drone
    waits WAIT_M above the pad until they are, then lands."""
    log, episode, info, _sun = _fly(19, _out_and_home(1.5, 0.0, speed=0.2))
    early = [r for r in log if r["phase"] in ("returning", "landing") and r["lids"] < 1.0]
    assert any(r["phase"] == "landing" for r in early)
    assert min(r["pos"][2] - episode.dock_position[2] for r in early) > dock.WAIT_M - 0.15
    assert episode.outcome.end_reason == "landed" and info["success"] is True


def test_the_landing_waits_clear_of_the_dock_and_its_lids(world):
    """An aircraft waiting WAIT_M above the pad stands over 0.5 m above DJI's closed dock height and above the lids
    at every point of their swing."""
    if not _M4TD_SHIPPED:
        pytest.skip(f"the installed swarm-worlds has no {airframe.URDF} yet")
    _ground(world)
    env = SimpleNamespace(CLIENT=world)
    model = dock3.spawn(env, 50.0, 50.0)
    rest, _yaw = dock3.rest_pose(model)
    swing = []
    for opening in np.linspace(0.0, 1.0, 41):
        dock3.set_opening(env, model, opening)
        swing.append(max(p.getAABB(model.uid, link, physicsClientId=world)[1][2] for link in (-1, *dock3.LID_JOINTS)))
    aircraft = p.loadURDF(os.path.join(swarm_worlds.robots_dir(), airframe.URDF), (rest + [0.0, 0.0, dock.WAIT_M]).tolist(),
                          physicsClientId=world)
    lowest = min(p.getAABB(aircraft, link, physicsClientId=world)[0][2]
                 for link in range(-1, p.getNumJoints(aircraft, physicsClientId=world)))
    assert lowest - max(swing) > 0.5
    assert lowest - (model.origin[2] + dock3.CLOSED_SIZE_M[2]) > 0.5


@pytest.mark.timeout(300)
def test_the_lids_shut_after_the_landing_without_changing_it(flat_park, monkeypatch):
    """After the landing the patrol runs on only while the lids shut, about 4 s, and ends with them closed. The
    landing keeps its step, end reason and outcome, so its score: sticks, take-off, reports and zooms pressed
    meanwhile do nothing, and no camera frame, zoom or laser reading is taken after the landing step."""
    taken = []
    for module, name in ((camera, "capture"), (zoom, "_draw"), (laser, "read")):
        _counted(monkeypatch, taken, module, name)
    home = _out_and_home(20.0, 30.0)

    def pilot(i, phase, env):
        """Out and home, then once landed push every stick and press take-off, report and zoom every other decision."""
        if env._solar.phase == "landed":
            pressed = float(i % 2)
            return _action(move_forward=1.0, move_up=1.0, turn=1.0, take_off=pressed, report=pressed, zoom=pressed)
        return home(i, phase, env)

    log, episode, info, _sun = _fly(22, pilot)
    landed = _rows(log, "landed")
    first = landed[0]
    landed_at = first["info"]["t_to_goal"]
    assert first["t"] - 0.1 < landed_at <= first["t"]
    assert first["lids"] > 0.9 and episode.dock["model"].opening == 0.0 and log[-1]["phase"] == "landed"
    assert len(landed) == pytest.approx(dock3.LID_TRAVEL_S / 0.1, abs=1)
    assert info["t_to_goal"] == landed_at and info["solar_outcome"] == first["info"]["solar_outcome"]
    assert episode.outcome.end_reason == "landed" and episode.outcome.landed_in_dock
    assert episode.outcome.returned_by_model and info["success"] is True and info["failure_reason"] == "NONE"
    assert max(taken) <= round(landed_at / SIM_DT)
    assert max(np.linalg.norm(r["pos"] - episode.dock_position) for r in landed[1:]) < 0.02


def test_a_clock_running_out_while_the_lids_shut_keeps_the_landing():
    """While the lids shut over a landed drone the patrol neither terminates nor truncates; when the 390 s pass before
    they are shut it ends there as the landing it was, with the landing's own time."""
    family = SolarPatrolChallengeFamily()
    ep = SolarEpisode(seed=0, phase="landed", step=int(round(388.0 / SIM_DT)))
    ep.dock = {"model": SimpleNamespace(opening=0.5)}
    ep.outcome.end_reason, ep.outcome.landed_in_dock, ep.outcome.returned_by_model = "landed", True, True
    env = SimpleNamespace(_solar=ep, EP_LEN_SEC=HORIZON_S, MAX_TILT_RAD=1.0, _time_alive=ep.time_s, _t_to_goal=None,
                          _success=False, _failure_reason="NONE")
    assert not family.compute_terminated(env)
    assert not family.compute_truncated(env, terminal_already=False, roll=0.0, pitch=0.0)
    ep.step = int(round(HORIZON_S / SIM_DT)) + 1
    env._time_alive = ep.time_s
    assert family.compute_terminated(env) and env._success
    assert family.compute_truncated(env, terminal_already=True, roll=0.0, pitch=0.0)
    assert ep.outcome.end_reason == "landed" and env._failure_reason == "NONE"
    assert env._t_to_goal == pytest.approx(388.0)


@pytest.mark.timeout(300)
def test_cancel_hovers_and_hands_back_then_a_second_return_still_lands(flat_park):
    """Cancel stops the return where it is and shuts the lids again: the model flies again from a hover, and pressing
    return home later still lands in the dock and earns the landing."""
    log, episode, info, _sun = _fly(25, _out_and_home(20.0, 25.0, cancel_after=20))
    hover = [r for r in log if r["phase"] == "flying" and r["i"] > _rows(log, "returning")[0]["i"]]
    assert hover and _speed(hover[-1]) < 0.3
    assert max(r["lids"] for r in _rows(log, "returning") if r["i"] < hover[0]["i"]) > 0.0
    assert hover[-1]["lids"] == 0.0
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
    """In a strong seeded wind the dock still lowers the drone into its body about the pad centre, going back up
    to the gate when a gust pushes it past the 3 cm the Dock 3 leaves the M4TD at rest, and lands it."""
    log, episode, info, _sun = _fly(seed, _out_and_home(20.0, 30.0))
    inside = [r for r in _rows(log, "landing") if r["pos"][2] - episode.dock_position[2] < 0.25]
    assert inside
    # The climb back starts once the drift passes 3 cm, so the drift itself may run a few millimetres further.
    assert max(np.hypot(*(r["pos"][:2] - episode.dock_position[:2])) for r in inside) < dock.FINAL_HOLD_M + 0.005
    assert episode.outcome.end_reason == "landed" and info["success"] is True
