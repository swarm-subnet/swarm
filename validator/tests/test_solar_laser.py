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

"""The laser point (task 12): distance and hit point against geometry at any tilt, DJI's four statuses and ranges,
what it can hit, its rate, and the reading a patrol hands the model by day and by night."""

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
from swarm.challenge_families.solar_patrol import airframe, drone_state, laser, park
from swarm.challenge_families.solar_patrol.contract import (
    ACTION_DIM,
    ACTION_INDEX,
    FAMILY_ID,
    FLIGHT_PHASES,
    LASER_STATUSES,
    STATE_SLICES,
    new_state,
)
from swarm.challenge_families.solar_patrol.episode import SolarEpisode
from swarm.challenge_families.solar_patrol.family import SolarPatrolChallengeFamily
from swarm.constants import SIM_DT
from swarm.utils.env_factory import make_env_with_initial_obs

ROBOTS = swarm_worlds.robots_dir()
needs_m4td = pytest.mark.skipif(not os.path.isfile(os.path.join(ROBOTS, airframe.URDF)),
                                reason=f"the installed swarm-worlds has no {airframe.URDF} yet")
HEIGHT_M = 20.0


class _Stub:
    """The environment fields the laser reads: the client and the aircraft's pose."""

    def __init__(self, cli: int, yaw_deg: float = 0.0, pitch_deg: float = 0.0, drone: int = -1):
        """An aircraft at HEIGHT_M over the origin, turned and pitched as given."""
        self.CLIENT = cli
        self.DRONE_IDS = [drone]
        self.pos = [np.array([0.0, 0.0, HEIGHT_M])]
        self.quat = [np.array(p.getQuaternionFromEuler([0.0, math.radians(pitch_deg), math.radians(yaw_deg)]))]


@pytest.fixture
def world():
    """A bare physics client, closed after the test."""
    cli = p.connect(p.DIRECT)
    yield cli
    p.disconnect(cli)


def _ground(cli: int, half: float = 1500.0, tile: float = 25.0) -> int:
    """Flat ground at z = 0 drawn as a mesh of tile-sized squares, the way the park's terrain is drawn."""
    n = int(2 * half / tile)
    xs = np.linspace(-half, half, n + 1)
    vertices = [[x, y, 0.0] for y in xs for x in xs]
    indices = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            indices += [a, a + 1, a + n + 2, a, a + n + 2, a + n + 1]
    shape = p.createVisualShape(p.GEOM_MESH, vertices=vertices, indices=indices, physicsClientId=cli)
    return p.createMultiBody(0, -1, shape, [0.0, 0.0, 0.0], physicsClientId=cli)


def _drawn_box(cli: int, centre, half_extents) -> int:
    """A box that is drawn but has no collision, like an intruder or a tree's crown."""
    shape = p.createVisualShape(p.GEOM_BOX, halfExtents=list(half_extents), physicsClientId=cli)
    return p.createMultiBody(0, -1, shape, list(centre), physicsClientId=cli)


def _ground_hit(env: _Stub, tilt_deg: float) -> tuple[float, np.ndarray]:
    """Where the beam meets z = 0, worked out from the beam's line alone: distance and point."""
    origin, direction = airframe.laser_pose(env, tilt_deg)
    t = -origin[2] / direction[2]
    return t, origin + direction * t


@pytest.mark.parametrize("yaw_deg, pitch_deg", [(0.0, 0.0), (30.0, 0.0), (-120.0, 12.0)])
def test_distance_and_ground_point_are_right_at_any_tilt(world, yaw_deg, pitch_deg):
    """Over flat ground, from straight down to a beam grazing the ground 570 m out, the distance and the hit point
    match the beam's own line to a centimetre, with the aircraft turned and pitched as it is in flight."""
    _ground(world)
    env = _Stub(world, yaw_deg, pitch_deg)
    for tilt in (-90.0, -75.0, -60.0, -45.0, -30.0, -15.0, -10.0, -5.0, -3.0, -2.0 + pitch_deg):
        reading = laser.read(env, tilt)
        distance, point = _ground_hit(env, tilt)
        assert reading["status"] == "normal", tilt
        assert reading["range_m"] == pytest.approx(distance, abs=0.01 + 2e-5 * distance), tilt
        assert reading["point"] == pytest.approx(point, abs=0.01 + 2e-5 * distance), tilt


def test_the_model_gets_the_point_from_the_dock(world):
    """The state carries the distance, the hit point in metres east, north and up from the dock, and status normal,
    within DJI's range error."""
    _ground(world)
    env = _Stub(world)
    ep = SolarEpisode(seed=0)
    ep.dock_position = np.array([5.0, -3.0, 1.0])
    ep.camera = {"tilt_deg": -60.0}
    laser.reset(env, ep)
    ep.step = 1
    laser.update(env, ep)
    state = new_state()
    laser.observe(env, ep, state)
    distance, point = _ground_hit(env, -60.0)
    bound = laser.ERROR_M + laser.ERROR_SHARE * distance + 0.01
    assert state[STATE_SLICES["laser_range_m"]][0] == pytest.approx(distance, abs=bound)
    assert state[STATE_SLICES["laser_point_m"]] == pytest.approx(point - ep.dock_position, abs=bound)
    assert LASER_STATUSES[int(state[STATE_SLICES["laser_status"]][0])] == "normal"


def test_open_sky_is_no_signal(world):
    """A beam that meets nothing reads no signal, with the distance and point zero, as DJI sends them."""
    _ground(world)
    env = _Stub(world)
    ep = SolarEpisode(seed=0)
    ep.camera = {"tilt_deg": 30.0}
    laser.reset(env, ep)
    ep.step = 1
    laser.update(env, ep)
    state = new_state()
    laser.observe(env, ep, state)
    assert LASER_STATUSES[int(state[STATE_SLICES["laser_status"]][0])] == "no_signal"
    assert not state[STATE_SLICES["laser_range_m"]].any() and not state[STATE_SLICES["laser_point_m"]].any()


def test_inside_the_blind_zone_is_too_close(world):
    """Something 0.5 m in front of the laser is inside DJI's 1 m blind zone: too close, distance and point zero."""
    env = _Stub(world)
    origin, _ = airframe.laser_pose(env, 0.0)
    _drawn_box(world, origin + [0.6, 0.0, 0.0], [0.1, 1.0, 1.0])
    ep = SolarEpisode(seed=0)
    ep.camera = {"tilt_deg": 0.0}
    laser.reset(env, ep)
    ep.step = 1
    laser.update(env, ep)
    state = new_state()
    laser.observe(env, ep, state)
    assert ep.laser["status"] == "too_close"
    assert LASER_STATUSES[int(state[STATE_SLICES["laser_status"]][0])] == "too_close"
    assert not state[STATE_SLICES["laser_range_m"]].any() and not state[STATE_SLICES["laser_point_m"]].any()


@pytest.mark.parametrize("distance, status", [(1.2, "normal"), (1700.0, "normal"), (1900.0, "too_far")])
def test_a_surface_facing_the_beam_reads_up_to_1800_m(world, distance, status):
    """A wall square to the beam reads from the edge of the blind zone out to 1,800 m, and past that is too far."""
    env = _Stub(world)
    origin, _ = airframe.laser_pose(env, 0.0)
    _drawn_box(world, origin + [distance + 1.0, 0.0, 0.0], [1.0, 100.0, 100.0])
    reading = laser.read(env, 0.0)
    assert reading["status"] == status
    assert reading["range_m"] == pytest.approx(distance, abs=0.01 + 2e-5 * distance)


@pytest.mark.parametrize("tilt, status", [(-2.0, "normal"), (-1.5, "too_far")])
def test_a_slanted_hit_reads_up_to_600_m(world, tilt, status):
    """The beam grazing flat ground meets it far shallower than 1:5, so DJI's 600 m range holds: 571 m reads, 762 m
    is too far although it is well inside 1,800 m."""
    _ground(world)
    env = _Stub(world)
    reading = laser.read(env, tilt)
    distance, _ = _ground_hit(env, tilt)
    assert reading["status"] == status
    assert reading["range_m"] == pytest.approx(distance, abs=0.1)


@pytest.mark.parametrize("slope, status", [(0.25, "normal"), (0.15, "too_far")])
def test_the_600_m_range_starts_at_a_1_in_5_slope(world, slope, status):
    """At 750 m a surface the beam meets at 1:4 still reads, and one it meets at 1:6.7 is too far."""
    env = _Stub(world)
    origin, _ = airframe.laser_pose(env, 0.0)
    turn = math.pi / 2.0 - math.atan(slope)
    wall = _drawn_box(world, origin + [750.0, 0.0, 0.0], [0.5, 2000.0, 400.0])
    p.resetBasePositionAndOrientation(wall, (origin + [750.0, 0.0, 0.0]).tolist(),
                                      p.getQuaternionFromEuler([0.0, 0.0, turn]), physicsClientId=world)
    reading = laser.read(env, 0.0)
    assert reading["status"] == status


def test_the_laser_hits_what_is_drawn_without_collision(world):
    """A person has no collision shape, so a physics ray passes through them to the ground; the laser stops on them,
    as the camera sees them, and names the body it hit."""
    _ground(world)
    slab = p.createCollisionShape(p.GEOM_BOX, halfExtents=[200.0, 200.0, 0.5], physicsClientId=world)
    floor = p.createMultiBody(0, slab, -1, [0.0, 0.0, -0.5], physicsClientId=world)
    env = _Stub(world)
    distance, point = _ground_hit(env, -30.0)
    person = _drawn_box(world, point + [0.0, 0.0, 0.9], [0.2, 0.3, 0.9])
    origin, direction = airframe.laser_pose(env, -30.0)
    assert p.rayTest(origin.tolist(), (origin + direction * 100.0).tolist(), physicsClientId=world)[0][0] == floor
    reading = laser.read(env, -30.0)
    assert reading["body"] == person
    assert reading["status"] == "normal"
    assert reading["range_m"] == pytest.approx(distance - 0.2 / math.cos(math.radians(30.0)), abs=0.01)


def test_a_reading_is_taken_once_a_second(world, monkeypatch):
    """The first reading comes on the first control step, the next ones every 50 steps (1 s at 50 Hz)."""
    taken = []
    monkeypatch.setattr(laser, "read", lambda env, tilt: {"status": "no_signal"})
    ep = SolarEpisode(seed=0)
    ep.camera = {"tilt_deg": 0.0}
    laser.reset(None, ep)
    for step in range(1, 201):
        ep.step = step
        laser.update(None, ep)
        if ep.laser["taken_step"] == step:
            taken.append(step)
    assert taken == [1, 51, 101, 151]
    assert laser.READING_STEPS * SIM_DT == pytest.approx(1.0)


@pytest.mark.parametrize("distance", [5.0, 40.0, 500.0, 1500.0])
def test_the_range_error_is_djis(distance):
    """Over 4,000 readings the error stays inside DJI's +-(0.2 m + 0.15 %), is centred on zero with that bound as
    two standard deviations, moves the point along the beam with it, and is the same for the same seed and step."""
    origin, direction = np.array([1.0, 2.0, 20.0]), np.array([0.6, 0.0, -0.8])
    clean = {"status": "normal", "range_m": distance, "origin": origin, "direction": direction,
             "point": origin + direction * distance, "body": 7}
    readings = [laser.with_error(clean, seed, 50 * k + 1) for seed in range(40) for k in range(100)]
    errors = np.array([r["range_m"] - distance for r in readings])
    bound = laser.ERROR_M + laser.ERROR_SHARE * distance
    assert np.abs(errors).max() <= bound + 1e-9
    assert abs(errors.mean()) < 0.05 * bound
    assert errors.std() == pytest.approx(bound / 2.0, rel=0.1)
    assert all(r["point"] == pytest.approx(origin + direction * r["range_m"]) for r in readings[:50])
    assert laser.with_error(clean, 3, 51)["range_m"] == readings[3 * 100 + 1]["range_m"]
    assert laser.with_error(dict(clean, status="too_far"), 3, 51)["range_m"] == distance


@needs_m4td
@pytest.mark.parametrize("tilt", [60.0, 70.0, 80.0, 90.0])
def test_the_beam_clears_the_aircraft_when_tilted_up(world, tilt):
    """Tilted up, where the aircraft's body fills much of the wide camera's view, the laser window still sees past
    it: the beam reads the open sky, not the drone's own body."""
    drone = p.loadURDF(os.path.join(ROBOTS, airframe.URDF), [0.0, 0.0, HEIGHT_M], physicsClientId=world)
    moving = p.loadURDF(os.path.join(ROBOTS, airframe.MOVING_URDF), [0.0, 0.0, HEIGHT_M], physicsClientId=world)
    joints = {p.getJointInfo(moving, j, physicsClientId=world)[1].decode(): j
              for j in range(p.getNumJoints(moving, physicsClientId=world))}
    p.resetJointState(moving, joints["gimbal_tilt"], math.radians(tilt), physicsClientId=world)
    assert laser.read(_Stub(world, drone=drone), tilt)["status"] == "no_signal"


def _action(**values) -> np.ndarray:
    """An action vector at rest, with the named fields set."""
    a = np.zeros(ACTION_DIM, dtype=np.float32)
    for name, value in values.items():
        a[ACTION_INDEX[name]] = value
    return a


@pytest.fixture
def flat_ground(monkeypatch):
    """The patrol's world as flat drawn ground, no movers, and a square fence around the dock spot."""
    class Still:
        """Movers for a park that has none."""

        body_uids = frozenset()

        def advance(self, step=None):
            """Nothing moves."""

    def build(seed=0, cli=0, asset_dir=None, groups=None):
        """Flat drawn ground standing in for the park's terrain, with a collision slab under it."""
        slab = p.createCollisionShape(p.GEOM_BOX, halfExtents=[200.0, 200.0, 0.5], physicsClientId=cli)
        floor = p.createMultiBody(0, slab, -1, [60.0, 100.0, -0.5], physicsClientId=cli)
        return {"bodies": {"terrain": [floor, _ground(cli, 400.0, 10.0)]}, "movers": [], "asset_dir": "flat"}

    monkeypatch.setattr(park, "build_solar_map", build)
    monkeypatch.setattr(park, "build_solar_movers", lambda world, seed=0, cli=0: Still())
    monkeypatch.setattr(park, "fence_line", lambda asset_dir: np.array([[0.0, 40.0], [120.0, 40.0],
                                                                        [120.0, 160.0], [0.0, 160.0]]))
    monkeypatch.setattr(drone_state, "survey", lambda asset_dir: (np.zeros((0, 5)), np.zeros((0, 5))))


@needs_m4td
@pytest.mark.timeout(300)
@pytest.mark.parametrize("night_share", [0.0, 1.0])
def test_a_patrol_reads_the_ground_by_day_and_by_night(flat_ground, monkeypatch, night_share):
    """In a real patrol, day or night, hovering at 20 m clear of the dock with the camera straight down, the model is
    handed the ground's distance and the point under the drone from the dock, with DJI's range error along the beam,
    refreshed once a second."""
    monkeypatch.setattr(SolarPatrolChallengeFamily, "seeded_sun", True)
    monkeypatch.setattr(SolarPatrolChallengeFamily, "night_share", night_share)
    task = build_benchmark_tasks(sim_dt=SIM_DT, seeds=[13], family_id=FAMILY_ID)[0]
    with contextlib.redirect_stdout(io.StringIO()):
        env, obs = make_env_with_initial_obs(task)
    try:
        assert bool(env._sun.night) == (night_share == 1.0)
        flying, hovering, readings = 0, 0, []
        for i in range(500):
            phase = FLIGHT_PHASES[int(obs["state"][STATE_SLICES["flight_phase"]][0])]
            flying += phase == "flying"
            away = 1.0 if 0 < flying <= 20 else 0.0
            action = _action(take_off=1.0 if i == 0 else 0.0, gimbal_tilt=-1.0, move_forward=away)
            obs, _r, terminated, truncated, _info = env.step(action[None, :])
            assert not (terminated or truncated)
            if flying > 20 and np.linalg.norm(env.vel[0]) < 0.05:
                hovering += 1
                if hovering > 30:
                    readings.append((env._solar.laser["taken_step"], obs["state"].copy(), env._solar))
            if len(readings) >= 25:
                break
        taken = sorted({r[0] for r in readings})
        assert len(taken) >= 2 and all(b - a == laser.READING_STEPS for a, b in zip(taken, taken[1:]))
        _step, state, ep = readings[-1]
        origin, direction = np.asarray(ep.laser["origin"]), np.asarray(ep.laser["direction"])
        distance = -origin[2] / direction[2]
        ground = origin + direction * distance
        error = ep.laser["error_m"]
        assert LASER_STATUSES[int(state[STATE_SLICES["laser_status"]][0])] == "normal"
        assert abs(error) <= laser.ERROR_M + laser.ERROR_SHARE * distance
        assert state[STATE_SLICES["laser_range_m"]][0] == pytest.approx(distance + error, abs=0.02)
        assert distance == pytest.approx(ep.dock_position[2] + 20.0, abs=1.0)
        assert state[STATE_SLICES["laser_point_m"]] == pytest.approx(ground + direction * error - ep.dock_position,
                                                                     abs=0.02)
    finally:
        env.close()
