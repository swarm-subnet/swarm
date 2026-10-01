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

"""The M4TD and the Dock 3 (task 3): DJI's numbers, the real outline, the dock's lids and pad, and the flight."""

from __future__ import annotations

import contextlib
import hashlib
import io
import math
import os
import subprocess
import sys
import types
import xml.etree.ElementTree as ET

import numpy as np
import pybullet as p
import pytest
import swarm_worlds

from swarm.challenge_families import build_benchmark_tasks
from swarm.challenge_families.solar_patrol import airframe, dock3, drone_state, park
from swarm.challenge_families.solar_patrol.contract import (
    ACTION_DIM,
    ACTION_INDEX,
    FAMILY_ID,
    FLIGHT_PHASES,
    MAX_CLIMB_MPS,
    MAX_DESCENT_MPS,
    MAX_HORIZONTAL_MPS,
    MAX_YAW_RATE_DEG_S,
    STATE_SLICES,
)
from swarm.constants import SIM_DT
from swarm.utils.env_factory import make_env_with_initial_obs
from validator.tests.test_solar_patrol_family import blank_camera  # noqa: F401

ROBOTS = swarm_worlds.robots_dir()
pytestmark = [pytest.mark.skipif(not os.path.isfile(os.path.join(ROBOTS, airframe.URDF)),
                                 reason=f"the installed swarm-worlds has no {airframe.URDF} yet"),
              pytest.mark.usefixtures("blank_camera")]
_FENCE = np.array([[0.0, 40.0], [120.0, 40.0], [120.0, 160.0], [0.0, 160.0]])
_CONTROL_ENV = types.SimpleNamespace(G=9.8, GRAVITY=18.13, KF=3.0833e-07, KM=4.9333e-09, HOVER_RPM=3834.08,
                                     DRAG_COEFF=np.array([0.00015, 0.00015, 0.00035]))


class _StillMovers:
    """Movers for a park that has none."""

    body_uids = frozenset()

    def advance(self, step=None):
        """Nothing moves."""


@pytest.fixture
def flat_park(monkeypatch):
    """Build the patrol's world as flat ground with a square fence around the dock spot."""
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
    """A bare physics client with gravity, closed after the test."""
    cli = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.8, physicsClientId=cli)
    yield cli
    p.disconnect(cli)


def _action(**values):
    """An action vector at rest, with the named fields set."""
    a = np.zeros(ACTION_DIM, dtype=np.float32)
    for name, value in values.items():
        a[ACTION_INDEX[name]] = value
    return a


def _collision_extent(urdf_path: str) -> np.ndarray:
    """Exact size of a body's collision shapes, read from its description (the engine meshes cylinders and pads
    every shape with a contact margin, so its own bounds overstate the size)."""
    corners = []
    link = ET.parse(urdf_path).getroot().find("link")
    for col in link.findall("collision"):
        origin = col.find("origin")
        pos = np.array([float(v) for v in origin.get("xyz").split()])
        rot = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(
            [float(v) for v in origin.get("rpy").split()]))).reshape(3, 3)
        geom = col.find("geometry")[0]
        if geom.tag == "box":
            half = np.array([float(v) for v in geom.get("size").split()]) / 2.0
        elif geom.tag == "cylinder":
            r, length = float(geom.get("radius")), float(geom.get("length"))
            half = np.array([r, r, length / 2.0])
        else:
            half = np.array([float(geom.get("radius"))] * 3)
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    corners.append(pos + rot @ (half * [sx, sy, sz]))
    corners = np.array(corners)
    return corners.max(0) - corners.min(0)


def _dock_bounds(uid: int, cli: int) -> np.ndarray:
    """World size of a dock over its body and both lids."""
    boxes = [p.getAABB(uid, link, physicsClientId=cli) for link in (-1, 0, 1)]
    lo = np.min([b[0] for b in boxes], axis=0)
    hi = np.max([b[1] for b in boxes], axis=0)
    return hi - lo


class _Stub:
    """The few environment fields the dock and airframe helpers read."""

    def __init__(self, cli: int, drone: int = -1):
        """Hold a client and an aircraft at 20 m, level."""
        self.CLIENT = cli
        self.DRONE_IDS = [drone]
        self.pos = [np.array([0.0, 0.0, 20.0])]
        self.quat = [np.array([0.0, 0.0, 0.0, 1.0])]


def test_the_aircraft_carries_dji_numbers():
    """1,850 g, the 498.5 mm wheelbase, DJI's motor spacings and a 6,300 rpm top speed at the thrust it was given."""
    root = ET.parse(os.path.join(ROBOTS, airframe.URDF)).getroot()
    props = root.find("properties").attrib
    assert float(root.find("link/inertial/mass").get("value")) == pytest.approx(1.85)
    rotors = [np.array([float(v) for v in root.find(f"link[@name='prop{i}_link']/inertial/origin").get("xyz").split()])
              for i in range(4)]
    assert np.linalg.norm(rotors[0][:2] - rotors[2][:2]) == pytest.approx(0.4985, abs=0.001)
    assert np.linalg.norm(rotors[0][:2] - rotors[3][:2]) == pytest.approx(0.383, abs=0.001)
    assert np.linalg.norm(rotors[1][:2] - rotors[2][:2]) == pytest.approx(0.343, abs=0.001)
    assert abs(rotors[0][0] - rotors[1][0]) == pytest.approx(0.3416, abs=0.001)
    kf, t2w = float(props["kf"]), float(props["thrust2weight"])
    assert math.sqrt(t2w * 1.85 * 9.8 / (4 * kf)) == pytest.approx(airframe.MAX_RPM, rel=0.001)


def test_the_collision_shape_is_the_real_outline():
    """The collision compound spans DJI's 377.7 x 416.2 mm within 1 %, and the CAD's height with its rear motors."""
    size = _collision_extent(os.path.join(ROBOTS, airframe.URDF))
    assert size[0] == pytest.approx(0.3777, rel=0.01)
    assert size[1] == pytest.approx(0.4162, rel=0.01)
    assert 0.2125 <= size[2] <= 0.2225


def test_the_moving_parts_never_collide(world):
    """Rotors, blades and the camera head carry no collision shape, so nothing can hit them or be hit by them."""
    uid = p.loadURDF(os.path.join(ROBOTS, airframe.MOVING_URDF), [0, 0, 1], physicsClientId=world)
    assert p.getNumJoints(uid, physicsClientId=world) == 13
    for link in range(-1, p.getNumJoints(uid, physicsClientId=world)):
        assert p.getCollisionShapeData(uid, link, physicsClientId=world) == ()


def test_the_dock_is_dji_size_closed_and_open(world):
    """Closed 640 x 745 mm and 625 mm to the lid (770 with the 145 mm wind gauge); open 1,760 mm with the gauge
    and 485 mm high."""
    dock = dock3.spawn(_Stub(world), 0.0, 0.0)
    closed = _dock_bounds(dock.uid, world)
    assert closed[0] == pytest.approx(0.640, abs=0.006)
    assert closed[1] == pytest.approx(0.745, abs=0.006)
    assert closed[2] == pytest.approx(0.770 - 0.145, abs=0.006)
    dock3.set_opening(_Stub(world), dock, 1.0)
    opened = _dock_bounds(dock.uid, world)
    assert opened[0] + 0.145 == pytest.approx(1.760, abs=0.008)
    assert opened[2] == pytest.approx(0.485, abs=0.006)


def test_the_dock_stands_level_on_a_slope(world):
    """On a 10 degree slope the dock stands upright on its concrete base: the base's level top is 100 mm over the high
    side of the ground and the base reaches below the low side, so nothing floats."""
    shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[5.0, 5.0, 0.5], physicsClientId=world)
    p.createMultiBody(0, shape, -1, [0, 0, -0.5], p.getQuaternionFromEuler([math.radians(10.0), 0, 0]),
                      physicsClientId=world)
    env = _Stub(world)
    low, high = dock3.ground_under(env, 0.0, 0.0)
    assert high - low == pytest.approx(dock3.BASE_SIZE_M * math.tan(math.radians(10.0)), abs=0.01)
    dock = dock3.spawn(env, 0.0, 0.0)
    assert dock.origin[2] == pytest.approx(high + dock3.BASE_ABOVE_M)
    lo, _hi = p.getAABB(dock.base_uid, -1, physicsClientId=world)
    assert lo[2] < low
    assert p.getBasePositionAndOrientation(dock.uid, physicsClientId=world)[1] == pytest.approx((0, 0, 0, 1))


def test_the_lids_take_four_seconds():
    """The lids move from closed to open in DJI's four seconds, and back."""
    cli = p.connect(p.DIRECT)
    try:
        dock = dock3.spawn(_Stub(cli), 0.0, 0.0)
        steps = 0
        while not dock3.move_lids(_Stub(cli), dock, True, 0.02):
            steps += 1
        assert (steps + 1) * 0.02 == pytest.approx(dock3.LID_TRAVEL_S, abs=0.03)
        while not dock3.move_lids(_Stub(cli), dock, False, 0.02):
            pass
        assert p.getJointState(dock.uid, 0, physicsClientId=cli)[0] == pytest.approx(0.0)
    finally:
        p.disconnect(cli)


def test_the_aircraft_rests_on_the_pad_as_dji_draws_it(world):
    """Set down on the pad, the aircraft settles on its four feet where DJI's CAD puts it, touching only the pad."""
    dock = dock3.spawn(_Stub(world), 0.0, 0.0)
    rest, _ = dock3.rest_pose(dock)
    drone = p.loadURDF(os.path.join(ROBOTS, airframe.URDF), rest.tolist(), flags=p.URDF_USE_INERTIA_FROM_FILE,
                       physicsClientId=world)
    for _ in range(480):
        p.stepSimulation(physicsClientId=world)
    pos, orn = p.getBasePositionAndOrientation(drone, physicsClientId=world)
    assert pos[2] == pytest.approx(rest[2], abs=0.003)
    assert max(abs(a) for a in p.getEulerFromQuaternion(orn)[:2]) < math.radians(0.5)
    touched = {c[2] for c in p.getContactPoints(bodyA=drone, physicsClientId=world) if c[9] > 0.01}
    assert touched == {dock.pad_uid}


def test_the_camera_and_laser_swing_about_the_gimbal_axis(world):
    """Level, the camera looks along the nose; at +90 it looks straight up; both stay on the tilt axis's circle."""
    env = _Stub(world)
    for tilt in (0.0, 45.0, 90.0):
        pos, forward, up = airframe.camera_pose(env, tilt)
        assert forward == pytest.approx([math.cos(math.radians(tilt)), 0.0, math.sin(math.radians(tilt))], abs=1e-9)
        assert np.dot(forward, up) == pytest.approx(0.0, abs=1e-9)
        pivot = env.pos[0] + airframe.GIMBAL_PIVOT
        radius = np.linalg.norm((airframe.WIDE_CAMERA - airframe.GIMBAL_PIVOT)[[0, 2]])
        assert np.linalg.norm((pos - pivot)[[0, 2]]) == pytest.approx(radius, abs=1e-9)
    origin, _ = airframe.laser_pose(env, 0.0)
    assert origin[2] < airframe.camera_pose(env, 0.0)[0][2]


@pytest.mark.parametrize("tilt, low, high", [(50, 0.0, 0.001), (60, 0.0, 0.03), (70, 0.25, 0.45), (90, 0.99, 1.0)])
def test_the_body_blocks_the_view_above_plus_70(world, tilt, low, high):
    """DJI: from +70 to +90 of tilt the aircraft's body blocks the wide camera. The share of the frame the body
    covers is next to nothing up to +60, a third at +70 and all of it at +90."""
    drone = p.loadURDF(os.path.join(ROBOTS, airframe.URDF), [0, 0, 20], physicsClientId=world)
    env = _Stub(world, drone)
    w, h = 320, 240
    vfov = 2 * math.degrees(math.atan(math.tan(math.radians(41.0)) * h / math.hypot(w, h)))
    pos, forward, up = airframe.camera_pose(env, tilt)
    view = p.computeViewMatrix(list(pos), list(pos + forward), list(up))
    proj = p.computeProjectionMatrixFOV(vfov, w / h, 0.005, 200.0)
    seg = np.asarray(p.getCameraImage(w, h, view, proj, renderer=p.ER_TINY_RENDERER, physicsClientId=world)[4])
    share = float(((seg.reshape(-1) & ((1 << 24) - 1)) == drone).mean())
    assert low <= share <= high


def _fly(pilot, max_decisions: int = 1500):
    """Fly one patrol on flat ground; returns the per-step log and the episode."""
    task = build_benchmark_tasks(sim_dt=SIM_DT, seeds=[13], family_id=FAMILY_ID)[0]
    with contextlib.redirect_stdout(io.StringIO()):
        env, obs = make_env_with_initial_obs(task)
    log = []
    try:
        for i in range(max_decisions):
            phase = FLIGHT_PHASES[int(obs["state"][STATE_SLICES["flight_phase"]][0])]
            obs, _r, terminated, truncated, _info = env.step(pilot(i, phase, env)[None, :])
            rpy = p.getEulerFromQuaternion(env.quat[0])
            log.append({"phase": phase, "tag": getattr(pilot, "tag", ""), "pos": np.array(env.pos[0]),
                        "vel": np.array(env.vel[0]), "tilt": max(abs(rpy[0]), abs(rpy[1])),
                        "yaw_rate": float(env.ang_v[0][2]), "lid": env._solar.dock["model"].opening,
                        "celsius": dict(env._solar.airframe["celsius"])})
            if terminated or truncated:
                break
        episode = env._solar
    finally:
        env.close()
    return log, episode


def _scripted(script):
    """A pilot that takes off, flies each (tag, decisions, action) leg once flying, then asks to come home."""
    def pilot(i, phase, env):
        """The next action for this decision."""
        if i == 0:
            return _action(take_off=1.0)
        if phase != "flying" or pilot.leg >= len(script):
            if phase == "flying" and not pilot.home:
                pilot.home, pilot.tag = True, "home"
                return _action(return_home=1.0)
            return _action()
        tag, count, fields = script[pilot.leg]
        pilot.tag = tag
        pilot.done += 1
        if pilot.done >= count:
            pilot.leg, pilot.done = pilot.leg + 1, 0
        return _action(**fields)
    pilot.leg, pilot.done, pilot.home, pilot.tag = 0, 0, False, "take_off"
    return pilot


@pytest.mark.timeout(300)
def test_the_aircraft_flies_inside_the_m4td_limits(flat_park):
    """Out of the dock and back into it: the aircraft sits level while the lids open, then climbs, descents,
    full-speed flight and a full-rate turn each reach what was asked within a few percent, never past 25 degrees of
    tilt, and the landing is in the pad."""
    log, episode = _fly(_scripted([("hover", 30, {}), ("climb", 40, {"move_up": 1.0}), ("settle", 20, {}),
                                   ("descend", 40, {"move_up": -1.0}), ("settle", 20, {}),
                                   ("forward", 100, {"move_forward": 1.0}), ("brake", 40, {}),
                                   ("turn", 30, {"turn": 1.0}), ("settle", 20, {})]))

    def settled(tag, key):
        """The mean of a value over the second half of a leg."""
        rows = [r for r in log if r["tag"] == tag]
        return np.mean([key(r) for r in rows[len(rows) // 2:]])

    lifted = [i for i, r in enumerate(log) if r["pos"][2] > log[0]["pos"][2] + 0.05][0]
    assert log[lifted]["lid"] == pytest.approx(1.0)
    assert max(r["tilt"] for r in log[:lifted - 10]) < math.radians(1.0)
    assert max(r["tilt"] for r in log) < airframe.MAX_TILT_RAD
    assert settled("climb", lambda r: r["vel"][2]) == pytest.approx(MAX_CLIMB_MPS, rel=0.05)
    assert settled("descend", lambda r: -r["vel"][2]) == pytest.approx(MAX_DESCENT_MPS, rel=0.05)
    assert settled("forward", lambda r: math.hypot(*r["vel"][:2])) == pytest.approx(MAX_HORIZONTAL_MPS, rel=0.05)
    assert settled("turn", lambda r: abs(math.degrees(r["yaw_rate"]))) == pytest.approx(MAX_YAW_RATE_DEG_S, rel=0.05)
    assert episode.outcome.end_reason == "landed" and episode.outcome.landed_in_dock
    assert log[-1]["celsius"]["motor"] > park.air_c(episode) + 15.0
    assert log[-1]["celsius"]["battery"] > airframe.BATTERY_BASE_C


@pytest.mark.timeout(300)
def test_home_from_full_speed_lands_in_the_pad(flat_park):
    """Asked home straight out of a full-speed leg, the aircraft stops over the dock without swinging past it and
    lands in the pad, never touching the dock's body or lids."""
    log, episode = _fly(_scripted([("turn", 20, {"turn": 1.0}), ("forward", 80, {"move_forward": 1.0})]))
    assert episode.outcome.end_reason == "landed" and episode.outcome.landed_in_dock
    landing = [r for r in log if r["phase"] == "landing"]
    offset = [np.linalg.norm(r["pos"][:2] - episode.dock_position[:2]) for r in landing]
    assert max(offset) < 0.5


@pytest.mark.timeout(300)
def test_a_crash_happens_at_the_real_size(flat_park, monkeypatch):
    """Flying sideways into a wall ends the patrol when the motor pods touch it, 208 mm from the centre line, where
    a drone the size of the old default body would still have 90 mm to go."""
    def pilot(i, phase, env):
        """Take off, then slide left at 0.75 m/s towards a wall standing 1.5 m beside the patrol height."""
        if i == 0:
            return _action(take_off=1.0)
        if phase == "flying":
            if pilot.wall is None:
                y = float(env.pos[0][1]) + 1.5
                shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[3.0, 0.05, 3.0], physicsClientId=env.CLIENT)
                pilot.wall = p.createMultiBody(0, shape, -1, [float(env.pos[0][0]), y + 0.05, float(env.pos[0][2])],
                                               physicsClientId=env.CLIENT)
                pilot.face = y
            return _action(move_right=-0.15)
        return _action()
    pilot.wall, pilot.face = None, 0.0
    log, episode = _fly(pilot)
    assert episode.outcome.end_reason == "collision"
    assert pilot.face - log[-1]["pos"][1] == pytest.approx(0.2085, abs=0.03)


class _NumpyControl(airframe.M4TDControl):
    """The controller as numpy and scipy ran it: the M4TD position loop in arrays and the gym's attitude loop."""

    _dslPIDAttitudeControl = airframe.DSLPIDControl._dslPIDAttitudeControl

    def _dslPIDPositionControl(self, control_timestep, cur_pos, cur_quat, cur_vel, target_pos, target_rpy,
                               target_vel, cur_rotation):
        """The position loop in numpy arrays and scipy's rotations."""
        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel
        self.integral_pos_e = np.clip(self.integral_pos_e + pos_e * control_timestep, -2.0, 2.0)
        self.integral_pos_e[2] = np.clip(self.integral_pos_e[2], -0.15, 0.15)
        near = np.abs(vel_e) < airframe.INTEGRAL_BAND_MPS
        leak = math.exp(-control_timestep / airframe.INTEGRAL_LEAK_S)
        self.integral_vel_e = np.clip(self.integral_vel_e * leak + np.where(near, vel_e, 0.0) * control_timestep,
                                      -airframe.INTEGRAL_LIMIT, airframe.INTEGRAL_LIMIT)
        target_thrust = (self.P_COEFF_FOR * pos_e + self.I_COEFF_FOR * self.integral_pos_e
                         + self.D_COEFF_FOR * vel_e + airframe.VEL_INTEGRAL * self.integral_vel_e
                         + self.drag_feedforward * target_vel + np.array([0.0, 0.0, self.GRAVITY]))
        target_thrust[2] = max(float(target_thrust[2]), 0.1 * self.GRAVITY)
        side = math.hypot(target_thrust[0], target_thrust[1])
        cap = target_thrust[2] * math.tan(airframe.MAX_TILT_RAD)
        if side > cap:
            target_thrust[:2] *= cap / side
        scalar_thrust = max(0.0, float(np.dot(target_thrust, cur_rotation[:, 2])))
        thrust = (math.sqrt(scalar_thrust / (4 * self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        target_z_ax = target_thrust / np.linalg.norm(target_thrust)
        target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0.0])
        zx_cross = np.cross(target_z_ax, target_x_c)
        target_y_ax = zx_cross / np.linalg.norm(zx_cross)
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        target_rotation = np.vstack([target_x_ax, target_y_ax, target_z_ax]).transpose()
        target_euler = airframe.Rotation.from_matrix(target_rotation).as_euler("XYZ", degrees=False)
        return thrust, target_euler, pos_e


def _random_flight(steps: int):
    """Controller inputs of a fixed random flight, well past the patrol's tilts, speeds and turns."""
    rng = np.random.default_rng(11)
    for _ in range(steps):
        quat = rng.normal(size=4) * [0.3, 0.3, 1.0, 1.0]
        pos = rng.normal(0.0, 50.0, 3)
        vel = rng.normal(0.0, 6.0, 3)
        target_vel = rng.normal(0.0, 8.0, 3) if rng.random() < 0.8 else np.zeros(3)
        if rng.random() < 0.2:
            vel = target_vel + rng.normal(0.0, 0.3, 3)
        yield dict(control_timestep=SIM_DT, cur_pos=pos, cur_quat=quat / np.sqrt((quat * quat).sum()), cur_vel=vel,
                   cur_ang_vel=rng.normal(size=3), target_pos=pos + rng.normal(size=3) * (rng.random() < 0.3),
                   target_rpy=np.array([0.0, 0.0, rng.uniform(-4.0, 4.0)]), target_vel=target_vel,
                   target_rpy_rates=np.array([0.0, 0.0, rng.normal()]))


def _flight_sha() -> str:
    """Hash of the controller's rotor speeds and yaw errors over the fixed random flight."""
    ctrl, digest = airframe.M4TDControl(_CONTROL_ENV), hashlib.sha256()
    for args in _random_flight(500):
        rpm, _, yaw_e = ctrl.computeControl(**args)
        digest.update(np.asarray(rpm, dtype=np.float64).tobytes() + np.float64(yaw_e).tobytes())
    return digest.hexdigest()


def test_the_controller_matches_the_numpy_and_scipy_loops():
    """Step after step over random flights, the plain-float controller gives the rotor speeds, errors and loop memory
    of the numpy and scipy loops it replaced, apart from the last bits their BLAS sums round differently."""
    fast, reference = airframe.M4TDControl(_CONTROL_ENV), _NumpyControl(_CONTROL_ENV)
    for args in _random_flight(3000):
        for got, want in zip(fast.computeControl(**args), reference.computeControl(**args)):
            np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)
        for name in ("integral_pos_e", "integral_vel_e", "last_rpy", "integral_rpy_e"):
            np.testing.assert_allclose(getattr(fast, name), getattr(reference, name), rtol=1e-9, atol=1e-9)


def test_the_controller_gives_the_same_bits_on_every_blas_kernel():
    """The fixed random flight hashes the same under the oldest and a fused multiply-add BLAS kernel, so validators on
    different CPU families fly the same path; the numpy loops it replaced did not."""
    script = "from validator.tests.test_solar_airframe import _flight_sha; print(_flight_sha())"
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    hashes = {subprocess.run([sys.executable, "-c", script], cwd=root, env=dict(os.environ, OPENBLAS_CORETYPE=core),
                             capture_output=True, text=True, check=True).stdout.split()[-1]
              for core in ("Prescott", "Haswell")}
    assert hashes == {_flight_sha()}


def test_the_rotation_steps_match_scipy_bit_for_bit():
    """The plain-float rotation steps give scipy's exact angles and matrix, and leave gimbal lock to scipy."""
    for m in airframe.Rotation.random(2000, random_state=np.random.default_rng(3)).as_matrix():
        euler = airframe.Rotation.from_matrix(m).as_euler("XYZ", degrees=False)
        assert np.array(airframe._euler_xyz(*m.transpose().tolist())).tobytes() == euler.tobytes()
        matrix = airframe.Rotation.from_quat(airframe.Rotation.from_euler("XYZ", euler).as_quat()).as_matrix()
        assert np.array(airframe._matrix_xyz(*euler.tolist())).tobytes() == matrix.tobytes()
    lock = airframe.Rotation.from_euler("XYZ", [0.3, math.pi / 2, -1.0]).as_matrix()
    assert airframe._euler_xyz(*lock.transpose().tolist()) is None
