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

"""M4TD airframe (task 3): the DJI Matrice 4TD as the patrol flies it and as the cameras see it.

The flying body is m4td.urdf in swarm-worlds: 1,850 g, the real outline as its collision shape (dome, fuselage,
feet, gull-wing arms, motor pods), thrust at the CAD motor positions. Swarm's physics flies it; a DSL PID controller
remapped to the 2611 motors' speed range and capped at the M4TD's 25 degree tilt holds each setpoint.

The parts that move on the real aircraft but not in its physics (the four rotors with their folding blades, and the
gimbal head) are a second body, m4td_moving.urdf, with no collision, posed on the aircraft every control step.
Motors, battery and gimbal carry temperatures for the thermal camera; every other surface stays on the engine's
passive model.
"""

from __future__ import annotations

import math
import os
from typing import Any, Optional, Tuple

import numpy as np
import pybullet as p
import swarm_worlds
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel
from scipy.spatial.transform import Rotation

from swarm.utils import gym_assets

from . import park
from .episode import SolarEpisode

URDF = "m4td.urdf"
MOVING_URDF = "m4td_moving.urdf"
MESH_DIR = "m4td"

MAX_RPM = 6300.0                        # DJI: maximum propeller speed
IDLE_RPM = 900.0                        # the slowest a spinning motor is commanded in flight
MAX_TILT_RAD = math.radians(25.0)       # DJI: Normal mode, the only mode with the dock
SPIN = (1, -1, 1, -1)                   # rotor turn sense seen from above, +1 counter-clockwise (the gym's torque signs)
MOTOR_XY = ((0.17065, -0.1915), (-0.17065, -0.1715), (-0.17065, 0.1715), (0.17065, 0.1915))  # DJI's CAD
# Controller gains, tuned on the patrol's physics: speed loop in newtons per m/s (and per m of accumulated speed
# error), attitude loop in PWM units.
VEL_GAIN = np.array([6.0, 6.0, 7.4])
VEL_INTEGRAL = np.array([1.0, 1.0, 2.0])  # a small, fading integral for what the drag feed-forward misses (wind)
INTEGRAL_BAND_MPS = 0.3                 # it only learns once the speed is this close to the target
INTEGRAL_LIMIT = 1.0
INTEGRAL_LEAK_S = 4.0
ATT_P = np.array([45000.0, 45000.0, 20000.0])
ATT_I = np.array([0.0, 0.0, 0.0])
ATT_D = np.array([9000.0, 9000.0, 24000.0])

# Drone frame (x nose, y left, z up, metres, gimbal level), from DJI's CAD.
GIMBAL_PIVOT = np.array([0.1295, 0.0, -0.0340])      # tilt axis, along y
WIDE_CAMERA = np.array([0.1579, 0.0162, -0.0373])    # front of the wide camera's glass
LASER_WINDOW = np.array([0.1454, -0.0095, -0.0560])  # centre of the laser rangefinder window
DOCKED_FOLD_RAD = math.radians(45.0)                 # blades set 90 degrees apart before the dock closes (DJI manual)

# Surface temperatures (deg C) for the thermal camera: (rise over air when warm, warm-up s, cool-down s, emissivity).
# Estimates: 20 to 35 K rise on drone motors measured in flight (arXiv 1906.04152); DJI docks keep the battery
# between 10 and 35 C and it warms through a flight; the gimbal's electronics run a few kelvin over the air.
HEAT = {"motor": (25.0, 60.0, 240.0, 0.85), "battery": (10.0, 400.0, 900.0, 0.92), "gimbal": (8.0, 120.0, 300.0, 0.92)}
BATTERY_BASE_C = 25.0                  # the dock conditions the battery before take-off
THERMAL = hasattr(p, "ER_SWARM_THERMAL")


class M4TDControl(DSLPIDControl):
    """DSL's PID controller on the M4TD: rotor speeds up to 6,300 rpm, gains for 1.85 kg, tilt capped at 25 deg."""

    def __init__(self, env: Any):
        """Take the parsed constants of the loaded body and set the M4TD's speed range and gains."""
        super().__init__(drone_model=DroneModel.CF2X, g=float(env.G))
        self.GRAVITY = float(env.GRAVITY)
        self.KF = float(env.KF)
        self.KM = float(env.KM)
        self.PWM2RPM_SCALE = MAX_RPM / 65535.0
        self.PWM2RPM_CONST = 0.0
        self.MIN_PWM = IDLE_RPM / self.PWM2RPM_SCALE
        self.MAX_PWM = 65535.0
        self.P_COEFF_FOR = np.zeros(3)
        self.I_COEFF_FOR = np.zeros(3)
        self.D_COEFF_FOR = VEL_GAIN.copy()
        self.P_COEFF_TOR = ATT_P.copy()
        self.I_COEFF_TOR = ATT_I.copy()
        self.D_COEFF_TOR = ATT_D.copy()
        # Rotor drag at hover speed, the environment's own law (force = coefficient x total rotor speed x airspeed):
        # the thrust tilted against it for the speed asked, so arriving and stopping leave nothing to unwind.
        rotor_rad_s = 4.0 * float(env.HOVER_RPM) * 2.0 * math.pi / 60.0
        self.drag_feedforward = np.asarray(env.DRAG_COEFF, dtype=float) * rotor_rad_s
        self.reset()

    def reset(self):
        """Clear the loops' memory, the speed integral with it."""
        super().reset()
        self.integral_vel_e = np.zeros(3)

    def _dslPIDPositionControl(self, control_timestep, cur_pos, cur_quat, cur_vel, target_pos, target_rpy,
                               target_vel, cur_rotation):
        """DSL's position loop holding a commanded speed: drag cancelled ahead of time, a small fading integral for
        wind, and the thrust vector held inside the tilt limit."""
        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel
        self.integral_pos_e = np.clip(self.integral_pos_e + pos_e * control_timestep, -2.0, 2.0)
        self.integral_pos_e[2] = np.clip(self.integral_pos_e[2], -0.15, 0.15)
        near = np.abs(vel_e) < INTEGRAL_BAND_MPS
        leak = math.exp(-control_timestep / INTEGRAL_LEAK_S)
        self.integral_vel_e = np.clip(self.integral_vel_e * leak + np.where(near, vel_e, 0.0) * control_timestep,
                                      -INTEGRAL_LIMIT, INTEGRAL_LIMIT)
        target_thrust = (self.P_COEFF_FOR * pos_e + self.I_COEFF_FOR * self.integral_pos_e
                         + self.D_COEFF_FOR * vel_e + VEL_INTEGRAL * self.integral_vel_e
                         + self.drag_feedforward * target_vel + np.array([0.0, 0.0, self.GRAVITY]))
        target_thrust[2] = max(float(target_thrust[2]), 0.1 * self.GRAVITY)
        side = math.hypot(target_thrust[0], target_thrust[1])
        cap = target_thrust[2] * math.tan(MAX_TILT_RAD)
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
        target_euler = Rotation.from_matrix(target_rotation).as_euler("XYZ", degrees=False)
        return thrust, target_euler, pos_e


def urdf(env: Any) -> Optional[str]:
    """Stage the M4TD's description and meshes where the gym loads bodies from, and return its name."""
    src_dir = swarm_worlds.robots_dir()
    dst_dir = gym_assets.stage_dir()
    mesh_dst = os.path.join(dst_dir, MESH_DIR)
    os.makedirs(mesh_dst, exist_ok=True)
    for name in sorted(os.listdir(os.path.join(src_dir, MESH_DIR))):
        gym_assets.copy_verified(os.path.join(src_dir, MESH_DIR, name), os.path.join(mesh_dst, name))
    dst = os.path.join(dst_dir, URDF)
    gym_assets.copy_verified(os.path.join(src_dir, URDF), dst)
    return gym_assets.register(URDF, dst)


def reset(env: Any, ep: SolarEpisode) -> None:
    """Fit the M4TD's controller, put the moving parts on the aircraft docked, and set cold temperatures."""
    env.ctrl = [M4TDControl(env) for _ in range(env.NUM_DRONES)]
    cli = env.CLIENT
    uid = int(p.loadURDF(os.path.join(swarm_worlds.robots_dir(), MOVING_URDF), [0.0, 0.0, -100.0],
                         flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=cli))
    for link in range(-1, p.getNumJoints(uid, physicsClientId=cli)):
        p.setCollisionFilterGroupMask(uid, link, 0, 0, physicsClientId=cli)
    joints = {p.getJointInfo(uid, j, physicsClientId=cli)[1].decode(): j for j in range(p.getNumJoints(uid, physicsClientId=cli))}
    ep.airframe = {"moving": uid, "joints": joints, "angles": [0.0] * 4, "warm": {k: 0.0 for k in HEAT},
                   "celsius": {}, "shown": {}, "shapes": _heated_shapes(env, uid)}
    update(env, ep)


def update(env: Any, ep: SolarEpisode) -> None:
    """After one control step: turn the rotors at their speed, fold or open the blades, tilt the camera head, pose
    the moving parts on the aircraft, and warm or cool the motors, battery and gimbal."""
    state = ep.airframe
    if state is None:
        return
    cli, uid, joints = env.CLIENT, state["moving"], state["joints"]
    dt = float(env.CTRL_TIMESTEP)
    rpm = np.asarray(getattr(env, "last_clipped_action", np.zeros((1, 4))), dtype=float).reshape(-1, 4)[0]
    spinning = bool(np.any(rpm > 1.0))
    for i in range(4):
        if spinning:
            state["angles"][i] = (state["angles"][i] + SPIN[i] * rpm[i] * 2.0 * math.pi / 60.0 * dt) % (2.0 * math.pi)
        else:
            state["angles"][i] = _docked_rotor_angle(i)
        p.resetJointState(uid, joints["rotor%d_spin" % i], state["angles"][i], physicsClientId=cli)
        fold = 0.0 if spinning else DOCKED_FOLD_RAD
        p.resetJointState(uid, joints["rotor%d_fold0" % i], -fold, physicsClientId=cli)
        p.resetJointState(uid, joints["rotor%d_fold1" % i], fold, physicsClientId=cli)
    tilt = float(ep.camera.get("tilt_deg", 0.0)) if isinstance(ep.camera, dict) else 0.0
    p.resetJointState(uid, joints["gimbal_tilt"], math.radians(tilt), physicsClientId=cli)
    pos, orn = p.getBasePositionAndOrientation(int(env.DRONE_IDS[0]), physicsClientId=cli)
    p.resetBasePositionAndOrientation(uid, pos, orn, physicsClientId=cli)
    _warm(env, ep, spinning, dt)


def camera_pose(env: Any, tilt_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World position, looking direction and up direction of the wide camera at a gimbal tilt (+90 straight up)."""
    return _gimbal_point(env, WIDE_CAMERA, tilt_deg)


def laser_pose(env: Any, tilt_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """World position and direction of the laser rangefinder at a gimbal tilt; its ray should skip the aircraft."""
    origin, forward, _ = _gimbal_point(env, LASER_WINDOW, tilt_deg)
    return origin, forward


def _gimbal_point(env: Any, point: np.ndarray, tilt_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A point on the camera head swung about the tilt axis, with the head's forward and up, in world axes."""
    t = math.radians(tilt_deg)
    c, s = math.cos(t), math.sin(t)
    d = point - GIMBAL_PIVOT
    local = GIMBAL_PIVOT + np.array([d[0] * c - d[2] * s, d[1], d[0] * s + d[2] * c])
    rot = np.array(p.getMatrixFromQuaternion(env.quat[0])).reshape(3, 3)
    return (np.asarray(env.pos[0], dtype=float) + rot @ local, rot @ np.array([c, 0.0, s]),
            rot @ np.array([-s, 0.0, c]))


def _docked_rotor_angle(i: int) -> float:
    """Rotor angle that opens the folded blades' right angle out along the arm, clear of the closing lids."""
    return math.atan2(MOTOR_XY[i][1], MOTOR_XY[i][0]) + SPIN[i] * math.pi / 2


def _heated_shapes(env: Any, moving: int) -> dict:
    """The visual shapes that carry a temperature, by heat source: (body id, link, shape index) triples."""
    out = {k: [] for k in HEAT}
    for body in (int(env.DRONE_IDS[0]), moving):
        counts: dict = {}
        for shape in p.getVisualShapeData(body, physicsClientId=env.CLIENT):
            link = int(shape[1])
            index = counts.get(link, 0)
            counts[link] = index + 1
            name = os.path.basename(shape[4].decode() if isinstance(shape[4], bytes) else str(shape[4]))
            if name.startswith("body_motor"):
                out["motor"].append((body, link, index))
            elif name.startswith("battery_"):
                out["battery"].append((body, link, index))
            elif name.startswith(("head_", "body_gimbal")):
                out["gimbal"].append((body, link, index))
    return out


def _warm(env: Any, ep: SolarEpisode, spinning: bool, dt: float) -> None:
    """Move each heat source towards hot while the motors run and back towards the air while they rest."""
    state = ep.airframe
    for name, (rise, heat_s, cool_s, emissivity) in HEAT.items():
        tau = heat_s if spinning else cool_s
        target = 1.0 if spinning else 0.0
        state["warm"][name] += (target - state["warm"][name]) * (1.0 - math.exp(-dt / tau))
        base = BATTERY_BASE_C if name == "battery" else park.air_c(ep)
        celsius = round(base + rise * state["warm"][name], 1)
        state["celsius"][name] = celsius
        if not THERMAL or state["shown"].get(name) == celsius:
            continue
        state["shown"][name] = celsius
        for body, link, index in state["shapes"][name]:
            p.changeVisualShape(body, link, shapeIndex=index, temperature=celsius, emissivity=emissivity,
                                physicsClientId=env.CLIENT)
