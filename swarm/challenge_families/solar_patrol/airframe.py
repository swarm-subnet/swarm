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
gimbal head) are a second body, m4td_moving.urdf, with no collision, posed on the aircraft before each picture.
Motors, battery and gimbal carry temperatures for the thermal camera; every other surface stays on the engine's
passive model.
"""

from __future__ import annotations

import ctypes
import math
import os
import types
from functools import lru_cache
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
from .fixed_order import rotate

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
INTEGRAL_BAND_MPS = 1.0                 # it only learns once the speed is this close to the target
INTEGRAL_LIMIT = 4.0                    # room for the steady push of a 12 m/s wind on top of the rotor drag
INTEGRAL_LEAK_S = 30.0
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
# The C library's hypot, the one scipy's rotations call: Python's math.hypot is its own algorithm and can round apart.
_HYPOT = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)(("hypot", ctypes.CDLL(None)))


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
        dt = control_timestep
        pos_e = target_pos - cur_pos
        leak = math.exp(-dt / INTEGRAL_LEAK_S)
        gains = zip(pos_e.tolist(), np.asarray(target_vel, dtype=float).tolist(), np.asarray(cur_vel).tolist(),
                    self.integral_pos_e.tolist(), self.integral_vel_e.tolist(), self.P_COEFF_FOR.tolist(),
                    self.I_COEFF_FOR.tolist(), self.D_COEFF_FOR.tolist(), VEL_INTEGRAL.tolist(),
                    self.drag_feedforward.tolist(), (0.0, 0.0, self.GRAVITY), (2.0, 2.0, 0.15))
        # Plain float operations in numpy's order, so every result keeps the bits the array version gave.
        integral_pos, integral_vel, target = [], [], []
        for pe, tv, cv, ip, iv, kp, ki, kd, kv, ff, g, ip_cap in gains:
            ve = tv - cv
            ip = min(max(min(max(ip + pe * dt, -2.0), 2.0), -ip_cap), ip_cap)
            iv = min(max(iv * leak + (ve if abs(ve) < INTEGRAL_BAND_MPS else 0.0) * dt, -INTEGRAL_LIMIT), INTEGRAL_LIMIT)
            integral_pos.append(ip)
            integral_vel.append(iv)
            target.append(kp * pe + ki * ip + kd * ve + kv * iv + ff * tv + g)
        self.integral_pos_e = np.array(integral_pos)
        self.integral_vel_e = np.array(integral_vel)
        t0, t1, t2 = target[0], target[1], max(target[2], 0.1 * self.GRAVITY)
        side = math.hypot(t0, t1)
        cap = t2 * math.tan(MAX_TILT_RAD)
        if side > cap:
            t0, t1 = t0 * (cap / side), t1 * (cap / side)
        # Dot products summed in a fixed order, not by BLAS, whose rounding changes with the CPU's kernel.
        up = cur_rotation[:, 2].tolist()
        scalar_thrust = max(0.0, t0 * up[0] + t1 * up[1] + t2 * up[2])
        thrust = (math.sqrt(scalar_thrust / (4 * self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        norm = math.sqrt(t0 * t0 + t1 * t1 + t2 * t2)
        z = (t0 / norm, t1 / norm, t2 / norm)
        c, s = math.cos(target_rpy[2]), math.sin(target_rpy[2])
        zx = (z[1] * 0.0 - z[2] * s, z[2] * c - z[0] * 0.0, z[0] * s - z[1] * c)
        norm = math.sqrt(zx[0] * zx[0] + zx[1] * zx[1] + zx[2] * zx[2])
        y = (zx[0] / norm, zx[1] / norm, zx[2] / norm)
        x = (y[1] * z[2] - y[2] * z[1], y[2] * z[0] - y[0] * z[2], y[0] * z[1] - y[1] * z[0])
        euler = _euler_xyz(x, y, z)
        if euler is None:
            euler = Rotation.from_matrix(np.vstack([x, y, z]).transpose()).as_euler("XYZ", degrees=False)
        return thrust, np.asarray(euler, dtype=float), pos_e

    def _dslPIDAttitudeControl(self, control_timestep, thrust, cur_quat, target_euler, target_rpy_rates,
                               cur_rotation):
        """DSL's attitude loop as the gym runs it, in plain float operations."""
        dt = control_timestep
        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        t, r = _matrix_xyz(*np.asarray(target_euler, dtype=float).tolist()), cur_rotation.tolist()
        # The three entries of target^T cur - cur^T target the loop reads, summed in a fixed order rather than by BLAS.
        rot_e = tuple(t[0][a] * r[0][b] + t[1][a] * r[1][b] + t[2][a] * r[2][b]
                      - (r[0][a] * t[0][b] + r[1][a] * t[1][b] + r[2][a] * t[2][b]) for a, b in ((2, 1), (0, 2), (1, 0)))
        terms = zip(rot_e, cur_rpy, self.last_rpy.tolist(), np.asarray(target_rpy_rates, dtype=float).tolist(),
                    self.integral_rpy_e.tolist(), self.P_COEFF_TOR.tolist(), self.D_COEFF_TOR.tolist(),
                    self.I_COEFF_TOR.tolist(), (1.0, 1.0, 1500.0))
        integral, torques = [], []
        for re, rpy, last, rate, ir, kp, kd, ki, ir_cap in terms:
            ir = min(max(min(max(ir - re * dt, -1500.0), 1500.0), -ir_cap), ir_cap)
            integral.append(ir)
            torques.append(min(max(-(kp * re) + kd * (rate - (rpy - last) / dt) + ki * ir, -3200.0), 3200.0))
        self.last_rpy = np.array(cur_rpy)
        self.integral_rpy_e = np.array(integral)
        pwm = [min(max(thrust + (m[0] * torques[0] + m[1] * torques[1] + m[2] * torques[2]), self.MIN_PWM), self.MAX_PWM)
               for m in self.MIXER_MATRIX.tolist()]
        return self.PWM2RPM_SCALE * np.array(pwm) + self.PWM2RPM_CONST


def _euler_xyz(x: tuple, y: tuple, z: tuple) -> Optional[list]:
    """Intrinsic XYZ angles of the rotation with columns x, y, z, as scipy's from_matrix and as_euler compute them,
    operation for operation; None where scipy would first orthogonalise the matrix or warn of gimbal lock."""
    m = tuple(zip(x, y, z))
    for r in range(3):
        for s in range(r, 3):
            # Well inside scipy's own check (atol 1e-12 off the diagonal, rtol 1e-5 on it), whatever its BLAS rounds.
            if not abs(m[r][0] * m[s][0] + m[r][1] * m[s][1] + m[r][2] * m[s][2] - (r == s)) <= (1e-6 if r == s else 1e-13):
                return None
    det = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
           + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
    if not det > 0.5:
        return None
    trace = m[0][0] + m[1][1] + m[2][2]
    decision = (m[0][0], m[1][1], m[2][2], trace)
    choice = 0
    for i in (1, 2, 3):
        if decision[i] > decision[choice]:
            choice = i
    if choice == 3:
        q = [m[2][1] - m[1][2], m[0][2] - m[2][0], m[1][0] - m[0][1], 1.0 + trace]
    else:
        i, j, k = choice, (choice + 1) % 3, (choice + 2) % 3
        q = [0.0] * 4
        q[i] = 1.0 - trace + 2.0 * m[i][i]
        q[j] = m[j][i] + m[i][j]
        q[k] = m[k][i] + m[i][k]
        q[3] = m[k][j] - m[j][k]
    norm = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    q = [v / norm for v in q]
    a, b, c, d = q[3] - q[1], q[2] + q[0] * -1.0, q[1] + q[3], q[0] * -1.0 - q[2]
    middle = 2.0 * math.atan2(_HYPOT(c, d), _HYPOT(a, b))
    if abs(middle) <= 1e-7 or abs(middle - math.pi) <= 1e-7:
        return None
    half_sum, half_diff = math.atan2(b, a), math.atan2(d, c)
    angles = [(half_sum + half_diff) * -1.0, middle - math.pi / 2, half_sum - half_diff]
    return [v + 2.0 * math.pi if v < -math.pi else v - 2.0 * math.pi if v > math.pi else v for v in angles]


def _matrix_xyz(roll: float, pitch: float, yaw: float) -> tuple:
    """Rotation matrix of intrinsic XYZ angles, as scipy's from_euler, from_quat and as_matrix compute it."""
    q = _compose((math.sin(roll / 2), 0.0, 0.0, math.cos(roll / 2)), (0.0, math.sin(pitch / 2), 0.0, math.cos(pitch / 2)))
    q = _compose(q, (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)))
    norm = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    x, y, z, w = (v / norm for v in q)
    x2, y2, z2, w2 = x * x, y * y, z * z, w * w
    xy, zw, xz, yw, yz, xw = x * y, z * w, x * z, y * w, y * z, x * w
    return ((x2 - y2 - z2 + w2, 2 * (xy - zw), 2 * (xz + yw)),
            (2 * (xy + zw), -x2 + y2 - z2 + w2, 2 * (yz - xw)),
            (2 * (xz - yw), 2 * (yz + xw), -x2 - y2 + z2 + w2))


def _compose(p: tuple, q: tuple) -> tuple:
    """Quaternion product p * q (x, y, z, w), in scipy's order of operations."""
    c0, c1, c2 = p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0]
    return (p[3] * q[0] + q[3] * p[0] + c0, p[3] * q[1] + q[3] * p[1] + c1, p[3] * q[2] + q[3] * p[2] + c2,
            p[3] * q[3] - p[0] * q[0] - p[1] * q[1] - p[2] * q[2])


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
    env._drag = types.MethodType(_still_air_drag, env)
    cli = env.CLIENT
    uid = int(p.loadURDF(os.path.join(swarm_worlds.robots_dir(), MOVING_URDF), [0.0, 0.0, -100.0],
                         flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=cli))
    for link in range(-1, p.getNumJoints(uid, physicsClientId=cli)):
        p.setCollisionFilterGroupMask(uid, link, 0, 0, physicsClientId=cli)
    joints = {p.getJointInfo(uid, j, physicsClientId=cli)[1].decode(): j for j in range(p.getNumJoints(uid, physicsClientId=cli))}
    ep.airframe = {"moving": uid, "joints": joints, "angles": [0.0] * 4, "warm": {k: 0.0 for k in HEAT},
                   "celsius": {}, "shown": {}, "shapes": _heated_shapes(env, uid)}
    update(env, ep)
    pose(env, ep)


def update(env: Any, ep: SolarEpisode) -> None:
    """After one control step: turn the rotors at their speed, fold or open the blades, tilt the camera head and keep
    that pose on the aircraft for the next picture, and warm or cool the motors, battery and gimbal."""
    state = ep.airframe
    if state is None:
        return
    cli = env.CLIENT
    dt = float(env.CTRL_TIMESTEP)
    rpm = rotor_speeds(env)
    spinning = any(speed > 1.0 for speed in rpm)
    for i in range(4):
        if spinning:
            state["angles"][i] = (state["angles"][i] + SPIN[i] * rpm[i] * 2.0 * math.pi / 60.0 * dt) % (2.0 * math.pi)
        else:
            state["angles"][i] = _docked_rotor_angle(i)
    tilt = float(ep.camera.get("tilt_deg", 0.0)) if isinstance(ep.camera, dict) else 0.0
    state["pose"] = (spinning, math.radians(tilt), p.getBasePositionAndOrientation(int(env.DRONE_IDS[0]), physicsClientId=cli))
    state["posed"] = False
    _warm(env, ep, spinning, dt)


def rotor_speeds(env: Any) -> list:
    """The four rotor speeds of the last control step as plain floats, zero before the first."""
    action = getattr(env, "last_clipped_action", None)
    if action is None:
        return [0.0, 0.0, 0.0, 0.0]
    return np.asarray(action, dtype=float).reshape(-1, 4)[0].tolist()


def pose(env: Any, ep: Optional[SolarEpisode]) -> None:
    """Put the moving parts where the last control step left them, once before the pictures that follow it; nothing
    but a picture sees them, as they have no collision."""
    state = ep.airframe if ep is not None else None
    if state is None or state["posed"]:
        return
    cli, uid, joints = env.CLIENT, state["moving"], state["joints"]
    spinning, tilt_rad, (pos, orn) = state["pose"]
    fold = 0.0 if spinning else DOCKED_FOLD_RAD
    for i in range(4):
        p.resetJointState(uid, joints["rotor%d_spin" % i], state["angles"][i], physicsClientId=cli)
        p.resetJointState(uid, joints["rotor%d_fold0" % i], -fold, physicsClientId=cli)
        p.resetJointState(uid, joints["rotor%d_fold1" % i], fold, physicsClientId=cli)
    p.resetJointState(uid, joints["gimbal_tilt"], tilt_rad, physicsClientId=cli)
    p.resetBasePositionAndOrientation(uid, pos, orn, physicsClientId=cli)
    state["posed"] = True


def _still_air_drag(env: Any, rpm: np.ndarray, nth_drone: int) -> None:
    """The gym's rotor drag for a seed without wind: its formula, its link and its force, with the turn into the
    body frame summed in a fixed order instead of the BLAS product whose rounding follows the CPU."""
    base_rot = np.array(p.getMatrixFromQuaternion(env.quat[nth_drone, :])).reshape(3, 3)
    drag_factors = -1 * env.DRAG_COEFF * np.sum(np.array(2 * np.pi * rpm / 60))
    drag = rotate(base_rot.T, drag_factors * np.array(env.vel[nth_drone, :]))
    # Link 4 is the URDF's center_of_mass_link, where the gym's drag has always pushed.
    p.applyExternalForce(env.DRONE_IDS[nth_drone], 4, forceObj=drag, posObj=[0, 0, 0], flags=p.LINK_FRAME,
                         physicsClientId=env.CLIENT)


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
    rot = p.getMatrixFromQuaternion(env.quat[0])
    return (np.asarray(env.pos[0], dtype=float) + rotate(rot, local), rotate(rot, (c, 0.0, s)),
            rotate(rot, (-s, 0.0, c)))


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


@lru_cache(maxsize=16)
def _approach(dt: float, tau: float) -> float:
    """The share of the gap to its target a heat source closes in one step of dt seconds with time constant tau."""
    return 1.0 - math.exp(-dt / tau)


def _warm(env: Any, ep: SolarEpisode, spinning: bool, dt: float) -> None:
    """Move each heat source towards hot while the motors run and back towards the air while they rest."""
    state = ep.airframe
    for name, (rise, heat_s, cool_s, emissivity) in HEAT.items():
        tau = heat_s if spinning else cool_s
        target = 1.0 if spinning else 0.0
        state["warm"][name] += (target - state["warm"][name]) * _approach(dt, tau)
        base = BATTERY_BASE_C if name == "battery" else park.air_c(ep)
        celsius = round(base + rise * state["warm"][name], 1)
        state["celsius"][name] = celsius
        if not THERMAL or state["shown"].get(name) == celsius:
            continue
        state["shown"][name] = celsius
        for body, link, index in state["shapes"][name]:
            p.changeVisualShape(body, link, shapeIndex=index, temperature=celsius, emissivity=emissivity,
                                physicsClientId=env.CLIENT)
