"""Frozen pre-optimization controller for exact regression comparisons."""

import math
from typing import Any

import numpy as np
import pybullet as p
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel
from scipy.spatial.transform import Rotation

from swarm.challenge_families.solar_patrol.airframe import (
    ATT_D,
    ATT_I,
    ATT_P,
    IDLE_RPM,
    INTEGRAL_BAND_MPS,
    INTEGRAL_LEAK_S,
    INTEGRAL_LIMIT,
    MAX_RPM,
    MAX_TILT_RAD,
    VEL_GAIN,
    VEL_INTEGRAL,
    _euler_xyz,
    _matrix_xyz,
)


class OriginalControl(DSLPIDControl):
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
