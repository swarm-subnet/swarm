"""Office interceptor challenge family (indoor Tello, body-frame RC actions).

The miner flies a Tello-class drone inside the fixed office map and must
intercept a target drone. Actions are the four Tello RC sticks
[lr, fb, ud, yaw]; there is no GPS-style world-frame control.

This module carries the family skeleton: map spawn, the Tello asset plumbing,
task generation over the office bounds, and a placeholder score (distance to
the target point). The detector emulator, the telemetry observation contract,
and the live target behaviour land in their own follow-up cards.
"""

from __future__ import annotations

import math
import os
import random
from importlib.resources import files as _pkg_files
from typing import Any

import numpy as np
import pybullet as p

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel

from swarm.constants import (
    OFFICE_CHALLENGE_TYPE,
    OFFICE_MAX_START_DISTANCE_M,
    OFFICE_MAX_TILT_DEG,
    OFFICE_MIN_START_DISTANCE_M,
    OFFICE_TELEM_ACCEL_BIAS,
    OFFICE_TELEM_ACCEL_NOISE,
    OFFICE_TELEM_ATTITUDE_NOISE_DEG,
    OFFICE_TELEM_BARO_NOISE_M,
    OFFICE_TELEM_BARO_WALK_M,
    OFFICE_TELEM_DELAY_STEPS,
    OFFICE_TELEM_DROP_PROB,
    OFFICE_TELEM_HEIGHT_NOISE_M,
    OFFICE_TELEM_PERIOD_STEPS,
    OFFICE_TELEM_SEED_OFFSET,
    OFFICE_TELEM_STALE_SEC,
    OFFICE_TELEM_TOF_MAX_M,
    OFFICE_TELEM_TOF_NOISE_M,
    OFFICE_TELEM_VELOCITY_NOISE,
    OFFICE_VPS_DRIFT_FORCE_N,
    OFFICE_W_SUCCESS,
    OFFICE_W_TIME,
)
from swarm.core.maps.office import OFFICE_X_RANGE, OFFICE_Y_RANGE, build_office_map
from swarm.domain_model import CHALLENGE_TYPE_TO_ENVIRONMENT_TYPE
from swarm.protocol import FailureReason, SCHEMA_VERSION
from swarm.validator.reward import (
    PARTICIPATION_REASONS,
    PARTICIPATION_REWARD,
    _calculate_office_target_time,
    _clamp,
    calculate_time_term,
)

from .base import ChallengeFamilyRuntime, ChallengeFamilyRuntimeProfile

TELLO_URDF = "tello.urdf"
TELLO_MESH_DIR = "tello"

# Interior margin that keeps generated points clear of walls.
_WALL_MARGIN_M = 0.9

_TEMPLATE_SLOT = {
    "challenge_type": OFFICE_CHALLENGE_TYPE,
    "distance_range": (OFFICE_MIN_START_DISTANCE_M, OFFICE_MAX_START_DISTANCE_M),
}

_tello_assets_ready = False

# Telemetry packet layout: [pitch, roll, sin_yaw, cos_yaw, vf, vr, vu, af, ar, au,
# tof, height, baro, age, valid]. Body axes follow the sticks: forward, right, up.
# Snapshots hold the raw yaw (index 2) that delivery expands into sin/cos.
_TELEM_DIM = 15
_ATT_Q = math.radians(1.0)  # the SDK reports attitude in whole degrees
_VEL_Q = 0.1                # the SDK reports velocity in dm/s
_ACC_Q = 0.01               # accelerometer granularity (~0.001 g)
_ALT_Q = 0.01               # the SDK reports heights in cm
_ATT_STD = math.radians(OFFICE_TELEM_ATTITUDE_NOISE_DEG)
_SNAP_NOISE_STD = np.array(
    [_ATT_STD] * 3
    + [OFFICE_TELEM_VELOCITY_NOISE] * 3
    + [OFFICE_TELEM_ACCEL_NOISE] * 3
    + [OFFICE_TELEM_TOF_NOISE_M, OFFICE_TELEM_HEIGHT_NOISE_M, OFFICE_TELEM_BARO_NOISE_M]
)
_SNAP_QUANT = np.array([_ATT_Q] * 3 + [_VEL_Q] * 3 + [_ACC_Q] * 3 + [_ALT_Q] * 3)


def tello_urdf_path() -> str:
    """Absolute path to the Tello URDF in the swarm package assets."""
    return str(_pkg_files("swarm").joinpath("assets", TELLO_URDF))


def ensure_tello_assets_in_gym_assets() -> str:
    """Make the Tello URDF + its mesh folder available in gym_pybullet_drones/assets
    so BaseAviary's hardcoded loader finds them via self.URDF. Content-verified and
    atomic so every validator parses byte-identical physical constants; the check
    runs once per process. Returns the URDF basename."""
    global _tello_assets_ready
    if _tello_assets_ready:
        return TELLO_URDF
    dst_dir = str(_pkg_files("gym_pybullet_drones").joinpath("assets"))
    src_mesh_dir = str(_pkg_files("swarm").joinpath("assets", TELLO_MESH_DIR))
    dst_mesh_dir = os.path.join(dst_dir, TELLO_MESH_DIR)
    os.makedirs(dst_mesh_dir, exist_ok=True)
    for name in sorted(os.listdir(src_mesh_dir)):
        _copy_verified(os.path.join(src_mesh_dir, name), os.path.join(dst_mesh_dir, name))
    _copy_verified(tello_urdf_path(), os.path.join(dst_dir, TELLO_URDF))
    _tello_assets_ready = True
    return TELLO_URDF


def _copy_verified(src: str, dst: str) -> None:
    with open(src, "rb") as f:
        src_bytes = f.read()
    if os.path.exists(dst):
        with open(dst, "rb") as f:
            if f.read() == src_bytes:
                return
    tmp = f"{dst}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(src_bytes)
    os.replace(tmp, dst)  # atomic, race-safe across workers
    with open(dst, "rb") as f:
        if f.read() != src_bytes:
            raise RuntimeError(f"{os.path.basename(dst)} content mismatch in gym assets")


def make_office_control(env: Any) -> DSLPIDControl:
    """A DSL PID controller carrying the Tello's parsed constants. The stock gains
    are tuned for the same weight class; refine against real flight logs later."""
    ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
    ctrl.GRAVITY = float(env.GRAVITY)
    ctrl.KF = float(env.KF)
    ctrl.KM = float(env.KM)
    ctrl.reset()
    return ctrl


def office_point(rng: random.Random, z_range: tuple) -> tuple:
    """A deterministic point inside the office flight volume, clear of the walls."""
    x = rng.uniform(OFFICE_X_RANGE[0] + _WALL_MARGIN_M, OFFICE_X_RANGE[1] - _WALL_MARGIN_M)
    y = rng.uniform(OFFICE_Y_RANGE[0] + _WALL_MARGIN_M, OFFICE_Y_RANGE[1] - _WALL_MARGIN_M)
    return (x, y, rng.uniform(*z_range))


class OfficeInterceptorChallengeFamily(ChallengeFamilyRuntime):
    """Indoor Tello interceptor inside the fixed office map."""

    family_id = "cf_office_interceptor"
    runtime_supported = True

    # ------------------------------------------------------------------ #
    # runtime profile
    # ------------------------------------------------------------------ #
    def runtime_profile(self, task) -> ChallengeFamilyRuntimeProfile:
        return ChallengeFamilyRuntimeProfile(
            family_id=self.family_id,
            profile_name="office_interceptor",
            resource_class="navigation",
            image_key="base",
            env_bootstrap={"sar_mode": False},
            docker_env={
                "SWARM_CHALLENGE_FAMILY_ID": self.family_id,
                "SWARM_RUNTIME_PROFILE": "office_interceptor",
                "SWARM_RUNTIME_RESOURCE_CLASS": "navigation",
                "SWARM_RUNTIME_IMAGE_KEY": "base",
                "SWARM_RUNTIME_ENV_BOOTSTRAP": "sar_mode=false",
            },
        )

    def env_kwargs_for_task(self, task) -> dict:
        return {"sar_mode": False}

    # ------------------------------------------------------------------ #
    # env lifecycle
    # ------------------------------------------------------------------ #
    def initialise_env_state(self, env, *, requested_mode: bool = False) -> None:
        env.sar_mode = False
        env.MAX_TILT_RAD = math.radians(OFFICE_MAX_TILT_DEG)

    def reset_env_state(self, env) -> None:
        n = env.NUM_DRONES
        seed = int(getattr(env.task, "map_seed", 0))
        env._office_telemetry = np.zeros((n, _TELEM_DIM), dtype=np.float32)
        # Age starts beyond the stale threshold: no packet has arrived yet.
        env._office_telemetry[:, 13] = 2.0 * OFFICE_TELEM_STALE_SEC
        env._office_telem_rng = np.random.default_rng(seed ^ OFFICE_TELEM_SEED_OFFSET)
        env._office_telem_step = 0
        env._office_pending = [None] * n
        env._office_prev_vel = np.array(env.vel, dtype=np.float64)
        env._office_accel_bias = env._office_telem_rng.normal(
            0.0, OFFICE_TELEM_ACCEL_BIAS, (n, 3)
        )
        env._office_baro_walk = np.zeros(n, dtype=np.float64)
        env._office_takeoff_z = np.array(env.pos[:, 2], dtype=np.float64)
        angle = env._office_telem_rng.uniform(0.0, 2.0 * math.pi)
        env._office_vps_force = [
            OFFICE_VPS_DRIFT_FORCE_N * math.cos(angle),
            OFFICE_VPS_DRIFT_FORCE_N * math.sin(angle),
            0.0,
        ]

    def spawn_task_world(self, env) -> None:
        build_office_map(seed=env.task.map_seed, cli=env.getPyBulletClient())

    def build_info(self, env) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "task_version": str(getattr(env.task, "version", "")),
        }

    # ------------------------------------------------------------------ #
    # telemetry link: noisy delayed packets at ~10 Hz, like the real SDK
    # ------------------------------------------------------------------ #
    def post_step_update(self, env) -> None:
        rng = getattr(env, "_office_telem_rng", None)
        if rng is None:
            return
        dt = float(env.CTRL_TIMESTEP)
        env._office_telem_step += 1
        step = env._office_telem_step
        # Snapshot only the step whose state the next packet will carry.
        if (step + OFFICE_TELEM_DELAY_STEPS) % OFFICE_TELEM_PERIOD_STEPS == 0:
            for d in range(env.NUM_DRONES):
                env._office_pending[d] = self._snapshot(env, d, dt)
        env._office_prev_vel[:] = env.vel
        telem = env._office_telemetry
        telem[:, 13] = np.minimum(telem[:, 13] + dt, 2.0 * OFFICE_TELEM_STALE_SEC)
        if step % OFFICE_TELEM_PERIOD_STEPS == 0:
            for d in range(env.NUM_DRONES):
                self._deliver_packet(env, d, rng, dt)
        telem[:, 14] = telem[:, 13] <= OFFICE_TELEM_STALE_SEC

    def _snapshot(self, env, d: int, dt: float) -> np.ndarray:
        # Lazy import: moving_drone imports this package, so the top level would cycle.
        from swarm.core.moving_drone import world_to_body

        vel = np.array(env.vel[d], dtype=np.float64)
        accel = (vel - env._office_prev_vel[d]) / dt + env._office_accel_bias[d]
        roll, pitch, yaw = (float(v) for v in env.rpy[d])
        vf, vr, vu = world_to_body(vel, yaw)
        af, ar, au = world_to_body(accel, yaw)
        tof = min(float(env._get_altitude_distance(d)), OFFICE_TELEM_TOF_MAX_M)
        height = float(env.pos[d, 2]) - float(env._office_takeoff_z[d])
        baro = height + float(env._office_baro_walk[d])
        return np.array([pitch, roll, yaw, vf, vr, vu, af, ar, au, tof, height, baro])

    def _deliver_packet(self, env, d: int, rng, dt: float) -> None:
        env._office_baro_walk[d] += rng.normal(0.0, OFFICE_TELEM_BARO_WALK_M)
        if rng.random() < OFFICE_TELEM_DROP_PROB:
            return
        snap = env._office_pending[d]
        if snap is None:
            return
        q = np.round((snap + rng.normal(0.0, _SNAP_NOISE_STD)) / _SNAP_QUANT) * _SNAP_QUANT
        telem = env._office_telemetry
        telem[d, 0:2] = q[0:2]
        telem[d, 2] = math.sin(q[2])
        telem[d, 3] = math.cos(q[2])
        telem[d, 4:13] = q[3:12]
        telem[d, 13] = OFFICE_TELEM_DELAY_STEPS * dt

    def apply_world_physics(self, env) -> None:
        force = getattr(env, "_office_vps_force", None)
        if force is None:
            return
        cli = env.CLIENT
        for d in range(env.NUM_DRONES):
            p.applyExternalForce(
                int(env.DRONE_IDS[d]), -1, force, env.pos[d].tolist(),
                p.WORLD_FRAME, physicsClientId=cli,
            )

    # ------------------------------------------------------------------ #
    # scoring (placeholder: reach the target point; replaced when the live
    # target behaviour card lands)
    # ------------------------------------------------------------------ #
    def build_rollout_metrics(self, *, task, success, t, horizon,
                              min_clearance, collision, failure_reason) -> dict:
        challenge_type = int(getattr(task, "challenge_type", OFFICE_CHALLENGE_TYPE))
        return {
            "challenge_type": challenge_type,
            "environment_type": CHALLENGE_TYPE_TO_ENVIRONMENT_TYPE.get(
                challenge_type, "unknown"
            ),
            "time_sec": float(t),
            "horizon_sec": float(horizon),
            "target_time_sec": _calculate_office_target_time(task),
            "min_clearance": float(min_clearance),
            "collision": bool(collision),
            "failure_reason": failure_reason,
            "success": bool(success),
        }

    def normalize_rollout_metrics(self, *, task, metrics) -> dict:
        horizon = float(metrics["horizon_sec"])
        if horizon <= 0.0:
            raise ValueError("'horizon' must be positive")
        success = bool(metrics["success"])
        collision = bool(metrics["collision"])
        failure_reason = str(metrics["failure_reason"])
        t = float(metrics["time_sec"])

        if failure_reason == FailureReason.EVAL_ERROR.value:
            return {"success_term": 0.0, "time_term": 0.0, "safety_term": 0.0,
                    "participation_term": 0.0, "final_score": 0.0}
        if not success:
            part = PARTICIPATION_REWARD if failure_reason in PARTICIPATION_REASONS else 0.0
            return {"success_term": 0.0, "time_term": 0.0, "safety_term": 0.0,
                    "participation_term": part, "final_score": part}
        if collision:
            part = PARTICIPATION_REWARD if t > 0.0 else 0.0
            return {"success_term": 0.0, "time_term": 0.0, "safety_term": 0.0,
                    "participation_term": part, "final_score": part}

        target_time = _calculate_office_target_time(task) if task is not None else None
        time_term = calculate_time_term(t=t, horizon=horizon, target_time=target_time)
        final = _clamp(OFFICE_W_SUCCESS * 1.0 + OFFICE_W_TIME * time_term)
        return {"success_term": 1.0, "time_term": float(time_term), "safety_term": 0.0,
                "participation_term": 0.0, "final_score": float(final)}

    # ------------------------------------------------------------------ #
    # task generation
    # ------------------------------------------------------------------ #
    def build_random_task(self, *, sim_dt: float, seed: int):
        from swarm.validator import task_gen
        return task_gen.random_task(sim_dt, seed, family_id=self.family_id)

    def screening_template(self) -> tuple:
        return (_TEMPLATE_SLOT,) * 8

    def benchmark_template(self) -> tuple:
        return (_TEMPLATE_SLOT,) * 100
