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

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel

from swarm.constants import (
    OFFICE_CHALLENGE_TYPE,
    OFFICE_MAX_START_DISTANCE_M,
    OFFICE_MAX_TILT_DEG,
    OFFICE_MIN_START_DISTANCE_M,
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

    def state_clue_dim(self, task) -> int:
        return 2

    # ------------------------------------------------------------------ #
    # env lifecycle
    # ------------------------------------------------------------------ #
    def initialise_env_state(self, env, *, requested_mode: bool = False) -> None:
        env.sar_mode = False
        env.MAX_TILT_RAD = math.radians(OFFICE_MAX_TILT_DEG)

    def spawn_task_world(self, env) -> None:
        build_office_map(seed=env.task.map_seed, cli=env.getPyBulletClient())

    def build_info(self, env) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "task_version": str(getattr(env.task, "version", "")),
        }

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
