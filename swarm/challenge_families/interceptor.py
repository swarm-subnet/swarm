from __future__ import annotations

import inspect
import math
import os
import random
from typing import Any, Optional

import numpy as np
import pkg_resources
import pybullet as p

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from swarm.constants import (
    INTERCEPTOR_ACQUIRE_SLACK_SEC,
    INTERCEPTOR_ALT_MAX_M,
    INTERCEPTOR_ALT_MIN_M,
    INTERCEPTOR_DEPTH_FAR_M,
    INTERCEPTOR_DEPTH_MAX_M,
    INTERCEPTOR_DEPTH_RES,
    INTERCEPTOR_DRONE_URDF,
    INTERCEPTOR_JINK_FREQ_MAX,
    INTERCEPTOR_JINK_FREQ_MIN,
    INTERCEPTOR_JINK_GAIN,
    INTERCEPTOR_KILL_RADIUS_M,
    INTERCEPTOR_MINER_SPEED,
    INTERCEPTOR_REACT_RANGE_M,
    INTERCEPTOR_SEARCH_RADIUS_MAX_M,
    INTERCEPTOR_SEARCH_RADIUS_MIN_M,
    INTERCEPTOR_SEARCH_REFRESH_SEC,
    INTERCEPTOR_SEED_OFFSET,
    INTERCEPTOR_START_PAD_HEIGHT,
    INTERCEPTOR_START_PAD_RADIUS,
    INTERCEPTOR_TAKEOFF_BUFFER,
    INTERCEPTOR_TARGET_CRUISE_FRAC,
    INTERCEPTOR_TARGET_FLEE_FRAC,
    INTERCEPTOR_TARGET_SELFCRASH_FORCE,
    INTERCEPTOR_TERRAIN_SIZE_M,
    INTERCEPTOR_W_SUCCESS,
    INTERCEPTOR_W_TIME,
)
from swarm.core.env_builder.body_tagger import BodyTagger
from swarm.core.env_builder.platform import build_start_platform, surface_z_at
from swarm.core.env_builder.sar_tagging import build_and_tag_map
from swarm.core.env_builder.surface_resolver import resolve_surface
from swarm.core.env_builder.victim import accepted_categories_for
from swarm.domain_model import CHALLENGE_TYPE_TO_ENVIRONMENT_TYPE
from swarm.protocol import FailureReason, SCHEMA_VERSION
from swarm.validator.reward import (
    PARTICIPATION_REASONS,
    PARTICIPATION_REWARD,
    _calculate_interceptor_target_time,
    _clamp,
    calculate_time_term,
)

from .base import ChallengeFamilyRuntime, ChallengeFamilyRuntimeProfile

# SWARM_DRONE PID gains, sized for the 36 cm drone.
_GAINS_P_FOR = np.array([2.3, 2.3, 7.1])
_GAINS_I_FOR = np.array([0.3, 0.3, 0.3])
_GAINS_D_FOR = np.array([1.1, 1.1, 2.9])
_GAINS_P_TOR = np.array([2.3e6, 2.3e6, 2.0e6])
_GAINS_I_TOR = np.array([0.0, 0.0, 16500.0])
_GAINS_D_TOR = np.array([6.6e5, 6.6e5, 4.0e5])

_PATROL_RADIUS_M = 55.0  # keep the target near its spawn so the chase stays bounded


def _supports_keyword_arg(callable_obj: Any, keyword: str) -> bool:
    try:
        return keyword in inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return True


def interceptor_urdf_path() -> str:
    """Absolute path to the 36 cm URDF in the swarm package assets."""
    return pkg_resources.resource_filename("swarm", os.path.join("assets", INTERCEPTOR_DRONE_URDF))


def ensure_interceptor_urdf_in_gym_assets() -> str:
    """Make the 36 cm URDF available in gym_pybullet_drones/assets (next to cf2.dae) so
    BaseAviary's hardcoded loader finds it via self.URDF. Content-verified and atomic so every
    validator parses byte-identical physical constants. Returns the URDF basename."""
    dst_dir = pkg_resources.resource_filename("gym_pybullet_drones", "assets")
    dst = os.path.join(dst_dir, INTERCEPTOR_DRONE_URDF)
    with open(interceptor_urdf_path(), "rb") as f:
        src_bytes = f.read()
    need = True
    if os.path.exists(dst):
        with open(dst, "rb") as f:
            need = f.read() != src_bytes
    if need:
        tmp = f"{dst}.tmp.{os.getpid()}"
        with open(tmp, "wb") as f:
            f.write(src_bytes)
        os.replace(tmp, dst)  # atomic, race-safe across workers
    with open(dst, "rb") as f:
        if f.read() != src_bytes:
            raise RuntimeError("interceptor_drone.urdf content mismatch in gym assets")
    return INTERCEPTOR_DRONE_URDF


def make_interceptor_control(env: Any) -> DSLPIDControl:
    """A DSL PID controller carrying the 36 cm drone's parsed constants + the validated gains.
    Used for both the chaser's VEL controller and the validator-flown target."""
    ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
    ctrl.GRAVITY = float(env.GRAVITY)
    ctrl.KF = float(env.KF)
    ctrl.KM = float(env.KM)
    ctrl.P_COEFF_FOR = _GAINS_P_FOR.copy()
    ctrl.I_COEFF_FOR = _GAINS_I_FOR.copy()
    ctrl.D_COEFF_FOR = _GAINS_D_FOR.copy()
    ctrl.P_COEFF_TOR = _GAINS_P_TOR.copy()
    ctrl.I_COEFF_TOR = _GAINS_I_TOR.copy()
    ctrl.D_COEFF_TOR = _GAINS_D_TOR.copy()
    ctrl.reset()
    return ctrl


class InterceptorChallengeFamily(ChallengeFamilyRuntime):
    family_id = "cf_interceptor"
    runtime_supported = True

    # ------------------------------------------------------------------ #
    # profile / contract
    # ------------------------------------------------------------------ #
    def runtime_profile(self, task: Any) -> ChallengeFamilyRuntimeProfile:
        _ = task
        return ChallengeFamilyRuntimeProfile(
            family_id=self.family_id,
            profile_name="interceptor",
            resource_class="navigation",
            image_key="base",
            env_bootstrap={"sar_mode": False},
            docker_env={
                "SWARM_CHALLENGE_FAMILY_ID": self.family_id,
                "SWARM_RUNTIME_PROFILE": "interceptor",
                "SWARM_RUNTIME_RESOURCE_CLASS": "navigation",
                "SWARM_RUNTIME_IMAGE_KEY": "base",
                "SWARM_RUNTIME_ENV_BOOTSTRAP": "sar_mode=false",
            },
            batch_timeout_multiplier=1.3,
        )

    def env_kwargs_for_task(self, task: Any) -> dict[str, Any]:
        _ = task
        return {"sar_mode": False}

    def state_clue_dim(self, task: Any) -> int:
        _ = task
        return 2

    # ------------------------------------------------------------------ #
    # per-episode state
    # ------------------------------------------------------------------ #
    def initialise_env_state(self, env: Any, *, requested_mode: bool = False) -> None:
        _ = requested_mode
        env.sar_mode = False
        env._target_uid = None
        env._target_ctrl = None
        self.reset_env_state(env)

    def reset_env_state(self, env: Any) -> None:
        env._target_pos = np.asarray(env.GOAL_POS, dtype=float).copy()
        env._target_vel = np.zeros(3, dtype=float)
        env._target_rpm = np.zeros(4, dtype=float)
        env._target_floor_z = 0.0
        env._target_home_xy = np.asarray(env.GOAL_POS, dtype=float)[:2].copy()
        env._intercept_caught = False
        env._target_crashed = False
        env._collision_exempt_uids = frozenset()
        env._intercept_min_dist = float("inf")
        env._search_area_center = np.asarray(env.GOAL_POS, dtype=float).copy()

        seed = int(getattr(env.task, "map_seed", 0))
        rng = random.Random((seed ^ INTERCEPTOR_SEED_OFFSET) & 0xFFFFFFFF)
        env._evader_jink_freq = rng.uniform(INTERCEPTOR_JINK_FREQ_MIN, INTERCEPTOR_JINK_FREQ_MAX)
        env._evader_jink_phase = rng.uniform(0.0, 2.0 * math.pi)
        env._evader_cruise_heading = rng.uniform(0.0, 2.0 * math.pi)
        env._search_radius = rng.uniform(INTERCEPTOR_SEARCH_RADIUS_MIN_M, INTERCEPTOR_SEARCH_RADIUS_MAX_M)
        if getattr(env, "_target_ctrl", None) is not None:
            env._target_ctrl.reset()

    # ------------------------------------------------------------------ #
    # world spawn
    # ------------------------------------------------------------------ #
    def spawn_task_world(self, env: Any) -> None:
        env.task.start = env._original_start
        env.task.goal = env._original_goal
        cli = getattr(env, "CLIENT", 0)

        # (a) load the target OFF-WORLD; it is flown into place after the map is built
        urdf = ensure_interceptor_urdf_in_gym_assets()
        urdf_path = pkg_resources.resource_filename("gym_pybullet_drones", os.path.join("assets", urdf))
        target_uid = p.loadURDF(
            urdf_path, [0.0, 0.0, -1000.0], p.getQuaternionFromEuler([0, 0, 0]),
            flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=cli,
        )
        env._target_uid = int(target_uid)
        p.resetBaseVelocity(target_uid, [0, 0, 0], [0, 0, 0], physicsClientId=cli)
        # the chaser physically rams the target (a real crash); exempt the target from the
        # chaser's obstacle-collision check so the ram is scored as a catch, not a chaser crash.
        env._collision_exempt_uids = frozenset({int(target_uid)})

        # (b) build the map / tagger
        tagger = build_and_tag_map(
            cli=cli, seed=env.task.map_seed, challenge_type=env.task.challenge_type,
            start=env.task.start, goal=env.task.goal, safe_zone_radius=0.0, sar_mode=False,
            terrain_size=INTERCEPTOR_TERRAIN_SIZE_M,
        )
        env._interceptor_world_tags = dict(tagger.body_tags)

        # (c) chaser takeoff pad
        sx, sy = float(env.task.start[0]), float(env.task.start[1])
        surf = surface_z_at(cli, sx, sy)
        pad_uids, pad_top = build_start_platform(
            tagger, cli, sx, sy, surf, int(env.task.challenge_type),
            radius=INTERCEPTOR_START_PAD_RADIUS, height=INTERCEPTOR_START_PAD_HEIGHT,
        )
        env._platform_uids = frozenset(pad_uids)
        env.task.start = (sx, sy, pad_top + INTERCEPTOR_TAKEOFF_BUFFER)
        p.resetBasePositionAndOrientation(
            env.DRONE_IDS[0], list(env.task.start), p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=cli,
        )

        # (d) airborne target altitude from the resolved floor (skips warehouse-style roofs)
        tx, ty = float(env.task.goal[0]), float(env.task.goal[1])
        hit = resolve_surface(cli, tx, ty, tagger.body_tags, accepted_categories_for(int(env.task.challenge_type)))
        floor_z = float(hit.surface_z) if hit is not None else surface_z_at(cli, tx, ty)
        env._target_floor_z = floor_z
        seed = int(getattr(env.task, "map_seed", 0))
        alt_rng = random.Random((seed ^ INTERCEPTOR_SEED_OFFSET ^ 0x5F5F5F5F) & 0xFFFFFFFF)
        tz = floor_z + alt_rng.uniform(INTERCEPTOR_ALT_MIN_M, INTERCEPTOR_ALT_MAX_M)
        env._target_pos = np.array([tx, ty, tz], dtype=float)
        env._target_vel = np.zeros(3, dtype=float)
        env._target_home_xy = np.array([tx, ty], dtype=float)

        # (e) the scored goal IS the live airborne target
        env.task.goal = (tx, ty, tz)
        env.GOAL_POS = env._target_pos.copy()
        self._refresh_search_clue(env)  # coarse clue from t=0 (never the exact target)

        # (f) place the target + its controller
        p.resetBasePositionAndOrientation(
            target_uid, [tx, ty, tz], p.getQuaternionFromEuler([0, 0, 0]), physicsClientId=cli,
        )
        env._target_ctrl = make_interceptor_control(env)

        plane_id = getattr(env, "PLANE_ID", None)
        if plane_id is not None:
            if int(getattr(env.task, "challenge_type", 0)) == 2:
                p.resetBasePositionAndOrientation(
                    plane_id, [0.0, 0.0, -1000.0], [0.0, 0.0, 0.0, 1.0], physicsClientId=cli,
                )
            p.changeVisualShape(plane_id, -1, rgbaColor=[0, 0, 0, 0], physicsClientId=cli)

        # (g) cull targets LAST so the (now-protected) target is never culled
        env._build_cull_targets()

    def protected_body_uids(self, env: Any) -> set[int]:
        base = set(getattr(env, "_platform_uids", frozenset()))
        if getattr(env, "_target_uid", None) is not None:
            base.add(int(env._target_uid))
        return base

    # ------------------------------------------------------------------ #
    # the evader: validator flies the target each control step
    # ------------------------------------------------------------------ #
    def _evader_velocity(self, env: Any, chaser_pos: np.ndarray, tpos: np.ndarray) -> np.ndarray:
        flee_speed = INTERCEPTOR_MINER_SPEED * INTERCEPTOR_TARGET_FLEE_FRAC
        cruise_speed = INTERCEPTOR_MINER_SPEED * INTERCEPTOR_TARGET_CRUISE_FRAC
        t = float(env._time_alive)

        home_off = tpos[:2] - env._target_home_xy
        if float(np.linalg.norm(home_off)) > _PATROL_RADIUS_M:
            # too far from home: head back in
            n = np.linalg.norm(home_off)
            dir2 = -home_off / n if n > 1e-6 else np.array([1.0, 0.0])
            speed = cruise_speed
        else:
            to_ch = chaser_pos[:2] - tpos[:2]
            dist = float(np.linalg.norm(to_ch))
            if dist > INTERCEPTOR_REACT_RANGE_M:
                ang = env._evader_cruise_heading + 0.2 * math.sin(0.1 * t)
                dir2 = np.array([math.cos(ang), math.sin(ang)])
                speed = cruise_speed
            else:
                away = -to_ch
                n = float(np.linalg.norm(away))
                away = away / n if n > 1e-6 else np.array([math.cos(env._evader_cruise_heading),
                                                           math.sin(env._evader_cruise_heading)])
                lateral = np.array([-away[1], away[0]])
                jink = math.sin(2.0 * math.pi * env._evader_jink_freq * t + env._evader_jink_phase)
                dir2 = away + INTERCEPTOR_JINK_GAIN * jink * lateral
                n2 = float(np.linalg.norm(dir2))
                dir2 = dir2 / n2 if n2 > 1e-6 else away
                speed = flee_speed

        lo = env._target_floor_z + INTERCEPTOR_ALT_MIN_M
        hi = env._target_floor_z + INTERCEPTOR_ALT_MAX_M
        vz = 0.0
        if tpos[2] < lo:
            vz = 0.5 * speed
        elif tpos[2] > hi:
            vz = -0.5 * speed
        return np.array([dir2[0] * speed, dir2[1] * speed, vz], dtype=float)

    def advance_world(self, env: Any) -> None:
        if getattr(env, "_target_uid", None) is None or getattr(env, "_target_ctrl", None) is None:
            return
        cli = getattr(env, "CLIENT", 0)
        dt = env._sim_dt
        chaser = env._getDroneStateVector(0)
        cpos = np.asarray(chaser[0:3], dtype=float)
        tpos, tquat = p.getBasePositionAndOrientation(env._target_uid, physicsClientId=cli)
        tvel, tang = p.getBaseVelocity(env._target_uid, physicsClientId=cli)
        tpos = np.asarray(tpos, dtype=float)
        # track the ground under the target; ray from below it, skipping both drones
        fhit = p.rayTest([float(tpos[0]), float(tpos[1]), float(tpos[2]) - 0.5],
                         [float(tpos[0]), float(tpos[1]), -30.0], physicsClientId=cli)
        if fhit and int(fhit[0][0]) not in (-1, int(env._target_uid), int(env.DRONE_IDS[0])):
            env._target_floor_z = float(fhit[0][3][2])
        desired = self._evader_velocity(env, cpos, tpos)
        look = tpos + desired * dt * 5.0
        rpm, _, _ = env._target_ctrl.computeControl(
            control_timestep=dt, cur_pos=tpos, cur_quat=np.asarray(tquat, dtype=float),
            cur_vel=np.asarray(tvel, dtype=float), cur_ang_vel=np.asarray(tang, dtype=float),
            target_pos=look, target_vel=desired,
        )
        env._target_rpm = np.asarray(rpm, dtype=float)

    def apply_world_physics(self, env: Any) -> None:
        if getattr(env, "_target_uid", None) is None:
            return
        if getattr(env, "PHYSICS", None) == Physics.DYN:
            return
        cli = getattr(env, "CLIENT", 0)
        rpm = np.asarray(env._target_rpm, dtype=float)
        forces = np.square(rpm) * float(env.KF)
        torques = np.square(rpm) * float(env.KM)
        z_torque = float(-torques[0] + torques[1] - torques[2] + torques[3])
        for i in range(4):
            p.applyExternalForce(env._target_uid, i, [0.0, 0.0, float(forces[i])], [0.0, 0.0, 0.0],
                                 p.LINK_FRAME, physicsClientId=cli)
        p.applyExternalTorque(env._target_uid, 4, [0.0, 0.0, z_torque], p.LINK_FRAME, physicsClientId=cli)

    # ------------------------------------------------------------------ #
    # post-physics: sync target, refresh the coarse clue, catch + self-crash
    # ------------------------------------------------------------------ #
    def _refresh_search_clue(self, env: Any) -> None:
        """Coarse search-area hint: a deterministic point within search_radius of the
        target's CURRENT position, re-sampled each refresh window. Never the exact target
        (the miner must find it with the camera). Read by the search_clue_offset channel."""
        seed = int(getattr(env.task, "map_seed", 0))
        ping = int(env._time_alive / INTERCEPTOR_SEARCH_REFRESH_SEC)
        prng = random.Random((seed ^ INTERCEPTOR_SEED_OFFSET ^ (ping * 0x9E3779B1)) & 0xFFFFFFFF)
        r = env._search_radius * math.sqrt(prng.random())
        th = 2.0 * math.pi * prng.random()
        env._search_area_center = np.array(
            [env._target_pos[0] + r * math.cos(th), env._target_pos[1] + r * math.sin(th), 0.0],
            dtype=float,
        )

    def post_step_update(self, env: Any) -> None:
        if getattr(env, "_target_uid", None) is None:
            return
        cli = getattr(env, "CLIENT", 0)
        tpos, _ = p.getBasePositionAndOrientation(env._target_uid, physicsClientId=cli)
        tvel, _ = p.getBaseVelocity(env._target_uid, physicsClientId=cli)
        env._target_pos = np.asarray(tpos, dtype=float)
        env._target_vel = np.asarray(tvel, dtype=float)
        self._refresh_search_clue(env)

        chaser = env._getDroneStateVector(0)
        cpos = np.asarray(chaser[0:3], dtype=float)
        dist = float(np.linalg.norm(cpos - env._target_pos))
        if dist < env._intercept_min_dist:
            env._intercept_min_dist = dist

        rammed = bool(p.getContactPoints(bodyA=int(env.DRONE_IDS[0]),
                                         bodyB=int(env._target_uid), physicsClientId=cli))
        if not env._success and (rammed or dist <= INTERCEPTOR_KILL_RADIUS_M):
            env._success = True
            env._intercept_caught = True
            env._t_to_goal = env._time_alive
            env._failure_reason = FailureReason.NONE.value
            return

        if not env._success and not env._target_crashed:
            contacts = p.getContactPoints(bodyA=env._target_uid, physicsClientId=cli)
            chaser_uid = int(env.DRONE_IDS[0])
            for c in contacts:
                if int(c[2]) in (-1, chaser_uid):
                    continue
                if c[9] > INTERCEPTOR_TARGET_SELFCRASH_FORCE:
                    env._target_crashed = True
                    if env._failure_reason == FailureReason.NONE.value:
                        env._failure_reason = FailureReason.INFEASIBLE.value
                    break

    def compute_terminated(self, env: Any) -> bool:
        if getattr(env, "_target_crashed", False) and not env._success:
            if env._failure_reason == FailureReason.NONE.value:
                env._failure_reason = FailureReason.INFEASIBLE.value
            return True
        if env._collision and not env._success:
            if env._failure_reason == FailureReason.NONE.value:
                env._failure_reason = FailureReason.OBSTACLE_COLLISION.value
        return False

    def compute_truncated(self, env: Any, *, terminal_already: bool, roll: float, pitch: float) -> bool:
        if abs(float(roll)) > float(env.MAX_TILT_RAD) or abs(float(pitch)) > float(env.MAX_TILT_RAD):
            if not terminal_already:
                env._failure_reason = FailureReason.TILT.value
            return True
        if env._time_alive >= env.EP_LEN_SEC:
            if not terminal_already:
                env._failure_reason = FailureReason.TIMEOUT.value
            return True
        return False

    def clue_offset(self, env: Any, state_vec: Any) -> Any:
        return env._search_area_center - state_vec[0:3]

    def build_info(self, env: Any) -> dict[str, Any]:
        return {
            "intercept_caught": bool(getattr(env, "_intercept_caught", False)),
            "target_crashed": bool(getattr(env, "_target_crashed", False)),
            "intercept_min_dist": float(getattr(env, "_intercept_min_dist", float("inf"))),
            "kill_radius_m": float(INTERCEPTOR_KILL_RADIUS_M),
            "search_radius_m": float(getattr(env, "_search_radius", 0.0)),
            "t_to_intercept": (
                float(env._t_to_goal) if env._success and env._t_to_goal is not None else None
            ),
            "schema_version": SCHEMA_VERSION,
            "task_version": str(getattr(env.task, "version", "")),
        }

    # ------------------------------------------------------------------ #
    # scoring: 0.5 caught + 0.5 time
    # ------------------------------------------------------------------ #
    def build_rollout_metrics(
        self, *, task: Any, success: bool, t: float, horizon: float,
        min_clearance: Optional[float], collision: bool, legitimate_model: bool, failure_reason: str,
    ) -> dict[str, Any]:
        challenge_type = int(getattr(task, "challenge_type", -1))
        target_time = _calculate_interceptor_target_time(task) if task is not None else None
        return {
            "challenge_type": challenge_type,
            "environment_type": CHALLENGE_TYPE_TO_ENVIRONMENT_TYPE.get(challenge_type, "unknown"),
            "time_sec": float(t),
            "horizon_sec": float(horizon),
            "target_time_sec": None if target_time is None else float(target_time),
            "min_clearance": None if min_clearance is None else float(min_clearance),
            "collision": bool(collision),
            "legitimate_model": bool(legitimate_model),
            "failure_reason": str(failure_reason),
            "success": bool(success),
        }

    def normalize_rollout_metrics(self, *, task: Any, metrics: dict[str, Any]) -> dict[str, float]:
        horizon = float(metrics["horizon_sec"])
        if horizon <= 0.0:
            raise ValueError("'horizon' must be positive")
        success = bool(metrics["success"])
        collision = bool(metrics["collision"])
        legitimate_model = bool(metrics["legitimate_model"])
        failure_reason = str(metrics["failure_reason"])
        t = float(metrics["time_sec"])

        if not legitimate_model or failure_reason == FailureReason.EVAL_ERROR.value:
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

        target_time = _calculate_interceptor_target_time(task) if task is not None else None
        time_term = calculate_time_term(t=t, horizon=horizon, target_time=target_time)
        final = _clamp(INTERCEPTOR_W_SUCCESS * 1.0 + INTERCEPTOR_W_TIME * time_term)
        return {"success_term": 1.0, "time_term": float(time_term), "safety_term": 0.0,
                "participation_term": 0.0, "final_score": float(final)}

    # ------------------------------------------------------------------ #
    # task builders
    # ------------------------------------------------------------------ #
    def build_random_task(self, *, sim_dt: float, seed: Optional[int]) -> Any:
        from swarm.validator import task_gen as legacy_task_gen

        kwargs = {"sim_dt": sim_dt, "seed": seed}
        if _supports_keyword_arg(legacy_task_gen.random_task, "family_id"):
            kwargs["family_id"] = self.family_id
        return legacy_task_gen.random_task(**kwargs)

    def screening_template(self) -> tuple[dict[str, Any], ...]:
        # interceptor runs on the open map only (challenge_type 2)
        return tuple({"challenge_type": 2, "distance_range": (25.0, 100.0)} for _ in range(8))

    def build_screening_tasks(
        self, *, sim_dt: float, seeds: list[int], offset: int = 0, total_seed_count: Optional[int] = None,
    ) -> list[Any]:
        from swarm.validator import task_gen as legacy_task_gen

        tasks = []
        for seed in seeds:
            kwargs = {"sim_dt": sim_dt, "seed": seed, "challenge_type": 2, "distance_range": (25.0, 100.0)}
            if _supports_keyword_arg(legacy_task_gen.screening_task, "family_id"):
                kwargs["family_id"] = self.family_id
            tasks.append(legacy_task_gen.screening_task(**kwargs))
        return tasks
