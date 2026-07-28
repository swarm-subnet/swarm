"""Office interceptor challenge family (indoor Tello, body-frame RC actions).

The miner flies a Tello-class drone inside the fixed office map and must
intercept a target drone. Actions are the four Tello RC sticks
[lr, fb, ud, yaw]; there is no GPS-style world-frame control.

The validator flies the target: a second Tello doing seeded person-style
waypoint legs above the furniture. The catch is a real physical hit —
chaser-target contact ends the episode as a success. The detector emulator
and the visual-input decision land in their own follow-up cards.
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
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from swarm.constants import (
    OFFICE_CHALLENGE_TYPE,
    OFFICE_KILL_RADIUS_M,
    OFFICE_MAX_START_DISTANCE_M,
    OFFICE_MAX_TILT_DEG,
    OFFICE_MIN_START_DISTANCE_M,
    OFFICE_TARGET_ALT_MAX_M,
    OFFICE_TARGET_ALT_MIN_M,
    OFFICE_TARGET_ARRIVE_M,
    OFFICE_TARGET_CLEAR_M,
    OFFICE_TARGET_MIN_LEG_M,
    OFFICE_TARGET_PAUSE_MAX_SEC,
    OFFICE_TARGET_PAUSE_MIN_SEC,
    OFFICE_TARGET_SEED_OFFSET,
    OFFICE_TARGET_SELFCRASH_FORCE,
    OFFICE_TARGET_SPEED,
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

# Target in-flight reflexes: the short forward guard runs often (it must beat the
# ~0.3 m stopping distance from cruise), the full-leg recheck reroutes early.
_GUARD_EVERY_STEPS = 2
_GUARD_RANGE_M = 0.6
_RECHECK_EVERY_STEPS = 10

# Telemetry packet layout: [pitch, roll, sin_yaw, cos_yaw, vf, vr, vd, af, ar, ad,
# tof, height, baro, age, valid]. Velocity/accel use the SDK body frame
# (forward, right, DOWN — vgz is positive when descending), and acceleration is
# the raw specific force: hover reads -g on the down axis, like the real IMU.
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
        env._office_target_uid = None
        env._office_target_ctrl = None

    def reset_env_state(self, env) -> None:
        n = env.NUM_DRONES
        seed = int(getattr(env.task, "map_seed", 0))
        env._office_target_rng = random.Random((seed ^ OFFICE_TARGET_SEED_OFFSET) & 0xFFFFFFFF)
        env._office_target_wp = None
        env._office_target_pause = 0.0
        env._office_target_forces = [0.0, 0.0, 0.0, 0.0]
        env._office_target_ztorque = 0.0
        env._office_target_pos = np.zeros(3, dtype=float)
        env._office_target_step = 0
        env._office_target_brakes = 0
        env._office_target_crashed = False
        env._collision_exempt_uids = frozenset()
        if getattr(env, "_office_target_ctrl", None) is not None:
            env._office_target_ctrl.reset()
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

    def _rays_clear(self, env, froms: list, tos: list, extra_ignore: tuple = ()) -> bool:
        ignore = {int(env._office_target_uid), -1, *extra_ignore}
        return all(int(h[0]) in ignore
                   for h in p.rayTestBatch(froms, tos, physicsClientId=env.CLIENT))

    def _point_is_clear(self, env, pos: np.ndarray, *, floor: bool) -> bool:
        """No geometry within a body length of the point (task points only respect
        wall margins, so an unlucky seed can land inside furniture)."""
        origin = pos + np.array([0.0, 0.0, 0.15 if floor else 0.0])
        dirs = [(0.4, 0, 0), (-0.4, 0, 0), (0, 0.4, 0), (0, -0.4, 0)]
        if floor:
            # Long up-ray: a takeoff column blocked by a furniture overhang is no spawn.
            dirs.append((0, 0, 1.2))
        else:
            dirs += [(0, 0, 0.4), (0, 0, -0.4)]
        froms = [origin.tolist()] * len(dirs)
        tos = [(origin + np.asarray(d)).tolist() for d in dirs]
        return self._rays_clear(env, froms, tos, extra_ignore=(int(env.DRONE_IDS[0]),))

    def _clear_spawn(self, env, pos: np.ndarray, *, floor: bool, rng,
                     anchor: np.ndarray) -> np.ndarray:
        """First clear point, preferring candidates that keep the task's start-goal
        separation band; falls back to any clear point rather than none."""
        z_range = (pos[2], pos[2]) if floor else (OFFICE_TARGET_ALT_MIN_M, OFFICE_TARGET_ALT_MAX_M)
        fallback = None
        for i in range(65):
            if self._point_is_clear(env, pos, floor=floor):
                gap = float(np.linalg.norm(pos[:2] - anchor[:2]))
                if OFFICE_MIN_START_DISTANCE_M <= gap <= OFFICE_MAX_START_DISTANCE_M:
                    return pos
                if fallback is None:
                    fallback = pos
            if i < 64:
                pos = np.array(office_point(rng, z_range), dtype=float)
        return fallback if fallback is not None else pos

    def spawn_task_world(self, env) -> None:
        cli = env.getPyBulletClient()
        env.task.start = env._original_start
        env.task.goal = env._original_goal
        seed = int(env.task.map_seed)

        urdf = ensure_tello_assets_in_gym_assets()
        urdf_path = str(_pkg_files("gym_pybullet_drones").joinpath("assets", urdf))
        uid = int(p.loadURDF(
            urdf_path, [0.0, 0.0, -1000.0], p.getQuaternionFromEuler([0, 0, 0]),
            flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=cli,
        ))
        env._office_target_uid = uid
        # Ramming the target is the catch, not a chaser crash.
        env._collision_exempt_uids = frozenset({uid})

        build_office_map(seed=seed, cli=cli)

        place_rng = random.Random((seed ^ OFFICE_TARGET_SEED_OFFSET ^ 0x9A7C) & 0xFFFFFFFF)
        start = self._clear_spawn(env, np.array(env.task.start, dtype=float),
                                  floor=True, rng=place_rng,
                                  anchor=np.array(env.task.goal, dtype=float))
        env.task.start = tuple(float(v) for v in start)
        p.resetBasePositionAndOrientation(
            int(env.DRONE_IDS[0]), start.tolist(),
            p.getQuaternionFromEuler([0, 0, 0]), physicsClientId=cli,
        )
        p.resetBaseVelocity(int(env.DRONE_IDS[0]), [0, 0, 0], [0, 0, 0], physicsClientId=cli)

        goal = self._clear_spawn(env, np.array(env.task.goal, dtype=float),
                                 floor=False, rng=place_rng, anchor=start)
        env.task.goal = tuple(float(v) for v in goal)
        env.GOAL_POS = goal.copy()
        p.resetBasePositionAndOrientation(
            uid, goal.tolist(), p.getQuaternionFromEuler([0, 0, 0]), physicsClientId=cli,
        )
        p.resetBaseVelocity(uid, [0, 0, 0], [0, 0, 0], physicsClientId=cli)
        env._office_target_pos = goal.copy()
        env._office_target_ctrl = make_office_control(env)

    def protected_body_uids(self, env) -> set:
        uid = getattr(env, "_office_target_uid", None)
        return {int(uid)} if uid is not None else set()

    def compute_terminated(self, env) -> bool:
        # A self-crashed target ends the seed as infeasible, like the outdoor family.
        if getattr(env, "_office_target_crashed", False) and not env._success:
            return True
        if env._collision and not env._success:
            if env._failure_reason == FailureReason.NONE.value:
                env._failure_reason = FailureReason.OBSTACLE_COLLISION.value
        return False

    def compute_truncated(self, env, *, terminal_already: bool, roll: float, pitch: float) -> bool:
        if abs(float(roll)) > float(env.MAX_TILT_RAD) or abs(float(pitch)) > float(env.MAX_TILT_RAD):
            if not terminal_already:
                env._failure_reason = FailureReason.TILT.value
            return True
        if env._time_alive >= env.EP_LEN_SEC:
            if not terminal_already:
                env._failure_reason = FailureReason.TIMEOUT.value
            return True
        return False

    def build_info(self, env) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "task_version": str(getattr(env.task, "version", "")),
        }

    # ------------------------------------------------------------------ #
    # target flight: seeded person-style waypoint legs with hover pauses
    # ------------------------------------------------------------------ #
    def _leg_is_clear(self, env, start: np.ndarray, end: np.ndarray,
                      ignore_chaser: bool = False) -> bool:
        """Ray-check the leg with side offsets so the body radius clears too.

        At planning time the chaser counts as an obstacle: a person would steer
        around a parked drone, and it stops a park-in-the-flight-path free catch.
        The in-flight guards ignore the chaser — dodging an approaching miner
        would be escape logic, which this target deliberately has none of."""
        cli = env.CLIENT
        d = end - start
        n = float(np.linalg.norm(d))
        if n < 1e-6:
            return False
        d = d / n
        side = np.cross(d, [0.0, 0.0, 1.0])
        sn = float(np.linalg.norm(side))
        side = side / sn if sn > 1e-6 else np.array([1.0, 0.0, 0.0])
        up = np.cross(side, d)
        froms, tos = [], []
        for off in (np.zeros(3), OFFICE_TARGET_CLEAR_M * side, -OFFICE_TARGET_CLEAR_M * side,
                    OFFICE_TARGET_CLEAR_M * up, -OFFICE_TARGET_CLEAR_M * up):
            froms.append((start + off).tolist())
            tos.append((end + off).tolist())
        ignore = {int(env._office_target_uid), -1}
        if ignore_chaser:
            ignore.add(int(env.DRONE_IDS[0]))
        for hit in p.rayTestBatch(froms, tos, physicsClientId=cli):
            if int(hit[0]) not in ignore:
                return False
        return True

    def _target_waypoint(self, env, tpos: np.ndarray) -> tuple:
        """Advance the waypoint state machine. Returns (anchor, moving): the point
        to track and whether to fly toward it (False = hover at it)."""
        wp = env._office_target_wp
        if wp is not None:
            if env._office_target_pause > 0.0:
                env._office_target_pause -= env._sim_dt
                if env._office_target_pause <= 0.0:
                    env._office_target_wp = None  # pick a fresh leg next step
                return wp, False
            if float(np.linalg.norm(tpos - wp)) < OFFICE_TARGET_ARRIVE_M:
                env._office_target_pause = env._office_target_rng.uniform(
                    OFFICE_TARGET_PAUSE_MIN_SEC, OFFICE_TARGET_PAUSE_MAX_SEC
                )
                return wp, False
            return wp, True
        rng = env._office_target_rng
        for _ in range(64):
            cand = np.array(
                office_point(rng, (OFFICE_TARGET_ALT_MIN_M, OFFICE_TARGET_ALT_MAX_M)),
                dtype=float,
            )
            if float(np.linalg.norm(cand - tpos)) < OFFICE_TARGET_MIN_LEG_M:
                continue
            if self._leg_is_clear(env, tpos, cand):
                env._office_target_wp = cand
                return cand, True
        return tpos.copy(), False  # boxed in this step: hover and retry next step

    def advance_world(self, env) -> None:
        uid = getattr(env, "_office_target_uid", None)
        if uid is None or getattr(env, "_office_target_ctrl", None) is None:
            return
        cli = env.CLIENT
        dt = env._sim_dt
        tpos, tquat = p.getBasePositionAndOrientation(uid, physicsClientId=cli)
        tvel, tang = p.getBaseVelocity(uid, physicsClientId=cli)
        tpos = np.asarray(tpos, dtype=float)
        env._office_target_step += 1
        anchor, moving = self._target_waypoint(env, tpos)
        # In-flight reflexes: legs are planned clear, but the PID can drift off the
        # line, so the flight is re-checked from where the drone ACTUALLY is.
        if moving and env._office_target_step % _RECHECK_EVERY_STEPS == 0:
            if not self._leg_is_clear(env, tpos, anchor, ignore_chaser=True):
                env._office_target_wp = None  # reroute smoothly, no stop
                anchor, moving = self._target_waypoint(env, tpos)
        if moving and env._office_target_step % _GUARD_EVERY_STEPS == 0:
            direction = anchor - tpos
            n = float(np.linalg.norm(direction))
            ahead = tpos + direction / n * min(_GUARD_RANGE_M, n) if n > 1e-6 else anchor
            if not self._leg_is_clear(env, tpos, ahead, ignore_chaser=True):
                # Something inside braking distance: stop now, replan next step.
                env._office_target_wp = None
                env._office_target_brakes += 1
                anchor, moving = tpos.copy(), False
        if moving:
            direction = anchor - tpos
            n = float(np.linalg.norm(direction))
            desired = direction / n * min(OFFICE_TARGET_SPEED, n / dt) if n > 1e-6 else np.zeros(3)
            look = tpos + desired * dt * 5.0
        else:
            desired, look = np.zeros(3), anchor
        rpm, _, _ = env._office_target_ctrl.computeControl(
            control_timestep=dt, cur_pos=tpos, cur_quat=np.asarray(tquat, dtype=float),
            cur_vel=np.asarray(tvel, dtype=float), cur_ang_vel=np.asarray(tang, dtype=float),
            target_pos=look, target_vel=desired,
        )
        # Thrust changes at control rate; cache plain floats for the 5x-rate substep hook.
        forces = np.square(np.asarray(rpm, dtype=float)) * float(env.KF)
        torques = np.square(np.asarray(rpm, dtype=float)) * float(env.KM)
        env._office_target_forces = [float(f) for f in forces]
        env._office_target_ztorque = float(-torques[0] + torques[1] - torques[2] + torques[3])

    # ------------------------------------------------------------------ #
    # post-physics: catch on real contact, target self-crash, telemetry
    # ------------------------------------------------------------------ #
    def _update_target(self, env) -> None:
        uid = getattr(env, "_office_target_uid", None)
        if uid is None:
            return
        cli = env.CLIENT
        tpos, _ = p.getBasePositionAndOrientation(uid, physicsClientId=cli)
        env._office_target_pos = np.asarray(tpos, dtype=float)

        if env._success:
            return
        chaser_uid = int(env.DRONE_IDS[0])
        contacts = p.getContactPoints(bodyA=uid, physicsClientId=cli)
        others = [(int(c[2]) if int(c[1]) == uid else int(c[1]), c[9]) for c in contacts]
        dist = float(np.linalg.norm(env.pos[0] - env._office_target_pos))
        if dist <= OFFICE_KILL_RADIUS_M or any(o == chaser_uid for o, _ in others):
            env._success = True
            env._t_to_goal = env._time_alive
            env._failure_reason = FailureReason.NONE.value
            return
        if not env._office_target_crashed:
            for other, force in others:
                if other != chaser_uid and force > OFFICE_TARGET_SELFCRASH_FORCE:
                    env._office_target_crashed = True
                    if env._failure_reason == FailureReason.NONE.value:
                        env._failure_reason = FailureReason.INFEASIBLE.value
                    break

    def post_step_update(self, env) -> None:
        self._update_target(env)
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
        accel = (vel - env._office_prev_vel[d]) / dt
        accel[2] += float(env.G)  # specific force, not linear accel: hover is not zero
        roll, pitch, yaw = (float(v) for v in env.rpy[d])
        vf, vr, vu = world_to_body(vel, yaw)
        af, ar, au = world_to_body(accel, yaw)
        bias = env._office_accel_bias[d]  # sensor-fixed, so applied in the body frame
        tof = min(float(env._get_altitude_distance(d)), OFFICE_TELEM_TOF_MAX_M)
        height = float(env.pos[d, 2]) - float(env._office_takeoff_z[d])
        baro = height + float(env._office_baro_walk[d])
        return np.array([pitch, roll, yaw, vf, vr, -vu,
                         af + bias[0], ar + bias[1], -au + bias[2], tof, height, baro])

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
        cli = env.CLIENT
        force = getattr(env, "_office_vps_force", None)
        if force is not None:
            for d in range(env.NUM_DRONES):
                p.applyExternalForce(
                    int(env.DRONE_IDS[d]), -1, force, env.pos[d].tolist(),
                    p.WORLD_FRAME, physicsClientId=cli,
                )
        uid = getattr(env, "_office_target_uid", None)
        if uid is None or getattr(env, "PHYSICS", None) == Physics.DYN:
            return
        forces = env._office_target_forces
        for i in range(4):
            p.applyExternalForce(uid, i, [0.0, 0.0, forces[i]], [0.0, 0.0, 0.0],
                                 p.LINK_FRAME, physicsClientId=cli)
        p.applyExternalTorque(uid, 4, [0.0, 0.0, env._office_target_ztorque],
                              p.LINK_FRAME, physicsClientId=cli)

    # ------------------------------------------------------------------ #
    # scoring
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
            "min_clearance": None if min_clearance is None else float(min_clearance),
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
