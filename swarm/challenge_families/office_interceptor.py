"""Office interceptor challenge family (indoor Tello, body-frame RC actions).

The miner flies a Tello-class drone inside the fixed office map and must
intercept a target drone. Actions are the four Tello RC sticks
[lr, fb, ud, yaw]; there is no GPS-style world-frame control.

The validator flies the target: a second Tello whose personality — cruise
speed, pause habits, leg style, and on most seeds a spook range it flees
from — is dealt by the seed. The catch is a real physical hit —
chaser-target contact ends the episode as a success. The miner senses the
world through the physical rig's channels — SDK telemetry packets, an
emulated YOLO detector, and a domain-randomized RGB camera — all seeded, all
bit-identical across validators. One deliberate deviation: the target is
invisible to camera and ToF, so the detector is the only sighting channel.
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
    ALTITUDE_RAY_INSET,
    CAMERA_EYE_FWD_M,
    CAMERA_EYE_UP_M,
    DRONE_HULL_RADIUS,
    MAX_RAY_DISTANCE,
    OFFICE_DRAG_COEF,
    OFFICE_GROUND_EFFECT_COEF,
    OFFICE_PROP_RADIUS_M,
    OFFICE_CHALLENGE_TYPE,
    OFFICE_DET_CONF_FLOOR,
    OFFICE_DET_DELAY_STEPS,
    OFFICE_DET_FP_RATE,
    OFFICE_DET_JITTER_SIZE,
    OFFICE_DET_MAX_BOXES,
    OFFICE_DET_MISS_PERSIST,
    OFFICE_DET_PERIOD_STEPS,
    OFFICE_DET_RECALL,
    OFFICE_DET_SEED_OFFSET,
    OFFICE_DET_STALE_SEC,
    OFFICE_DRIFT_SEED_OFFSET,
    OFFICE_KILL_RADIUS_M,
    OFFICE_MAX_START_DISTANCE_M,
    OFFICE_MAX_TILT_DEG,
    OFFICE_MIN_START_DISTANCE_M,
    OFFICE_RC_SPEED,
    OFFICE_TARGET_ACCEL,
    OFFICE_TARGET_ALT_MAX_M,
    OFFICE_TARGET_ALT_MIN_M,
    OFFICE_TARGET_ARRIVE_M,
    OFFICE_TARGET_AWARE_PROB,
    OFFICE_TARGET_BRAKE_DECEL,
    OFFICE_TARGET_CLEAR_M,
    OFFICE_TARGET_DODGE_REPLAN_STEPS,
    OFFICE_TARGET_FLEE_MAX,
    OFFICE_TARGET_FLEE_MIN,
    OFFICE_TARGET_GUARD_SAFETY,
    OFFICE_TARGET_LEG_MIN_HIGH,
    OFFICE_TARGET_LEG_MIN_LOW,
    OFFICE_TARGET_PAUSE_MAX_SEC,
    OFFICE_TARGET_PAUSE_MIN_SEC,
    OFFICE_TARGET_REACT_MAX_M,
    OFFICE_TARGET_REACT_MIN_M,
    OFFICE_TARGET_SPEED_MAX,
    OFFICE_TARGET_SPEED_MIN,
    OFFICE_TARGET_TURN_SPEED,
    OFFICE_APPEARANCE_SEED_OFFSET,
    OFFICE_CAMERA_RES,
    OFFICE_RGB_BRIGHT_HIGH,
    OFFICE_RGB_BRIGHT_LOW,
    OFFICE_TARGET_SEED_OFFSET,
    OFFICE_TARGET_SELFCRASH_FORCE,
    OFFICE_TINT_LOW,
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
from swarm.core.maps.office import (
    OFFICE_CEILING_M,
    OFFICE_X_RANGE,
    OFFICE_Y_RANGE,
    build_office_map,
)
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

# Target reflexes: the guard range grows as v^2/2a, the full-leg recheck reroutes early.
_GUARD_EVERY_STEPS = 2
_RECHECK_EVERY_STEPS = 10

# Placement navigation grid (fixed map => computed once per process).
_NAV_PITCH = 0.45
_NAV_Z_LEVELS = (1.4, 2.0)
_NAV_MAIN = None

# Waypoint sampling: flee candidates first, then any clear exit so a cornered target moves.
_WP_FLEE_TRIES = 64
_WP_ANY_TRIES = 32

# Physical target dimensions: box sizes fall out of these and the camera focal.
_DET_TARGET_W_M = 0.18   # Tello body width seen by the camera
_DET_TARGET_H_M = 0.08   # body + guards height
# Detection block layout: [n_boxes, age, (cx, cy, w, h, conf) x MAX_BOXES], normalized.
_DET_DIM = 2 + 5 * OFFICE_DET_MAX_BOXES
# Placeholder bin shapes (visibility, distance, edge) until the calibration
# recordings; only the marginal recall/precision are measured numbers. Good
# conditions carry no penalty so the documented marginal actually emerges.
_DET_VIS_BINS = ((0.7, 1.0), (0.3, 0.6), (0.0, 0.15))   # (min visible fraction, factor)
_DET_DIST_BINS = ((8.0, 1.0), (99.0, 0.8))               # (max distance m, factor)
_DET_EDGE_MARGIN = 0.1   # the outer frame band where detection weakens...
_DET_EDGE_FACTOR = 0.85  # ...by this factor
_DET_CONF_TOP = 0.97     # ceiling of sampled confidences
_DET_FP_ANCHOR_P = 0.7   # false positives favor fixed scene spots, like real YOLO ghosts
_DET_CENTER_JITTER_FRAC = 0.15  # center noise as a fraction of box size, like real YOLO


def _det_marginal_to_eval_p(marginal: float, persist: float) -> float:
    """Per-evaluation detect probability whose miss-streak chain has the given
    stationary detection rate (misses persist outright with prob `persist`)."""
    miss = 1.0 - marginal
    return 1.0 - miss * (1.0 - persist) / (1.0 - miss * persist)


def _det_true_conf(rng, vis: float, w_px: float) -> float:
    """Confidence degrades with visibility AND apparent size: small far boxes
    sink toward the rig's threshold, overlapping the false-positive range.
    The size scale matches the obs camera (~128 px focal): a box under ~8 px
    is a hard detection."""
    quality = (0.3 + 0.7 * vis) * float(np.clip(w_px / 8.0, 0.2, 1.0))
    return OFFICE_DET_CONF_FLOOR + (_DET_CONF_TOP - OFFICE_DET_CONF_FLOOR) * (
        quality * rng.uniform(0.6, 1.0)
    )


def _det_fp_box(rng, anchors, focal: float) -> tuple:
    """A drone-plausible ghost box in the obs camera frame: sized like the
    target at a fake distance, usually at a persistent anchor spot."""
    res = float(OFFICE_CAMERA_RES)
    if anchors and rng.random() < _DET_FP_ANCHOR_P:
        u, v = anchors[int(rng.integers(len(anchors)))]
        cx = u + rng.normal(0.0, 0.015 * res)
        cy = v + rng.normal(0.0, 0.012 * res)
    else:
        cx = rng.uniform(0.0, res)
        cy = rng.uniform(0.0, res)
    fake_dist = rng.uniform(2.0, 12.0)
    w = abs(focal * _DET_TARGET_W_M / fake_dist * (1.0 + rng.normal(0.0, 0.2)))
    h = abs(focal * _DET_TARGET_H_M / fake_dist * (1.0 + rng.normal(0.0, 0.2)))
    conf = OFFICE_DET_CONF_FLOOR + (0.75 - OFFICE_DET_CONF_FLOOR) * rng.random() ** 2
    return cx, cy, w, h, conf

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


def _brake_guard_range(speed: float, dt: float) -> float:
    """Guard distance: safety x v^2/2a stopping distance + travel between checks."""
    return (OFFICE_TARGET_GUARD_SAFETY * speed * speed
            / (2.0 * OFFICE_TARGET_BRAKE_DECEL) + speed * _GUARD_EVERY_STEPS * dt)


def office_target_profile(map_seed: int) -> dict:
    """The seed deals the target its personality for the episode. Pure function of
    the seed so the reward side can rebuild it without an env; own salt so the
    draw is independent of the flight and placement streams."""
    rng = random.Random((int(map_seed) ^ OFFICE_TARGET_SEED_OFFSET ^ 0x9E12) & 0xFFFFFFFF)
    aware = rng.random() < OFFICE_TARGET_AWARE_PROB
    return {
        "cruise": rng.uniform(OFFICE_TARGET_SPEED_MIN, OFFICE_TARGET_SPEED_MAX),
        "pause_prob": rng.uniform(0.0, 1.0),
        "leg_min": rng.uniform(OFFICE_TARGET_LEG_MIN_LOW, OFFICE_TARGET_LEG_MIN_HIGH),
        "react_range": rng.uniform(OFFICE_TARGET_REACT_MIN_M, OFFICE_TARGET_REACT_MAX_M)
        if aware else 0.0,
        "flee_frac": rng.uniform(OFFICE_TARGET_FLEE_MIN, OFFICE_TARGET_FLEE_MAX),
    }


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
            # RGB rendering makes this the heaviest family per step (~91 ms);
            # budgets mirror the outdoor interceptor's, which shares the horizon.
            batch_timeout_multiplier=1.3,
            global_eval_base_sec=600.0,
            global_eval_per_seed_sec=300.0,
            global_eval_cap_sec=3600.0,
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
        env._office_map_uids = ()
        # Safe default so the obs path works before the first spawn.
        env._office_rgb_bright = 1.0

    def reset_env_state(self, env) -> None:
        n = env.NUM_DRONES
        seed = int(getattr(env.task, "map_seed", 0))
        env._office_target_rng = random.Random((seed ^ OFFICE_TARGET_SEED_OFFSET) & 0xFFFFFFFF)
        env._office_target_profile = office_target_profile(seed)
        env._office_target_last_dodge = -(10 ** 9)
        env._office_target_leg_step = 0
        env._office_target_wp = None
        env._office_target_pause = 0.0
        env._office_target_forces = [0.0, 0.0, 0.0, 0.0]
        env._office_target_ztorque = 0.0
        env._office_target_pos = np.zeros(3, dtype=float)
        env._office_target_step = 0
        env._office_target_brakes = 0
        env._office_target_crashed = False
        env._collision_exempt_uids = frozenset()
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
        env._office_detection = np.zeros(_DET_DIM, dtype=np.float32)
        # No frame seen yet: stale from the first step, like the real rig booting.
        env._office_detection[1] = 2.0 * OFFICE_DET_STALE_SEC
        env._office_det_rng = np.random.default_rng(seed ^ OFFICE_DET_SEED_OFFSET)
        env._office_det_pending = None
        env._office_det_missed = False
        env._office_rgb_capture = None  # the held video frame never crosses episodes
        env._min_clearance_episode = None  # clearance is never scanned here (audit: None)
        # The one camera's focal length (px); env._fov is fixed per env at __init__.
        env._office_det_focal = (float(OFFICE_CAMERA_RES) / 2.0) / math.tan(
            math.radians(float(env._fov)) / 2.0
        )
        env._office_det_fp_anchors = [
            (env._office_det_rng.uniform(0.0, float(OFFICE_CAMERA_RES)),
             env._office_det_rng.uniform(0.0, float(OFFICE_CAMERA_RES)))
            for _ in range(2)
        ]
        # Own stream: the drift direction must not shift when telemetry draws change.
        drift_rng = np.random.default_rng(seed ^ OFFICE_DRIFT_SEED_OFFSET)
        angle = drift_rng.uniform(0.0, 2.0 * math.pi)
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
        froms = [origin.tolist()] * len(dirs)
        if floor:
            # Drone-width up-ray ring: the whole takeoff column must clear overhangs.
            for ox, oy in ((0, 0), (0.15, 0), (-0.15, 0), (0, 0.15), (0, -0.15)):
                froms.append([origin[0] + ox, origin[1] + oy, origin[2]])
                dirs.append((0, 0, 1.2))
            # Low ring at body height: furniture feet sit below the main rays.
            low = pos + np.array([0.0, 0.0, 0.03])
            for d in ((0.4, 0, 0), (-0.4, 0, 0), (0, 0.4, 0), (0, -0.4, 0)):
                froms.append(low.tolist())
                dirs.append(d)
            tos = [(np.asarray(f) + np.asarray(d)).tolist() for f, d in zip(froms, dirs)]
            if not self._rays_clear(env, froms, tos,
                                    extra_ignore=(int(env.DRONE_IDS[0]),)):
                return False
            # Rays cannot see a flat floor object under the point; probe with the body.
            cli = env.CLIENT
            cid = int(env.DRONE_IDS[0])
            keep_pos, keep_quat = p.getBasePositionAndOrientation(cid, physicsClientId=cli)
            p.resetBasePositionAndOrientation(cid, pos.tolist(), keep_quat,
                                              physicsClientId=cli)
            touching = any(
                p.getClosestPoints(cid, uid, 0.02, physicsClientId=cli)
                for uid in getattr(env, "_office_map_uids", ())
            )
            p.resetBasePositionAndOrientation(cid, keep_pos, keep_quat,
                                              physicsClientId=cli)
            return not touching
        dirs += [(0, 0, 0.4), (0, 0, -0.4)]
        froms = [origin.tolist()] * len(dirs)
        tos = [(origin + np.asarray(d)).tolist() for d in dirs]
        return self._rays_clear(env, froms, tos, extra_ignore=(int(env.DRONE_IDS[0]),))

    def _nav_main_component(self, env) -> tuple:
        """Grid the flight volume once and label its connected components with the
        family's own leg clearance. The office is fixed geometry, so the result is
        cached for the process and identical on every validator. Placement uses it
        to keep tasks out of sealed rooms (the office has a closed corridor)."""
        global _NAV_MAIN
        if _NAV_MAIN is not None:
            return _NAV_MAIN
        xs = np.arange(OFFICE_X_RANGE[0] + 0.5, OFFICE_X_RANGE[1] - 0.5, _NAV_PITCH)
        ys = np.arange(OFFICE_Y_RANGE[0] + 0.5, OFFICE_Y_RANGE[1] - 0.5, _NAV_PITCH)
        cells = [(ix, iy, iz) for ix in range(len(xs)) for iy in range(len(ys))
                 for iz in range(len(_NAV_Z_LEVELS))]
        index = {c: i for i, c in enumerate(cells)}
        nodes = [(float(xs[ix]), float(ys[iy]), float(_NAV_Z_LEVELS[iz]))
                 for ix, iy, iz in cells]
        parent = list(range(len(cells)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for (ix, iy, iz), i in index.items():
            for nb in ((ix + 1, iy, iz), (ix, iy + 1, iz), (ix, iy, iz + 1)):
                j = index.get(nb)
                if j is not None and self._leg_is_clear(
                        env, np.array(nodes[i]), np.array(nodes[j]), ignore_chaser=True):
                    parent[find(i)] = find(j)
        comps: dict = {}
        for i in range(len(nodes)):
            comps.setdefault(find(i), []).append(i)
        main = max(comps.values(), key=len)
        _NAV_MAIN = (nodes, frozenset(main))
        return _NAV_MAIN

    def _nav_in_main(self, env, pos: np.ndarray) -> bool:
        """True when a clear leg connects pos to a main-component nav node
        among the nearest few grid cells."""
        nodes, main = self._nav_main_component(env)
        order = sorted(range(len(nodes)),
                       key=lambda i: float(np.linalg.norm(np.array(nodes[i]) - pos)))
        for i in order[:8]:
            if i in main and self._leg_is_clear(env, pos, np.array(nodes[i]),
                                                ignore_chaser=True):
                return True
        return False

    def _clear_spawn(self, env, pos: np.ndarray, *, floor: bool, rng,
                     anchor: np.ndarray) -> np.ndarray:
        """First clear point in the main open component, preferring candidates that
        keep the task's start-goal separation band; falls back to any clear point."""
        z_range = (pos[2], pos[2]) if floor else (OFFICE_TARGET_ALT_MIN_M, OFFICE_TARGET_ALT_MAX_M)
        fallback = None
        for i in range(65):
            if self._point_is_clear(env, pos, floor=floor):
                # A drone placed in a sealed room makes the task impossible.
                probe = np.array([pos[0], pos[1], _NAV_Z_LEVELS[0]]) if floor else pos
                if self._nav_in_main(env, probe):
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
        # The detector is the only sighting channel: the target never appears on camera.
        for shape in p.getVisualShapeData(uid, physicsClientId=cli):
            p.changeVisualShape(uid, shape[1], rgbaColor=[0, 0, 0, 0],
                                physicsClientId=cli)
        # Ramming the target is the catch, not a chaser crash.
        env._collision_exempt_uids = frozenset({uid})

        map_info = build_office_map(seed=seed, cli=cli)
        # Spawn-probe obstacles: every map body except the floor the drone rests on.
        env._office_map_uids = tuple(int(u) for name, u in sorted(map_info["bodies"].items())
                                     if name != "floor") + (int(map_info["window_plug"]),)
        self._apply_appearance(env, map_info, seed)

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

    def _apply_appearance(self, env, map_info: dict, seed: int) -> None:
        """Per-episode seeded skin: the policy must learn geometry, not color.
        Draw order is fixed — reordering changes every episode's look."""
        rng = random.Random((seed ^ OFFICE_APPEARANCE_SEED_OFFSET) & 0xFFFFFFFF)
        cli = env.CLIENT
        plane = getattr(env, "PLANE_ID", None)
        if plane is not None:
            # The gym plane z-fights the office floor in color renders.
            p.changeVisualShape(plane, -1, rgbaColor=[0, 0, 0, 0], physicsClientId=cli)
        for name in sorted(map_info["bodies"]):
            tint = [rng.uniform(OFFICE_TINT_LOW, 1.0) for _ in range(3)] + [1.0]
            p.changeVisualShape(map_info["bodies"][name], -1, rgbaColor=tint,
                                physicsClientId=cli)
        env._light_direction = [rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0),
                                rng.uniform(0.35, 1.0)]
        env._light_color = [rng.uniform(0.55, 1.0) for _ in range(3)]
        env._office_rgb_bright = rng.uniform(OFFICE_RGB_BRIGHT_LOW, OFFICE_RGB_BRIGHT_HIGH)

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
    # target flight: seeded personality — waypoint legs, pauses, spook response
    # ------------------------------------------------------------------ #
    def _leg_is_clear(self, env, start: np.ndarray, end: np.ndarray,
                      ignore_chaser: bool = False) -> bool:
        """Ray-check the leg with side offsets so the body radius clears too.

        At planning time the chaser counts as an obstacle: a person would steer
        around a parked drone, and it stops a park-in-the-flight-path free catch.
        The in-flight guards ignore the chaser — escape behavior lives in the
        seeded spook response, never in the clearance checks, so parking in the
        flight path stays a valid catch tactic."""
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

    def _target_waypoint(self, env, tpos: np.ndarray,
                         flee_from: np.ndarray | None = None) -> tuple:
        """Advance the waypoint state machine. Returns (anchor, moving): the point
        to track and whether to fly toward it (False = hover at it). A spooked
        target passes flee_from: legs away from it are preferred, but a cornered
        target takes any clear exit rather than freezing."""
        profile = env._office_target_profile
        wp = env._office_target_wp
        if wp is not None:
            if env._office_target_pause > 0.0:
                env._office_target_pause -= env._sim_dt
                if env._office_target_pause <= 0.0:
                    env._office_target_wp = None  # pick a fresh leg next step
                return wp, False
            if float(np.linalg.norm(tpos - wp)) < OFFICE_TARGET_ARRIVE_M:
                if env._office_target_rng.random() < profile["pause_prob"]:
                    env._office_target_pause = env._office_target_rng.uniform(
                        OFFICE_TARGET_PAUSE_MIN_SEC, OFFICE_TARGET_PAUSE_MAX_SEC
                    )
                    return wp, False
                env._office_target_wp = None  # restless pilot: straight to a new leg
            else:
                return wp, True
        rng = env._office_target_rng
        for i in range(_WP_FLEE_TRIES + _WP_ANY_TRIES):
            cand = np.array(
                office_point(rng, (OFFICE_TARGET_ALT_MIN_M, OFFICE_TARGET_ALT_MAX_M)),
                dtype=float,
            )
            if float(np.linalg.norm(cand - tpos)) < profile["leg_min"]:
                continue
            if flee_from is not None and i < _WP_FLEE_TRIES:
                if float(np.dot(cand[:2] - tpos[:2], tpos[:2] - flee_from[:2])) <= 0.0:
                    continue
            if self._leg_is_clear(env, tpos, cand):
                env._office_target_wp = cand
                env._office_target_leg_step = env._office_target_step
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
        profile = env._office_target_profile
        cpos = np.asarray(env.pos[0], dtype=float)
        spooked = (profile["react_range"] > 0.0
                   and float(np.linalg.norm(cpos - tpos)) < profile["react_range"])
        # A spooked brisk pilot never slows down to flee.
        speed = (max(profile["flee_frac"] * OFFICE_RC_SPEED, profile["cruise"])
                 if spooked else profile["cruise"])
        flee_from = cpos if spooked else None
        if spooked and (env._office_target_step - env._office_target_last_dodge
                        >= OFFICE_TARGET_DODGE_REPLAN_STEPS):
            wp = env._office_target_wp
            # Rebuilding a still-good flee leg would reset the ramp and cap the flee speed.
            if (env._office_target_pause > 0.0 or wp is None
                    or float(np.dot(wp[:2] - tpos[:2], tpos[:2] - cpos[:2])) <= 0.0):
                # Spooked pilot: drop any hover and bolt away from the chaser.
                env._office_target_last_dodge = env._office_target_step
                env._office_target_pause = 0.0
                env._office_target_wp = None
        anchor, moving = self._target_waypoint(env, tpos, flee_from)
        # Corner ramp: slow into turns, keep the speed component already down the leg.
        leg_age = (env._office_target_step - env._office_target_leg_step) * dt
        leg_vec = anchor - tpos
        n0 = float(np.linalg.norm(leg_vec))
        carry = max(0.0, float(np.dot(np.asarray(tvel, dtype=float),
                                      leg_vec / n0))) if n0 > 1e-6 else 0.0
        speed = min(speed, max(OFFICE_TARGET_TURN_SPEED, carry)
                    + OFFICE_TARGET_ACCEL * leg_age)
        # In-flight reflexes: legs are planned clear, but the PID can drift off the
        # line, so the flight is re-checked from where the drone ACTUALLY is.
        if moving and env._office_target_step % _RECHECK_EVERY_STEPS == 0:
            if not self._leg_is_clear(env, tpos, anchor, ignore_chaser=True):
                env._office_target_wp = None  # reroute smoothly, no stop
                anchor, moving = self._target_waypoint(env, tpos, flee_from)
        if moving and env._office_target_step % _GUARD_EVERY_STEPS == 0:
            guard = _brake_guard_range(speed, dt)
            direction = anchor - tpos
            n = float(np.linalg.norm(direction))
            ahead = tpos + direction / n * min(guard, n) if n > 1e-6 else anchor
            blocked = not self._leg_is_clear(env, tpos, ahead, ignore_chaser=True)
            # Guard along actual motion too: a clear plan can still clip a door frame.
            v = np.asarray(tvel, dtype=float)
            vn = float(np.linalg.norm(v))
            if not blocked and vn > 0.2:
                blocked = not self._leg_is_clear(
                    env, tpos, tpos + v / vn * _brake_guard_range(vn, dt),
                    ignore_chaser=True)
            if blocked:
                # Something inside braking distance: stop now, replan next step.
                env._office_target_wp = None
                env._office_target_brakes += 1
                anchor, moving = tpos.copy(), False
        if moving:
            direction = anchor - tpos
            n = float(np.linalg.norm(direction))
            desired = direction / n * min(speed, n / dt) if n > 1e-6 else np.zeros(3)
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

    # ------------------------------------------------------------------ #
    # detector emulator: what the YOLO rig would say, nothing more
    # ------------------------------------------------------------------ #
    def _detector_capture(self, env) -> dict | None:
        """Project the target into the SAME camera the rgb observation renders
        through (eye offset, seeded FOV, square frame) and measure visibility.
        One camera, like the real rig: boxes land exactly on obs["rgb"].
        Pure geometry + 5 rays; the frame is never rendered."""
        cpos = np.array(env.pos[0], dtype=float)
        rot = np.array(p.getMatrixFromQuaternion(env.quat[0])).reshape(3, 3)
        fwd, left, up = rot[:, 0], rot[:, 1], rot[:, 2]
        eye = cpos + fwd * CAMERA_EYE_FWD_M + up * CAMERA_EYE_UP_M
        rel = env._office_target_pos - eye
        depth = float(np.dot(rel, fwd))
        if depth < 0.2:
            return None
        res = float(OFFICE_CAMERA_RES)
        focal = env._office_det_focal
        px = focal * float(np.dot(rel, -left)) / depth + res / 2.0
        py = focal * float(np.dot(rel, -up)) / depth + res / 2.0
        if not (0.0 <= px <= res and 0.0 <= py <= res):
            return None
        w_px = focal * _DET_TARGET_W_M / depth
        h_px = focal * _DET_TARGET_H_M / depth
        tpos = env._office_target_pos
        offs = (np.zeros(3), 0.09 * -left, 0.09 * left, 0.045 * up, 0.045 * -up)
        froms = [eye.tolist()] * len(offs)
        tos = [(tpos + off).tolist() for off in offs]
        ignore = {int(env.DRONE_IDS[0]), int(env._office_target_uid), -1}
        hits = p.rayTestBatch(froms, tos, physicsClientId=env.CLIENT)
        vis = sum(1 for h in hits if int(h[0]) in ignore) / len(hits)
        if vis == 0.0:
            return None
        return {"px": px, "py": py, "w": w_px, "h": h_px, "dist": depth, "vis": vis}

    def _detector_deliver(self, env, rng, dt: float) -> None:
        det = env._office_detection
        det[2:] = 0.0
        truth = env._office_det_pending
        boxes = []
        if truth is None:
            env._office_det_missed = False  # a streak must not survive a visibility gap
        else:
            if env._office_det_missed and rng.random() < OFFICE_DET_MISS_PERSIST:
                detected = False  # real detectors lose a target for streaks of frames
            else:
                marginal = OFFICE_DET_RECALL
                for lo, factor in _DET_VIS_BINS:
                    if truth["vis"] >= lo:
                        marginal *= factor
                        break
                for hi, factor in _DET_DIST_BINS:
                    if truth["dist"] <= hi:
                        marginal *= factor
                        break
                res = float(OFFICE_CAMERA_RES)
                margin_x = min(truth["px"], res - truth["px"]) / res
                margin_y = min(truth["py"], res - truth["py"]) / res
                if min(margin_x, margin_y) < _DET_EDGE_MARGIN:
                    marginal *= _DET_EDGE_FACTOR
                # Convert the target marginal to a per-eval p the streak chain lands on.
                prob = _det_marginal_to_eval_p(marginal, OFFICE_DET_MISS_PERSIST)
                detected = rng.random() < prob
            env._office_det_missed = not detected
            if detected:
                cx = truth["px"] + rng.normal(0.0, _DET_CENTER_JITTER_FRAC * truth["w"])
                cy = truth["py"] + rng.normal(0.0, _DET_CENTER_JITTER_FRAC * truth["h"])
                w = truth["w"] * (1.0 + rng.normal(0.0, OFFICE_DET_JITTER_SIZE))
                h = truth["h"] * (1.0 + rng.normal(0.0, OFFICE_DET_JITTER_SIZE))
                conf = _det_true_conf(rng, truth["vis"], truth["w"])
                boxes.append((cx, cy, w, h, conf))
        if rng.random() < OFFICE_DET_FP_RATE:
            boxes.append(_det_fp_box(rng, env._office_det_fp_anchors, env._office_det_focal))
        # The real rig sorts by confidence; slot order must not reveal which is real.
        boxes.sort(key=lambda b: b[4], reverse=True)
        boxes = boxes[:OFFICE_DET_MAX_BOXES]
        res = float(OFFICE_CAMERA_RES)
        for i, (cx, cy, w, h, conf) in enumerate(boxes):
            base = 2 + 5 * i
            det[base] = np.clip(cx / res, 0.0, 1.0)
            det[base + 1] = np.clip(cy / res, 0.0, 1.0)
            det[base + 2] = np.clip(w / res, 0.0, 1.0)
            det[base + 3] = np.clip(h / res, 0.0, 1.0)
            det[base + 4] = conf
        det[0] = float(len(boxes))
        if boxes:
            det[1] = OFFICE_DET_DELAY_STEPS * dt

    def post_step_update(self, env) -> None:
        # A full-speed drone can tunnel through the thin ceiling sheet between
        # substeps; leaving the room is hitting the shell, whatever physics missed.
        pos = env.pos[0]
        if (not env._collision
                and (pos[2] > OFFICE_CEILING_M + 0.3 or pos[2] < -0.3
                     or not (OFFICE_X_RANGE[0] - 0.3 < pos[0] < OFFICE_X_RANGE[1] + 0.3)
                     or not (OFFICE_Y_RANGE[0] - 0.3 < pos[1] < OFFICE_Y_RANGE[1] + 0.3))):
            env._collision = True
            if env._failure_reason == FailureReason.NONE.value:
                env._failure_reason = FailureReason.OBSTACLE_COLLISION.value
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

        det = env._office_detection
        det[1] = min(det[1] + dt, 2.0 * OFFICE_DET_STALE_SEC)
        if (step + OFFICE_DET_DELAY_STEPS) % OFFICE_DET_PERIOD_STEPS == 0:
            env._office_det_pending = self._detector_capture(env)
        if step % OFFICE_DET_PERIOD_STEPS == 0:
            self._detector_deliver(env, env._office_det_rng, dt)

    def _office_tof(self, env, d: int) -> float:
        """Downward ToF that never returns the target — the detector must stay
        the only sighting channel, so a target underneath reads as the floor.
        Same geometry and body-center distance as _get_altitude_distance."""
        cli = env.CLIENT
        pos = env.pos[d]
        offset = getattr(env, "_alt_ray_origin_offset",
                         DRONE_HULL_RADIUS - ALTITUDE_RAY_INSET)
        start = [float(pos[0]), float(pos[1]), float(pos[2]) - offset]
        end = [float(pos[0]), float(pos[1]), float(pos[2]) - MAX_RAY_DISTANCE]
        hit_uid, _, hit_frac, hit_pos, _ = p.rayTest(start, end, physicsClientId=cli)[0]
        tuid = getattr(env, "_office_target_uid", None)
        if tuid is None or int(hit_uid) != int(tuid):
            # Bit-identical to _get_altitude_distance when no target is in the ray.
            if int(hit_uid) != -1:
                seg_len = MAX_RAY_DISTANCE - offset
                return min(MAX_RAY_DISTANCE, offset + float(hit_frac) * seg_len)
            return MAX_RAY_DISTANCE
        # Re-cast from below the target's live AABB: at any attitude the ray
        # cannot hit it again, so the reading is what the floor would give.
        start[2] = p.getAABB(int(tuid), physicsClientId=cli)[0][2] - 1e-3
        hit_uid, _, _, hit_pos, _ = p.rayTest(start, end, physicsClientId=cli)[0]
        if int(hit_uid) == -1:
            return MAX_RAY_DISTANCE
        return min(MAX_RAY_DISTANCE, float(pos[2]) - float(hit_pos[2]))

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
        tof = min(self._office_tof(env, d), OFFICE_TELEM_TOF_MAX_M)
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
        for d in range(env.NUM_DRONES):
            pos = env.pos[d]
            v = np.asarray(env.vel[d], dtype=float)
            vn = float(np.linalg.norm(v))
            if vn > 1e-3:
                # Quadratic body drag; the velocity PID must now fight the air.
                drag = (-OFFICE_DRAG_COEF * vn * v).tolist()
                p.applyExternalForce(int(env.DRONE_IDS[d]), -1, drag,
                                     pos.tolist(), p.WORLD_FRAME, physicsClientId=cli)
            z = max(float(pos[2]), OFFICE_PROP_RADIUS_M)
            if z < 8.0 * OFFICE_PROP_RADIUS_M:
                # Ground cushion: extra lift grows as (prop_r/4z)^2 close to the floor.
                lift = (OFFICE_GROUND_EFFECT_COEF * float(env.M) * float(env.G)
                        * (OFFICE_PROP_RADIUS_M / (4.0 * z)) ** 2)
                p.applyExternalForce(int(env.DRONE_IDS[d]), -1, [0.0, 0.0, lift],
                                     pos.tolist(), p.WORLD_FRAME, physicsClientId=cli)
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
