#!/usr/bin/env python3
"""
Swarm V4 Video Generator
=========================
Renders flight replay videos for a given miner model + seed + map type.

Camera Modes
------------
depth       Drone's onboard depth sensor (128x128, Inferno colormap, upscaled)
fpv         First-person RGB from the drone's nose camera
chase       Cinematic third-person follow camera
overview    Slowly orbiting bird's-eye view centred on the flight path
all         Generate every mode above in a single run

The script is designed for two use-cases:
  1. CLI — run directly to produce .mp4 files on disk.
  2. Library — ``from generate_video import record_flight`` for programmatic
     use inside validator endpoints that serve videos to the frontend.

Examples
--------
    python3 scripts/generate_video.py --model UID_178.zip --seed 42 --type 1 --mode chase
    python3 scripts/generate_video.py --model UID_178.zip --seed 42 --type 5 --mode all --out ./videos
    python3 scripts/generate_video.py --model UID_178.zip --seed 42 --type 1 --mode depth,fpv --width 1920 --height 1080
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import time
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap — allow importing the V4 swarm package from the sibling repo
# without requiring a global install.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ═══════════════════════════════════════════════════════════════════════════════
#  Action log I/O
# ═══════════════════════════════════════════════════════════════════════════════


def _action_log_path(directory: Path, seed: int, challenge_type: int) -> Path:
    label = {1: "city", 2: "open", 3: "mountain", 4: "village", 5: "warehouse", 6: "forest"}.get(
        challenge_type, f"type{challenge_type}"
    )
    return Path(directory) / f"seed{seed}_{label}_actions.json"


def _save_action_log(
    path: Path, seed: int, challenge_type: int, actions: List[List[float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"seed": seed, "challenge_type": challenge_type, "actions": actions}, f)


def _load_action_log(path: Path) -> List[np.ndarray]:
    with open(path, "r") as f:
        data = json.load(f)
    return [np.asarray(a, dtype=np.float32) for a in data["actions"]]


# ═══════════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════════

VALID_MODES: List[str] = ["depth", "fpv", "chase", "overview"]

TYPE_LABELS: Dict[int, str] = {
    1: "city",
    2: "open",
    3: "mountain",
    4: "village",
    5: "warehouse",
    6: "forest",
}
BENCH_GROUP_ORDER: List[str] = [
    "type1_city",
    "type2_open",
    "type3_mountain",
    "type4_village",
    "type5_warehouse",
    "type6_forest",
]
BENCH_GROUP_TO_TYPE: Dict[str, int] = {
    "type1_city": 1,
    "type2_open": 2,
    "type3_mountain": 3,
    "type4_village": 4,
    "type5_warehouse": 5,
    "type6_forest": 6,
}

# --- output defaults ---
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 25

# --- chase camera ---
CHASE_DISTANCE_BACK_M = 2.5
CHASE_HEIGHT_ABOVE_M = 1.0
CHASE_FOV_DEG = 65.0
CHASE_SMOOTHING = 0.92  # exponential smoothing (higher = smoother)

# --- first-person view ---
FPV_FOV_DEG = 90.0
FPV_OFFSET_FORWARD_M = 0.15
FPV_OFFSET_UP_M = 0.02
FPV_SMOOTHING = 0.85

# --- overview ---
OVERVIEW_FOV_DEG = 60.0
OVERVIEW_PITCH_DEG = -35.0
OVERVIEW_ORBIT_DEG_SEC = 5.0  # slow rotation around the scene

# --- depth visualisation ---
DEPTH_SENSOR_RES = 128  # native sensor resolution (square)
DEPTH_COLORMAP_NAME = "inferno"


# ═══════════════════════════════════════════════════════════════════════════════
#  Data classes
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class FlightTask:
    """Lightweight stand-in for ``swarm.protocol.MapTask``.

    We avoid importing the real protocol module because it transitively pulls
    in ``bittensor`` which hijacks ``argparse`` and takes several seconds to
    initialise — neither of which is acceptable for a video tool.
    """

    map_seed: int
    start: Tuple[float, float, float]
    goal: Tuple[float, float, float]
    sim_dt: float
    horizon: float
    challenge_type: int
    search_radius: float = 10.0
    moving_platform: bool = False
    version: str = "1"


@dataclass(frozen=True, slots=True)
class VideoResult:
    """Metadata returned for every successfully written video file."""

    seed: int
    challenge_type: int
    mode: str
    path: str
    frames: int
    duration_sec: float
    success: bool
    sim_time_sec: float
    wall_time_sec: float


@dataclass(frozen=True, slots=True)
class VideoJob:
    seed: int
    challenge_type: int


@dataclass(frozen=True, slots=True)
class BenchmarkExpectation:
    success: bool
    score: float
    sim_time_sec: float


def _outcome_label(success: bool) -> str:
    return "SUCCESS" if bool(success) else "FAILED"


def _summary_tag(*, success: bool, verified: bool) -> str:
    if verified:
        return "MATCH_OK" if bool(success) else "MATCH_FAIL"
    return "OK" if bool(success) else "FAILED"


@contextmanager
def _temporary_env(overrides: Dict[str, Optional[str]]):
    previous = {k: os.environ.get(k) for k in overrides}
    try:
        for key, value in overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# ═══════════════════════════════════════════════════════════════════════════════
#  Task construction  (mirrors swarm.validator.task_gen without bittensor)
# ═══════════════════════════════════════════════════════════════════════════════


def _get_type3_world_range(seed: int) -> float:
    from swarm.constants import TYPE_3_WORLD_RANGE_RATIO
    from swarm.core.mountain_generator import get_global_scale

    gs = get_global_scale(seed)
    return (250.0 * gs) * TYPE_3_WORLD_RANGE_RATIO


def _get_type3_surface_z(x: float, y: float, seed: int) -> float:
    from swarm.core.mountain_generator import get_global_scale, get_terrain_z

    gs = get_global_scale(seed)
    return get_terrain_z(x, y, seed, gs)


def _load_type_params(challenge_type: int, seed: int) -> Dict[str, float]:
    """Return current map-generation parameters for *challenge_type*."""
    from swarm import constants as C

    _MAP = {
        1: dict(
            world_range=C.TYPE_1_WORLD_RANGE,
            r_min=C.TYPE_1_R_MIN,
            r_max=C.TYPE_1_R_MAX,
            h_min=C.TYPE_1_H_MIN,
            h_max=C.TYPE_1_H_MAX,
            start_h_min=C.TYPE_1_START_H_MIN,
            start_h_max=C.TYPE_1_START_H_MAX,
            horizon=C.TYPE_1_HORIZON,
        ),
        2: dict(
            world_range=C.TYPE_2_WORLD_RANGE,
            r_min=C.TYPE_2_R_MIN,
            r_max=C.TYPE_2_R_MAX,
            h_min=C.TYPE_2_H_MIN,
            h_max=C.TYPE_2_H_MAX,
            start_h_min=C.TYPE_2_START_H_MIN,
            start_h_max=C.TYPE_2_START_H_MAX,
            horizon=C.TYPE_2_HORIZON,
        ),
        3: dict(
            world_range=_get_type3_world_range(seed),
            r_min=C.TYPE_3_R_MIN,
            r_max=C.TYPE_3_R_MAX,
            h_min=C.TYPE_3_H_MIN,
            h_max=C.TYPE_3_H_MAX,
            start_h_min=C.TYPE_3_START_H_MIN,
            start_h_max=C.TYPE_3_START_H_MAX,
            horizon=C.TYPE_3_HORIZON,
        ),
        4: dict(
            world_range=C.TYPE_3_VILLAGE_RANGE,
            r_min=C.TYPE_3_R_MIN,
            r_max=C.TYPE_3_R_MAX,
            h_min=C.TYPE_3_H_MIN,
            h_max=C.TYPE_3_H_MAX,
            start_h_min=C.TYPE_3_START_H_MIN,
            start_h_max=C.TYPE_3_START_H_MAX,
            horizon=C.TYPE_3_HORIZON,
        ),
        5: dict(
            world_range_x=C.TYPE_4_WORLD_RANGE_X,
            world_range_y=C.TYPE_4_WORLD_RANGE_Y,
            r_min=C.TYPE_4_R_MIN,
            r_max=C.TYPE_4_R_MAX,
            h_min=C.TYPE_4_H_MIN,
            h_max=C.TYPE_4_H_MAX,
            start_h_min=C.TYPE_4_START_H_MIN,
            start_h_max=C.TYPE_4_START_H_MAX,
            horizon=C.TYPE_4_HORIZON,
        ),
        6: dict(
            world_range=C.TYPE_6_WORLD_RANGE,
            r_min=C.TYPE_6_R_MIN,
            r_max=C.TYPE_6_R_MAX,
            h_min=C.TYPE_6_H_MIN,
            h_max=C.TYPE_6_H_MAX,
            start_h_min=C.TYPE_6_START_H_MIN,
            start_h_max=C.TYPE_6_START_H_MAX,
            horizon=C.TYPE_6_HORIZON,
        ),
    }
    raw = _MAP.get(challenge_type, _MAP[1])
    return {k: float(v) for k, v in raw.items()}


def _sample_start(
    rng, params: Dict[str, float], challenge_type: int, seed: int
) -> Tuple[float, float, float]:
    from swarm import constants as C

    if challenge_type == 5:
        x = rng.uniform(-params["world_range_x"], params["world_range_x"])
        y = rng.uniform(-params["world_range_y"], params["world_range_y"])
        pz = rng.uniform(params["start_h_min"], params["start_h_max"])
        return float(x), float(y), float(pz + float(C.START_PLATFORM_TAKEOFF_BUFFER))

    wr = params["world_range"]
    x = rng.uniform(-wr, wr)
    y = rng.uniform(-wr, wr)
    if challenge_type == 3:
        rng.uniform(0, 1)
        z = _get_type3_surface_z(x, y, seed) + float(C.START_PLATFORM_TAKEOFF_BUFFER)
    elif challenge_type == 4:
        z = float(C.START_PLATFORM_TAKEOFF_BUFFER)
    elif getattr(C, "START_PLATFORM", False):
        if getattr(C, "START_PLATFORM_RANDOMIZE", False):
            pz = rng.uniform(
                float(C.START_PLATFORM_MIN_Z), float(C.START_PLATFORM_MAX_Z)
            )
        else:
            pz = float(C.START_PLATFORM_SURFACE_Z)
        z = pz + float(C.START_PLATFORM_TAKEOFF_BUFFER)
    else:
        z = rng.uniform(params["start_h_min"], params["start_h_max"])
    return float(x), float(y), float(z)


def _sample_goal_warehouse(
    rng, start: Tuple[float, float, float], params: Dict[str, float]
) -> Tuple[float, float, float]:
    sx, sy, _ = start
    wx, wy = params["world_range_x"], params["world_range_y"]
    for _ in range(100):
        angle = rng.uniform(0, 2 * math.pi)
        ca, sa = math.cos(angle), math.sin(angle)

        max_rx = float("inf")
        max_ry = float("inf")
        if abs(ca) > 1e-8:
            max_rx = ((wx if ca > 0 else -wx) - sx) / ca
        if abs(sa) > 1e-8:
            max_ry = ((wy if sa > 0 else -wy) - sy) / sa

        max_r = min(max_rx, max_ry, params["r_max"])
        if max_r < params["r_min"]:
            continue

        r = rng.uniform(params["r_min"], min(max_r * 0.999, params["r_max"]))
        gx = sx + r * ca
        gy = sy + r * sa
        gz = rng.uniform(params["h_min"], params["h_max"])

        if -wx <= gx <= wx and -wy <= gy <= wy:
            return float(gx), float(gy), float(gz)

    angle = rng.uniform(0, 2 * math.pi)
    r = rng.uniform(params["r_min"], params["r_max"])
    gx = max(-wx, min(wx, sx + r * math.cos(angle)))
    gy = max(-wy, min(wy, sy + r * math.sin(angle)))
    gz = rng.uniform(params["h_min"], params["h_max"])
    return float(gx), float(gy), float(gz)


def _sample_goal(
    rng, start: Tuple[float, float, float], params: Dict[str, float], challenge_type: int, seed: int
) -> Tuple[float, float, float]:
    from swarm import constants as C

    if challenge_type == 5:
        return _sample_goal_warehouse(rng, start, params)

    sx, sy, sz = start
    wr = params["world_range"]
    start_surface_z = (
        sz - float(C.START_PLATFORM_TAKEOFF_BUFFER) if challenge_type in (3, 4) else sz
    )

    for _ in range(100):
        angle = rng.uniform(0, 2 * math.pi)
        ca, sa = math.cos(angle), math.sin(angle)

        max_rx = float("inf")
        max_ry = float("inf")
        if abs(ca) > 1e-8:
            max_rx = ((wr if ca > 0 else -wr) - sx) / ca
        if abs(sa) > 1e-8:
            max_ry = ((wr if sa > 0 else -wr) - sy) / sa

        max_r = min(max_rx, max_ry, params["r_max"])
        if max_r < params["r_min"]:
            continue

        r = rng.uniform(params["r_min"], min(max_r * 0.999, params["r_max"]))
        gx = sx + r * ca
        gy = sy + r * sa

        if challenge_type in (3, 4):
            gz = _get_type3_surface_z(gx, gy, seed) if challenge_type == 3 else 0.0
            dist_3d = math.sqrt((gx - sx) ** 2 + (gy - sy) ** 2 + (gz - start_surface_z) ** 2)
            if params["r_min"] <= dist_3d <= params["r_max"] and -wr <= gx <= wr and -wr <= gy <= wr:
                return float(gx), float(gy), float(gz)
        else:
            gz = rng.uniform(params["h_min"], params["h_max"])
            if -wr <= gx <= wr and -wr <= gy <= wr:
                return float(gx), float(gy), float(gz)

    angle = rng.uniform(0, 2 * math.pi)
    r = rng.uniform(params["r_min"], params["r_max"])
    gx = max(-wr, min(wr, sx + r * math.cos(angle)))
    gy = max(-wr, min(wr, sy + r * math.sin(angle)))
    if challenge_type in (3, 4):
        gz = _get_type3_surface_z(gx, gy, seed) if challenge_type == 3 else 0.0
    else:
        gz = rng.uniform(params["h_min"], params["h_max"])
    return float(gx), float(gy), float(gz)


def build_task(seed: int, challenge_type: int) -> FlightTask:
    """Deterministically build a :class:`FlightTask` for *seed* / *challenge_type*."""
    import random as _random
    from swarm.constants import (
        MOVING_PLATFORM_PROB,
        MOVING_PLATFORM_SEED_OFFSET,
        SEARCH_RADIUS_MAX,
        SEARCH_RADIUS_MIN,
        SIM_DT,
    )

    params = _load_type_params(challenge_type, seed)
    rng = _random.Random(seed)
    search_rng = _random.Random(seed + 888888)
    platform_rng = _random.Random((seed + MOVING_PLATFORM_SEED_OFFSET) & 0xFFFFFFFF)

    start = _sample_start(rng, params, challenge_type, seed)
    goal = _sample_goal(rng, start, params, challenge_type, seed)
    return FlightTask(
        map_seed=seed,
        start=start,
        goal=goal,
        sim_dt=float(SIM_DT),
        horizon=float(params["horizon"]),
        challenge_type=int(challenge_type),
        search_radius=float(search_rng.uniform(SEARCH_RADIUS_MIN, SEARCH_RADIUS_MAX)),
        moving_platform=bool(
            platform_rng.random() < MOVING_PLATFORM_PROB.get(challenge_type, 0.0)
        ),
    )


def _load_seed_jobs(seed_file: Path) -> List[VideoJob]:
    raw = json.loads(Path(seed_file).read_text())
    if not isinstance(raw, dict):
        raise ValueError("Seed file must contain a JSON object mapping benchmark groups to seed lists.")

    jobs: List[VideoJob] = []
    missing = [group for group in BENCH_GROUP_ORDER if group not in raw]
    if missing:
        raise ValueError(f"Seed file missing groups: {', '.join(missing)}")

    for group in BENCH_GROUP_ORDER:
        seeds = raw.get(group)
        if not isinstance(seeds, list) or not seeds:
            raise ValueError(f"Seed group {group} must be a non-empty list.")
        challenge_type = BENCH_GROUP_TO_TYPE[group]
        for seed in seeds:
            jobs.append(VideoJob(seed=int(seed), challenge_type=challenge_type))
    return jobs


# ═══════════════════════════════════════════════════════════════════════════════
#  Submission loading
# ═══════════════════════════════════════════════════════════════════════════════


def _extract_zip(zip_path: Path, dest: Path) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    if not (dest / "drone_agent.py").exists():
        raise FileNotFoundError(f"Submission zip missing drone_agent.py: {zip_path}")
    return dest


def _link_workspace(extracted: Path) -> None:
    """Expose ONNX assets under a Docker-like workspace path when needed."""
    onnx_files = list(extracted.glob("*.onnx"))
    if not onnx_files:
        return

    ws = Path(os.environ.get("SWARM_VIDEO_WORKSPACE", "/workspace/submission"))
    ws.mkdir(parents=True, exist_ok=True)
    for onnx in onnx_files:
        dst = ws / onnx.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            dst.symlink_to(onnx)
        except OSError:
            shutil.copy2(onnx, dst)


def _load_agent(extracted: Path):
    """Import ``DroneFlightController`` from the extracted submission."""
    if str(extracted) not in sys.path:
        sys.path.insert(0, str(extracted))
    prev_cwd = os.getcwd()
    os.chdir(str(extracted))
    try:
        spec = importlib.util.spec_from_file_location(
            "drone_agent", str(extracted / "drone_agent.py")
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load drone_agent.py from submission")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        agent = mod.DroneFlightController()
        for init_method in ("_ensure_loaded", "reset"):
            fn = getattr(agent, init_method, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
        return agent
    finally:
        os.chdir(prev_cwd)


# ═══════════════════════════════════════════════════════════════════════════════
#  PyBullet rendering helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _render_rgb(cli: int, view, proj, w: int, h: int) -> np.ndarray:
    """RGB render with shadows disabled for performance."""
    import pybullet as p

    _, _, rgba, _, _ = p.getCameraImage(
        width=w,
        height=h,
        viewMatrix=view,
        projectionMatrix=proj,
        renderer=p.ER_TINY_RENDERER,
        shadow=0,
        lightDirection=[0.4, 0.4, 1.0],
        physicsClientId=cli,
    )
    return np.asarray(rgba, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]


def _render_depth(
    cli: int, view, proj, w: int, h: int, near: float, far: float
) -> np.ndarray:
    """Return depth in **metres** (float32 array, shape ``(h, w)``)."""
    import pybullet as p

    _, _, _, zbuf, _ = p.getCameraImage(
        width=w,
        height=h,
        viewMatrix=view,
        projectionMatrix=proj,
        renderer=p.ER_TINY_RENDERER,
        shadow=0,
        flags=p.ER_NO_SEGMENTATION_MASK,
        physicsClientId=cli,
    )
    zbuf = np.asarray(zbuf, dtype=np.float32).reshape(h, w)
    denom = np.maximum(far - (far - near) * zbuf, near * 1e-6)
    return far * near / denom


def _colourise_depth(
    depth_m: np.ndarray, clip_min: float = 0.5, clip_max: float = 20.0
) -> np.ndarray:
    """Map a depth image (metres) to a uint8 RGB frame via the Inferno colourmap."""
    normalised = np.clip((depth_m - clip_min) / (clip_max - clip_min), 0.0, 1.0)
    try:
        import matplotlib.cm as cm

        cmap = cm.colormaps.get_cmap(DEPTH_COLORMAP_NAME)
        rgb = cmap(1.0 - normalised)[:, :, :3]  # invert → close = bright
        return (rgb * 255).astype(np.uint8)
    except Exception:  # matplotlib unavailable
        grey = (255 * (1.0 - normalised)).astype(np.uint8)
        return np.stack([grey, grey, grey], axis=-1)


# ═══════════════════════════════════════════════════════════════════════════════
#  Camera controllers
# ═══════════════════════════════════════════════════════════════════════════════


class _CameraBase:
    """Common interface for all camera modes."""

    def __init__(self, cli: int, width: int, height: int, fov: float):
        import pybullet as p

        self._p = p
        self._cli = cli
        self._w = width
        self._h = height
        self._fov = fov

    def _proj(self, near: float = 0.1, far: float = 500.0):
        return self._p.computeProjectionMatrixFOV(
            fov=self._fov,
            aspect=self._w / self._h,
            nearVal=near,
            farVal=far,
            physicsClientId=self._cli,
        )

    def _view(self, eye, target, up=(0, 0, 1)):
        return self._p.computeViewMatrix(
            cameraEyePosition=list(eye),
            cameraTargetPosition=list(target),
            cameraUpVector=list(up),
            physicsClientId=self._cli,
        )

    def capture(
        self, drone_pos: np.ndarray, drone_quat: np.ndarray, rot: np.ndarray, dt: float
    ) -> np.ndarray:
        raise NotImplementedError


class DepthCamera(_CameraBase):
    """Replicates the drone's onboard 128x128 depth sensor, colourised."""

    def __init__(self, cli: int, width: int, height: int):
        super().__init__(cli, DEPTH_SENSOR_RES, DEPTH_SENSOR_RES, fov=90.0)
        from swarm.constants import DEPTH_FAR, DEPTH_MIN_M, DEPTH_MAX_M

        self._near = 0.05
        self._far = float(DEPTH_FAR)
        self._clip_min = float(DEPTH_MIN_M)
        self._clip_max = float(DEPTH_MAX_M)
        self._out_w = width
        self._out_h = height

    def capture(self, drone_pos, drone_quat, rot, dt):
        fwd = rot @ np.array([1.0, 0.0, 0.0])
        fwd /= np.linalg.norm(fwd) + 1e-9
        up = rot @ np.array([0.0, 0.0, 1.0])
        eye = drone_pos + fwd * 0.35 + up * 0.05
        tgt = eye + fwd * 20.0

        view = self._view(eye, tgt, up)
        proj = self._proj(near=self._near, far=self._far)

        depth_m = _render_depth(
            self._cli,
            view,
            proj,
            DEPTH_SENSOR_RES,
            DEPTH_SENSOR_RES,
            self._near,
            self._far,
        )
        depth_rgb = _colourise_depth(depth_m, self._clip_min, self._clip_max)

        from PIL import Image

        return np.asarray(
            Image.fromarray(depth_rgb).resize((self._out_w, self._out_h), Image.NEAREST)
        )


class FPVCamera(_CameraBase):
    """First-person view with exponential smoothing for stable footage."""

    def __init__(self, cli: int, width: int, height: int, fov: float = FPV_FOV_DEG):
        super().__init__(cli, width, height, fov)
        self._smooth_fwd: Optional[np.ndarray] = None
        self._smooth_up: Optional[np.ndarray] = None

    def capture(self, drone_pos, drone_quat, rot, dt):
        cur_fwd = rot @ np.array([1.0, 0.0, 0.0])
        cur_fwd /= np.linalg.norm(cur_fwd) + 1e-9
        cur_up = rot @ np.array([0.0, 0.0, 1.0])
        cur_up /= np.linalg.norm(cur_up) + 1e-9

        a = FPV_SMOOTHING
        if self._smooth_fwd is not None:
            fwd = a * self._smooth_fwd + (1 - a) * cur_fwd
            up = a * self._smooth_up + (1 - a) * cur_up
        else:
            fwd, up = cur_fwd, cur_up
        fwd /= np.linalg.norm(fwd) + 1e-9
        up /= np.linalg.norm(up) + 1e-9
        self._smooth_fwd = fwd
        self._smooth_up = up

        eye = drone_pos + fwd * FPV_OFFSET_FORWARD_M + up * FPV_OFFSET_UP_M
        tgt = eye + fwd * 20.0
        return _render_rgb(
            self._cli, self._view(eye, tgt, up), self._proj(), self._w, self._h
        )


class ChaseCamera(_CameraBase):
    """Cinematic third-person follow camera with smoothed heading."""

    def __init__(
        self,
        cli: int,
        width: int,
        height: int,
        fov: float = CHASE_FOV_DEG,
        back: float = CHASE_DISTANCE_BACK_M,
        up: float = CHASE_HEIGHT_ABOVE_M,
    ):
        super().__init__(cli, width, height, fov)
        self._back = back
        self._up = up
        self._smooth_fwd: Optional[np.ndarray] = None

    def capture(self, drone_pos, drone_quat, rot, dt):
        cur_fwd = rot @ np.array([1.0, 0.0, 0.0])
        cur_fwd /= np.linalg.norm(cur_fwd) + 1e-9

        a = CHASE_SMOOTHING
        if self._smooth_fwd is not None:
            fwd = a * self._smooth_fwd + (1 - a) * cur_fwd
        else:
            fwd = cur_fwd
        fwd /= np.linalg.norm(fwd) + 1e-9
        self._smooth_fwd = fwd

        eye = drone_pos - fwd * self._back + np.array([0.0, 0.0, self._up])
        tgt = drone_pos + np.array([0.0, 0.0, 0.15])
        return _render_rgb(
            self._cli, self._view(eye, tgt), self._proj(), self._w, self._h
        )


class OverviewCamera(_CameraBase):
    """Slowly orbiting bird's-eye camera centred between drone and goal."""

    def __init__(
        self,
        cli: int,
        width: int,
        height: int,
        goal: Tuple[float, float, float],
        fov: float = OVERVIEW_FOV_DEG,
    ):
        super().__init__(cli, width, height, fov)
        self._goal = np.asarray(goal)
        self._yaw = 0.0

    def capture(self, drone_pos, drone_quat, rot, dt):
        mid = (drone_pos + self._goal) * 0.5
        span = float(np.linalg.norm(drone_pos - self._goal))
        cam_dist = max(15.0, span * 1.3)

        self._yaw += OVERVIEW_ORBIT_DEG_SEC * dt
        yaw_r = math.radians(self._yaw)
        pitch_r = math.radians(OVERVIEW_PITCH_DEG)

        eye = np.array(
            [
                mid[0] + cam_dist * math.cos(yaw_r) * math.cos(pitch_r),
                mid[1] + cam_dist * math.sin(yaw_r) * math.cos(pitch_r),
                mid[2] - cam_dist * math.sin(pitch_r),
            ]
        )
        return _render_rgb(
            self._cli, self._view(eye, mid), self._proj(far=1000.0), self._w, self._h
        )


class _Cv2VideoWriter:
    """Small adapter matching the `imageio` writer API used below."""

    def __init__(self, path: Path, fps: int, width: int, height: int):
        import cv2

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._cv2 = cv2
        self._writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"OpenCV failed to open video writer for {path}")

    def append_data(self, frame: np.ndarray) -> None:
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.ndim == 2:
            frame = np.repeat(frame[..., None], 3, axis=2)
        self._writer.write(self._cv2.cvtColor(frame, self._cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        self._writer.release()


def _open_video_writer(path: Path, fps: int, width: int, height: int) -> Any:
    try:
        import imageio

        return imageio.get_writer(str(path), fps=fps)
    except ModuleNotFoundError:
        return _Cv2VideoWriter(path, fps=fps, width=width, height=height)


def _ensure_local_ansible_temp() -> None:
    ansible_tmp = Path(os.environ.get("ANSIBLE_LOCAL_TEMP", "/tmp/swarm_ansible"))
    ansible_tmp.mkdir(parents=True, exist_ok=True)
    os.environ["ANSIBLE_LOCAL_TEMP"] = str(ansible_tmp)


class _FlightRecorder:
    def __init__(
        self,
        *,
        seed: int,
        challenge_type: int,
        modes: List[str],
        out_dir: Path,
        width: int,
        height: int,
        fps: int,
        chase_back: float,
        chase_up: float,
        chase_fov: float,
        fpv_fov: float,
        overview_fov: float,
        progress_file: Optional[Path],
    ) -> None:
        self._seed = int(seed)
        self._challenge_type = int(challenge_type)
        self._modes = list(modes)
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._width = int(width)
        self._height = int(height)
        self._fps = int(fps)
        self._frame_dt = 1.0 / float(fps) if fps > 0 else 0.0
        self._chase_back = float(chase_back)
        self._chase_up = float(chase_up)
        self._chase_fov = float(chase_fov)
        self._fpv_fov = float(fpv_fov)
        self._overview_fov = float(overview_fov)
        self._progress_file = Path(progress_file) if progress_file else None

        self._initialized = False
        self._cameras: Dict[str, _CameraBase] = {}
        self._writers: Dict[str, Any] = {}
        self._out_paths: Dict[str, Path] = {}
        self._tmp_paths: Dict[str, Path] = {}
        self._frame_count = 0
        self._next_frame_t = 0.0
        self._t_wall_start = 0.0
        self._last_sim_time = 0.0

    @property
    def expected_paths(self) -> List[Path]:
        type_label = TYPE_LABELS.get(self._challenge_type, f"type{self._challenge_type}")
        return [
            self._out_dir / f"seed{self._seed}_{type_label}_{mode}.mp4"
            for mode in self._modes
        ]

    def start(self, env: object, goal: Tuple[float, float, float], horizon: float) -> None:
        if self._initialized:
            return
        cli = getattr(env, "CLIENT", 0)
        self._t_wall_start = time.time()
        self._last_sim_time = 0.0
        self._next_frame_t = 0.0
        self._frame_count = 0

        if "depth" in self._modes:
            self._cameras["depth"] = DepthCamera(cli, self._width, self._height)
        if "fpv" in self._modes:
            self._cameras["fpv"] = FPVCamera(cli, self._width, self._height, self._fpv_fov)
        if "chase" in self._modes:
            self._cameras["chase"] = ChaseCamera(
                cli,
                self._width,
                self._height,
                self._chase_fov,
                self._chase_back,
                self._chase_up,
            )
        if "overview" in self._modes:
            self._cameras["overview"] = OverviewCamera(
                cli,
                self._width,
                self._height,
                goal,
                self._overview_fov,
            )

        type_label = TYPE_LABELS.get(self._challenge_type, f"type{self._challenge_type}")
        for mode in self._modes:
            fname = f"seed{self._seed}_{type_label}_{mode}.mp4"
            self._out_paths[mode] = self._out_dir / fname
            self._tmp_paths[mode] = self._out_dir / f".tmp_{fname}"
            self._writers[mode] = _open_video_writer(
                self._tmp_paths[mode],
                fps=self._fps,
                width=self._width,
                height=self._height,
            )

        if self._progress_file:
            total_frames = int(float(horizon) * self._fps)
            _write_progress(
                self._progress_file,
                {
                    "status": "generating",
                    "frames_rendered": 0,
                    "total_frames": total_frames,
                    "start_time": self._t_wall_start,
                    "last_update": self._t_wall_start,
                },
            )
        self._initialized = True

    def capture_step(self, env: object, sim_time_sec: float) -> None:
        if not self._initialized or self._frame_dt <= 0:
            self._last_sim_time = float(sim_time_sec)
            return
        import pybullet as p

        drone_pos = np.asarray(env._getDroneStateVector(0)[:3])
        drone_quat = np.asarray(env.quat[0])
        rot = np.array(p.getMatrixFromQuaternion(drone_quat)).reshape(3, 3)

        while float(sim_time_sec) >= self._next_frame_t:
            for mode, cam in self._cameras.items():
                frame = cam.capture(drone_pos, drone_quat, rot, self._frame_dt)
                self._writers[mode].append_data(frame)
            self._frame_count += 1
            self._next_frame_t += self._frame_dt
            if self._progress_file and self._frame_count % 20 == 0:
                _write_progress(
                    self._progress_file,
                    {
                        "status": "generating",
                        "frames_rendered": self._frame_count,
                        "total_frames": 0,
                        "start_time": self._t_wall_start,
                        "last_update": time.time(),
                    },
                )
        self._last_sim_time = float(sim_time_sec)

    def finish(self, success: bool, sim_time_sec: float) -> List[VideoResult]:
        wall_sec = max(0.0, time.time() - self._t_wall_start) if self._initialized else 0.0
        video_sec = self._frame_count / float(self._fps) if self._fps > 0 else 0.0

        for writer in self._writers.values():
            try:
                writer.close()
            except Exception:
                pass
        for mode in self._modes:
            src = self._tmp_paths.get(mode)
            dst = self._out_paths.get(mode)
            if src is not None and dst is not None and src.exists():
                try:
                    src.replace(dst)
                except Exception:
                    pass

        results: List[VideoResult] = []
        for mode in self._modes:
            out_path = self._out_paths[mode]
            size_mb = out_path.stat().st_size / (1024 * 1024) if out_path.exists() else 0.0
            results.append(
                VideoResult(
                    seed=self._seed,
                    challenge_type=self._challenge_type,
                    mode=mode,
                    path=str(out_path),
                    frames=self._frame_count,
                    duration_sec=video_sec,
                    success=bool(success),
                    sim_time_sec=float(sim_time_sec),
                    wall_time_sec=wall_sec,
                )
            )
            print(f"  {mode:<10}  {out_path.name}  ({size_mb:.1f} MB)")
        return results

    def abort(self) -> None:
        for writer in self._writers.values():
            try:
                writer.close()
            except Exception:
                pass
        for tmp_path in self._tmp_paths.values():
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass


def _infer_uid_from_model_path(model_path: Path) -> int:
    for candidate in (Path(model_path).stem, Path(model_path).name):
        match = re.search(r"uid[_-]?(\d+)", candidate, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                pass
    return 0


def _load_benchmark_expectations(summary_json: Path) -> Dict[Tuple[int, int], BenchmarkExpectation]:
    payload = json.loads(Path(summary_json).read_text())
    raw_groups = payload.get("group_results")
    if not isinstance(raw_groups, dict):
        raise ValueError("Summary JSON missing group_results.")

    expectations: Dict[Tuple[int, int], BenchmarkExpectation] = {}
    for group_name in BENCH_GROUP_ORDER:
        rows = raw_groups.get(group_name, [])
        if not isinstance(rows, list):
            raise ValueError(f"Summary JSON group_results[{group_name}] must be a list.")
        challenge_type = BENCH_GROUP_TO_TYPE[group_name]
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"Summary JSON row for {group_name} is not an object.")
            seed = int(row["seed"])
            expectations[(seed, challenge_type)] = BenchmarkExpectation(
                success=bool(row["success"]),
                score=float(row["score"]),
                sim_time_sec=float(row["sim_time"]),
            )
    return expectations


def _assert_replay_matches_expected(
    *,
    job: VideoJob,
    expected: BenchmarkExpectation,
    success: bool,
    score: float,
    sim_time_sec: float,
    score_tol: float = 1e-6,
    sim_tol: float = 1e-6,
) -> None:
    mismatches: List[str] = []
    if bool(success) != bool(expected.success):
        mismatches.append(f"success expected={expected.success} actual={success}")
    if abs(float(score) - float(expected.score)) > score_tol:
        mismatches.append(f"score expected={expected.score:.6f} actual={score:.6f}")
    if abs(float(sim_time_sec) - float(expected.sim_time_sec)) > sim_tol:
        mismatches.append(
            f"sim_time expected={expected.sim_time_sec:.6f} actual={sim_time_sec:.6f}"
        )
    if mismatches:
        raise RuntimeError(
            f"Benchmark replay mismatch for seed={job.seed} type={job.challenge_type}: "
            + "; ".join(mismatches)
        )


def _video_benchmark_env_overrides() -> Dict[str, Optional[str]]:
    # Video rendering runs inside the same Docker/RPC evaluator path as benchmark,
    # but host-side frame capture makes each seed much slower than pure scoring.
    # Use a generous timeout envelope while preserving the exact simulation logic.
    return {
        "SWARM_BATCH_TIMEOUT_MULT": "20.0",
        "SWARM_BATCH_TIMEOUT_HARD_CAP_SEC": "7200.0",
        "SWARM_BATCH_TIMEOUT_EXTEND_ON_PROGRESS": "1",
        "SWARM_BATCH_TIMEOUT_EXTEND_SEC": "60.0",
        "SWARM_BATCH_TIMEOUT_PROGRESS_STALE_SEC": "15.0",
        "SWARM_BATCH_TIMEOUT_PROGRESS_MIN_SIM_ADVANCE": "0.02",
        "SWARM_BATCH_TIMEOUT_MAX_TOTAL_SEC": "7200.0",
    }


def record_flight_benchmark(
    model_path: Path,
    seed: int,
    challenge_type: int,
    modes: List[str],
    out_dir: Path,
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fps: int = DEFAULT_FPS,
    chase_back: float = CHASE_DISTANCE_BACK_M,
    chase_up: float = CHASE_HEIGHT_ABOVE_M,
    chase_fov: float = CHASE_FOV_DEG,
    fpv_fov: float = FPV_FOV_DEG,
    overview_fov: float = OVERVIEW_FOV_DEG,
    save_actions_dir: Optional[Path] = None,
    progress_file: Optional[Path] = None,
) -> Tuple[List[VideoResult], bool, float, float]:
    from swarm.constants import SIM_DT
    from swarm.validator.docker.docker_evaluator import DockerSecureEvaluator
    from swarm.validator.task_gen import task_for_seed_and_type

    task = task_for_seed_and_type(sim_dt=SIM_DT, seed=seed, challenge_type=challenge_type)
    uid = _infer_uid_from_model_path(model_path)
    recorder = _FlightRecorder(
        seed=seed,
        challenge_type=challenge_type,
        modes=modes,
        out_dir=out_dir,
        width=width,
        height=height,
        fps=fps,
        chase_back=chase_back,
        chase_up=chase_up,
        chase_fov=chase_fov,
        fpv_fov=fpv_fov,
        overview_fov=overview_fov,
        progress_file=progress_file,
    )

    recorded_actions: List[List[float]] = []

    def _rollout_observer(event: Dict[str, object]) -> None:
        event_type = str(event.get("event", ""))
        env = event.get("env")
        if env is None:
            return
        if event_type == "seed_ready":
            goal = getattr(env, "GOAL_POS", getattr(task, "goal", (0.0, 0.0, 0.0)))
            recorder.start(env, tuple(goal), float(getattr(task, "horizon", 0.0)))
            return
        if event_type == "step":
            recorder.capture_step(env, float(event.get("sim_time_sec", 0.0)))
            action = event.get("action")
            if action is not None:
                recorded_actions.append(list(action) if not isinstance(action, list) else action)

    evaluator = DockerSecureEvaluator()
    if not getattr(evaluator, "_base_ready", False):
        raise RuntimeError("Docker evaluator base image is not ready.")

    print(
        f"[video] exact benchmark replay seed={seed}  type={challenge_type} "
        f"modes={modes}"
    )
    try:
        with _temporary_env(_video_benchmark_env_overrides()):
            results = asyncio.run(
                evaluator.evaluate_seeds_batch(
                    tasks=[task],
                    uid=uid,
                    model_path=Path(model_path),
                    worker_id=0,
                    rollout_observer=_rollout_observer,
                    task_offset=0,
                    task_total=1,
                )
            )
        if len(results) != 1:
            raise RuntimeError(f"Unexpected result count from replay: {len(results)}")
        result = results[0]
        video_results = recorder.finish(
            success=bool(result.success),
            sim_time_sec=float(result.time_sec),
        )
        if save_actions_dir is not None and recorded_actions:
            ap = _action_log_path(save_actions_dir, seed, challenge_type)
            _save_action_log(ap, seed, challenge_type, recorded_actions)
            print(f"  actions    {ap.name}  ({len(recorded_actions)} steps)")

        status = _outcome_label(bool(result.success))
        print(
            f"[video] RECORDED  outcome={status}  frames={video_results[0].frames if video_results else 0}  "
            f"video={(video_results[0].duration_sec if video_results else 0.0):.1f}s  "
            f"sim={float(result.time_sec):.1f}s"
        )
        return video_results, bool(result.success), float(result.score), float(result.time_sec)
    except Exception:
        recorder.abort()
        raise


# ═══════════════════════════════════════════════════════════════════════════════
#  Core recording function  (importable by validators)
# ═══════════════════════════════════════════════════════════════════════════════


def _write_progress(progress_file: Optional[Path], data: dict) -> None:
    if not progress_file:
        return
    try:
        import json as _json
        tmp = str(progress_file) + ".tmp"
        with open(tmp, "w") as f:
            _json.dump(data, f)
        os.replace(tmp, str(progress_file))
    except Exception:
        pass


def record_flight(
    model_path: Path,
    seed: int,
    challenge_type: int,
    modes: List[str],
    out_dir: Path,
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fps: int = DEFAULT_FPS,
    chase_back: float = CHASE_DISTANCE_BACK_M,
    chase_up: float = CHASE_HEIGHT_ABOVE_M,
    chase_fov: float = CHASE_FOV_DEG,
    fpv_fov: float = FPV_FOV_DEG,
    overview_fov: float = OVERVIEW_FOV_DEG,
    progress_file: Optional[Path] = None,
    save_actions_dir: Optional[Path] = None,
    replay_actions_dir: Optional[Path] = None,
) -> List[VideoResult]:
    """Record one flight and return metadata for each requested camera mode.

    Parameters
    ----------
    model_path     : path to the miner's ``submission.zip``
    seed           : map seed (deterministic world generation)
    challenge_type : 1-6  (city / open / mountain / village / warehouse / forest)
    modes          : subset of ``VALID_MODES``
    out_dir        : directory where ``.mp4`` files will be written
    progress_file  : optional path to write JSON progress updates

    Returns
    -------
    list[VideoResult] — one entry per requested mode.
    """
    import pybullet as p
    from gym_pybullet_drones.utils.enums import ActionType
    from swarm.constants import SIM_DT, SPEED_LIMIT

    model_path = Path(model_path).resolve()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- agent ----------------------------------------------------------
    work_dir = out_dir / f".work_seed{seed}"
    extracted = _extract_zip(model_path, work_dir)
    _link_workspace(extracted)
    agent = _load_agent(extracted)

    # --- environment ----------------------------------------------------
    task = build_task(seed, challenge_type)
    _ensure_local_ansible_temp()
    from swarm.utils.env_factory import make_env

    env = make_env(task, gui=False)
    obs, _ = env.reset(seed=task.map_seed)
    cli = getattr(env, "CLIENT", 0)
    act_lo = env.action_space.low.flatten()
    act_hi = env.action_space.high.flatten()

    replay_actions: Optional[List[np.ndarray]] = None
    if replay_actions_dir is not None:
        replay_path = _action_log_path(replay_actions_dir, seed, challenge_type)
        replay_actions = _load_action_log(replay_path)
        print(f"[video] replaying {len(replay_actions)} recorded actions")
    recorded_actions: List[List[float]] = []
    replay_idx = 0

    # --- cameras --------------------------------------------------------
    cameras: Dict[str, _CameraBase] = {}
    if "depth" in modes:
        cameras["depth"] = DepthCamera(cli, width, height)
    if "fpv" in modes:
        cameras["fpv"] = FPVCamera(cli, width, height, fpv_fov)
    if "chase" in modes:
        cameras["chase"] = ChaseCamera(
            cli, width, height, chase_fov, chase_back, chase_up
        )
    if "overview" in modes:
        cameras["overview"] = OverviewCamera(
            cli, width, height, task.goal, overview_fov
        )

    # --- writers --------------------------------------------------------
    type_label = TYPE_LABELS.get(challenge_type, f"type{challenge_type}")
    writers: Dict[str, Any] = {}
    out_paths: Dict[str, Path] = {}
    tmp_paths: Dict[str, Path] = {}

    for mode in modes:
        fname = f"seed{seed}_{type_label}_{mode}.mp4"
        out_paths[mode] = out_dir / fname
        tmp_paths[mode] = out_dir / f".tmp_{fname}"
        writers[mode] = _open_video_writer(tmp_paths[mode], fps=fps, width=width, height=height)

    # --- simulation loop ------------------------------------------------
    frame_dt = 1.0 / fps
    next_frame_t = 0.0
    frame_count = 0
    total_frames = int(task.horizon * fps)
    t_sim = 0.0
    success = False
    t_wall_start = time.time()

    if progress_file:
        progress_file = Path(progress_file)
        _write_progress(progress_file, {
            "status": "generating", "frames_rendered": 0,
            "total_frames": total_frames, "start_time": t_wall_start,
            "last_update": t_wall_start,
        })

    dist_m = math.dist(task.start, task.goal)
    print(
        f"[video] seed={seed}  type={challenge_type} ({type_label})  "
        f"dist={dist_m:.1f}m  modes={modes}"
    )

    try:
        while t_sim < task.horizon:
            # --- agent step ---
            if replay_actions is not None and replay_idx < len(replay_actions):
                act = np.clip(replay_actions[replay_idx], act_lo, act_hi)
                replay_idx += 1
            else:
                try:
                    raw = agent.act(obs)
                    if raw is None:
                        raw = np.zeros(5, dtype=np.float32)
                except Exception:
                    raw = np.zeros(5, dtype=np.float32)

                act = np.clip(np.asarray(raw, dtype=np.float32).flatten(), act_lo, act_hi)
                if getattr(env, "ACT_TYPE", None) == ActionType.VEL:
                    norm = max(float(np.linalg.norm(act[:3])), 1e-6)
                    act[:3] *= min(1.0, float(SPEED_LIMIT) / norm)
                    act = np.clip(act, act_lo, act_hi)

            recorded_actions.append(act.tolist())

            obs, _, terminated, truncated, info = env.step(act[None, :])
            t_sim += float(SIM_DT)

            # --- render frame ---
            if t_sim >= next_frame_t:
                drone_pos = np.asarray(env._getDroneStateVector(0)[:3])
                drone_quat = np.asarray(env.quat[0])
                rot = np.array(p.getMatrixFromQuaternion(drone_quat)).reshape(3, 3)

                for mode, cam in cameras.items():
                    frame = cam.capture(drone_pos, drone_quat, rot, frame_dt)
                    writers[mode].append_data(frame)

                frame_count += 1
                next_frame_t += frame_dt

                if progress_file and frame_count % 20 == 0:
                    _write_progress(progress_file, {
                        "status": "generating",
                        "frames_rendered": frame_count,
                        "total_frames": total_frames,
                        "start_time": t_wall_start,
                        "last_update": time.time(),
                    })

            if terminated or truncated:
                success = bool(info.get("success", False))
                break

    finally:
        # close writers before anything else to flush buffers
        for w in writers.values():
            try:
                w.close()
            except Exception:
                pass
        try:
            env.close()
        except Exception:
            pass

        # atomic rename: tmp → final
        for mode in modes:
            src = tmp_paths[mode]
            if src.exists():
                try:
                    src.replace(out_paths[mode])
                except Exception:
                    pass

        if save_actions_dir is not None and recorded_actions:
            ap = _action_log_path(save_actions_dir, seed, challenge_type)
            _save_action_log(ap, seed, challenge_type, recorded_actions)
            print(f"  actions    {ap.name}  ({len(recorded_actions)} steps)")

        # cleanup extraction directory
        try:
            if work_dir.exists():
                shutil.rmtree(work_dir)
        except Exception:
            pass

    wall_sec = time.time() - t_wall_start
    video_sec = frame_count / fps if fps else 0.0

    # --- summary --------------------------------------------------------
    results: List[VideoResult] = []
    for mode in modes:
        fpath = out_paths[mode]
        size = fpath.stat().st_size / (1024 * 1024) if fpath.exists() else 0.0
        results.append(
            VideoResult(
                seed=seed,
                challenge_type=challenge_type,
                mode=mode,
                path=str(fpath),
                frames=frame_count,
                duration_sec=video_sec,
                success=success,
                sim_time_sec=t_sim,
                wall_time_sec=wall_sec,
            )
        )
        print(f"  {mode:<10}  {fpath.name}  ({size:.1f} MB)")

    status = _outcome_label(success)
    print(
        f"[video] RECORDED  outcome={status}  frames={frame_count}  video={video_sec:.1f}s  "
        f"sim={t_sim:.1f}s  wall={wall_sec:.1f}s"
    )
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI entry-point
# ═══════════════════════════════════════════════════════════════════════════════


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="generate_video",
        description="Swarm V4 — render drone flight videos for a given model + seed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            camera modes
              depth     onboard 128x128 depth sensor (Inferno colourmap, upscaled)
              fpv       first-person RGB from the drone nose
              chase     cinematic third-person follow camera
              overview  slowly orbiting bird's-eye view
              all       generate every mode in one run

            examples
              %(prog)s --model UID_178.zip --seed 42 --type 1 --mode chase
              %(prog)s --model UID_178.zip --seed 42 --type 5 --mode all --out ./videos
              %(prog)s --model UID_178.zip --seed 42 --type 1 --mode depth,fpv --width 1920 --height 1080
        """
        ),
    )

    req = ap.add_argument_group("required arguments")
    req.add_argument(
        "--model", type=Path, required=True, metavar="ZIP", help="miner submission zip"
    )
    req.add_argument(
        "--seed",
        type=int,
        default=None,
        help="map seed for deterministic world generation",
    )
    req.add_argument(
        "--type",
        type=int,
        default=None,
        choices=[1, 2, 3, 4, 5, 6],
        metavar="TYPE",
        help="challenge type  (1=City 2=Open 3=Mountain 4=Village 5=Warehouse 6=Forest)",
    )
    req.add_argument(
        "--seed-file",
        type=Path,
        default=None,
        metavar="JSON",
        help="benchmark seed JSON generated by swarm benchmark --save-seed-file",
    )

    vid = ap.add_argument_group("video options")
    vid.add_argument(
        "--mode",
        type=str,
        default="chase",
        help="camera mode(s), comma-separated  (default: chase)",
    )
    vid.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="DIR",
        help="output directory  (default: ./videos)",
    )
    vid.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=f"frame width   (default: {DEFAULT_WIDTH})",
    )
    vid.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help=f"frame height  (default: {DEFAULT_HEIGHT})",
    )
    vid.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"frames/sec    (default: {DEFAULT_FPS})",
    )
    vid.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip a seed if all requested output files already exist",
    )
    vid.add_argument(
        "--backend",
        choices=["local", "benchmark"],
        default="benchmark",
        help="Replay backend: local fast replay, or exact benchmark Docker/RPC replay.",
    )
    vid.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        metavar="JSON",
        help="Benchmark summary JSON from swarm benchmark --summary-json-out; replay results must match when provided.",
    )

    cam = ap.add_argument_group("camera tuning")
    cam.add_argument(
        "--chase-back",
        type=float,
        default=CHASE_DISTANCE_BACK_M,
        metavar="M",
        help="chase: distance behind drone (m)",
    )
    cam.add_argument(
        "--chase-up",
        type=float,
        default=CHASE_HEIGHT_ABOVE_M,
        metavar="M",
        help="chase: height above drone (m)",
    )
    cam.add_argument(
        "--chase-fov",
        type=float,
        default=CHASE_FOV_DEG,
        metavar="DEG",
        help="chase: field of view",
    )
    cam.add_argument(
        "--fpv-fov",
        type=float,
        default=FPV_FOV_DEG,
        metavar="DEG",
        help="fpv: field of view",
    )
    cam.add_argument(
        "--overview-fov",
        type=float,
        default=OVERVIEW_FOV_DEG,
        metavar="DEG",
        help="overview: field of view",
    )

    ap.add_argument(
        "--progress-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="write JSON progress to this file (for API integration)",
    )

    replay = ap.add_argument_group("action replay")
    replay.add_argument(
        "--save-actions",
        type=Path,
        default=None,
        metavar="DIR",
        help="save recorded actions per seed for deterministic replay",
    )
    replay.add_argument(
        "--replay-actions",
        type=Path,
        default=None,
        metavar="DIR",
        help="replay pre-recorded actions instead of running the policy",
    )
    return ap


def _resolve_modes(raw_mode: str) -> List[str]:
    raw = raw_mode.strip().lower()
    if raw == "all":
        modes = list(VALID_MODES)
    else:
        modes = [m.strip() for m in raw.split(",") if m.strip()]
        invalid = [m for m in modes if m not in VALID_MODES]
        if invalid:
            raise ValueError(f"Unknown mode(s): {invalid}  (valid: {VALID_MODES})")
    if not modes:
        raise ValueError(f"No modes selected. Choose from: {VALID_MODES}")
    return modes


def _resolve_jobs(args: argparse.Namespace) -> List[VideoJob]:
    if args.seed_file is not None:
        if args.seed is not None or args.type is not None:
            raise ValueError("--seed-file cannot be combined with --seed or --type")
        return _load_seed_jobs(args.seed_file)
    if args.seed is None or args.type is None:
        raise ValueError("Provide either --seed-file, or both --seed and --type")
    return [VideoJob(seed=int(args.seed), challenge_type=int(args.type))]


def _expected_output_paths(
    out_dir: Path,
    jobs: Iterable[VideoJob],
    modes: List[str],
) -> Dict[VideoJob, List[Path]]:
    out_dir = Path(out_dir)
    expected: Dict[VideoJob, List[Path]] = {}
    for job in jobs:
        type_label = TYPE_LABELS.get(job.challenge_type, f"type{job.challenge_type}")
        expected[job] = [
            out_dir / f"seed{job.seed}_{type_label}_{mode}.mp4"
            for mode in modes
        ]
    return expected


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)

    model = args.model.resolve()
    if not model.exists():
        raise FileNotFoundError(f"Model not found: {model}")
    if args.summary_json is not None and args.backend != "benchmark":
        raise ValueError("--summary-json requires --backend benchmark")

    modes = _resolve_modes(args.mode)
    jobs = _resolve_jobs(args)
    out_dir = args.out or (_SCRIPT_DIR / "videos")
    expected_paths = _expected_output_paths(out_dir, jobs, modes)

    print("=" * 64)
    print("  Swarm V4 Video Generator")
    print("=" * 64)
    print(f"  Model       {model}")
    if args.seed_file is not None:
        print(f"  Seed file   {args.seed_file}")
        print(f"  Jobs        {len(jobs)} seeds")
    else:
        tname = TYPE_LABELS.get(args.type, "?")
        print(f"  Seed        {args.seed}")
        print(f"  Type        {args.type} ({tname})")
    print(f"  Modes       {', '.join(modes)}")
    print(f"  Resolution  {args.width}x{args.height} @ {args.fps} fps")
    print(f"  Backend     {args.backend}")
    print(f"  Output      {out_dir}")
    if args.skip_existing:
        print("  Skip exists yes")
    if args.summary_json is not None:
        print(f"  Summary     {args.summary_json}")
    print("=" * 64)
    print()

    t0 = time.time()
    results: List[VideoResult] = []
    failures: List[str] = []
    generated_jobs = 0
    skipped_jobs = 0
    verified_jobs = 0
    verified_job_keys: set[tuple[int, int]] = set()
    mismatch_job_keys: set[tuple[int, int]] = set()
    expectations = (
        _load_benchmark_expectations(args.summary_json)
        if args.summary_json is not None
        else {}
    )
    for index, job in enumerate(jobs, start=1):
        job_paths = expected_paths[job]
        if args.skip_existing and job_paths and all(path.exists() for path in job_paths):
            print(
                f"[video] skip {index}/{len(jobs)} seed={job.seed} "
                f"type={job.challenge_type} (all outputs already exist)"
            )
            skipped_jobs += 1
            continue
        print(f"[video] job {index}/{len(jobs)}")
        job_results: List[VideoResult] = []
        try:
            if args.backend == "benchmark":
                job_results, success, score, sim_time_sec = record_flight_benchmark(
                    model_path=model,
                    seed=job.seed,
                    challenge_type=job.challenge_type,
                    modes=modes,
                    out_dir=out_dir,
                    width=args.width,
                    height=args.height,
                    fps=args.fps,
                    chase_back=args.chase_back,
                    chase_up=args.chase_up,
                    chase_fov=args.chase_fov,
                    fpv_fov=args.fpv_fov,
                    overview_fov=args.overview_fov,
                    progress_file=args.progress_file if len(jobs) == 1 else None,
                    save_actions_dir=args.save_actions,
                )
            else:
                job_results = record_flight(
                    model_path=model,
                    seed=job.seed,
                    challenge_type=job.challenge_type,
                    modes=modes,
                    out_dir=out_dir,
                    width=args.width,
                    height=args.height,
                    fps=args.fps,
                    chase_back=args.chase_back,
                    chase_up=args.chase_up,
                    chase_fov=args.chase_fov,
                    fpv_fov=args.fpv_fov,
                    overview_fov=args.overview_fov,
                    progress_file=args.progress_file if len(jobs) == 1 else None,
                    save_actions_dir=args.save_actions,
                    replay_actions_dir=args.replay_actions,
                )
                success = bool(job_results[0].success) if job_results else False
                sim_time_sec = float(job_results[0].sim_time_sec) if job_results else 0.0
                score = 1.0 if success else 0.0

            results.extend(job_results)
            expected = expectations.get((job.seed, job.challenge_type))
            if expected is not None:
                _assert_replay_matches_expected(
                    job=job,
                    expected=expected,
                    success=success,
                    score=score,
                    sim_time_sec=sim_time_sec,
                )
                verified_jobs += 1
                verified_job_keys.add((job.seed, job.challenge_type))
                print(
                    f"[video] VERIFIED  seed={job.seed}  type={job.challenge_type}  "
                    f"outcome={_outcome_label(success)}  score={score:.6f}  sim={sim_time_sec:.2f}s"
                )
            generated_jobs += 1
        except Exception as exc:
            if (job.seed, job.challenge_type) in expectations:
                mismatch_job_keys.add((job.seed, job.challenge_type))
            failures.append(
                f"seed={job.seed} type={job.challenge_type}: {type(exc).__name__}: {exc}"
            )
            print(f"[video] FAILED  {failures[-1]}")

    print()
    print("=" * 64)
    print(f"  Finished in {time.time() - t0:.1f}s")
    print(f"  Jobs done   {generated_jobs}")
    if skipped_jobs:
        print(f"  Jobs skipped {skipped_jobs}")
    if verified_jobs:
        print(f"  Jobs verified {verified_jobs}")
    if failures:
        print(f"  Jobs failed {len(failures)}")
    for r in results:
        job_key = (r.seed, r.challenge_type)
        if job_key in mismatch_job_keys:
            tag = "MISMATCH"
        else:
            tag = _summary_tag(
                success=r.success,
                verified=job_key in verified_job_keys,
            )
        print(f"  [{tag}]  {r.mode:<10}  {r.path}")
    for failure in failures:
        print(f"  [FAIL] {failure}")
    print("=" * 64)
    if failures:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
import textwrap  # noqa: E402  (kept at bottom to avoid top-level clutter)

if __name__ == "__main__":
    main()
