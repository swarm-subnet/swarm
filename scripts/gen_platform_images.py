#!/usr/bin/env python3
"""
Platform Image Generator
========================
Generates multi-angle renders of each challenge type (City, Open, Mountain)
for documentation, README images, and visual inspection.

Five camera views per seed:
  - overview:     elevated 3/4 view of the full scene
  - start_close:  zoomed on the start platform
  - goal_close:   zoomed on the goal platform
  - side:         lateral perspective
  - top:          near-orthographic top-down

Usage:
    python3 scripts/gen_platform_images.py --per-type 2
    python3 scripts/gen_platform_images.py --per-type 1 --width 1920 --height 1080
    python3 scripts/gen_platform_images.py --per-type 3 --out-dir ./renders --seeds 442893 442884 442894
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

CHALLENGE_NAMES = {
    1: "city",
    2: "open",
    3: "mountain",
    4: "village",
    5: "warehouse",
    6: "forest",
}


@dataclass
class SeedTask:
    challenge_type: int
    seed: int


def _ensure_local_ansible_temp() -> None:
    ansible_tmp = Path(os.environ.get("ANSIBLE_LOCAL_TEMP", "/tmp/swarm_ansible"))
    ansible_tmp.mkdir(parents=True, exist_ok=True)
    os.environ["ANSIBLE_LOCAL_TEMP"] = str(ansible_tmp)


def _choose_challenge_type(seed: int) -> int:
    from swarm.constants import CHALLENGE_TYPE_DISTRIBUTION

    challenge_types = list(CHALLENGE_TYPE_DISTRIBUTION.keys())
    probabilities = list(CHALLENGE_TYPE_DISTRIBUTION.values())
    type_rng = random.Random(seed + 999999)
    return int(type_rng.choices(challenge_types, weights=probabilities, k=1)[0])


def _capture_image(
    cli: int,
    target: list,
    distance: float,
    yaw: float,
    pitch: float,
    width: int,
    height: int,
) -> Image.Image:
    """Render a single camera view and return as PIL Image."""
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target,
        distance=distance,
        yaw=yaw,
        pitch=pitch,
        roll=0,
        upAxisIndex=2,
    )
    projection = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=width / height,
        nearVal=0.1,
        farVal=1000.0,
    )
    img = p.getCameraImage(
        width,
        height,
        viewMatrix=view,
        projectionMatrix=projection,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=cli,
    )
    rgba = np.array(img[2], dtype=np.uint8).reshape(height, width, 4)
    return Image.fromarray(rgba[:, :, :3])


def _pick_seeds(
    per_type: int,
    max_scan: int,
    sim_dt: float,
    explicit_seeds: Optional[List[int]] = None,
    explicit_seed_specs: Optional[List[str]] = None,
) -> Dict[int, List[int]]:
    """Select seeds covering all challenge types.

    If explicit_seeds is given, classify those directly.
    Otherwise scan sequentially until per_type seeds per type are found.
    """

    picked: Dict[int, List[int]] = {k: [] for k in CHALLENGE_NAMES}

    if explicit_seed_specs:
        for spec in explicit_seed_specs:
            raw_type, raw_seed = spec.split(":", 1)
            ct = int(raw_type)
            seed = int(raw_seed)
            if ct not in picked:
                raise ValueError(f"Unknown challenge type in seed spec: {ct}")
            picked[ct].append(seed)
        return picked

    if explicit_seeds:
        for s in explicit_seeds:
            ct = _choose_challenge_type(s)
            if ct in picked:
                picked[ct].append(s)
        return picked

    seed = 0
    while seed < max_scan and any(len(v) < per_type for v in picked.values()):
        ct = _choose_challenge_type(seed)
        if ct in picked and len(picked[ct]) < per_type:
            picked[ct].append(seed)
        seed += 1

    return picked


def _save_images_for_seed(
    seed_task: SeedTask,
    out_root: Path,
    width: int,
    height: int,
    sim_dt: float,
) -> int:
    """Build world for one seed and save all camera views. Returns images saved."""
    from scripts.generate_video import build_task
    from swarm.core.env_builder import build_world

    _ensure_local_ansible_temp()
    task = build_task(seed=seed_task.seed, challenge_type=seed_task.challenge_type)
    task.sim_dt = float(sim_dt)
    sx, sy, sz = task.start
    gx, gy, gz = task.goal

    cli = p.connect(p.DIRECT)
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=cli)
        p.loadURDF("plane.urdf", physicsClientId=cli)

        result = build_world(
            seed=seed_task.seed,
            cli=cli,
            start=(sx, sy, sz),
            goal=(gx, gy, gz),
            challenge_type=seed_task.challenge_type,
        )

        if len(result) == 4:
            _, _, start_surface_z, goal_surface_z = result
        else:
            start_surface_z = None
            goal_surface_z = None

        actual_sz = float(start_surface_z) if start_surface_z is not None else float(sz)
        actual_gz = float(goal_surface_z) if goal_surface_z is not None else float(gz)

        mid_x = (sx + gx) * 0.5
        mid_y = (sy + gy) * 0.5
        mid_z = (actual_sz + actual_gz) * 0.5
        dist_2d = math.hypot(gx - sx, gy - sy)
        overview_distance = max(40.0, min(180.0, dist_2d * 1.6))

        views = [
            ("overview", [mid_x, mid_y, mid_z], overview_distance, 35.0, -40.0),
            ("start_close", [sx, sy, actual_sz + 1.8], 18.0, 20.0, -30.0),
            ("goal_close", [gx, gy, actual_gz + 1.8], 18.0, 20.0, -30.0),
            ("side", [mid_x, mid_y, mid_z], overview_distance, 110.0, -28.0),
            ("top", [mid_x, mid_y, mid_z], max(25.0, dist_2d * 0.9), 0.0, -89.0),
        ]

        type_name = CHALLENGE_NAMES.get(
            seed_task.challenge_type, f"type_{seed_task.challenge_type}"
        )
        type_dir = out_root / type_name
        type_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for name, target, distance, yaw, pitch in views:
            image = _capture_image(cli, target, distance, yaw, pitch, width, height)
            out_path = type_dir / f"seed_{seed_task.seed}_{name}.png"
            image.save(str(out_path))
            saved += 1
        return saved
    finally:
        p.disconnect(cli)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate multi-angle platform images for each challenge type.",
    )
    parser.add_argument(
        "--per-type",
        type=int,
        default=1,
        help="Number of seeds to render per challenge type (default: 1).",
    )
    parser.add_argument(
        "--max-scan",
        type=int,
        default=50000,
        help="Max seeds to scan when auto-selecting (default: 50000).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit seed list (auto-classified by type). Overrides --per-type.",
    )
    parser.add_argument(
        "--seed-specs",
        type=str,
        nargs="+",
        default=None,
        help="Explicit type:seed pairs (e.g. 1:323517 6:431623).",
    )
    parser.add_argument(
        "--sim-dt",
        type=float,
        default=1 / 240.0,
        help="Simulation timestep for task generation (default: 1/240).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=960,
        help="Output image width in pixels (default: 960).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Output image height in pixels (default: 720).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <cwd>/platform_images).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = args.out_dir or Path.cwd() / "platform_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    picked = _pick_seeds(
        args.per_type,
        args.max_scan,
        args.sim_dt,
        args.seeds,
        args.seed_specs,
    )

    if not args.seeds and not args.seed_specs and any(
        len(v) < args.per_type for v in picked.values()
    ):
        missing = {
            k: args.per_type - len(v)
            for k, v in picked.items()
            if len(v) < args.per_type
        }
        raise RuntimeError(
            f"Could not find enough seeds (missing: {missing}). Try --max-scan larger."
        )

    total_seeds = sum(len(v) for v in picked.values())
    print(
        f"Seeds selected: {total_seeds} across {sum(1 for v in picked.values() if v)} types"
    )
    for ct in sorted(picked):
        if picked[ct]:
            print(f"  {CHALLENGE_NAMES.get(ct, ct)}: {picked[ct]}")

    total_images = 0
    for ctype in sorted(picked):
        for seed in picked[ctype]:
            saved = _save_images_for_seed(
                SeedTask(challenge_type=ctype, seed=seed),
                out_root=out_dir,
                width=args.width,
                height=args.height,
                sim_dt=args.sim_dt,
            )
            total_images += saved
            print(f"  [{CHALLENGE_NAMES.get(ctype, ctype)}] seed={seed}  views={saved}")

    print(f"\nDone: {total_images} images saved to {out_dir}")


if __name__ == "__main__":
    main()
