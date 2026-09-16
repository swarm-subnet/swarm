#!/usr/bin/env python3
"""
Segmentation mask parity
========================
Renders the identity orbit of every verify_render_identity scene on TinyRenderer and on the
ray-cast backend with the object-and-link mask on, and reports per scene how many pixels carry
the same id on both. Needs a swarm-bullet3 wheel with ER_SWARM_RAYCAST.

    SWARM_RENDER_THREADS=2 python3 validator/scripts/compare_render_masks.py --json masks.json
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))

import numpy as np
import pybullet as p

from swarm.constants import DEPTH_NEAR, SIM_DT
from swarm.utils.env_factory import make_env_with_initial_obs
from swarm.validator.task_gen import task_for_seed_and_type
from validator.scripts.verify_render_identity import CONFIGS, ORBIT_POSES


def _mask(env, view, proj, flags):
    """Object-and-link segmentation mask of one getCameraImage call as an int array."""
    w, h = int(env.IMG_RES[0]), int(env.IMG_RES[1])
    image = p.getCameraImage(
        width=w, height=h, shadow=0, renderer=p.ER_TINY_RENDERER, viewMatrix=view,
        projectionMatrix=proj, flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX | flags,
        physicsClientId=env.CLIENT,
    )
    return np.asarray(image[4]).reshape(h, w)


def _scene_parity(env):
    """Pixel counts over the identity orbit: total, same id, one hit the other miss, both hit but different id."""
    cli = env.CLIENT
    w, h = int(env.IMG_RES[0]), int(env.IMG_RES[1])
    proj = p.computeProjectionMatrixFOV(
        fov=getattr(env, "_fov", 91.0), aspect=w / h, nearVal=DEPTH_NEAR,
        farVal=float(getattr(env, "_depth_far_m", 20.0)), physicsClientId=cli,
    )
    total = same = hit_miss = other_id = 0
    ids = set()
    for k in range(ORBIT_POSES):
        ang = 2.0 * math.pi * k / ORBIT_POSES
        r = 3.0 + 12.0 * (k % 5)
        eye = [r * math.cos(ang), r * math.sin(ang), 1.0 + 2.0 * (k % 7)]
        target = [0.0, 0.0, 1.0 + 0.5 * (k % 3)]
        view = p.computeViewMatrix(eye, target, [0, 0, 1], physicsClientId=cli)
        tiny = _mask(env, view, proj, 0)
        raycast = _mask(env, view, proj, p.ER_SWARM_RAYCAST)
        equal = tiny == raycast
        miss_differs = (tiny < 0) != (raycast < 0)
        total += equal.size
        same += int(np.count_nonzero(equal))
        hit_miss += int(np.count_nonzero(miss_differs))
        other_id += int(np.count_nonzero(~equal & ~miss_differs))
        ids.update(np.unique(tiny).tolist())
    return {
        "pixels": total, "same": same, "hit_miss": hit_miss, "other_id": other_id,
        "ids": len(ids), "agreement": same / total,
    }


def main():
    """Run every scene and print one line per scene plus a JSON record."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="render_masks.json")
    args = ap.parse_args()
    out = {"render_threads_env": os.environ.get("SWARM_RENDER_THREADS", "<unset>")}
    for family, ctype, seed in CONFIGS:
        task = task_for_seed_and_type(SIM_DT, seed=seed, challenge_type=ctype, family_id=family)
        env, _obs = make_env_with_initial_obs(task)
        key = f"{family}/t{ctype}/s{seed}"
        out[key] = _scene_parity(env)
        env.close()
        r = out[key]
        print(
            f"{key}: ids={r['ids']} agreement={100.0 * r['agreement']:.4f}% "
            f"hit/miss={r['hit_miss']} other_id={r['other_id']} of {r['pixels']}", flush=True,
        )
    with open(args.json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"JSON written to {args.json}")


if __name__ == "__main__":
    main()
