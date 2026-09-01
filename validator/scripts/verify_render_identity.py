#!/usr/bin/env python3
"""
Render identity verifier
========================
Proves a renderer change is pixel-identical. Two checks:

1. render-only: build each scene, then (no physics stepping) render a fixed
   orbit of camera poses with getCameraImage and hash the raw depth buffers.
   Valid to compare across wheels AND across SWARM_RENDER_THREADS values.
2. episode: run a deterministic action sequence and hash every full
   observation. Valid to compare across thread counts on the same wheel.

Run it once per configuration and diff the JSON outputs:
    SWARM_RENDER_THREADS=1 python3 validator/scripts/verify_render_identity.py --json a.json
    SWARM_RENDER_THREADS=4 python3 validator/scripts/verify_render_identity.py --json b.json
"""

import argparse
import hashlib
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))

import numpy as np
import pybullet as p

from swarm.constants import SIM_DT, DEPTH_NEAR
from swarm.utils.env_factory import make_env_with_initial_obs
from swarm.validator.task_gen import task_for_seed_and_type

CONFIGS = [
    ("cf_autopilot", 1, 101),
    ("cf_autopilot", 3, 103),
    ("cf_autopilot", 5, 105),
    ("cf_autopilot", 6, 106),
    ("cf_search_and_rescue", 2, 104),
    ("cf_swarm_sar", 2, 1040),
    ("cf_interceptor_office", 7, 7),
]
ORBIT_POSES = 40
EPISODE_STEPS = 60


def _orbit_hash(env):
    cli = getattr(env, "CLIENT", 0)
    w, h = int(env.IMG_RES[0]), int(env.IMG_RES[1])
    far = float(getattr(env, "_depth_far_m", 20.0))
    proj = p.computeProjectionMatrixFOV(
        fov=getattr(env, "_fov", 91.0), aspect=w / h,
        nearVal=DEPTH_NEAR, farVal=far, physicsClientId=cli,
    )
    seg_flag = p.ER_NO_SEGMENTATION_MASK
    depth_only = getattr(p, "ER_DEPTH_ONLY", None)
    if depth_only is not None:
        seg_flag |= depth_only

    digest = hashlib.sha256()
    for k in range(ORBIT_POSES):
        ang = 2.0 * math.pi * k / ORBIT_POSES
        r = 3.0 + 12.0 * (k % 5)
        eye = [r * math.cos(ang), r * math.sin(ang), 1.0 + 2.0 * (k % 7)]
        target = [0.0, 0.0, 1.0 + 0.5 * (k % 3)]
        view = p.computeViewMatrix(eye, target, [0, 0, 1], physicsClientId=cli)
        _w, _h, _rgb, dep, _seg = p.getCameraImage(
            width=w, height=h, shadow=0, renderer=p.ER_TINY_RENDERER,
            viewMatrix=view, projectionMatrix=proj,
            lightDirection=getattr(env, "_light_direction", [1, 1, 1]),
            flags=seg_flag, physicsClientId=cli,
        )
        digest.update(np.ascontiguousarray(dep).tobytes())
    return digest.hexdigest()


def _episode_hash(env, obs):
    digest = hashlib.sha256()
    n_act = int(np.prod(env.action_space.shape))
    for t in range(EPISODE_STEPS):
        for key in sorted(obs):
            digest.update(key.encode())
            digest.update(np.ascontiguousarray(obs[key]).tobytes())
        flat = np.zeros(n_act, dtype=np.float32)
        flat[0::6] = 0.4 * math.sin(0.11 * t)
        flat[1::6] = 0.4 * math.cos(0.07 * t)
        flat[2::6] = 0.2 * math.sin(0.05 * t)
        flat[3::6] = 0.6
        action = flat.reshape(env.action_space.shape)
        obs, _r, term, trunc, _i = env.step(action)
        if term or trunc:
            break
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="render_identity.json")
    args = ap.parse_args()

    out = {
        "render_threads_env": os.environ.get("SWARM_RENDER_THREADS", "<unset>"),
        "pybullet": getattr(p, "__file__", "?"),
    }
    for family, ctype, seed in CONFIGS:
        task = task_for_seed_and_type(SIM_DT, seed=seed, challenge_type=ctype, family_id=family)
        env, obs = make_env_with_initial_obs(task)
        key = f"{family}/t{ctype}/s{seed}"
        out[key] = {
            "orbit": _orbit_hash(env),
            "episode": _episode_hash(env, obs),
        }
        env.close()
        print(f"{key}: orbit={out[key]['orbit'][:16]} episode={out[key]['episode'][:16]}", flush=True)

    with open(args.json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"JSON written to {args.json}")


if __name__ == "__main__":
    main()
