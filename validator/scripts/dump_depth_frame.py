#!/usr/bin/env python3
# The MIT License (MIT)
# Copyright © 2026 Swarm

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Dump raw depth frames for a few fixed camera poses to .npy for pixel diffing."""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))

import numpy as np
import pybullet as p

from swarm.constants import DEPTH_NEAR, SIM_DT
from swarm.utils.env_factory import make_env_with_initial_obs
from swarm.validator.task_gen import task_for_seed_and_type


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="cf_autopilot")
    ap.add_argument("--ctype", type=int, default=1)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    task = task_for_seed_and_type(SIM_DT, seed=args.seed, challenge_type=args.ctype, family_id=args.family)
    env, _obs = make_env_with_initial_obs(task)
    cli = getattr(env, "CLIENT", 0)
    w, h = int(env.IMG_RES[0]), int(env.IMG_RES[1])
    proj = p.computeProjectionMatrixFOV(
        fov=getattr(env, "_fov", 91.0), aspect=w / h,
        nearVal=DEPTH_NEAR, farVal=float(getattr(env, "_depth_far_m", 20.0)),
        physicsClientId=cli,
    )
    seg_flag = p.ER_NO_SEGMENTATION_MASK
    if getattr(p, "ER_DEPTH_ONLY", None) is not None:
        seg_flag |= p.ER_DEPTH_ONLY

    frames = []
    for k in range(8):
        ang = 2.0 * math.pi * k / 8
        eye = [12 * math.cos(ang), 12 * math.sin(ang), 2.0]
        view = p.computeViewMatrix(eye, [0, 0, 1], [0, 0, 1], physicsClientId=cli)
        _w, _h, _rgb, dep, _seg = p.getCameraImage(
            width=w, height=h, shadow=0, renderer=p.ER_TINY_RENDERER,
            viewMatrix=view, projectionMatrix=proj,
            lightDirection=getattr(env, "_light_direction", [1, 1, 1]),
            flags=seg_flag, physicsClientId=cli,
        )
        frames.append(np.array(dep, dtype=np.float32).reshape(h, w))
    env.close()
    np.save(args.out, np.stack(frames))
    print("saved", args.out)


if __name__ == "__main__":
    main()
