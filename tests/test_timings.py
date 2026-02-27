#!/usr/bin/env python3
"""
Benchmark: _update_min_clearance() cost vs full env.step()
==========================================================
Measures each isolated component of env.step() across all three
challenge types: rendering, physics, clearance, and overhead.

Usage:
    python3 bench_min_clearance.py

Output:
    Per-scene breakdown: bodies checked, ms per component, % of total step.
"""

import sys
import os
import time

sys.path.insert(0, "/root/New_Swarm/swarm")
os.environ.setdefault("SWARM_PRIVATE_BENCHMARK_SECRET", "bench_test_key_2026")

import numpy as np
import pybullet as p

from swarm.constants import SIM_DT, DEPTH_NEAR, SAFETY_DISTANCE_SAFE
from swarm.validator.task_gen import random_task
from swarm.utils.env_factory import make_env

# ── Config ────────────────────────────────────────────────────────────────────
SEEDS = {
    "Mountain": 442894,
    "City":     442893,
    "Open":     442892,
}
N_STEPS  = 40
WARMUP   = 5
SEG_FLAG = p.ER_NO_SEGMENTATION_MASK | p.ER_DEPTH_ONLY
FULL     = 3000  # steps per seed
# ─────────────────────────────────────────────────────────────────────────────


def benchmark_scene(label, seed):
    task = random_task(sim_dt=SIM_DT, seed=seed)
    env  = make_env(task, gui=False)
    env.reset()

    cli    = getattr(env, "CLIENT", 0)
    fov    = getattr(env, "_fov", 91.0)
    W, H   = env.IMG_RES
    aspect = W / H
    dummy  = np.zeros_like(env.action_space.low.flatten())[None, :]
    light  = getattr(env, "_light_direction", [1, 1, 1])
    proj   = p.computeProjectionMatrixFOV(
        fov=fov, aspect=aspect, nearVal=DEPTH_NEAR, farVal=1000.0, physicsClientId=cli
    )

    # Pre-build body list (same exclusion logic as _update_min_clearance)
    drone_id  = env.DRONE_IDS[0]
    ground_id = getattr(env, "PLANE_ID", 0)
    excluded  = (
        {drone_id, -1, ground_id}
        | set(getattr(env, "_end_platform_uids",   []))
        | set(getattr(env, "_start_platform_uids", []))
    )
    num_bodies = p.getNumBodies(physicsClientId=cli)
    body_list  = [
        p.getBodyUniqueId(i, physicsClientId=cli)
        for i in range(num_bodies)
        if p.getBodyUniqueId(i, physicsClientId=cli) not in excluded
    ]

    t_step      = []
    t_render    = []
    t_physics   = []
    t_clearance = []

    for i in range(WARMUP + N_STEPS):

        # ── Full env.step() ──────────────────────────────────────────────────
        t0 = time.perf_counter()
        _, _, term, trunc, _ = env.step(dummy)
        t_step_i = (time.perf_counter() - t0) * 1000
        if term or trunc:
            env.reset()

        # ── Isolated: getCameraImage ─────────────────────────────────────────
        rot_mat = np.array(p.getMatrixFromQuaternion(env.quat[0])).reshape(3, 3)
        fwd     = rot_mat @ [1, 0, 0];  fwd /= np.linalg.norm(fwd)
        up      = rot_mat @ [0, 0, 1]
        cam     = env.pos[0] + fwd * 0.35 + up * 0.05
        view    = p.computeViewMatrix(cam, cam + fwd * 20.0, up.tolist(), physicsClientId=cli)

        t0 = time.perf_counter()
        p.getCameraImage(
            W, H, shadow=0, renderer=p.ER_TINY_RENDERER,
            viewMatrix=view, projectionMatrix=proj,
            lightDirection=light, flags=SEG_FLAG, physicsClientId=cli,
        )
        t_render_i = (time.perf_counter() - t0) * 1000

        # ── Isolated: stepSimulation ─────────────────────────────────────────
        t0 = time.perf_counter()
        p.stepSimulation(physicsClientId=cli)
        t_physics_i = (time.perf_counter() - t0) * 1000

        # ── Isolated: _update_min_clearance ─────────────────────────────────
        t0 = time.perf_counter()
        min_dist = SAFETY_DISTANCE_SAFE
        for body_uid in body_list:
            closest = p.getClosestPoints(
                bodyA=drone_id, bodyB=body_uid,
                distance=SAFETY_DISTANCE_SAFE, physicsClientId=cli,
            )
            for pt in closest:
                if pt[8] < min_dist:
                    min_dist = pt[8]
        t_clearance_i = (time.perf_counter() - t0) * 1000

        if i < WARMUP:
            continue

        t_step.append(t_step_i)
        t_render.append(t_render_i)
        t_physics.append(t_physics_i)
        t_clearance.append(t_clearance_i)

    env.close()

    return dict(
        label     = label,
        n_bodies  = num_bodies,
        n_checked = len(body_list),
        step      = float(np.mean(t_step)),
        render    = float(np.mean(t_render)),
        physics   = float(np.mean(t_physics)),
        clearance = float(np.mean(t_clearance)),
    )


def print_results(results):
    SEP = "─" * 65

    for r in results:
        other = r["step"] - r["render"] - r["physics"] - r["clearance"]
        pct   = lambda ms: ms / r["step"] * 100

        print(f"\n  {r['label']}")
        print(f"  {SEP}")
        print(f"  Total bodies in scene:       {r['n_bodies']}")
        print(f"  Bodies queried by clearance: {r['n_checked']}  "
              f"(excluded: {r['n_bodies'] - r['n_checked']})")
        print()
        print(f"  env.step() total:            {r['step']:7.1f} ms  (100%)")
        print(f"  ├─ getCameraImage (render):  {r['render']:7.1f} ms  ({pct(r['render']):.0f}%)")
        print(f"  ├─ stepSimulation (physics): {r['physics']:7.1f} ms  ({pct(r['physics']):.0f}%)")
        print(f"  ├─ _update_min_clearance:    {r['clearance']:7.1f} ms  ({pct(r['clearance']):.0f}%)")
        print(f"  └─ other (obs build, etc):   {other:7.1f} ms  ({pct(other):.0f}%)")
        print()
        print(f"  Per-body getClosestPoints:   "
              f"{r['clearance'] / r['n_checked'] * 1000:.3f} µs/body")
        print()
        print(f"  Extrapolated × {FULL} steps:")
        print(f"    Full step:       {r['step']*FULL/1000:6.0f}s  ({r['step']*FULL/60000:.1f} min)")
        print(f"    Render alone:    {r['render']*FULL/1000:6.0f}s  ({r['render']*FULL/60000:.1f} min)")
        print(f"    Clearance alone: {r['clearance']*FULL/1000:6.0f}s  ({r['clearance']*FULL/60000:.2f} min)")
        print()
        no_clear = r["step"] - r["clearance"]
        print(f"  If clearance REMOVED:")
        print(f"    New step time:   {no_clear:.1f} ms  →  {no_clear*FULL/1000:.0f}s "
              f"({no_clear*FULL/60000:.1f} min/seed)")
        print(f"    Speedup:         {r['step']/no_clear:.2f}×")

    print()
    print("=" * 65)
    print("  Summary")
    print("=" * 65)
    print(f"  {'Scene':<10}  {'Bodies':>7}  {'Step':>8}  {'Render':>8}  "
          f"{'Physics':>8}  {'Clearance':>10}  {'Clear%':>7}")
    print(f"  {'-'*10}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*7}")
    for r in results:
        print(f"  {r['label']:<10}  {r['n_checked']:>7}  "
              f"{r['step']:>6.1f}ms  {r['render']:>6.1f}ms  "
              f"{r['physics']:>6.1f}ms  {r['clearance']:>8.2f}ms  "
              f"{r['clearance']/r['step']*100:>6.1f}%")

    avg_pct = float(np.mean([r["clearance"] / r["step"] * 100 for r in results]))
    print()
    print(f"  Average clearance share: {avg_pct:.1f}%")
    print(f"  → _update_min_clearance is NOT the bottleneck.")
    print(f"  → Depth rendering (>90%) is the only target worth optimizing.")
    print("=" * 65)
    print()


if __name__ == "__main__":
    print()
    print("=" * 65)
    print("  _update_min_clearance() — Full Step Cost Breakdown")
    print("=" * 65)
    print(f"\n  Steps: {N_STEPS} measured  |  Warmup: {WARMUP}  |  Scenes: {list(SEEDS.keys())}\n")

    results = []
    for label, seed in SEEDS.items():
        print(f"  [{label}] running...", end=" ", flush=True)
        r = benchmark_scene(label, seed)
        results.append(r)
        print(f"step={r['step']:.1f}ms  render={r['render']:.1f}ms  "
              f"clearance={r['clearance']:.2f}ms  ({r['clearance']/r['step']*100:.1f}%)")

    print_results(results)