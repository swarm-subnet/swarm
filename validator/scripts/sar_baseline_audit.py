#!/usr/bin/env python3
"""D.3.2 audit — baseline hover-and-sweep policy success rates per map.

Runs a scripted policy (fly to search_centre, sweep, hover at the victim)
and reports success rate / mean confirm time / failure_reason distribution
per environment type. Plan asks for 1000 seeds per environment.

This is a deployment-only sanity check; the threshold is set during the
cutover runbook, not by code, because the baseline policy is intentionally
sub-optimal and the absolute success rate is a calibration target.

Usage:
    python3 scripts/sar_baseline_audit.py                  # full 1000 / map
    python3 scripts/sar_baseline_audit.py --n-seeds 50     # smoke
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
import time
from collections import Counter

import numpy as np
import pybullet as p

from swarm.constants import HORIZON_SEC, SAR_DWELL_SEC, SAR_HOVER_BAND, SPEED_LIMIT
from swarm.protocol import MapTask


CHALLENGE_TYPES = {
    "city":      1,
    "open":      2,
    "mountain":  3,
    "village":   4,
    "warehouse": 5,
    "forest":    6,
}

DEFAULT_N_SEEDS = 1000


def _scripted_step(env, target_xy):
    pos, _ = env._sar_drone_state()
    dx = float(target_xy[0]) - float(pos[0])
    dy = float(target_xy[1]) - float(pos[1])
    dz = (env.sar_world.victim_aabb[1][2] + sum(SAR_HOVER_BAND) / 2.0) - float(pos[2])
    norm = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
    shape = env.action_space.shape
    if norm < 1e-6:
        return np.zeros(shape, dtype=np.float32)
    speed = min(1.0, SPEED_LIMIT / max(SPEED_LIMIT, norm))
    raw = np.array([dx / norm, dy / norm, dz / norm, speed, 0.0], dtype=np.float32)
    dim = int(np.prod(shape))
    if dim != raw.size:
        raw = raw[:dim] if dim < raw.size else np.pad(raw, (0, dim - raw.size))
    return raw.reshape(shape)


def run_one_episode(map_seed: int, challenge_type: int) -> dict:
    from swarm.core.moving_drone import MovingDroneAviary

    task = MapTask(
        map_seed=map_seed,
        start=(0.0, 0.0, 1.5),
        goal=(8.0, 8.0, 1.5),
        sim_dt=1 / 30,
        horizon=HORIZON_SEC,
        challenge_type=challenge_type,
        version="5.0.0",
    )
    with contextlib.redirect_stdout(io.StringIO()):
        env = MovingDroneAviary(task, ctrl_freq=30, pyb_freq=30, sar_mode=True)
        env.reset(seed=map_seed)
    try:
        if env.sar_world is None:
            return {"failure_reason": "SPAWN_FAILURE", "t": 0.0, "success": False}
        target = env.sar_world.victim_centre_xy
        for _ in range(int(HORIZON_SEC * 30) + 10):
            with contextlib.redirect_stdout(io.StringIO()):
                obs, _r, terminated, truncated, info = env.step(_scripted_step(env, target))
            if terminated or truncated:
                break
        return {
            "failure_reason": info.get("failure_reason", "NONE"),
            "t": float(env._time_alive),
            "success": bool(info.get("success", False)),
        }
    finally:
        try:
            env.close()
        except Exception:
            pass


def audit_one_map(name: str, ctype: int, n_seeds: int) -> dict:
    started = time.time()
    successes = 0
    reasons = Counter()
    times = []
    for seed in range(n_seeds):
        rec = run_one_episode(seed, ctype)
        reasons[rec["failure_reason"]] += 1
        if rec["success"]:
            successes += 1
            times.append(rec["t"])
    elapsed = time.time() - started
    return {
        "n_seeds": n_seeds,
        "successes": successes,
        "success_rate": successes / n_seeds if n_seeds else 0.0,
        "mean_confirm_time": (sum(times) / len(times)) if times else None,
        "reasons": dict(reasons),
        "elapsed_sec": elapsed,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    ap.add_argument("--maps", default=",".join(CHALLENGE_TYPES.keys()))
    args = ap.parse_args(argv)

    selected = [m.strip() for m in args.maps.split(",") if m.strip()]
    for m in selected:
        if m not in CHALLENGE_TYPES:
            print(f"unknown map: {m!r}")
            return 2

    print(f"sar baseline audit · n_seeds={args.n_seeds}")
    print("-" * 72)
    for name in selected:
        ctype = CHALLENGE_TYPES[name]
        r = audit_one_map(name, ctype, args.n_seeds)
        mct = f"{r['mean_confirm_time']:.1f}s" if r["mean_confirm_time"] is not None else "n/a"
        print(
            f"  {name:<10} success {r['successes']}/{r['n_seeds']} "
            f"({r['success_rate']:.2%}), mean_t={mct}, "
            f"reasons={r['reasons']}, elapsed={r['elapsed_sec']:.0f}s"
        )
    print("-" * 72)
    print("baseline metrics emitted; calibrate thresholds in the cutover runbook")
    return 0


if __name__ == "__main__":
    sys.exit(main())
