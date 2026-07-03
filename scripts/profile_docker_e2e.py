#!/usr/bin/env python3
"""
Real Docker RPC end-to-end wall-time measurement
================================================
Runs a hover agent (zero action, never terminates early, no pip deps) through
the production Docker evaluator for every challenge family, so each seed runs
the full horizon and the numbers include container setup, Cap'n Proto RPC,
observation transport, and act() round trips.

Compare per-seed walls against scripts/profile_walltime.py (in-process) to
attribute the RPC/container share.

Usage:
    python3 scripts/profile_docker_e2e.py --json docker_e2e.json
"""

import argparse
import asyncio
import io
import json
import os
import random
import re
import statistics
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

os.environ.setdefault("SWARM_LOG_RPC_TRACE", "1")
os.environ.setdefault("SWARM_LOG_RPC_TRACE_EVERY", "500")

from swarm.constants import SIM_DT, SWARM_COUNT_SEED_OFFSET, SWARM_MAX_DRONES, SWARM_MIN_DRONES
from swarm.validator.task_gen import task_for_seed_and_type

HOVER_AGENT = '''import numpy as np


class DroneFlightController:
    def __init__(self):
        pass

    def act(self, observation):
        state = observation.get("state") if isinstance(observation, dict) else None
        n = state.shape[0] if state is not None and getattr(state, "ndim", 1) > 1 else 1
        return np.zeros((n, 6), dtype=np.float32) if n > 1 else np.zeros(6, dtype=np.float32)

    def reset(self):
        pass
'''

ENV_BUILT_RE = re.compile(r"env built in ([\d.]+)s")
ACT_RE = re.compile(r"act_ok=([\d.]+)ms")
OVERHEAD_RE = re.compile(r"overhead=([\d.]+)ms")


def _relax_timeouts():
    import swarm.constants as C
    import swarm.validator.docker.docker_evaluator as DE
    import swarm.validator.docker.docker_evaluator_parts.batch as BATCH
    import swarm.validator.docker.docker_evaluator_parts.rpc as RPC

    for var in ("SWARM_BATCH_TIMEOUT_HARD_CAP_SEC", "SWARM_BATCH_TIMEOUT_MAX_TOTAL_SEC"):
        os.environ.pop(var, None)

    overrides = {
        "GLOBAL_EVAL_BASE_SEC": 36000.0,
        "GLOBAL_EVAL_PER_SEED_SEC": 3600.0,
        "GLOBAL_EVAL_CAP_SEC": 36000.0,
        "CALIBRATION_RECAL_INTERVAL": 0,
    }
    for mod in (C, DE, RPC, BATCH):
        for name, val in overrides.items():
            if hasattr(mod, name):
                setattr(mod, name, val)


def seed_with_drone_count(target_n, start=1000):
    seed = start
    while True:
        rng = random.Random((seed + SWARM_COUNT_SEED_OFFSET) & 0xFFFFFFFF)
        if rng.randint(SWARM_MIN_DRONES, SWARM_MAX_DRONES) == target_n:
            return seed
        seed += 1


class _Tee(io.TextIOBase):
    def __init__(self, real):
        self.real = real
        self.parts = []

    def write(self, s):
        self.real.write(s)
        self.parts.append(s)
        return len(s)

    def flush(self):
        self.real.flush()

    def text(self):
        return "".join(self.parts)


def build_configs():
    n5_a = seed_with_drone_count(5)
    n5_b = seed_with_drone_count(5, start=n5_a + 1)
    return [
        ("cf_autopilot", 2, [104, 105]),
        ("cf_search_and_rescue", 2, [104, 105]),
        ("cf_swarm_autopilot", 2, [n5_a, n5_b]),
        ("cf_swarm_sar", 2, [n5_a, n5_b]),
        ("cf_interceptor", 2, [7, 21]),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="docker_e2e.json")
    ap.add_argument("--workdir", default="/tmp/walltime_e2e")
    ap.add_argument("--families", default="", help="comma-separated family filter")
    ap.add_argument("--seeds-per-family", type=int, default=0, help="limit seeds per family")
    args = ap.parse_args()

    _relax_timeouts()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    model_path = workdir / "hover_model.zip"
    with zipfile.ZipFile(model_path, "w") as zf:
        zf.writestr("drone_agent.py", HOVER_AGENT)
    print(f"Hover model written to {model_path}")

    from swarm.validator.docker.docker_evaluator import DockerSecureEvaluator

    evaluator = DockerSecureEvaluator()
    summary = []

    configs = build_configs()
    if args.families:
        wanted = {f.strip() for f in args.families.split(",") if f.strip()}
        configs = [c for c in configs if c[0] in wanted]
    if args.seeds_per_family > 0:
        configs = [(f, c, seeds[: args.seeds_per_family]) for f, c, seeds in configs]

    for family, ctype, seeds in configs:
        tasks = [
            task_for_seed_and_type(SIM_DT, seed=s, challenge_type=ctype, family_id=family)
            for s in seeds
        ]
        payloads = []

        def on_seed_complete(payload=None):
            if payload:
                payloads.append(payload)

        print(f"\n>>> {family} type={ctype} seeds={seeds}", flush=True)
        tee = _Tee(sys.stdout)
        sys.stdout = tee
        t0 = time.perf_counter()
        try:
            results = asyncio.run(
                evaluator.evaluate_seeds_batch(
                    tasks, 0, model_path, worker_id=0, on_seed_complete=on_seed_complete
                )
            )
        finally:
            sys.stdout = tee.real
        batch_wall = time.perf_counter() - t0
        log = tee.text()

        env_builds = [float(x) for x in ENV_BUILT_RE.findall(log)]
        acts = [float(x) for x in ACT_RE.findall(log)]
        overheads = [float(x) for x in OVERHEAD_RE.findall(log)]
        seed_walls = [p["seed_wall_sec"] for p in payloads]
        steps = [p["step_idx"] for p in payloads]

        row = {
            "family": family,
            "ctype": ctype,
            "seeds": seeds,
            "batch_wall_s": batch_wall,
            "seed_walls_s": seed_walls,
            "setup_wall_s": batch_wall - sum(seed_walls),
            "steps_per_seed": steps,
            "statuses": [p["status"] for p in payloads],
            "sim_time_sec": [p["sim_time_sec"] for p in payloads],
            "env_build_s": env_builds,
            "act_ms_p50": statistics.median(acts) if acts else 0.0,
            "act_samples": len(acts),
            "rpc_overhead_ms": statistics.median(overheads) if overheads else 0.0,
            "scores": [float(getattr(r, "score", 0.0)) for r in results],
        }
        summary.append(row)
        ms_per_step = [
            1000.0 * w / s for w, s in zip(seed_walls, steps) if s
        ]
        print(
            f"<<< {family}: batch={batch_wall:.0f}s setup={row['setup_wall_s']:.0f}s "
            f"seed_walls={[round(w) for w in seed_walls]}s steps={steps} "
            f"({[round(m, 1) for m in ms_per_step]} ms/step) "
            f"act_p50={row['act_ms_p50']:.1f}ms rpc_overhead={row['rpc_overhead_ms']:.1f}ms",
            flush=True,
        )

    with open(args.json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nJSON written to {args.json}")


if __name__ == "__main__":
    main()
