#!/usr/bin/env python3
"""
Validator wall-time profiler
============================
Attributes evaluation cost per stage for every challenge family and map type,
using the real validator code path: task_gen -> env_factory -> env.step(),
with the real Cap'n Proto observation serialization from the docker evaluator.

Per config it reports:
  task build / env build (world gen) / per-step render / physics / clearance /
  other (obs assembly etc.) / observation serialize+parse / payload size,
then extrapolates to a full seed (horizon steps) and a full 1,100-seed eval.

Usage:
    python3 scripts/profile_walltime.py                 # full sweep
    python3 scripts/profile_walltime.py --quick         # smoke test
    python3 scripts/profile_walltime.py --json out.json
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import capnp
import numpy as np
import pybullet as p

from swarm.constants import (
    SIM_DT,
    SWARM_COUNT_SEED_OFFSET,
    SWARM_MAX_DRONES,
    SWARM_MIN_DRONES,
    BENCHMARK_TOTAL_SEED_COUNT,
)
from swarm.utils.env_factory import make_env_with_initial_obs
from swarm.validator.docker.docker_evaluator_parts._shared import _submission_template_dir
from swarm.validator.docker.docker_evaluator_parts.submission import _serialize_observation
from swarm.validator.task_gen import task_for_seed_and_type

MAP_LABELS = {1: "City", 2: "Open", 3: "Mountain", 4: "Village", 5: "Warehouse", 6: "Forest"}
SINGLE_FAMILIES = ("cf_autopilot", "cf_search_and_rescue")
SWARM_FAMILIES = ("cf_swarm_autopilot", "cf_swarm_sar")

_TIMED_CALLS = ("getCameraImage", "stepSimulation", "getClosestPoints", "rayTest")


class BulletTimer:
    """Accumulates wall time spent inside selected pybullet C calls."""

    def __init__(self):
        self.ms = defaultdict(float)
        self.calls = defaultdict(int)
        self._orig = {}

    def _wrap(self, name, fn):
        def timed(*args, **kwargs):
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            self.ms[name] += (time.perf_counter() - t0) * 1000.0
            self.calls[name] += 1
            return out

        return timed

    def __enter__(self):
        for name in _TIMED_CALLS:
            fn = getattr(p, name)
            self._orig[name] = fn
            setattr(p, name, self._wrap(name, fn))
        return self

    def __exit__(self, *exc):
        for name, fn in self._orig.items():
            setattr(p, name, fn)
        self._orig.clear()

    def snapshot(self):
        return dict(self.ms), dict(self.calls)


def _delta(after, before):
    return {k: after.get(k, 0.0) - before.get(k, 0.0) for k in set(after) | set(before)}


def seed_with_drone_count(target_n, start=1000):
    seed = start
    while True:
        rng = random.Random((seed + SWARM_COUNT_SEED_OFFSET) & 0xFFFFFFFF)
        if rng.randint(SWARM_MIN_DRONES, SWARM_MAX_DRONES) == target_n:
            return seed
        seed += 1


def profile_config(agent_capnp, family, ctype, seed, steps, warmup):
    t0 = time.perf_counter()
    task = task_for_seed_and_type(SIM_DT, seed=seed, challenge_type=ctype, family_id=family)
    task_ms = (time.perf_counter() - t0) * 1000.0

    result = {
        "family": family,
        "ctype": ctype,
        "map": MAP_LABELS[ctype],
        "seed": seed,
        "task_ms": task_ms,
    }

    with BulletTimer() as timer:
        t0 = time.perf_counter()
        env, obs = make_env_with_initial_obs(task)
        build_ms = (time.perf_counter() - t0) * 1000.0
        build_bullet_ms, _ = timer.snapshot()

        cli = getattr(env, "CLIENT", 0)
        result.update(
            build_ms=build_ms,
            build_render_ms=build_bullet_ms.get("getCameraImage", 0.0),
            n_drones=int(getattr(env, "NUM_DRONES", 1)),
            img_res=[int(x) for x in getattr(env, "IMG_RES", (0, 0))],
            bodies=int(p.getNumBodies(physicsClientId=cli)),
            horizon=float(task.horizon),
            steps_per_seed=int(round(task.horizon / task.sim_dt)),
        )

        action = np.zeros(env.action_space.shape, dtype=np.float32)
        step_ms, ser_ms, parse_ms = [], [], []
        stage_ms = defaultdict(list)
        stage_calls = defaultdict(list)
        obs_bytes = 0

        for i in range(warmup + steps):
            before_ms, before_calls = timer.snapshot()
            t0 = time.perf_counter()
            obs, _r, term, trunc, _info = env.step(action)
            step_i = (time.perf_counter() - t0) * 1000.0
            after_ms, after_calls = timer.snapshot()

            t0 = time.perf_counter()
            msg = _serialize_observation(agent_capnp, obs)
            data = msg.to_bytes()
            ser_i = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            with agent_capnp.Observation.from_bytes(data) as parsed:
                rebuilt = 0
                for entry in parsed.entries:
                    shape = tuple(entry.tensor.shape)
                    dtype = np.dtype(entry.tensor.dtype)
                    if len(entry.tensor.data) == 0:
                        rebuilt += int(np.zeros(shape, dtype=dtype).size)
                    else:
                        rebuilt += int(
                            np.frombuffer(entry.tensor.data, dtype=dtype).reshape(shape).size
                        )
            parse_i = (time.perf_counter() - t0) * 1000.0
            assert rebuilt >= 1

            if i >= warmup:
                step_ms.append(step_i)
                ser_ms.append(ser_i)
                parse_ms.append(parse_i)
                obs_bytes = len(data)
                for name, val in _delta(after_ms, before_ms).items():
                    stage_ms[name].append(val)
                for name, val in _delta(after_calls, before_calls).items():
                    stage_calls[name].append(val)

            if term or trunc:
                env.reset(seed=task.map_seed)

        env.close()

    step = float(np.mean(step_ms))
    render = float(np.mean(stage_ms.get("getCameraImage", [0.0])))
    physics = float(np.mean(stage_ms.get("stepSimulation", [0.0])))
    clearance = float(
        np.mean(stage_ms.get("getClosestPoints", [0.0]))
        + np.mean(stage_ms.get("rayTest", [0.0]))
    )
    ser = float(np.mean(ser_ms))
    parse = float(np.mean(parse_ms))
    per_step_total = step + ser + parse
    per_seed_s = (result["build_ms"] + result["steps_per_seed"] * per_step_total) / 1000.0

    result.update(
        step_ms_mean=step,
        step_ms_std=float(np.std(step_ms)),
        render_ms=render,
        physics_ms=physics,
        clearance_ms=clearance,
        other_ms=step - render - physics - clearance,
        serialize_ms=ser,
        parse_ms=parse,
        obs_kb=obs_bytes / 1024.0,
        render_calls_per_step=float(np.mean(stage_calls.get("getCameraImage", [0.0]))),
        clearance_calls_per_step=float(
            np.mean(stage_calls.get("getClosestPoints", [0.0]))
            + np.mean(stage_calls.get("rayTest", [0.0]))
        ),
        per_step_total_ms=per_step_total,
        per_seed_s=per_seed_s,
    )
    return result


def terrain_rebuild_check(steps_seed_a=13, seed_b=42):
    """Cost of building the Mountain env: cold, warm same seed, different seed."""
    out = []
    for label, seed in (("cold seed A", steps_seed_a), ("same seed A again", steps_seed_a), ("different seed B", seed_b)):
        task = task_for_seed_and_type(SIM_DT, seed=seed, challenge_type=3, family_id="cf_autopilot")
        t0 = time.perf_counter()
        env, _obs = make_env_with_initial_obs(task)
        ms = (time.perf_counter() - t0) * 1000.0
        env.close()
        out.append({"label": label, "seed": seed, "build_ms": ms})
    return out


def build_sweep(quick):
    sweep = []
    if quick:
        sweep.append(("cf_autopilot", 2, 4))
        sweep.append(("cf_autopilot", 6, 106))
        sweep.append(("cf_interceptor", 2, 7))
        sweep.append(("cf_swarm_sar", 2, seed_with_drone_count(5)))
        sweep.append(("cf_swarm_sar", 6, seed_with_drone_count(5)))
        return sweep

    for family in SINGLE_FAMILIES:
        for ctype in MAP_LABELS:
            sweep.append((family, ctype, 100 + ctype))

    for family in SWARM_FAMILIES:
        for ctype in MAP_LABELS:
            sweep.append((family, ctype, seed_with_drone_count(5)))
        for n in (2, 8):
            sweep.append((family, 2, seed_with_drone_count(n)))

    sweep.append(("cf_interceptor", 2, 7))
    sweep.append(("cf_interceptor", 2, 21))
    return sweep


def print_table(results):
    hdr = (
        f"{'family':<22} {'map':<10} {'n':>2} {'res':>5} {'bodies':>6} "
        f"{'build_s':>8} {'step_ms':>8} {'render':>7} {'physic':>7} {'clear':>7} "
        f"{'other':>7} {'ser':>6} {'obs_kb':>7} {'seed_s':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['family']:<22} {r['map']:<10} {r['n_drones']:>2} {r['img_res'][0]:>5} "
            f"{r['bodies']:>6} {r['build_ms']/1000:>8.2f} {r['step_ms_mean']:>8.2f} "
            f"{r['render_ms']:>7.2f} {r['physics_ms']:>7.2f} {r['clearance_ms']:>7.2f} "
            f"{r['other_ms']:>7.2f} {r['serialize_ms']+r['parse_ms']:>6.2f} "
            f"{r['obs_kb']:>7.0f} {r['per_seed_s']:>7.1f}"
        )


def print_family_totals(results):
    per_family = defaultdict(list)
    for r in results:
        per_family[r["family"]].append(r["per_seed_s"])
    print("\nValidator-side cost only: excludes the RPC round trip and miner compute")
    print("(see scripts/profile_docker_e2e.py for end-to-end numbers).")
    print(f"\n{'family':<22} {'avg seed_s':>10} {'x1100 core-h':>13} {'@6 workers':>11}")
    print("-" * 60)
    for family, seeds_s in per_family.items():
        avg = float(np.mean(seeds_s))
        core_h = avg * BENCHMARK_TOTAL_SEED_COUNT / 3600.0
        print(f"{family:<22} {avg:>10.1f} {core_h:>13.1f} {core_h/6:>10.1f}h")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--json", default="walltime_profile.json")
    args = ap.parse_args()

    if args.quick:
        args.steps = min(args.steps, 25)
        args.warmup = min(args.warmup, 4)

    agent_capnp = capnp.load(str(_submission_template_dir() / "agent.capnp"))
    sweep = build_sweep(args.quick)

    results = []
    for family, ctype, seed in sweep:
        label = f"{family}/{MAP_LABELS[ctype]}/seed={seed}"
        print(f"[{len(results)+1}/{len(sweep)}] {label} ...", flush=True)
        t0 = time.perf_counter()
        r = profile_config(agent_capnp, family, ctype, seed, args.steps, args.warmup)
        r["config_wall_s"] = time.perf_counter() - t0
        results.append(r)
        print(
            f"    n={r['n_drones']} res={r['img_res']} step={r['step_ms_mean']:.2f}ms "
            f"(render {r['render_ms']:.2f} / physics {r['physics_ms']:.2f} / "
            f"clear {r['clearance_ms']:.2f}) ser={r['serialize_ms']+r['parse_ms']:.2f}ms "
            f"-> {r['per_seed_s']:.1f}s/seed",
            flush=True,
        )

    print("\nMountain terrain rebuild check:")
    terrain = terrain_rebuild_check()
    for t in terrain:
        print(f"    {t['label']:<22} build={t['build_ms']/1000:.2f}s")

    print()
    print_table(results)
    print_family_totals(results)

    with open(args.json, "w") as f:
        json.dump({"results": results, "terrain_check": terrain}, f, indent=2)
    print(f"\nJSON written to {args.json}")


if __name__ == "__main__":
    main()
