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
    python3 validator/scripts/profile_walltime.py                 # full sweep
    python3 validator/scripts/profile_walltime.py --quick         # smoke test
    python3 validator/scripts/profile_walltime.py --json out.json
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))

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

_TIMED_CALLS = (
    "getCameraImage",
    "getDepthImagesBatch",
    "stepSimulation",
    "getClosestPoints",
    "rayTest",
    "applyExternalForce",
    "applyExternalTorque",
    "getBasePositionAndOrientation",
    "getBaseVelocity",
    "getContactPoints",
)

_PY_PHASES = (
    ("ctrl", "_preprocessAction"),
    ("obs_total", "_computeObs"),
    ("reward", "_computeReward"),
    ("terminated", "_computeTerminated"),
    ("truncated", "_computeTruncated"),
    ("info", "_computeInfo"),
    ("bookkeeping", "_process_step_updates"),
    ("rgb_serve", "_update_rgb_requests"),
)


class PhaseTimer:
    """Accumulates wall time of selected env methods via instance wrappers."""

    def __init__(self, env):
        self.ms = defaultdict(float)
        self._env = env
        self._orig = {}
        for phase, name in _PY_PHASES:
            fn = getattr(env, name, None)
            if fn is None:
                continue
            self._orig[name] = fn
            setattr(env, name, self._wrap(phase, fn))

    def _wrap(self, phase, fn):
        def timed(*args, **kwargs):
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            self.ms[phase] += (time.perf_counter() - t0) * 1000.0
            return out

        return timed

    def snapshot(self):
        return dict(self.ms)

    def restore(self):
        for name, fn in self._orig.items():
            setattr(self._env, name, fn)


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
            fn = getattr(p, name, None)
            if fn is None:
                continue
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
        phase_ms = defaultdict(list)
        obs_bytes = 0
        phases = PhaseTimer(env)

        for i in range(warmup + steps):
            before_ms, before_calls = timer.snapshot()
            before_phases = phases.snapshot()
            t0 = time.perf_counter()
            obs, _r, term, trunc, _info = env.step(action)
            step_i = (time.perf_counter() - t0) * 1000.0
            after_ms, after_calls = timer.snapshot()
            after_phases = phases.snapshot()

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
                for name, val in _delta(after_phases, before_phases).items():
                    phase_ms[name].append(val)

            if term or trunc:
                env.reset(seed=task.map_seed)

        phases.restore()
        env.close()

    def _avg(table, name):
        return float(np.mean(table.get(name, [0.0])))

    step = float(np.mean(step_ms))
    render = _avg(stage_ms, "getCameraImage") + _avg(stage_ms, "getDepthImagesBatch")
    physics = _avg(stage_ms, "stepSimulation")
    phys_apply = _avg(stage_ms, "applyExternalForce") + _avg(stage_ms, "applyExternalTorque")
    kinematics = _avg(stage_ms, "getBasePositionAndOrientation") + _avg(stage_ms, "getBaseVelocity")
    contacts = _avg(stage_ms, "getContactPoints")
    clearance = _avg(stage_ms, "getClosestPoints") + _avg(stage_ms, "rayTest")
    ctrl = _avg(phase_ms, "ctrl")
    obs_total = _avg(phase_ms, "obs_total")
    obs_build = max(0.0, obs_total - render)
    scoring = (
        _avg(phase_ms, "reward")
        + _avg(phase_ms, "terminated")
        + _avg(phase_ms, "truncated")
        + _avg(phase_ms, "info")
    )
    bookkeeping = _avg(phase_ms, "bookkeeping") + _avg(phase_ms, "rgb_serve")
    leftover = step - ctrl - obs_total - scoring - bookkeeping - physics - phys_apply - kinematics
    ser = float(np.mean(ser_ms))
    parse = float(np.mean(parse_ms))
    per_step_total = step + ser + parse
    per_seed_s = (result["build_ms"] + result["steps_per_seed"] * per_step_total) / 1000.0

    result.update(
        step_ms_mean=step,
        step_ms_std=float(np.std(step_ms)),
        render_ms=render,
        physics_ms=physics,
        phys_apply_ms=phys_apply,
        kinematics_ms=kinematics,
        contacts_ms=contacts,
        clearance_ms=clearance,
        ctrl_ms=ctrl,
        obs_build_ms=obs_build,
        scoring_ms=scoring,
        bookkeeping_ms=bookkeeping,
        leftover_ms=leftover,
        other_ms=step - render - physics - clearance,
        serialize_ms=ser,
        parse_ms=parse,
        obs_kb=obs_bytes / 1024.0,
        render_calls_per_step=float(
            np.mean(stage_calls.get("getCameraImage", [0.0]))
            + np.mean(stage_calls.get("getDepthImagesBatch", [0.0]))
        ),
        batch_depth_active=bool(getattr(env, "_use_batch_depth", False)),
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
        sweep.append(("cf_interceptor_office", 7, 7))
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

    sweep.append(("cf_interceptor_office", 7, 7))
    sweep.append(("cf_interceptor_office", 7, 21))
    return sweep


def print_table(results):
    hdr = (
        f"{'family':<20} {'map':<9} {'n':>2} {'batch':>5} {'step':>7} {'render':>7} "
        f"{'ctrl':>6} {'obsbld':>7} {'physic':>7} {'apply':>6} {'kin':>6} "
        f"{'score':>6} {'book':>6} {'left':>6} {'ser':>5} {'seed_s':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['family'].replace('cf_', ''):<20} {r['map']:<9} {r['n_drones']:>2} "
            f"{'Y' if r.get('batch_depth_active') else 'n':>5} "
            f"{r['step_ms_mean']:>7.2f} {r['render_ms']:>7.2f} "
            f"{r['ctrl_ms']:>6.2f} {r['obs_build_ms']:>7.2f} {r['physics_ms']:>7.2f} "
            f"{r['phys_apply_ms']:>6.2f} {r['kinematics_ms']:>6.2f} "
            f"{r['scoring_ms']:>6.2f} {r['bookkeeping_ms']:>6.2f} {r['leftover_ms']:>6.2f} "
            f"{r['serialize_ms']+r['parse_ms']:>5.2f} {r['per_seed_s']:>7.1f}"
        )


def print_family_totals(results):
    per_family = defaultdict(list)
    for r in results:
        per_family[r["family"]].append(r["per_seed_s"])
    print("\nValidator-side cost only: excludes the RPC round trip and miner compute.")
    print(f"\n{'family':<22} {'avg seed_s':>10} {'x1100 core-h':>13} {'@6 workers':>11}")
    print("-" * 60)
    for family, seeds_s in per_family.items():
        avg = float(np.mean(seeds_s))
        core_h = avg * BENCHMARK_TOTAL_SEED_COUNT / 3600.0
        print(f"{family:<22} {avg:>10.1f} {core_h:>13.1f} {core_h/6:>10.1f}h")


def _next_seed_same_count(family, prev_seed):
    if family in SWARM_FAMILIES:
        rng = random.Random((prev_seed + SWARM_COUNT_SEED_OFFSET) & 0xFFFFFFFF)
        n = rng.randint(SWARM_MIN_DRONES, SWARM_MAX_DRONES)
        return seed_with_drone_count(n, start=prev_seed + 1)
    return prev_seed + 17


def _aggregate_seed_runs(runs):
    agg = dict(runs[0])
    numeric = [k for k, v in runs[0].items() if isinstance(v, (int, float)) and k != "seed"]
    for k in numeric:
        agg[k] = float(np.mean([r[k] for r in runs]))
    agg["seeds_profiled"] = len(runs)
    agg["step_ms_min"] = float(min(r["step_ms_mean"] for r in runs))
    agg["step_ms_max"] = float(max(r["step_ms_mean"] for r in runs))
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=1, help="seeds per family/map config")
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
        print(f"[{len(results)+1}/{len(sweep)}] {label} x{args.seeds} seeds ...", flush=True)
        t0 = time.perf_counter()
        runs = []
        s = seed
        for k in range(args.seeds):
            runs.append(profile_config(agent_capnp, family, ctype, s, args.steps, args.warmup))
            s = _next_seed_same_count(family, s)
        r = _aggregate_seed_runs(runs)
        r["config_wall_s"] = time.perf_counter() - t0
        results.append(r)
        print(
            f"    n={r['n_drones']} res={r['img_res']} batch={'Y' if r.get('batch_depth_active') else 'n'} "
            f"step={r['step_ms_mean']:.2f}ms "
            f"(render {r['render_ms']:.2f} / ctrl {r['ctrl_ms']:.2f} / obs {r['obs_build_ms']:.2f} / "
            f"physics {r['physics_ms']:.2f}) ser={r['serialize_ms']+r['parse_ms']:.2f}ms "
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
