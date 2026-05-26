#!/usr/bin/env python3
"""D.3.2 audit — SAR spawn-pipeline failure rate per environment type.

Target: ≤ 0.5% SARSpawnError rate per map at the full 5000-seed sample
size. Run before the cutover.

Usage:
    python3 scripts/sar_spawn_audit.py                       # full 5000 / map
    python3 scripts/sar_spawn_audit.py --n-seeds 500         # smoke
    python3 scripts/sar_spawn_audit.py --maps city,warehouse # subset

Exit code 0 when every environment type clears the threshold, 1 otherwise.
"""
from __future__ import annotations

import argparse
import sys
import time

import pybullet as p

from swarm.core.env_builder.sar_tagging import build_and_tag_map
from swarm.core.env_builder.spawn_pipeline import SARSpawnError, find_spawn_xy


CHALLENGE_TYPES = {
    "city":      1,
    "open":      2,
    "mountain":  3,
    "village":   4,
    "warehouse": 5,
    "forest":    6,
}

DEFAULT_THRESHOLD = 0.005
DEFAULT_N_SEEDS = 5000


def audit_one_map(cli: int, name: str, ctype: int, n_seeds: int) -> tuple[int, int, float]:
    fails = 0
    started = time.time()
    for seed in range(n_seeds):
        p.resetSimulation(physicsClientId=cli)
        tagger = build_and_tag_map(
            cli, seed=seed, challenge_type=ctype,
            start=(0.0, 0.0, 1.5), goal=(8.0, 8.0, 1.5),
        )
        try:
            find_spawn_xy(
                cli,
                map_seed=seed,
                challenge_type=ctype,
                body_tags=tagger.body_tags,
            )
        except SARSpawnError:
            fails += 1
    elapsed = time.time() - started
    return fails, n_seeds, elapsed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument(
        "--maps",
        default=",".join(CHALLENGE_TYPES.keys()),
        help="comma-separated map names",
    )
    args = ap.parse_args(argv)

    selected = [m.strip() for m in args.maps.split(",") if m.strip()]
    for m in selected:
        if m not in CHALLENGE_TYPES:
            print(f"unknown map: {m!r}; expected one of {list(CHALLENGE_TYPES)}")
            return 2

    cli = p.connect(p.DIRECT)
    overall_pass = True
    print(f"sar spawn audit · n_seeds={args.n_seeds} · threshold={args.threshold:.2%}")
    print("-" * 72)
    try:
        for name in selected:
            ctype = CHALLENGE_TYPES[name]
            fails, n, elapsed = audit_one_map(cli, name, ctype, args.n_seeds)
            rate = fails / n if n else 0.0
            verdict = "PASS" if rate <= args.threshold else "FAIL"
            if verdict == "FAIL":
                overall_pass = False
            print(
                f"  {name:<10} {fails:>5} / {n:<5} fails  ({rate:.4%}) "
                f"in {elapsed:>5.0f}s  {verdict}"
            )
    finally:
        p.disconnect(cli)
    print("-" * 72)
    print("OVERALL: PASS" if overall_pass else "OVERALL: FAIL")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
