#!/usr/bin/env python3
"""D.3.2 audit — episode-horizon timeout fraction at max distance.

Runs the scripted baseline at the farthest-distance mountain scenario and
measures timeout fraction. Plan target: 1000 seeds, < 30% TIMEOUT.

Usage:
    python3 scripts/sar_horizon_audit.py                  # full 1000
    python3 scripts/sar_horizon_audit.py --n-seeds 50     # smoke
"""
from __future__ import annotations

import argparse
import sys

from sar_baseline_audit import run_one_episode


DEFAULT_N_SEEDS = 1000
DEFAULT_THRESHOLD = 0.30


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = ap.parse_args(argv)

    mountain = 3
    timeouts = 0
    for seed in range(args.n_seeds):
        rec = run_one_episode(seed, mountain)
        if rec["failure_reason"] == "TIMEOUT":
            timeouts += 1
    rate = timeouts / args.n_seeds if args.n_seeds else 0.0
    verdict = "PASS" if rate < args.threshold else "FAIL"
    print(
        f"sar horizon audit (mountain) · timeouts {timeouts}/{args.n_seeds} "
        f"({rate:.2%}) vs threshold {args.threshold:.2%}  {verdict}"
    )
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
