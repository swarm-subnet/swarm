#!/usr/bin/env python3
"""Debug-only repeated-seed report for score idempotency.

Runs one exact seed per map group multiple times through the normal Docker
evaluator path and prints a benchmark-like grouped report.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

BENCH_GROUP_ORDER = [
    "type1_city",
    "type2_open",
    "type3_mountain",
    "type4_village",
    "type5_warehouse",
    "type6_forest",
]
BENCH_GROUP_TO_TYPE = {
    "type1_city": 1,
    "type2_open": 2,
    "type3_mountain": 3,
    "type4_village": 4,
    "type5_warehouse": 5,
    "type6_forest": 6,
}


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _find_seeds(seeds_per_group: int):
    from swarm.benchmark.engine import _find_seeds as _real_find_seeds

    return _real_find_seeds(seeds_per_group)


def _load_type_seeds(seed_file: Path):
    from swarm.benchmark.engine import _load_type_seeds as _real_load_type_seeds

    return _real_load_type_seeds(seed_file)


def _infer_uid_from_model_path(model_path: Path):
    from swarm.benchmark.engine import _infer_uid_from_model_path as _real_infer_uid

    return _real_infer_uid(model_path)


def _apply_relaxed_overrides():
    from swarm.benchmark.engine_parts.config import _apply_relaxed_overrides as _real_relax

    return _real_relax()


def run_idempotency(**kwargs):
    from swarm.benchmark.idempotency import run_idempotency as _real_run_idempotency

    return _real_run_idempotency(**kwargs)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeat one exact seed per map group and print an idempotency report.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to submission zip (for example model/UID_178.zip).",
    )
    parser.add_argument(
        "--runs-per-map",
        type=int,
        default=5,
        help="How many times to repeat the chosen seed for each map group (default: 5).",
    )
    parser.add_argument(
        "--uid",
        type=int,
        default=None,
        help="Miner UID. If omitted, inferred from the model filename when possible.",
    )
    parser.add_argument(
        "--seed-file",
        type=Path,
        default=None,
        help=(
            "Optional benchmark seed file in the normal group->seed-list JSON format. "
            "The first seed from each group is reused."
        ),
    )
    parser.add_argument(
        "--moving-platform",
        choices=["auto", "on", "off"],
        default="auto",
        help="Override moving-platform mode for all repeated tasks (default: auto).",
    )
    parser.add_argument(
        "--relax-timeouts",
        action="store_true",
        help="Enable the same slow-machine timeout overrides used by benchmark.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path for machine-readable JSON output.",
    )
    return parser.parse_args(argv)


def _resolve_group_seeds(seed_file: Optional[Path]) -> Dict[str, int]:
    if seed_file is None:
        raw = _find_seeds(1)
    else:
        raw = _load_type_seeds(seed_file)

    resolved: Dict[str, int] = {}
    for group in BENCH_GROUP_ORDER:
        values = raw.get(group)
        if not values:
            raise ValueError(f"Seed group {group} is missing or empty.")
        resolved[group] = int(values[0])
    return resolved


def _print_report(group_summaries: Dict[str, Dict[str, Any]]) -> None:
    print()
    print(f"  {'Group':<18} {'Seed':>8} {'Run':>5} {'Score':>7} {'OK?':>5} {'SimT':>7} {'WallT':>7}")
    print(f"  {'-'*18} {'-'*8} {'-'*5} {'-'*7} {'-'*5} {'-'*7} {'-'*7}")

    for group in BENCH_GROUP_ORDER:
        summary = group_summaries[group]
        rows = summary["runs"]
        for row in rows:
            ok = "Y" if row["success"] else "N"
            print(
                f"  {group:<18} {summary['seed']:>8} {row['run']:>5} "
                f"{row['score']:>7.4f} {ok:>5} {row['time_sec']:>6.2f}s {row['wall_time_sec']:>6.1f}s"
            )
        avg_score = sum(float(row["score"]) for row in rows) / len(rows)
        avg_wall = sum(float(row["wall_time_sec"]) for row in rows) / len(rows)
        print(
            f"  {'  -> AVG':<18} {'':>8} {'':>5} "
            f"{avg_score:>7.4f} {'':>5} {'':>6} {avg_wall:>6.1f}s"
        )
        print(
            f"  {'  -> SCORE IDEMP':<18} {'':>8} {'':>5} "
            f"{'Y' if summary['idempotent_score'] else 'N':>7}"
        )
        print(f"  {'  -> UNIQUE SCORES':<18} {str(summary['unique_scores'])}")
        print(f"  {'  -> UNIQUE SimT':<18} {str(summary['unique_sim_times'])}")
        print(f"  {'  -> UNIQUE WallT':<18} {str(summary['unique_wall_times'])}")
        print()


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    model_path = args.model.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if args.runs_per_map <= 0:
        raise ValueError("--runs-per-map must be positive")

    inferred_uid = _infer_uid_from_model_path(model_path)
    uid = int(args.uid) if args.uid is not None else int(inferred_uid or 0)
    moving_platform: bool | None
    if args.moving_platform == "auto":
        moving_platform = None
    else:
        moving_platform = args.moving_platform == "on"

    overrides: Dict[str, object] = {}
    if args.relax_timeouts:
        overrides = _apply_relaxed_overrides()

    group_seeds = _resolve_group_seeds(args.seed_file)

    print(f"[{_ts()}] === IDEMPOTENCY DEBUG REPORT ===")
    print(f"[{_ts()}] Model: {model_path}")
    print(f"[{_ts()}] UID: {uid}")
    print(f"[{_ts()}] Runs per map: {args.runs_per_map}")
    print(f"[{_ts()}] Seed source: {args.seed_file if args.seed_file is not None else 'auto-find one seed per group'}")
    if overrides:
        print(f"[{_ts()}] Relaxed timeouts: {overrides}")
    print(f"[{_ts()}] Selected seeds:")
    for group in BENCH_GROUP_ORDER:
        print(f"  {group}: {group_seeds[group]}")

    run_start = time.time()
    group_summaries: Dict[str, Dict[str, Any]] = {}
    for worker_id, group in enumerate(BENCH_GROUP_ORDER):
        summary = run_idempotency(
            model_path=model_path,
            uid=uid,
            seed=group_seeds[group],
            challenge_type=BENCH_GROUP_TO_TYPE[group],
            runs=int(args.runs_per_map),
            moving_platform=moving_platform,
            worker_id=worker_id,
        )
        group_summaries[group] = summary

    elapsed = time.time() - run_start

    print()
    print(f"[{_ts()}] === RESULTS ===")
    _print_report(group_summaries)

    score_idempotent_groups = sum(
        1 for group in BENCH_GROUP_ORDER if group_summaries[group]["idempotent_score"]
    )
    strict_idempotent_groups = sum(
        1 for group in BENCH_GROUP_ORDER if group_summaries[group]["strict_idempotent"]
    )
    print("  Run summary:")
    print(f"    Groups tested:             {len(BENCH_GROUP_ORDER)}")
    print(f"    Runs per group:            {args.runs_per_map}")
    print(f"    Score-idempotent groups:   {score_idempotent_groups}/{len(BENCH_GROUP_ORDER)}")
    print(f"    Strict-idempotent groups:  {strict_idempotent_groups}/{len(BENCH_GROUP_ORDER)}")
    print(f"    Total wall-clock:          {elapsed:.1f}s ({elapsed / 60.0:.1f} min)")

    if args.json_out is not None:
        payload = {
            "uid": uid,
            "model_path": str(model_path),
            "runs_per_map": int(args.runs_per_map),
            "group_seeds": group_seeds,
            "group_summaries": group_summaries,
            "wall_clock_sec": elapsed,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"    JSON report:               {args.json_out}")

    all_score_idempotent = score_idempotent_groups == len(BENCH_GROUP_ORDER)
    return 0 if all_score_idempotent else 1


if __name__ == "__main__":
    raise SystemExit(main())
