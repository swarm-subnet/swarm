#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_ANSIBLE_TMP = Path("/tmp") / "swarm_ansible_tmp"
_ANSIBLE_TMP.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("ANSIBLE_LOCAL_TEMP", str(_ANSIBLE_TMP))
os.environ.setdefault("ANSIBLE_REMOTE_TEMP", str(_ANSIBLE_TMP))

from swarm.benchmark.engine import main as benchmark_main
from swarm.constants import (
    BENCHMARK_SCREENING_SEED_COUNT,
    BENCHMARK_TOTAL_SEED_COUNT,
    SIM_DT,
)
from swarm.validator.task_gen import random_task


BENCH_GROUP_ORDER = [
    "type1_city",
    "type2_open",
    "type3_mountain",
    "type4_village",
    "type5_warehouse",
    "type6_forest",
]

TYPE_TO_GROUP = {
    1: "type1_city",
    2: "type2_open",
    3: "type3_mountain",
    4: "type4_village",
    5: "type5_warehouse",
    6: "type6_forest",
}

_MAX_SEED_EXCLUSIVE = 2**32


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _default_run_dir() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return Path("bench_logs") / f"stress_benchmark_{stamp}"


def _resolve_run_dir(requested: Path | None) -> Path:
    if requested is None:
        return _default_run_dir().resolve()

    resolved = Path(requested).resolve()
    if not resolved.exists():
        return resolved
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return resolved.parent / f"{resolved.name}_{stamp}"


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run repeated validator-style benchmark samples for one model and compare "
            "the run-to-run variance of average score."
        ),
    )
    parser.add_argument("--model", type=Path, required=True, help="Path to submission zip.")
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="How many independent benchmark runs to execute (default: 5).",
    )
    parser.add_argument(
        "--seed-count",
        type=int,
        default=1000,
        help="How many raw validator-style seeds to evaluate per run (default: 1000).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Parallel benchmark workers per run (default: 2).",
    )
    parser.add_argument(
        "--uid",
        type=int,
        default=None,
        help="Optional explicit UID. Otherwise inferred from the model filename.",
    )
    parser.add_argument(
        "--rpc-verbosity",
        choices=["low", "mid", "high"],
        default="mid",
        help="Benchmark RPC trace verbosity.",
    )
    parser.add_argument(
        "--relax-timeouts",
        action="store_true",
        help="Forward --relax-timeouts to the benchmark engine.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Artifact directory. Default: bench_logs/stress_benchmark_<timestamp>.",
    )
    parser.add_argument(
        "--seed-rng",
        type=int,
        default=None,
        help="Optional base RNG seed for reproducible raw-seed generation.",
    )
    parser.add_argument(
        "--same-seeds",
        action="store_true",
        help="Reuse the same raw seed set for every repetition.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort immediately if any repetition fails.",
    )
    return parser.parse_args(argv)


def _infer_uid_from_model_path(model_path: Path) -> int:
    from swarm.benchmark.engine_parts.seeds import _infer_uid_from_model_path as _real_infer_uid

    inferred = _real_infer_uid(model_path)
    return int(inferred or 0)


def _screening_count_for_total(total: int) -> int:
    if total <= 0:
        raise ValueError("seed-count must be positive")
    ratio = float(BENCHMARK_SCREENING_SEED_COUNT) / float(BENCHMARK_TOTAL_SEED_COUNT)
    screening = int(round(total * ratio))
    if total == 1:
        return 1
    return max(1, min(total - 1, screening))


def _make_rng(seed: Optional[int]) -> random.Random:
    if seed is None:
        return random.SystemRandom()
    return random.Random(int(seed))


def _generate_unique_raw_seeds(count: int, rng: random.Random) -> List[int]:
    if count <= 0:
        raise ValueError("count must be positive")
    if count > _MAX_SEED_EXCLUSIVE:
        raise ValueError(f"count exceeds available seed space ({_MAX_SEED_EXCLUSIVE})")

    seen = set()
    seeds: List[int] = []
    while len(seeds) < count:
        seed = int(rng.randrange(_MAX_SEED_EXCLUSIVE))
        if seed in seen:
            continue
        seen.add(seed)
        seeds.append(seed)
    return seeds


def _classify_raw_seeds(raw_seeds: List[int], screening_count: int) -> tuple[Dict[str, List[int]], List[Dict[str, Any]]]:
    grouped: Dict[str, List[int]] = {group: [] for group in BENCH_GROUP_ORDER}
    manifest: List[Dict[str, Any]] = []

    for index, seed in enumerate(raw_seeds):
        task = random_task(sim_dt=SIM_DT, seed=seed)
        challenge_type = int(task.challenge_type)
        group = TYPE_TO_GROUP.get(challenge_type)
        if group is None:
            raise ValueError(f"Unsupported challenge type {challenge_type} for seed {seed}")
        grouped[group].append(int(seed))
        manifest.append(
            {
                "index": int(index),
                "seed": int(seed),
                "challenge_type": challenge_type,
                "group": group,
                "phase": "screening" if index < screening_count else "benchmark",
            }
        )
    return grouped, manifest


def _build_seed_set(count: int, rng: random.Random, screening_count: int) -> tuple[List[int], Dict[str, List[int]], List[Dict[str, Any]]]:
    if count < len(BENCH_GROUP_ORDER):
        raise ValueError(f"seed-count must be at least {len(BENCH_GROUP_ORDER)} to cover all map groups.")

    for _attempt in range(1, 129):
        raw_seeds = _generate_unique_raw_seeds(count, rng)
        grouped, manifest = _classify_raw_seeds(raw_seeds, screening_count)
        if all(grouped[group] for group in BENCH_GROUP_ORDER):
            return raw_seeds, grouped, manifest
    raise RuntimeError("Could not generate a seed set that covers all benchmark groups.")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _metric_stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "variance": 0.0,
            "stdev": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "range": 0.0,
        }
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "variance": float(statistics.pvariance(values)) if len(values) > 1 else 0.0,
        "stdev": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
        "median": float(statistics.median(values)),
        "range": float(max(values) - min(values)),
    }


def _average(rows: List[Dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return float(statistics.fmean(float(row["score"]) for row in rows))


def _summarize_run(
    *,
    summary: Dict[str, Any],
    manifest: List[Dict[str, Any]],
    raw_seed_count: int,
    screening_count: int,
) -> Dict[str, Any]:
    rows_by_seed: Dict[int, Dict[str, Any]] = {}
    group_averages: Dict[str, float] = {}
    group_counts: Dict[str, int] = {}

    group_results = summary.get("group_results", {})
    for group in BENCH_GROUP_ORDER:
        rows = list(group_results.get(group, []))
        group_counts[group] = len(rows)
        group_averages[group] = _average(rows)
        for row in rows:
            rows_by_seed[int(row["seed"])] = row

    ordered_rows: List[Dict[str, Any]] = []
    for item in manifest:
        seed = int(item["seed"])
        row = rows_by_seed.get(seed)
        if row is None:
            raise RuntimeError(f"Missing benchmark result row for seed {seed}")
        ordered_rows.append(
            {
                **item,
                "score": float(row["score"]),
                "success": bool(row["success"]),
                "sim_time": float(row["sim_time"]),
                "wall_time": float(row["wall_time"]),
                "processing_wall_time": float(row.get("processing_wall_time", row["wall_time"])),
                "execution_status": str(row.get("execution_status", "unknown")),
                "execution_ok": bool(row.get("execution_ok", False)),
            }
        )

    screening_rows = ordered_rows[:screening_count]
    benchmark_rows = ordered_rows[screening_count:]
    overall_avg = _average(ordered_rows)
    screening_avg = _average(screening_rows)
    benchmark_avg = _average(benchmark_rows)
    overall_success_rate = float(
        statistics.fmean(1.0 if bool(row["success"]) else 0.0 for row in ordered_rows)
    ) if ordered_rows else 0.0

    return {
        "raw_seed_count": int(raw_seed_count),
        "screening_seed_count": int(len(screening_rows)),
        "benchmark_seed_count": int(len(benchmark_rows)),
        "overall_avg_score": overall_avg,
        "screening_avg_score": screening_avg,
        "benchmark_avg_score": benchmark_avg,
        "overall_success_rate": overall_success_rate,
        "group_averages": group_averages,
        "group_counts": group_counts,
        "ordered_rows": ordered_rows,
        "run_metrics": summary.get("run_metrics", {}),
        "execution_status_counts": summary.get("execution_status_counts", {}),
        "wall_clock_sec": float(summary.get("wall_clock_sec", 0.0)),
    }


def _aggregate_successful_runs(successful_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    overall_avgs = [float(run["metrics"]["overall_avg_score"]) for run in successful_runs]
    screening_avgs = [float(run["metrics"]["screening_avg_score"]) for run in successful_runs]
    benchmark_avgs = [float(run["metrics"]["benchmark_avg_score"]) for run in successful_runs]
    success_rates = [float(run["metrics"]["overall_success_rate"]) for run in successful_runs]
    wall_clocks = [float(run["metrics"]["wall_clock_sec"]) for run in successful_runs]

    per_group_stats: Dict[str, Dict[str, Any]] = {}
    for group in BENCH_GROUP_ORDER:
        values = [
            float(run["metrics"]["group_averages"][group])
            for run in successful_runs
            if group in run["metrics"]["group_averages"]
        ]
        per_group_stats[group] = _metric_stats(values)

    return {
        "successful_runs": len(successful_runs),
        "overall_avg_score": _metric_stats(overall_avgs),
        "screening_avg_score": _metric_stats(screening_avgs),
        "benchmark_avg_score": _metric_stats(benchmark_avgs),
        "overall_success_rate": _metric_stats(success_rates),
        "wall_clock_sec": _metric_stats(wall_clocks),
        "group_avg_scores": per_group_stats,
    }


def _build_benchmark_argv(
    *,
    model_path: Path,
    uid: int,
    workers: int,
    grouped_seed_file: Path,
    log_path: Path,
    summary_path: Path,
    rpc_verbosity: str,
    relax_timeouts: bool,
) -> List[str]:
    argv = [
        "--model", str(model_path),
        "--uid", str(uid),
        "--workers", str(workers),
        "--seed-file", str(grouped_seed_file),
        "--log-out", str(log_path),
        "--summary-json-out", str(summary_path),
        "--rpc-verbosity", str(rpc_verbosity),
    ]
    if relax_timeouts:
        argv.append("--relax-timeouts")
    return argv


def _run_single_benchmark(argv: List[str]) -> None:
    benchmark_main(argv)


def _write_report_text(path: Path, payload: Dict[str, Any]) -> None:
    args = payload["config"]
    agg = payload["aggregate"]
    runs = payload["runs"]

    lines: List[str] = []
    lines.append("Swarm Stress Benchmark Comparison")
    lines.append("=" * 64)
    lines.append(f"Model:             {args['model']}")
    lines.append(f"UID:               {args['uid']}")
    lines.append(f"Repetitions:       {args['repetitions']}")
    lines.append(f"Seed count / run:  {args['seed_count']}")
    lines.append(f"Screening split:   {args['screening_seed_count']} / {args['benchmark_seed_count']}")
    lines.append(f"Workers:           {args['workers']}")
    lines.append(f"RPC verbosity:     {args['rpc_verbosity']}")
    lines.append(f"Relaxed timeouts:  {args['relax_timeouts']}")
    lines.append("")

    lines.append("Run-level averages")
    lines.append(f"{'Run':<6} {'Overall':>10} {'Screen':>10} {'Bench':>10} {'Succ%':>8} {'Wall(s)':>10}")
    for run in runs:
        if run["status"] != "ok":
            lines.append(f"{run['run_index']:<6} {'FAILED':>10} {'':>10} {'':>10} {'':>8} {'':>10}")
            continue
        metrics = run["metrics"]
        lines.append(
            f"{run['run_index']:<6} "
            f"{metrics['overall_avg_score']:>10.4f} "
            f"{metrics['screening_avg_score']:>10.4f} "
            f"{metrics['benchmark_avg_score']:>10.4f} "
            f"{metrics['overall_success_rate'] * 100.0:>7.1f}% "
            f"{metrics['wall_clock_sec']:>10.1f}"
        )
    lines.append("")

    if agg["successful_runs"] > 0:
        lines.append("Aggregate statistics across successful runs")
        for label, key in [
            ("Overall avg score", "overall_avg_score"),
            ("Screening avg", "screening_avg_score"),
            ("Benchmark avg", "benchmark_avg_score"),
            ("Success rate", "overall_success_rate"),
            ("Wall clock (s)", "wall_clock_sec"),
        ]:
            stats = agg[key]
            lines.append(
                f"{label:<18} mean={stats['mean']:.6f}  stdev={stats['stdev']:.6f}  "
                f"var={stats['variance']:.6f}  min={stats['min']:.6f}  max={stats['max']:.6f}"
            )
        lines.append("")
        lines.append("Per-group average score statistics")
        for group in BENCH_GROUP_ORDER:
            stats = agg["group_avg_scores"][group]
            lines.append(
                f"{group:<18} mean={stats['mean']:.6f}  stdev={stats['stdev']:.6f}  "
                f"min={stats['min']:.6f}  max={stats['max']:.6f}"
            )

    path.write_text("\n".join(lines) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    model_path = args.model.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if args.repetitions <= 0:
        raise ValueError("--repetitions must be positive")

    run_dir = _resolve_run_dir(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = run_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    uid = int(args.uid) if args.uid is not None else _infer_uid_from_model_path(model_path)
    screening_count = _screening_count_for_total(int(args.seed_count))
    benchmark_count = int(args.seed_count) - screening_count

    base_rng = _make_rng(args.seed_rng)
    shared_bundle = None
    if args.same_seeds:
        shared_rng_seed = int(base_rng.randrange(_MAX_SEED_EXCLUSIVE))
        shared_rng = random.Random(shared_rng_seed)
        raw_seeds, grouped_seeds, manifest = _build_seed_set(
            int(args.seed_count),
            shared_rng,
            screening_count,
        )
        shared_bundle = {
            "rng_seed": shared_rng_seed,
            "raw_seeds": raw_seeds,
            "grouped_seeds": grouped_seeds,
            "manifest": manifest,
        }

    config_payload = {
        "model": str(model_path),
        "uid": int(uid),
        "repetitions": int(args.repetitions),
        "seed_count": int(args.seed_count),
        "screening_seed_count": int(screening_count),
        "benchmark_seed_count": int(benchmark_count),
        "workers": int(args.workers),
        "rpc_verbosity": str(args.rpc_verbosity),
        "relax_timeouts": bool(args.relax_timeouts),
        "seed_rng": args.seed_rng,
        "same_seeds": bool(args.same_seeds),
        "run_dir": str(run_dir),
    }
    _write_json(run_dir / "config.json", config_payload)

    print(f"[{_ts()}] === STRESS BENCHMARK COMPARISON ===")
    print(f"[{_ts()}] Model: {model_path}")
    print(f"[{_ts()}] UID: {uid}")
    print(f"[{_ts()}] Repetitions: {args.repetitions}")
    print(f"[{_ts()}] Seed count / run: {args.seed_count}")
    print(f"[{_ts()}] Screening split: {screening_count} / {benchmark_count}")
    print(f"[{_ts()}] Workers: {args.workers}")
    print(f"[{_ts()}] Relaxed timeouts: {bool(args.relax_timeouts)}")
    if args.seed_rng is not None:
        print(f"[{_ts()}] Base RNG seed: {args.seed_rng}")
    if args.same_seeds:
        print(f"[{_ts()}] Seed mode: SAME seed set reused across repetitions")
    else:
        print(f"[{_ts()}] Seed mode: independent seed set per repetition")
    print(f"[{_ts()}] Run dir: {run_dir}")

    runs_payload: List[Dict[str, Any]] = []
    any_failures = False
    overall_start = time.time()

    for run_index in range(1, int(args.repetitions) + 1):
        print()
        print(f"[{_ts()}] --- RUN {run_index}/{args.repetitions} ---")
        repetition_dir = runs_dir / f"run_{run_index:02d}"
        repetition_dir.mkdir(parents=True, exist_ok=True)

        if shared_bundle is not None:
            rng_seed = int(shared_bundle["rng_seed"])
            raw_seeds = list(shared_bundle["raw_seeds"])
            grouped_seeds = {
                group: list(values) for group, values in shared_bundle["grouped_seeds"].items()
            }
            manifest = [dict(row) for row in shared_bundle["manifest"]]
        else:
            rng_seed = int(base_rng.randrange(_MAX_SEED_EXCLUSIVE))
            repetition_rng = random.Random(rng_seed)
            raw_seeds, grouped_seeds, manifest = _build_seed_set(
                int(args.seed_count),
                repetition_rng,
                screening_count,
            )

        raw_seed_path = repetition_dir / "raw_seeds.json"
        grouped_seed_path = repetition_dir / "grouped_seeds.json"
        log_path = repetition_dir / "benchmark.log"
        summary_path = repetition_dir / "summary.json"
        run_meta_path = repetition_dir / "run_manifest.json"

        _write_json(
            raw_seed_path,
            {
                "rng_seed": rng_seed,
                "raw_seeds": raw_seeds,
                "screening_seed_count": screening_count,
                "benchmark_seed_count": benchmark_count,
            },
        )
        _write_json(grouped_seed_path, grouped_seeds)
        _write_json(
            run_meta_path,
            {
                "rng_seed": rng_seed,
                "manifest": manifest,
            },
        )

        argv_run = _build_benchmark_argv(
            model_path=model_path,
            uid=uid,
            workers=max(1, int(args.workers)),
            grouped_seed_file=grouped_seed_path,
            log_path=log_path,
            summary_path=summary_path,
            rpc_verbosity=str(args.rpc_verbosity),
            relax_timeouts=bool(args.relax_timeouts),
        )

        run_started = time.time()
        try:
            print(f"[{_ts()}] RNG seed: {rng_seed}")
            print(f"[{_ts()}] Launching benchmark with {args.seed_count} seeds...")
            _run_single_benchmark(argv_run)
            summary = json.loads(summary_path.read_text())
            metrics = _summarize_run(
                summary=summary,
                manifest=manifest,
                raw_seed_count=int(args.seed_count),
                screening_count=screening_count,
            )
            run_payload = {
                "run_index": run_index,
                "status": "ok",
                "rng_seed": rng_seed,
                "artifact_dir": str(repetition_dir),
                "raw_seed_file": str(raw_seed_path),
                "grouped_seed_file": str(grouped_seed_path),
                "summary_file": str(summary_path),
                "log_file": str(log_path),
                "metrics": metrics,
                "elapsed_wall_clock_sec": time.time() - run_started,
            }
            runs_payload.append(run_payload)
            print(
                f"[{_ts()}] Run {run_index} complete: "
                f"overall_avg={metrics['overall_avg_score']:.4f} "
                f"screening_avg={metrics['screening_avg_score']:.4f} "
                f"benchmark_avg={metrics['benchmark_avg_score']:.4f}"
            )
        except Exception as exc:
            any_failures = True
            run_payload = {
                "run_index": run_index,
                "status": "failed",
                "rng_seed": rng_seed,
                "artifact_dir": str(repetition_dir),
                "raw_seed_file": str(raw_seed_path),
                "grouped_seed_file": str(grouped_seed_path),
                "summary_file": str(summary_path),
                "log_file": str(log_path),
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                "elapsed_wall_clock_sec": time.time() - run_started,
            }
            runs_payload.append(run_payload)
            _write_json(repetition_dir / "failure.json", run_payload["error"])
            print(f"[{_ts()}] Run {run_index} failed: {type(exc).__name__}: {exc}")
            if args.fail_fast:
                break

    successful_runs = [run for run in runs_payload if run["status"] == "ok"]
    aggregate = _aggregate_successful_runs(successful_runs)

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": config_payload,
        "runs": runs_payload,
        "aggregate": aggregate,
        "total_wall_clock_sec": time.time() - overall_start,
    }
    _write_json(run_dir / "report.json", payload)
    _write_report_text(run_dir / "report.txt", payload)

    print()
    print(f"[{_ts()}] === STRESS REPORT ===")
    print(f"[{_ts()}] Successful runs: {aggregate['successful_runs']}/{len(runs_payload)}")
    if aggregate["successful_runs"] > 0:
        overall_stats = aggregate["overall_avg_score"]
        print(
            f"[{_ts()}] Overall avg score: mean={overall_stats['mean']:.6f} "
            f"stdev={overall_stats['stdev']:.6f} var={overall_stats['variance']:.6f} "
            f"min={overall_stats['min']:.6f} max={overall_stats['max']:.6f}"
        )
    print(f"[{_ts()}] JSON report: {run_dir / 'report.json'}")
    print(f"[{_ts()}] Text report: {run_dir / 'report.txt'}")

    return 1 if any_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
