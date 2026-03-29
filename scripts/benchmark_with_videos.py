#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


DEFAULT_VIDEO_WIDTH = 960
DEFAULT_VIDEO_HEIGHT = 540
DEFAULT_VIDEO_FPS = 15

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


def _default_run_dir() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return Path("bench_logs") / f"benchmark_video_{stamp}"


def _timestamp_suffix() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _resolve_run_dir(requested: Path | None) -> Path:
    if requested is None:
        return _default_run_dir().resolve()

    resolved = Path(requested).resolve()
    if not resolved.exists():
        return resolved
    return resolved.parent / f"{resolved.name}_{_timestamp_suffix()}"


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="benchmark_with_videos",
        description=(
            "Run swarm benchmark, save seeds/summary, render exact benchmark "
            "replay videos, and fail if replayed results do not match."
        ),
    )
    ap.add_argument("--model", type=Path, required=True, help="Miner submission zip.")
    ap.add_argument("--uid", type=int, default=None, help="Optional explicit UID.")
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Benchmark worker count. Default: 1 for local debug stability.",
    )
    ap.add_argument(
        "--seeds-per-group",
        type=int,
        default=1,
        help="Seeds per map group when not using --seed-file.",
    )
    ap.add_argument(
        "--seed-file",
        type=Path,
        default=None,
        help="Optional benchmark seed JSON to reuse instead of searching new seeds.",
    )
    ap.add_argument(
        "--seed-search-rng",
        type=int,
        default=None,
        help="Optional RNG seed for deterministic benchmark seed search.",
    )
    ap.add_argument(
        "--rpc-verbosity",
        choices=["low", "mid", "high"],
        default="mid",
        help="Benchmark RPC trace verbosity.",
    )
    ap.add_argument(
        "--relax-timeouts",
        action="store_true",
        help="Pass benchmark --relax-timeouts through unchanged.",
    )
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Artifact directory. Default: bench_logs/benchmark_video_<timestamp>.",
    )
    ap.add_argument(
        "--mode",
        type=str,
        default="chase",
        help="Video mode(s) for swarm video. Default: chase.",
    )
    ap.add_argument(
        "--only-type",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        default=None,
        help=(
            "Optional benchmark/video filter. When set, both benchmarking and replay "
            "are limited to this challenge type only."
        ),
    )
    ap.add_argument(
        "--width",
        type=int,
        default=DEFAULT_VIDEO_WIDTH,
        help=f"Video width. Default: {DEFAULT_VIDEO_WIDTH}.",
    )
    ap.add_argument(
        "--height",
        type=int,
        default=DEFAULT_VIDEO_HEIGHT,
        help=f"Video height. Default: {DEFAULT_VIDEO_HEIGHT}.",
    )
    ap.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_VIDEO_FPS,
        help=f"Video frames per second. Default: {DEFAULT_VIDEO_FPS}.",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Forward --skip-existing to swarm video.",
    )
    ap.add_argument(
        "--chase-back",
        type=float,
        default=2.5,
        help="Chase camera distance behind the drone in metres.",
    )
    ap.add_argument(
        "--chase-up",
        type=float,
        default=1.0,
        help="Chase camera height above the drone in metres.",
    )
    ap.add_argument(
        "--chase-fov",
        type=float,
        default=65.0,
        help="Chase camera field of view.",
    )
    ap.add_argument(
        "--fpv-fov",
        type=float,
        default=90.0,
        help="FPV camera field of view.",
    )
    ap.add_argument(
        "--overview-fov",
        type=float,
        default=55.0,
        help="Overview camera field of view.",
    )
    return ap


def _build_benchmark_argv(
    *,
    model: Path,
    uid: int | None,
    workers: int,
    seeds_per_group: int,
    seed_file: Path | None,
    seed_search_rng: int | None,
    rpc_verbosity: str,
    relax_timeouts: bool,
    log_out: Path,
    saved_seed_file: Path,
    summary_json_out: Path,
) -> list[str]:
    argv = ["--model", str(model), "--workers", str(workers), "--log-out", str(log_out)]
    if uid is not None:
        argv.extend(["--uid", str(uid)])
    if seed_file is not None:
        argv.extend(["--seed-file", str(seed_file)])
    else:
        argv.extend(["--seeds-per-group", str(seeds_per_group)])
        if seed_search_rng is not None:
            argv.extend(["--seed-search-rng", str(seed_search_rng)])
    argv.extend(["--save-seed-file", str(saved_seed_file)])
    argv.extend(["--summary-json-out", str(summary_json_out)])
    argv.extend(["--rpc-verbosity", str(rpc_verbosity)])
    if relax_timeouts:
        argv.append("--relax-timeouts")
    return argv


def _build_video_argv(
    *,
    model: Path,
    seed_file: Path,
    summary_json: Path,
    out_dir: Path,
    mode: str,
    width: int,
    height: int,
    fps: int,
    skip_existing: bool,
    chase_back: float,
    chase_up: float,
    chase_fov: float,
    fpv_fov: float,
    overview_fov: float,
) -> list[str]:
    argv = [
        "--model",
        str(model),
        "--seed-file",
        str(seed_file),
        "--summary-json",
        str(summary_json),
        "--backend",
        "benchmark",
        "--mode",
        str(mode),
        "--out",
        str(out_dir),
        "--width",
        str(width),
        "--height",
        str(height),
        "--fps",
        str(fps),
        "--chase-back",
        str(chase_back),
        "--chase-up",
        str(chase_up),
        "--chase-fov",
        str(chase_fov),
        "--fpv-fov",
        str(fpv_fov),
        "--overview-fov",
        str(overview_fov),
    ]
    if skip_existing:
        argv.append("--skip-existing")
    return argv


def _build_video_argv_with_actions(base_argv: list[str], actions_dir: Path) -> list[str]:
    return base_argv + ["--save-actions", str(actions_dir)]


def _load_jobs_for_type(seed_file: Path, challenge_type: int) -> list[int]:
    raw = json.loads(seed_file.read_text())
    group = TYPE_TO_GROUP[int(challenge_type)]
    if not isinstance(raw, dict):
        raise ValueError("Seed file must contain a JSON object mapping benchmark groups to seed lists.")
    seeds = raw.get(group)
    if not isinstance(seeds, list) or not seeds:
        raise ValueError(f"Seed group {group} is missing or empty.")
    return [int(seed) for seed in seeds]


def _find_seeds_for_type(
    *,
    challenge_type: int,
    seeds_per_group: int,
    seed_search_rng: int | None,
) -> list[int]:
    import random

    from swarm.constants import SIM_DT
    from swarm.validator.task_gen import random_task

    rng = random.Random(int(seed_search_rng)) if seed_search_rng is not None else random.Random()
    seeds: list[int] = []
    seed = rng.randint(100000, 900000)
    max_search = seed + 500000
    while seed < max_search:
        task = random_task(sim_dt=SIM_DT, seed=seed)
        if int(task.challenge_type) == int(challenge_type):
            seeds.append(int(seed))
            if len(seeds) >= int(seeds_per_group):
                return seeds
        seed += 1

    raise RuntimeError(
        f"Could not find {seeds_per_group} seeds for challenge type {challenge_type}."
    )


def _run_filtered_benchmark(
    *,
    model: Path,
    uid: int | None,
    workers: int,
    challenge_type: int,
    seeds: list[int],
    rpc_verbosity: str,
    relax_timeouts: bool,
    log_out: Path,
    saved_seed_file: Path,
    summary_json_out: Path,
) -> None:
    import swarm.benchmark.engine as engine
    from swarm.benchmark.engine_parts.config import (
        _Tee,
        _active_runtime_overrides,
        _apply_relaxed_overrides,
        _resolve_run_options,
        _ts,
    )
    from swarm.benchmark.engine_parts.reporting import _print_results

    inferred_uid = engine._infer_uid_from_model_path(model)
    effective_uid = int(uid) if uid is not None else int(inferred_uid or 0)
    args_ns = argparse.Namespace(rpc_verbosity=str(rpc_verbosity))
    run_opts = _resolve_run_options(args_ns)
    group_name = TYPE_TO_GROUP[int(challenge_type)]
    type_seeds = {group_name: [int(seed) for seed in seeds]}
    saved_seed_file.parent.mkdir(parents=True, exist_ok=True)
    saved_seed_file.write_text(json.dumps(type_seeds, indent=2, sort_keys=True))

    log_out.parent.mkdir(parents=True, exist_ok=True)
    log_fh = None
    out_stream = sys.__stdout__
    err_stream = sys.__stderr__
    summary: dict | None = None
    run_error: BaseException | None = None
    report_error: BaseException | None = None
    task_meta = []
    results = []
    seed_times = []
    seed_wall_by_key = {}
    seed_status_by_key = {}
    full_wall_by_key = {}
    batch_stats = []
    elapsed = 0.0
    eval_start = time.time()
    launched_workers = max(1, int(workers))
    overrides: dict = {}

    try:
        log_fh = open(log_out, "w")
        sys.stdout = _Tee(sys.__stdout__, log_fh)
        sys.stderr = _Tee(sys.__stderr__, log_fh)

        if relax_timeouts:
            overrides = _apply_relaxed_overrides()

        print(f"[{_ts()}] === FILTERED EVALUATION BENCHMARK ===")
        print(f"[{_ts()}] Model: {model}")
        if uid is None and inferred_uid is not None:
            print(f"[{_ts()}] UID: {effective_uid} (inferred from model filename)")
        elif uid is None:
            print(f"[{_ts()}] UID: {effective_uid} (default fallback)")
        else:
            print(f"[{_ts()}] UID: {effective_uid} (from --uid)")
        print(f"[{_ts()}] Workers requested: {workers}")
        print(f"[{_ts()}] Workers effective:  {workers}")
        print(f"[{_ts()}] Profile: debug")
        print(f"[{_ts()}] RPC verbosity: {rpc_verbosity}")
        print(f"[{_ts()}] Host parallelism: process")
        print(f"[{_ts()}] Container mode: single-seed")
        if overrides:
            print(f"[{_ts()}] Relaxed timeouts: {overrides}")
        runtime_overrides = _active_runtime_overrides()
        if runtime_overrides:
            print(f"[{_ts()}] Runtime overrides: {runtime_overrides}")
        print(f"[{_ts()}] Map types selected: {group_name}")
        print(f"[{_ts()}] Log file: {log_out}")
        print()
        print(f"[{_ts()}] Saved seed file: {saved_seed_file}")
        print(f"  {group_name}: {seeds}")
        print(f"  Total: {len(seeds)}")
        print()

        (
            task_meta,
            results,
            seed_times,
            seed_wall_by_key,
            seed_status_by_key,
            full_wall_by_key,
            batch_stats,
            elapsed,
            eval_start,
            launched_workers,
        ) = asyncio.run(
            engine._run_benchmark(
                model,
                effective_uid,
                type_seeds,
                max(1, int(workers)),
                run_opts=run_opts,
            )
        )
    except BaseException as exc:
        run_error = exc
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        sys.stdout = out_stream
        sys.stderr = err_stream

    try:
        print(f"\n[{_ts()}] === RESULTS ===", flush=True)
        if task_meta and results:
            if run_error is not None:
                print(f"[{_ts()}] Printing partial results collected before failure.", flush=True)
            try:
                summary = _print_results(
                    task_meta,
                    results,
                    seed_times,
                    seed_wall_by_key,
                    seed_status_by_key,
                    full_wall_by_key,
                    batch_stats,
                    elapsed,
                    eval_start,
                    launched_workers,
                    host_parallelism="process",
                )
            except BaseException as exc:
                report_error = exc
                print(f"[{_ts()}] Report generation failed: {type(exc).__name__}: {exc}", flush=True)

        if run_error is not None:
            print(
                f"[{_ts()}] Benchmark failed before report generation: {type(run_error).__name__}: {run_error}",
                flush=True,
            )
            raise run_error
        if report_error is not None:
            raise report_error
        print(f"[{_ts()}] === BENCHMARK COMPLETE ===", flush=True)
    finally:
        if log_fh is not None:
            try:
                log_fh.flush()
            except Exception:
                pass
            log_fh.close()

    if summary is not None:
        summary_json_out.parent.mkdir(parents=True, exist_ok=True)
        summary_json_out.write_text(json.dumps(summary, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    model = args.model.resolve()
    if not model.exists():
        print(f"Model not found: {model}", file=sys.stderr)
        return 1

    run_dir = _resolve_run_dir(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = run_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    actions_dir = run_dir / "actions"
    actions_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "benchmark.log"
    saved_seed_file = run_dir / "seeds.json"
    summary_json = run_dir / "summary.json"

    benchmark_argv = _build_benchmark_argv(
        model=model,
        uid=args.uid,
        workers=max(1, int(args.workers)),
        seeds_per_group=max(1, int(args.seeds_per_group)),
        seed_file=args.seed_file.resolve() if args.seed_file is not None else None,
        seed_search_rng=args.seed_search_rng,
        rpc_verbosity=args.rpc_verbosity,
        relax_timeouts=bool(args.relax_timeouts),
        log_out=log_path,
        saved_seed_file=saved_seed_file,
        summary_json_out=summary_json,
    )
    video_argv = _build_video_argv(
        model=model,
        seed_file=saved_seed_file,
        summary_json=summary_json,
        out_dir=videos_dir,
        mode=args.mode,
        width=int(args.width),
        height=int(args.height),
        fps=int(args.fps),
        skip_existing=bool(args.skip_existing),
        chase_back=float(args.chase_back),
        chase_up=float(args.chase_up),
        chase_fov=float(args.chase_fov),
        fpv_fov=float(args.fpv_fov),
        overview_fov=float(args.overview_fov),
    )

    print("=" * 64)
    print("  Swarm Benchmark + Video Replay")
    print("=" * 64)
    print(f"  Model       {model}")
    print(f"  Run dir     {run_dir}")
    print(f"  Benchmark   {log_path}")
    print(f"  Seeds       {saved_seed_file}")
    print(f"  Summary     {summary_json}")
    print(f"  Videos      {videos_dir}")
    print(f"  Actions     {actions_dir}")
    print(f"  Video       {args.width}x{args.height} @ {args.fps} fps  mode={args.mode}")
    if args.only_type is not None:
        print(f"  Replay type {args.only_type} only")
    print("=" * 64)

    try:
        from swarm.benchmark.engine import main as benchmark_main
        from scripts.generate_video import main as video_main

        print("[runner] benchmark phase")
        if args.only_type is None:
            benchmark_main(benchmark_argv)
        else:
            if args.seed_file is not None:
                filtered_seeds = _load_jobs_for_type(args.seed_file.resolve(), int(args.only_type))
            else:
                if args.seed_search_rng is not None:
                    print(f"[runner] seed search RNG: {args.seed_search_rng}")
                print(
                    f"[runner] finding {args.seeds_per_group} seed(s) for type {args.only_type} only"
                )
                filtered_seeds = _find_seeds_for_type(
                    challenge_type=int(args.only_type),
                    seeds_per_group=max(1, int(args.seeds_per_group)),
                    seed_search_rng=args.seed_search_rng,
                )
            _run_filtered_benchmark(
                model=model,
                uid=args.uid,
                workers=max(1, int(args.workers)),
                challenge_type=int(args.only_type),
                seeds=filtered_seeds,
                rpc_verbosity=args.rpc_verbosity,
                relax_timeouts=bool(args.relax_timeouts),
                log_out=log_path,
                saved_seed_file=saved_seed_file,
                summary_json_out=summary_json,
            )

        if args.only_type is None:
            print("[runner] video replay phase")
        else:
            filtered_seeds = _load_jobs_for_type(saved_seed_file, int(args.only_type))
            print(
                f"[runner] video replay phase (type {args.only_type}, {len(filtered_seeds)} seed(s))"
            )
        video_main(_build_video_argv_with_actions(video_argv, actions_dir))

        print("[runner] verification passed: benchmark and video replay matched.")
        return 0
    except (SystemExit, KeyboardInterrupt) as exc:
        return int(exc.code) if isinstance(exc, SystemExit) and isinstance(exc.code, int) else 1
    except Exception as exc:
        print(f"benchmark_with_videos failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
