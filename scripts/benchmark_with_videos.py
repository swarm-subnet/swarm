#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    print("=" * 64)

    try:
        from swarm.benchmark.engine import main as benchmark_main
        from scripts.generate_video import main as video_main

        print("[runner] benchmark phase")
        benchmark_main(benchmark_argv)

        print("[runner] video replay phase")
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
