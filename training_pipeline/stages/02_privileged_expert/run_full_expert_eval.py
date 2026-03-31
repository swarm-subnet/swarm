"""Run a full validator-like privileged-expert evaluation in one command.

This script is a thin stage-02 orchestrator:
1. Generate a validator-like task list with ``N`` sampled tasks.
2. Save the task list and exact run configuration under one artifact folder.
3. Invoke ``build_privileged_expert.py`` on that task list.

It keeps the stage-02 evaluation path single-sourced while making full runs
easy to launch repeatedly.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys

CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
for root in (CURRENT_DIR, DEFAULT_MODEL_ROOT, REPO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from training_lib.common import ensure_dir, load_json, save_json
from discover_validator_like_tasks import build_validator_like_task_list_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-runs", type=int, required=True, help="Number of validator-like expert episodes to evaluate.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional artifact folder name. Defaults to a timestamped validator-like name.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_MODEL_ROOT / "artifacts" / "02_privileged_expert" / "full_runs",
        help="Root directory for generated full-run artifacts.",
    )
    parser.add_argument("--seed-mode", type=str, choices=("deterministic", "system"), default="deterministic")
    parser.add_argument("--random-seed", type=int, default=20260330)
    parser.add_argument("--max-steps", type=int, default=None, help="Optional per-episode step cap passed through to build_privileged_expert.py.")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=1, help="Number of worker processes for stage-02 evaluation. Use small values like 2-4.")
    parser.add_argument("--gui", action="store_true")

    parser.add_argument("--fast-privileged", action="store_true", help="Stage-02 only speed mode: disable depth obstacle checks and relax privileged ray-cast refresh.")
    parser.add_argument("--disable-depth-obstacle-checks", action="store_true")
    parser.add_argument("--privileged-raycast-stride", type=int, default=1)
    return parser.parse_args()


def default_run_name(args: argparse.Namespace) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "system" if args.seed_mode == "system" else f"seed{args.random_seed}"
    suffix = "fast" if args.fast_privileged else "full"
    return f"validator_like_{args.num_runs}_strong_{mode}_{suffix}_{stamp}"


def build_eval_command(args: argparse.Namespace, *, task_list_path: Path, output_dir: Path) -> list[str]:
    script_path = DEFAULT_MODEL_ROOT / "stages" / "02_privileged_expert" / "build_privileged_expert.py"
    command = [
        sys.executable,
        str(script_path),
        "--task-source",
        "task_list",
        "--task-list-path",
        str(task_list_path),
        "--profile",
        "full",
        "--progress-every",
        str(args.progress_every),
        "--num-workers",
        str(args.num_workers),
        "--output-dir",
        str(output_dir),
    ]
    if args.max_steps is not None:
        command.extend(["--max-steps", str(args.max_steps)])
    if args.gui:
        command.append("--gui")

    disable_depth_obstacle_checks = bool(args.disable_depth_obstacle_checks)
    privileged_raycast_stride = int(args.privileged_raycast_stride)
    if args.fast_privileged:
        disable_depth_obstacle_checks = True
        privileged_raycast_stride = max(privileged_raycast_stride, 10)

    if disable_depth_obstacle_checks:
        command.append("--disable-depth-obstacle-checks")
    if privileged_raycast_stride != 1:
        command.extend(["--privileged-raycast-stride", str(privileged_raycast_stride)])
    return command


def main() -> None:
    args = parse_args()

    run_name = args.run_name or default_run_name(args)
    run_dir = ensure_dir(args.output_root / run_name)
    task_list_path = run_dir / "validator_like_tasks.json"

    task_payload = build_validator_like_task_list_payload(
        name=run_name,
        num_tasks=int(args.num_runs),
        seed_mode=str(args.seed_mode),
        random_seed=int(args.random_seed),
    )
    save_json(task_list_path, task_payload)

    command = build_eval_command(args, task_list_path=task_list_path, output_dir=run_dir)
    run_config = {
        "run_name": run_name,
        "task_list_path": str(task_list_path),
        "output_dir": str(run_dir),
        "num_runs": int(args.num_runs),
        "expert": "strong",
        "seed_mode": str(args.seed_mode),
        "random_seed": int(args.random_seed),
        "max_steps": int(args.max_steps) if args.max_steps is not None else None,
        "progress_every": int(args.progress_every),
        "num_workers": int(args.num_workers),
        "gui": bool(args.gui),
        "fast_privileged": bool(args.fast_privileged),
        "skip_depth_render": bool(args.skip_depth_render) or bool(args.fast_privileged),
        "disable_depth_obstacle_checks": bool(args.disable_depth_obstacle_checks) or bool(args.fast_privileged),
        "privileged_raycast_stride": max(int(args.privileged_raycast_stride), 10) if args.fast_privileged else int(args.privileged_raycast_stride),
        "eval_command": command,
    }
    save_json(run_dir / "run_config.json", run_config)

    print(f"Run directory: {run_dir}")
    print(f"Generated task list: {task_list_path}")
    print(f"Challenge histogram: {task_payload['challenge_type_histogram']}")
    print(f"Motion histogram: {task_payload['motion_histogram']}")
    print("Launching stage-02 evaluator:")
    print(" ".join(command))

    completed = subprocess.run(command, cwd=REPO_ROOT)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    summary_path = run_dir / "expert_eval_summary.json"
    if summary_path.exists():
        summary = load_json(summary_path)
        metrics = summary.get("aggregate_metrics", {})
        print("Completed successfully.")
        print(f"Summary: {summary_path}")
        print(f"success_rate={metrics.get('success_rate')}")
        print(f"mean_score={metrics.get('mean_score')}")
        print(f"mean_final_distance_to_platform_m={metrics.get('mean_final_distance_to_platform_m')}")
        print(f"aggregate_selection_score={metrics.get('aggregate_selection_score')}")


if __name__ == "__main__":
    main()
