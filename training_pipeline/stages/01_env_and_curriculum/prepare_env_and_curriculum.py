"""Build and verify deterministic curriculum manifests for the training stack.

This stage establishes the fixed task schedule that every later stage should
reuse. The script does four concrete things:

1. Build tasks by exact map type through the repo's task generator.
2. Decide static vs moving-objective semantics per curriculum stage.
3. Create deterministic train/val/test seed splits.
4. Save a manifest that can be consumed by later folders without resampling.

The script also validates the manifest against the source task generator and
can run a small environment smoke test to prove the tasks actually boot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

DEFAULT_MODEL_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
for root in (DEFAULT_MODEL_ROOT, REPO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from training_env import (
    CHALLENGE_TYPES,
    CURRICULUM_STAGES,
    DEFAULT_SIM_DT,
    FIXED_STATE_DIM,
    VALIDATOR_SIM_DT,
    build_task,
    make_training_env,
    serialize_task,
    task_from_payload,
)
from training_lib.common import ensure_dir, load_json, save_json, seed_everything

SPLIT_ORDER = ("train", "val", "test")
STAGE_DESCRIPTIONS = {
    "open_static": "Single-type warm-up on the open terrain map with a fixed landing objective.",
    "city_forest_static": "Static-goal obstacle curriculum over dense outdoor maps.",
    "mountain_village_static": "Static-goal terrain curriculum over elevation-heavy outdoor maps.",
    "warehouse_static": "Indoor static-goal curriculum for tight obstacle structure and height changes.",
    "mixed_static": "Type-balanced static-goal curriculum across all six map families.",
    "mixed_dynamic": "Type-balanced moving-platform curriculum across the map families that support moving goals.",
    "benchmark_like": "Type-balanced curriculum that keeps exact map-type control while delegating moving/static resolution to the repo generator.",
}
PRIVILEGED_KEYS = {
    "adjusted_goal",
    "adjusted_start",
    "challenge_type",
    "collision",
    "goal_position",
    "moving_platform",
    "platform_position",
    "platform_velocity",
    "relative_goal",
    "relative_platform",
    "relative_search_center",
    "search_area_center",
    "success",
    "teacher_state",
    "time_alive",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_MODEL_ROOT / "artifacts" / "01_env_and_curriculum",
    )
    parser.add_argument("--train-per-stage", type=int, default=32)
    parser.add_argument("--val-per-stage", type=int, default=8)
    parser.add_argument("--test-per-stage", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--seed-stride", type=int, default=10000)
    parser.add_argument("--sim-dt", type=float, default=DEFAULT_SIM_DT)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument(
        "--smoke-test-per-split",
        type=int,
        default=1,
        help="Number of tasks per split and per stage to boot and step once.",
    )
    parser.add_argument(
        "--skip-smoke-test",
        action="store_true",
        help="Generate and validate the manifest without constructing environments.",
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")


def manifest_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def moving_policy_name(moving_platform: bool | None) -> str:
    if moving_platform is True:
        return "fixed_moving"
    if moving_platform is False:
        return "fixed_static"
    return "repo_default"


def empty_type_histogram(challenge_types: Iterable[int]) -> dict[str, int]:
    return {str(int(challenge_type)): 0 for challenge_type in challenge_types}


def empty_moving_histogram() -> dict[str, int]:
    return {"moving": 0, "static": 0}


def increment_moving_histogram(histogram: dict[str, int], *, moving_platform: bool) -> None:
    histogram["moving" if moving_platform else "static"] += 1


def split_counts_from_args(args: argparse.Namespace) -> dict[str, int]:
    return {
        "train": int(args.train_per_stage),
        "val": int(args.val_per_stage),
        "test": int(args.test_per_stage),
    }


def split_seed_start(
    *,
    stage_seed_base: int,
    split_name: str,
    split_counts: dict[str, int],
) -> int:
    if split_name == "train":
        return stage_seed_base
    if split_name == "val":
        return stage_seed_base + split_counts["train"]
    if split_name == "test":
        return stage_seed_base + split_counts["train"] + split_counts["val"]
    raise ValueError(f"Unsupported split: {split_name}")


def split_seed_summary(
    *,
    stage_seed_base: int,
    split_name: str,
    split_counts: dict[str, int],
) -> dict[str, int]:
    seed_start = split_seed_start(
        stage_seed_base=stage_seed_base,
        split_name=split_name,
        split_counts=split_counts,
    )
    count = split_counts[split_name]
    return {
        "seed_start": seed_start,
        "seed_stop_exclusive": seed_start + count,
        "count": count,
    }


def expected_challenge_type(challenge_types: tuple[int, ...], offset: int) -> int:
    return int(challenge_types[offset % len(challenge_types)])


def build_split_tasks(
    *,
    stage,
    split_name: str,
    stage_seed_base: int,
    split_counts: dict[str, int],
    sim_dt: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary = split_seed_summary(
        stage_seed_base=stage_seed_base,
        split_name=split_name,
        split_counts=split_counts,
    )
    tasks: list[dict[str, Any]] = []
    type_histogram = empty_type_histogram(stage.challenge_types)
    moving_histogram = empty_moving_histogram()
    map_seeds: list[int] = []
    split_offset = summary["seed_start"] - stage_seed_base

    for local_offset in range(summary["count"]):
        global_offset = split_offset + local_offset
        seed = summary["seed_start"] + local_offset
        challenge_type = expected_challenge_type(stage.challenge_types, global_offset)
        task = build_task(
            seed=seed,
            challenge_type=challenge_type,
            sim_dt=sim_dt,
            moving_platform=stage.moving_platform,
        )
        payload = serialize_task(task)
        tasks.append(payload)
        map_seeds.append(int(payload["map_seed"]))
        type_histogram[str(int(payload["challenge_type"]))] += 1
        increment_moving_histogram(
            moving_histogram,
            moving_platform=bool(payload["moving_platform"]),
        )

    split_summary = {
        **summary,
        "map_seeds": map_seeds,
        "challenge_type_histogram": type_histogram,
        "moving_platform_histogram": moving_histogram,
    }
    return tasks, split_summary


def build_stage_manifest(
    *,
    stage_index: int,
    stage,
    split_counts: dict[str, int],
    seed_start: int,
    seed_stride: int,
    sim_dt: float,
) -> dict[str, Any]:
    stage_seed_base = seed_start + stage_index * seed_stride
    splits: dict[str, list[dict[str, Any]]] = {}
    split_summaries: dict[str, dict[str, Any]] = {}

    for split_name in SPLIT_ORDER:
        split_tasks, split_summary = build_split_tasks(
            stage=stage,
            split_name=split_name,
            stage_seed_base=stage_seed_base,
            split_counts=split_counts,
            sim_dt=sim_dt,
        )
        splits[split_name] = split_tasks
        split_summaries[split_name] = split_summary

    return {
        "stage_name": stage.name,
        "description": STAGE_DESCRIPTIONS.get(stage.name, ""),
        "stage_index": stage_index,
        "challenge_types": list(stage.challenge_types),
        "moving_platform": stage.moving_platform,
        "moving_platform_policy": moving_policy_name(stage.moving_platform),
        "seed_base": stage_seed_base,
        "seed_stride": seed_stride,
        "split_summaries": split_summaries,
        "splits": splits,
    }


def aggregate_manifest_summary(stages: list[dict[str, Any]], split_counts: dict[str, int]) -> dict[str, Any]:
    type_histogram = empty_type_histogram(CHALLENGE_TYPES)
    moving_histogram = empty_moving_histogram()
    total_tasks = 0
    total_by_split = {split_name: 0 for split_name in SPLIT_ORDER}
    total_by_stage: dict[str, int] = {}

    for stage in stages:
        stage_task_count = 0
        for split_name in SPLIT_ORDER:
            split_tasks = stage["splits"][split_name]
            total_by_split[split_name] += len(split_tasks)
            stage_task_count += len(split_tasks)
            total_tasks += len(split_tasks)
            for payload in split_tasks:
                type_histogram[str(int(payload["challenge_type"]))] += 1
                increment_moving_histogram(
                    moving_histogram,
                    moving_platform=bool(payload["moving_platform"]),
                )
        total_by_stage[stage["stage_name"]] = stage_task_count

    return {
        "total_tasks": total_tasks,
        "total_by_split": total_by_split,
        "total_by_stage": total_by_stage,
        "challenge_type_histogram": type_histogram,
        "moving_platform_histogram": moving_histogram,
        "per_stage_split_counts": dict(split_counts),
    }


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    split_counts = split_counts_from_args(args)
    stages = [
        build_stage_manifest(
            stage_index=stage_index,
            stage=stage,
            split_counts=split_counts,
            seed_start=args.seed_start,
            seed_stride=args.seed_stride,
            sim_dt=args.sim_dt,
        )
        for stage_index, stage in enumerate(CURRICULUM_STAGES)
    ]

    manifest = {
        "schema_version": 2,
        "description": "Deterministic curriculum manifest for the Swarm training pipeline.",
        "generator": {
            "script": Path(__file__).name,
            "sim_dt": args.sim_dt,
            "seed_start": args.seed_start,
            "seed_stride": args.seed_stride,
            "random_seed": args.random_seed,
            "split_counts": split_counts,
            "smoke_test_per_split": args.smoke_test_per_split,
        },
        "summary": aggregate_manifest_summary(stages, split_counts),
        "stage_count": len(stages),
        "stages": stages,
    }
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    return manifest


def compare_payloads(expected: dict[str, Any], actual: dict[str, Any], *, context: str) -> None:
    if expected != actual:
        raise ValueError(
            f"{context} does not match generator output.\n"
            f"Expected: {json.dumps(expected, indent=2, sort_keys=True)}\n"
            f"Actual: {json.dumps(actual, indent=2, sort_keys=True)}"
        )


def validate_stage_manifest(
    stage_manifest: dict[str, Any],
    *,
    stage,
    split_counts: dict[str, int],
    seed_start: int,
    seed_stride: int,
    sim_dt: float,
) -> None:
    require(stage_manifest["stage_name"] == stage.name, f"Stage name mismatch for {stage.name}.")
    require(
        stage_manifest["challenge_types"] == list(stage.challenge_types),
        f"Challenge types mismatch for {stage.name}.",
    )
    require(
        stage_manifest["moving_platform"] == stage.moving_platform,
        f"Moving-platform policy mismatch for {stage.name}.",
    )
    require(
        stage_manifest["moving_platform_policy"] == moving_policy_name(stage.moving_platform),
        f"Moving-platform policy label mismatch for {stage.name}.",
    )

    stage_index = int(stage_manifest["stage_index"])
    stage_seed_base = seed_start + stage_index * seed_stride
    require(
        int(stage_manifest["seed_base"]) == stage_seed_base,
        f"Seed base mismatch for {stage.name}.",
    )
    require(
        int(stage_manifest["seed_stride"]) == seed_stride,
        f"Seed stride mismatch for {stage.name}.",
    )

    stage_seen_seeds: set[int] = set()

    for split_name in SPLIT_ORDER:
        split_tasks = stage_manifest["splits"][split_name]
        split_summary = stage_manifest["split_summaries"][split_name]
        expected_summary = split_seed_summary(
            stage_seed_base=stage_seed_base,
            split_name=split_name,
            split_counts=split_counts,
        )
        require(
            int(split_summary["seed_start"]) == expected_summary["seed_start"],
            f"{stage.name}/{split_name} seed_start mismatch.",
        )
        require(
            int(split_summary["seed_stop_exclusive"]) == expected_summary["seed_stop_exclusive"],
            f"{stage.name}/{split_name} seed_stop_exclusive mismatch.",
        )
        require(
            int(split_summary["count"]) == split_counts[split_name],
            f"{stage.name}/{split_name} count mismatch.",
        )
        require(
            len(split_tasks) == split_counts[split_name],
            f"{stage.name}/{split_name} task count mismatch.",
        )
        require(
            len(split_summary["map_seeds"]) == split_counts[split_name],
            f"{stage.name}/{split_name} map seed count mismatch.",
        )

        type_histogram = empty_type_histogram(stage.challenge_types)
        moving_histogram = empty_moving_histogram()
        expected_seed_list = list(
            range(expected_summary["seed_start"], expected_summary["seed_stop_exclusive"])
        )
        require(
            split_summary["map_seeds"] == expected_seed_list,
            f"{stage.name}/{split_name} seed list is not deterministic.",
        )

        split_offset = expected_summary["seed_start"] - stage_seed_base
        for local_offset, payload in enumerate(split_tasks):
            expected_seed = expected_summary["seed_start"] + local_offset
            expected_type = expected_challenge_type(
                stage.challenge_types,
                split_offset + local_offset,
            )
            require(
                int(payload["map_seed"]) == expected_seed,
                f"{stage.name}/{split_name}[{local_offset}] seed mismatch.",
            )
            require(
                int(payload["challenge_type"]) == expected_type,
                f"{stage.name}/{split_name}[{local_offset}] challenge_type mismatch.",
            )
            if stage.moving_platform is not None:
                require(
                    bool(payload["moving_platform"]) == bool(stage.moving_platform),
                    f"{stage.name}/{split_name}[{local_offset}] moving flag mismatch.",
                )

            expected_payload = serialize_task(
                build_task(
                    seed=expected_seed,
                    challenge_type=expected_type,
                    sim_dt=sim_dt,
                    moving_platform=stage.moving_platform,
                )
            )
            compare_payloads(
                expected_payload,
                payload,
                context=f"{stage.name}/{split_name}[{local_offset}]",
            )

            type_histogram[str(int(payload["challenge_type"]))] += 1
            increment_moving_histogram(
                moving_histogram,
                moving_platform=bool(payload["moving_platform"]),
            )
            stage_seen_seeds.add(int(payload["map_seed"]))

        require(
            split_summary["challenge_type_histogram"] == type_histogram,
            f"{stage.name}/{split_name} type histogram mismatch.",
        )
        require(
            split_summary["moving_platform_histogram"] == moving_histogram,
            f"{stage.name}/{split_name} moving histogram mismatch.",
        )

        counts = list(type_histogram.values())
        if counts:
            require(
                max(counts) - min(counts) <= 1,
                f"{stage.name}/{split_name} is not type-balanced.",
            )

    require(
        len(stage_seen_seeds) == sum(split_counts.values()),
        f"{stage.name} contains overlapping seeds across splits.",
    )
    stage_seed_stop = stage_seed_base + sum(split_counts.values())
    require(
        stage_seed_stop <= stage_seed_base + seed_stride,
        f"{stage.name} exceeds the configured seed stride; increase --seed-stride.",
    )


def validate_manifest(manifest: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    split_counts = split_counts_from_args(args)
    require(args.seed_stride >= sum(split_counts.values()), "--seed-stride is too small for the split sizes.")
    require(manifest["stage_count"] == len(CURRICULUM_STAGES), "Stage count mismatch.")
    require(len(manifest["stages"]) == len(CURRICULUM_STAGES), "Stage list length mismatch.")
    require(
        manifest["generator"]["split_counts"] == split_counts,
        "Manifest split counts do not match CLI arguments.",
    )
    require(float(manifest["generator"]["sim_dt"]) == float(args.sim_dt), "sim_dt mismatch.")
    require(
        manifest["manifest_sha256"] == manifest_sha256({k: v for k, v in manifest.items() if k != "manifest_sha256"}),
        "Manifest SHA256 is inconsistent with its content.",
    )

    stage_names = [stage["stage_name"] for stage in manifest["stages"]]
    require(len(stage_names) == len(set(stage_names)), "Stage names are not unique.")

    all_seen_seeds: set[int] = set()
    for stage, stage_manifest in zip(CURRICULUM_STAGES, manifest["stages"], strict=True):
        validate_stage_manifest(
            stage_manifest,
            stage=stage,
            split_counts=split_counts,
            seed_start=args.seed_start,
            seed_stride=args.seed_stride,
            sim_dt=args.sim_dt,
        )
        for split_name in SPLIT_ORDER:
            for payload in stage_manifest["splits"][split_name]:
                seed = int(payload["map_seed"])
                require(seed not in all_seen_seeds, f"Seed {seed} is reused across stages.")
                all_seen_seeds.add(seed)

    expected_summary = aggregate_manifest_summary(manifest["stages"], split_counts)
    require(manifest["summary"] == expected_summary, "Manifest summary does not match the task contents.")
    require(
        manifest["summary"]["total_tasks"] == len(all_seen_seeds),
        "Total task count does not match the number of unique seeds.",
    )

    return {
        "validated_stage_count": len(manifest["stages"]),
        "validated_total_tasks": manifest["summary"]["total_tasks"],
        "validated_total_by_split": manifest["summary"]["total_by_split"],
    }


def validate_runtime_observation(env, task, obs: dict[str, Any], info: dict[str, Any]) -> None:
    require(isinstance(obs, dict), "Environment observation is not a dict.")
    require("depth" in obs and "state" in obs, "Observation is missing depth/state keys.")
    depth = np.asarray(obs["depth"])
    state = np.asarray(obs["state"])
    require(depth.shape == (128, 128, 1), f"Unexpected depth shape: {depth.shape}.")
    require(
        state.shape == (FIXED_STATE_DIM,),
        f"Unexpected state shape: {state.shape}; expected {(FIXED_STATE_DIM,)}.",
    )
    require(np.isfinite(depth).all(), "Depth observation contains non-finite values.")
    require(np.isfinite(state).all(), "State observation contains non-finite values.")
    require(
        tuple(env.observation_space["state"].shape) == tuple(state.shape),
        "Observation-space state shape does not match runtime state shape.",
    )
    require(isinstance(info, dict), "reset() info must be a dict.")
    require("privileged" in info, "Training env did not attach privileged info.")
    privileged = info["privileged"]
    require(PRIVILEGED_KEYS.issubset(set(privileged.keys())), "Privileged info is missing required keys.")
    require(
        int(privileged["challenge_type"]) == int(task.challenge_type),
        "Privileged challenge_type does not match the task.",
    )
    require(
        bool(privileged["moving_platform"]) == bool(task.moving_platform),
        "Privileged moving_platform does not match the task.",
    )
    require(
        np.asarray(privileged["teacher_state"], dtype=np.float32).shape == (15,),
        "Teacher-state shape is unexpected.",
    )


def smoke_test_manifest(manifest: dict[str, Any], *, smoke_test_per_split: int) -> list[dict[str, Any]]:
    if smoke_test_per_split <= 0:
        return []

    rows: list[dict[str, Any]] = []

    for stage in manifest["stages"]:
        for split_name in SPLIT_ORDER:
            for payload in stage["splits"][split_name][:smoke_test_per_split]:
                task = task_from_payload(payload)
                env = make_training_env(task, gui=False, privileged=True)
                try:
                    obs, info = env.reset(seed=task.map_seed)
                    validate_runtime_observation(env, task, obs, info)

                    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
                    require(
                        zero_action.shape == tuple(env.action_space.shape),
                        f"Action shape mismatch for {stage['stage_name']}/{split_name}.",
                    )
                    zero_action.reshape(-1)[0] = 1.0

                    next_obs, reward, terminated, truncated, next_info = env.step(zero_action)
                    validate_runtime_observation(env, task, next_obs, next_info)
                    require(np.isfinite(float(reward)), "Step reward is not finite.")

                    rows.append(
                        {
                            "stage_name": stage["stage_name"],
                            "split": split_name,
                            "map_seed": int(task.map_seed),
                            "challenge_type": int(task.challenge_type),
                            "moving_platform": bool(task.moving_platform),
                            "depth_shape": list(np.asarray(obs["depth"]).shape),
                            "state_shape": list(np.asarray(obs["state"]).shape),
                            "reward_after_one_step": float(reward),
                            "terminated_after_one_step": bool(terminated),
                            "truncated_after_one_step": bool(truncated),
                        }
                    )
                finally:
                    env.close()

    return rows


def save_outputs(
    *,
    output_dir: Path,
    manifest: dict[str, Any],
    validation_summary: dict[str, Any],
    smoke_test_rows: list[dict[str, Any]] | None,
) -> dict[str, Path]:
    ensure_dir(output_dir)
    manifest_path = output_dir / "curriculum_manifest.json"
    summary_path = output_dir / "curriculum_summary.json"
    smoke_test_path = output_dir / "smoke_test_results.json"

    smoke_rows_to_save: list[dict[str, Any]]
    if smoke_test_rows is None:
        if smoke_test_path.exists():
            smoke_rows_to_save = list(load_json(smoke_test_path))
        else:
            smoke_rows_to_save = []
    else:
        smoke_rows_to_save = smoke_test_rows

    save_json(manifest_path, manifest)
    save_json(
        summary_path,
        {
            "manifest_sha256": manifest["manifest_sha256"],
            "stage_count": manifest["stage_count"],
            "summary": manifest["summary"],
            "validation": validation_summary,
            "smoke_test_count": len(smoke_rows_to_save),
            "smoke_test_ran": smoke_test_rows is not None,
        },
    )
    save_json(smoke_test_path, smoke_rows_to_save)

    saved_manifest = load_json(manifest_path)
    require(saved_manifest == manifest, "Saved manifest differs from in-memory manifest.")

    return {
        "manifest_path": manifest_path,
        "summary_path": summary_path,
        "smoke_test_path": smoke_test_path,
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.random_seed)

    split_counts = split_counts_from_args(args)
    for split_name, count in split_counts.items():
        require(count >= 0, f"{split_name} split size must be non-negative.")
    require(sum(split_counts.values()) > 0, "At least one task per stage is required.")
    require(args.seed_start >= 0, "--seed-start must be non-negative.")
    require(args.seed_stride > 0, "--seed-stride must be positive.")
    require(args.sim_dt > 0.0, "--sim-dt must be positive.")
    require(
        np.isclose(float(args.sim_dt), float(VALIDATOR_SIM_DT)),
        f"The training pipeline curriculum is pinned to validator SIM_DT={VALIDATOR_SIM_DT}.",
    )
    require(args.smoke_test_per_split >= 0, "--smoke-test-per-split must be non-negative.")

    manifest = build_manifest(args)
    manifest_again = build_manifest(args)
    require(manifest == manifest_again, "Manifest generation is not deterministic.")

    validation_summary = validate_manifest(manifest, args)
    smoke_test_rows: list[dict[str, Any]] | None = None
    if not args.skip_smoke_test:
        smoke_test_rows = smoke_test_manifest(
            manifest,
            smoke_test_per_split=args.smoke_test_per_split,
        )

    saved_paths = save_outputs(
        output_dir=args.output_dir,
        manifest=manifest,
        validation_summary=validation_summary,
        smoke_test_rows=smoke_test_rows,
    )

    print(
        json.dumps(
            {
                "manifest_path": str(saved_paths["manifest_path"]),
                "summary_path": str(saved_paths["summary_path"]),
                "smoke_test_path": str(saved_paths["smoke_test_path"]),
                "stage_count": manifest["stage_count"],
                "total_tasks": manifest["summary"]["total_tasks"],
                "total_by_split": manifest["summary"]["total_by_split"],
                "challenge_type_histogram": manifest["summary"]["challenge_type_histogram"],
                "moving_platform_histogram": manifest["summary"]["moving_platform_histogram"],
                "smoke_test_count": 0 if smoke_test_rows is None else len(smoke_test_rows),
                "manifest_sha256": manifest["manifest_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
