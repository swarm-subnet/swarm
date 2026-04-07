"""Collect expert trajectories and derive perception labels from simulator truth."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

DEFAULT_MODEL_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
for root in (DEFAULT_MODEL_ROOT, REPO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from training_env import make_training_env
from training_env import task_from_payload
from training_lib.common import ensure_dir, load_json, rollout_episode, save_json, seed_everything, summary_as_dict
from training_lib.dataset import (
    DATASET_WEIGHTING_POLICIES,
    build_weighted_dataset_manifest,
    save_episode_dataset,
    validate_episode_dataset,
)
from training_lib.experts import (
    ExpertIdentity,
    PrivilegedExpertPolicy,
    build_specialist_policy,
    default_mode_vocabulary_payload,
    evaluate_quality_gate,
    load_expert_config,
    load_quality_gate,
    load_specialist_registry,
    map_category_label,
    make_expert_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum-manifest", type=Path, default=DEFAULT_MODEL_ROOT / "artifacts" / "01_env_and_curriculum" / "curriculum_manifest.json")
    parser.add_argument("--expert-config", type=Path, default=DEFAULT_MODEL_ROOT / "artifacts" / "02_privileged_expert" / "expert_config.json")
    parser.add_argument("--expert-registry", type=Path, default=None, help="Optional specialist registry. When set, experts are selected by map category.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_ROOT / "artifacts" / "03_dataset_and_labels")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="train")
    parser.add_argument("--episodes-per-stage", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--stage-name", action="append", default=None, help="Repeatable stage filter.")
    parser.add_argument("--map-seed", action="append", type=int, default=None, help="Repeatable exact task-seed filter.")
    parser.add_argument("--max-episodes-total", type=int, default=None)
    parser.add_argument("--teacher-id", type=str, default="expert_shared")
    parser.add_argument("--teacher-version", type=str, default="v0")
    parser.add_argument("--only-success", action="store_true", help="Save only successful episodes into the training dataset.")
    parser.add_argument(
        "--dataset-weighting-policy",
        choices=DATASET_WEIGHTING_POLICIES,
        default="balanced_by_map_category",
    )
    parser.add_argument("--require-quality-gates", action="store_true")
    parser.add_argument("--random-seed", type=int, default=11)
    parser.add_argument("--gui", action="store_true")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    seed_everything(args.random_seed)
    ensure_dir(args.output_dir)

    manifest = load_json(args.curriculum_manifest)
    specialist_registry = load_specialist_registry(args.expert_registry) if args.expert_registry else None
    shared_single_config = load_expert_config(args.expert_config) if args.expert_registry is None else None
    selected_stage_names = set(args.stage_name) if args.stage_name else None
    selected_map_seeds = set(args.map_seed) if args.map_seed else None
    expert_cache: dict[str, tuple[PrivilegedExpertPolicy, dict[str, object]]] = {}
    rows = []
    rejected_rows = []
    episode_index = 0
    attempted_episode_count = 0
    split_episode_counts: dict[str, int] = {}
    category_episode_counts: dict[str, int] = {}

    save_json(args.output_dir / "mode_vocabulary.json", default_mode_vocabulary_payload())

    for stage in manifest["stages"]:
        stage_name = str(stage["stage_name"])
        if selected_stage_names is not None and stage_name not in selected_stage_names:
            continue
        split_names = [args.split] if args.split != "all" else ["train", "val", "test"]
        for split_name in split_names:
            payloads = stage["splits"][split_name][: args.episodes_per_stage]
            for payload in payloads:
                if selected_map_seeds is not None and int(payload["map_seed"]) not in selected_map_seeds:
                    continue
                if args.max_episodes_total is not None and episode_index >= int(args.max_episodes_total):
                    break
                task = task_from_payload(payload)
                map_category = map_category_label(task.challenge_type, task.moving_platform)
                if stage_name != map_category:
                    raise ValueError(
                        f"Manifest stage_name={stage_name!r} does not match task map_category={map_category!r}"
                    )
                if map_category not in expert_cache:
                    if specialist_registry is not None:
                        policy, spec = build_specialist_policy(
                            registry_path=args.expert_registry,
                            map_category=map_category,
                        )
                        expert_info: dict[str, object] = {
                            "teacher_id": spec.teacher_id,
                            "teacher_version": spec.teacher_version,
                            "dataset_weight": float(spec.dataset_weight),
                            "quality_gate_report": None,
                        }
                        if args.require_quality_gates:
                            if spec.eval_summary_path is None or spec.quality_gate_path is None:
                                raise ValueError(
                                    f"Specialist {spec.teacher_id} for {map_category} is missing "
                                    "eval_summary_path or quality_gate_path"
                                )
                            registry_root = args.expert_registry.resolve().parent
                            summary_path = (registry_root / spec.eval_summary_path).resolve()
                            gate_path = (registry_root / spec.quality_gate_path).resolve()
                            quality_gate = load_quality_gate(gate_path)
                            gate_report = evaluate_quality_gate(
                                summary=load_json(summary_path),
                                map_category=map_category,
                                quality_gate=quality_gate,
                                teacher_id=spec.teacher_id,
                                teacher_version=spec.teacher_version,
                            )
                            if not bool(gate_report["accepted"]):
                                raise ValueError(
                                    f"Specialist {spec.teacher_id}@{spec.teacher_version} "
                                    f"failed its quality gate for {map_category}: {gate_report}"
                                )
                            expert_info["quality_gate_report"] = gate_report
                    else:
                        policy = make_expert_policy(
                            shared_single_config or load_expert_config(args.expert_config),
                            identity=ExpertIdentity(
                                teacher_id=str(args.teacher_id),
                                teacher_version=str(args.teacher_version),
                                map_category=map_category,
                            ),
                        )
                        expert_info = {
                            "teacher_id": str(args.teacher_id),
                            "teacher_version": str(args.teacher_version),
                            "dataset_weight": 1.0,
                            "quality_gate_report": None,
                        }
                    expert_cache[map_category] = (policy, expert_info)
                expert, expert_info = expert_cache[map_category]
                split_dir = ensure_dir(
                    args.output_dir
                    / split_name
                    / map_category
                    / str(expert_info["teacher_id"])
                )
                env = make_training_env(task, gui=args.gui, privileged=True)
                try:
                    step_records, summary = rollout_episode(
                        env,
                        expert,
                        episode_index=episode_index,
                        seed=task.map_seed,
                        max_steps=args.max_steps,
                        record_steps=True,
                    )
                    attempted_episode_count += 1
                    summary_row = summary_as_dict(summary)
                    summary_row["stage_name"] = stage_name
                    summary_row["map_category"] = map_category
                    summary_row["split"] = split_name
                    summary_row["teacher_id"] = str(expert_info["teacher_id"])
                    summary_row["teacher_version"] = str(expert_info["teacher_version"])
                    summary_row["dataset_weight"] = float(expert_info["dataset_weight"])
                    if expert_info["quality_gate_report"] is not None:
                        summary_row["quality_gate_report"] = expert_info["quality_gate_report"]
                    if args.only_success and not bool(summary.success):
                        rejected_rows.append(summary_row)
                        continue
                    episode_name = f"{map_category}_{expert_info['teacher_id']}_seed_{task.map_seed}"
                    npz_path = save_episode_dataset(
                        split_dir,
                        episode_name,
                        step_records,
                        summary,
                        extra_metadata={
                            "dataset_weight": float(expert_info["dataset_weight"]),
                            "teacher_id": str(expert_info["teacher_id"]),
                            "teacher_version": str(expert_info["teacher_version"]),
                            "map_category": map_category,
                            "stage_name": stage_name,
                            "split": split_name,
                        },
                    )
                    dataset_shapes = validate_episode_dataset(npz_path)
                    row = summary_row
                    row["dataset_path"] = str(npz_path)
                    row["dataset_shapes"] = {key: list(shape) for key, shape in dataset_shapes.items()}
                    rows.append(row)
                    split_episode_counts[split_name] = split_episode_counts.get(split_name, 0) + 1
                    category_episode_counts[map_category] = category_episode_counts.get(map_category, 0) + 1
                finally:
                    env.close()
                episode_index += 1
            if args.max_episodes_total is not None and episode_index >= int(args.max_episodes_total):
                break
        if args.max_episodes_total is not None and episode_index >= int(args.max_episodes_total):
            break

    dataset_manifest = {
        "format_version": 2,
        "curriculum_manifest": str(args.curriculum_manifest),
        "expert_registry": str(args.expert_registry) if args.expert_registry else None,
        "expert_config": str(args.expert_config) if args.expert_registry is None else None,
        "split_request": args.split,
        "only_success": bool(args.only_success),
        "attempted_episode_count": attempted_episode_count,
        "dataset_weighting_policy": str(args.dataset_weighting_policy),
        "episode_count": len(rows),
        "episodes_by_split": split_episode_counts,
        "episodes_by_map_category": category_episode_counts,
        "episodes": rows,
    }
    save_json(args.output_dir / "dataset_manifest.json", dataset_manifest)
    for split_name in ("train", "val", "test"):
        split_root = args.output_dir / split_name
        if split_root.exists() and list(split_root.rglob("*.npz")):
            build_weighted_dataset_manifest(
                split_root,
                weighting_policy=str(args.dataset_weighting_policy),
                output_path=args.output_dir / f"{split_name}_sampling_manifest.json",
            )
    save_json(
        args.output_dir / "dataset_summary.json",
        {
            "episode_count": len(rows),
            "attempted_episode_count": attempted_episode_count,
            "rejected_episode_count": len(rejected_rows),
            "episodes_by_split": split_episode_counts,
            "episodes_by_map_category": category_episode_counts,
            "only_success": bool(args.only_success),
            "dataset_weighting_policy": str(args.dataset_weighting_policy),
        },
    )
    if rejected_rows:
        save_json(
            args.output_dir / "rejected_episodes.json",
            {
                "only_success": bool(args.only_success),
                "episode_count": len(rejected_rows),
                "episodes": rejected_rows,
            },
        )
    print(f"Saved dataset manifest to {args.output_dir / 'dataset_manifest.json'}")


if __name__ == "__main__":
    main()
