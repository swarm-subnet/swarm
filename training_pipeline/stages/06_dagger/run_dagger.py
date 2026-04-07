"""Run DAgger on top of the behavior-cloned student."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys

DEFAULT_MODEL_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
for root in (DEFAULT_MODEL_ROOT, REPO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from training_env import make_training_env, task_from_payload
from training_lib.bc import BehaviorCloningConfig, run_behavior_cloning
from training_lib.common import ensure_dir, load_json, rollout_episode, save_json, seed_everything, summary_as_dict
from training_lib.dataset import DATASET_WEIGHTING_POLICIES, build_weighted_dataset_manifest, save_episode_dataset, validate_episode_dataset
from training_lib.experts import ExpertIdentity, load_expert_config, make_expert_policy, map_category_label
from training_lib.models import StudentInferencePolicy, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum-manifest", type=Path, default=DEFAULT_MODEL_ROOT / "artifacts" / "01_env_and_curriculum" / "curriculum_manifest.json")
    parser.add_argument("--base-dataset-root", type=Path, default=DEFAULT_MODEL_ROOT / "artifacts" / "03_dataset_and_labels" / "train")
    parser.add_argument("--expert-config", type=Path, default=DEFAULT_MODEL_ROOT / "artifacts" / "02_privileged_expert" / "expert_config.json")
    parser.add_argument("--student-checkpoint", type=Path, default=DEFAULT_MODEL_ROOT / "artifacts" / "05_behavior_cloning" / "best_student.pt")
    parser.add_argument("--model-config-path", type=Path, default=DEFAULT_MODEL_ROOT / "artifacts" / "04_student_model" / "student_model_config.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_ROOT / "artifacts" / "06_dagger")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--episodes-per-stage", type=int, default=4)
    parser.add_argument("--stage-name", action="append", default=None, help="Optional repeatable stage filter for rollout collection.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional per-episode step cap for smoke runs.")
    parser.add_argument("--bc-epochs", type=int, default=5, help="Retraining epochs after each DAgger iteration.")
    parser.add_argument("--bc-seq-len", type=int, default=16, help="Sequence length for BC retraining windows.")
    parser.add_argument("--bc-stride", type=int, default=8, help="Window stride for BC retraining.")
    parser.add_argument("--bc-batch-size", type=int, default=16, help="Batch size for BC retraining.")
    parser.add_argument("--bc-learning-rate", type=float, default=3e-4, help="Learning rate for BC retraining.")
    parser.add_argument("--bc-max-episodes", type=int, default=None, help="Optional cap on episodes used during BC retraining.")
    parser.add_argument(
        "--dataset-weighting-policy",
        choices=DATASET_WEIGHTING_POLICIES,
        default="balanced_by_map_category",
        help="How merged datasets are reweighted for BC retraining.",
    )
    parser.add_argument("--teacher-id", type=str, default="expert_shared")
    parser.add_argument("--teacher-version", type=str, default="v0")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--random-seed", type=int, default=17)
    return parser.parse_args()

def copy_dataset_tree(source_root: Path, destination_root: Path) -> None:
    destination_root.mkdir(parents=True, exist_ok=True)
    for episode in source_root.rglob("*.npz"):
        rel = episode.relative_to(source_root)
        target = destination_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(episode, target)
        meta = episode.with_suffix(".json")
        if meta.exists():
            shutil.copy2(meta, target.with_suffix(".json"))


def main() -> None:
    args = parse_args()
    seed_everything(args.random_seed)
    ensure_dir(args.output_dir)

    manifest = load_json(args.curriculum_manifest)
    expert_config = load_expert_config(args.expert_config)
    expert_cache = {}
    current_checkpoint = Path(args.student_checkpoint)
    merged_dataset_root = ensure_dir(args.output_dir / "merged_dataset")
    copy_dataset_tree(args.base_dataset_root, merged_dataset_root)
    merged_dataset_manifest_path = args.output_dir / "merged_dataset_sampling_manifest.json"
    build_weighted_dataset_manifest(
        merged_dataset_root,
        weighting_policy=str(args.dataset_weighting_policy),
        output_path=merged_dataset_manifest_path,
    )
    source_episode_count = len(list(merged_dataset_root.rglob("*.npz")))
    stage_filter = set(args.stage_name) if args.stage_name else None

    iteration_rows = []
    for iteration in range(args.iterations):
        model, _ = load_checkpoint(current_checkpoint, map_location=args.device)
        student_policy = StudentInferencePolicy(model, device=args.device)
        iteration_dataset_root = ensure_dir(args.output_dir / f"iteration_{iteration:03d}" / "new_rollouts")

        episode_index = 0
        for stage in manifest["stages"]:
            if stage_filter is not None and stage["stage_name"] not in stage_filter:
                continue
            for payload in stage["splits"]["train"][: args.episodes_per_stage]:
                task = task_from_payload(payload)
                map_category = map_category_label(task.challenge_type, task.moving_platform)
                if map_category not in expert_cache:
                    expert_cache[map_category] = make_expert_policy(
                        expert_config,
                        identity=ExpertIdentity(
                            teacher_id=str(args.teacher_id),
                            teacher_version=str(args.teacher_version),
                            map_category=map_category,
                        ),
                    )
                expert = expert_cache[map_category]
                env = make_training_env(task, gui=False, privileged=True)
                try:
                    step_records, summary = rollout_episode(
                        env,
                        student_policy,
                        episode_index=episode_index,
                        seed=task.map_seed,
                        max_steps=args.max_steps,
                        record_steps=True,
                    )

                    expert.reset()
                    relabeled_records = []
                    for record in step_records:
                        observation = record["observation"]
                        info = record["info"]
                        expert_action = expert.act(observation, info)
                        relabeled_records.append(
                            {
                                "observation": observation,
                                "info": info,
                                "action": expert_action,
                                "reward": float(record["reward"]),
                                "terminated": bool(record["terminated"]),
                                "truncated": bool(record["truncated"]),
                                "metadata": expert.get_last_metadata(),
                            }
                        )

                    episode_name = f"{map_category}_{args.teacher_id}_iter_{iteration:03d}_seed_{task.map_seed}"
                    rollout_dir = ensure_dir(iteration_dataset_root / map_category / str(args.teacher_id))
                    saved_path = save_episode_dataset(
                        rollout_dir,
                        episode_name,
                        relabeled_records,
                        summary,
                        extra_metadata={
                            "dataset_weight": 1.0,
                            "teacher_id": str(args.teacher_id),
                            "teacher_version": str(args.teacher_version),
                            "map_category": map_category,
                        },
                    )
                    dataset_shapes = validate_episode_dataset(saved_path)

                    merge_target = merged_dataset_root / saved_path.relative_to(iteration_dataset_root)
                    merge_target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(saved_path, merge_target)
                    shutil.copy2(saved_path.with_suffix(".json"), merge_target.with_suffix(".json"))

                    row = summary_as_dict(summary)
                    row["stage_name"] = stage["stage_name"]
                    row["map_category"] = map_category
                    row["iteration"] = iteration
                    row["teacher_id"] = str(args.teacher_id)
                    row["teacher_version"] = str(args.teacher_version)
                    row["merged_dataset_path"] = str(merge_target)
                    row["dataset_shapes"] = {key: list(shape) for key, shape in dataset_shapes.items()}
                    iteration_rows.append(row)
                finally:
                    env.close()
                episode_index += 1

        bc_output_dir = ensure_dir(args.output_dir / f"iteration_{iteration:03d}" / "bc_retrain")
        build_weighted_dataset_manifest(
            merged_dataset_root,
            weighting_policy=str(args.dataset_weighting_policy),
            output_path=merged_dataset_manifest_path,
        )
        bc_result = run_behavior_cloning(
            BehaviorCloningConfig(
                dataset_root=str(merged_dataset_root),
                dataset_manifest=str(merged_dataset_manifest_path),
                output_dir=str(bc_output_dir),
                model_config_path=str(args.model_config_path),
                init_checkpoint=str(current_checkpoint),
                epochs=args.bc_epochs,
                seq_len=args.bc_seq_len,
                stride=args.bc_stride,
                batch_size=args.bc_batch_size,
                learning_rate=args.bc_learning_rate,
                device=args.device,
                max_episodes=args.bc_max_episodes,
            )
        )
        current_checkpoint = Path(bc_result["checkpoint_path"])

    save_json(args.output_dir / "dagger_manifest.json", iteration_rows)
    save_json(
        args.output_dir / "dagger_latest_checkpoint.json",
        {
            "checkpoint_path": str(current_checkpoint),
            "source_episode_count": source_episode_count,
            "iteration_count": args.iterations,
        },
    )
    print(f"Saved DAgger outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
