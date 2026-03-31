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
from training_lib.dataset import save_episode_dataset, validate_episode_dataset
from training_lib.experts import PrivilegedExpertPolicy, load_expert_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum-manifest", type=Path, default=DEFAULT_MODEL_ROOT / "artifacts" / "01_env_and_curriculum" / "curriculum_manifest.json")
    parser.add_argument("--expert-config", type=Path, default=DEFAULT_MODEL_ROOT / "artifacts" / "02_privileged_expert" / "expert_config.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_ROOT / "artifacts" / "03_dataset_and_labels")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="train")
    parser.add_argument("--episodes-per-stage", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=11)
    parser.add_argument("--gui", action="store_true")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    seed_everything(args.random_seed)
    ensure_dir(args.output_dir)

    manifest = load_json(args.curriculum_manifest)
    expert = PrivilegedExpertPolicy(load_expert_config(args.expert_config))
    rows = []
    episode_index = 0

    for stage in manifest["stages"]:
        split_names = [args.split] if args.split != "all" else ["train", "val", "test"]
        for split_name in split_names:
            payloads = stage["splits"][split_name][: args.episodes_per_stage]
            split_dir = ensure_dir(args.output_dir / split_name / stage["stage_name"])
            for payload in payloads:
                task = task_from_payload(payload)
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
                    episode_name = f"{stage['stage_name']}_seed_{task.map_seed}"
                    npz_path = save_episode_dataset(split_dir, episode_name, step_records, summary)
                    dataset_shapes = validate_episode_dataset(npz_path)
                    row = summary_as_dict(summary)
                    row["stage_name"] = stage["stage_name"]
                    row["split"] = split_name
                    row["dataset_path"] = str(npz_path)
                    row["dataset_shapes"] = {key: list(shape) for key, shape in dataset_shapes.items()}
                    rows.append(row)
                finally:
                    env.close()
                episode_index += 1

    save_json(args.output_dir / "dataset_manifest.json", rows)
    print(f"Saved dataset manifest to {args.output_dir / 'dataset_manifest.json'}")


if __name__ == "__main__":
    main()
