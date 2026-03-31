"""Train the student by behavior cloning on the expert dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

DEFAULT_MODEL_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
for root in (DEFAULT_MODEL_ROOT, REPO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from training_lib.bc import BehaviorCloningConfig, run_behavior_cloning
from training_lib.common import ensure_dir
from training_lib.dataset import list_episode_files, validate_episode_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_MODEL_ROOT / "artifacts" / "03_dataset_and_labels" / "train")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_ROOT / "artifacts" / "05_behavior_cloning")
    parser.add_argument("--model-config-path", type=Path, default=DEFAULT_MODEL_ROOT / "artifacts" / "04_student_model" / "student_model_config.json")
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_MODEL_ROOT / "artifacts" / "04_student_model" / "student_init.pt")
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-episodes", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    dataset_root = Path(args.dataset_root)
    episode_files = list_episode_files(dataset_root)
    if not episode_files:
        raise FileNotFoundError(f"No episode files found under {dataset_root}")

    sample_shapes = validate_episode_dataset(episode_files[0])
    smoke_manifest = {
        "dataset_root": str(dataset_root),
        "episode_count": len(episode_files),
        "sample_episode": str(episode_files[0]),
        "sample_shapes": {key: list(shape) for key, shape in sample_shapes.items()},
    }
    (args.output_dir / "dataset_smoke.json").write_text(json.dumps(smoke_manifest, indent=2, sort_keys=True), encoding="utf-8")

    config = BehaviorCloningConfig(
        dataset_root=str(args.dataset_root),
        output_dir=str(args.output_dir),
        model_config_path=str(args.model_config_path),
        init_checkpoint=str(args.init_checkpoint) if args.init_checkpoint else None,
        seq_len=args.seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        max_episodes=args.max_episodes,
    )
    result = run_behavior_cloning(config)
    with (args.output_dir / "bc_result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(result)


if __name__ == "__main__":
    main()
