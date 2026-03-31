"""Bootstrap a residual landing head from expert corrections.

This is a practical open-source skeleton:
- filter for near-goal states
- compute expert minus base-policy action deltas
- fit a small residual model

It is a good precursor to later online residual RL.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

DEFAULT_MODEL_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
for root in (DEFAULT_MODEL_ROOT, REPO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import numpy as np
import torch
import torch.nn as nn

from training_env import make_training_env
from training_env import task_from_payload
from training_lib.common import ensure_dir, load_json, save_json, seed_everything
from training_lib.experts import PrivilegedExpertPolicy, load_expert_config
from training_lib.models import StudentInferencePolicy, load_checkpoint


@dataclass
class ResidualLandingConfig:
    near_goal_distance_m: float = 3.0
    hidden_dim: int = 128
    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 1e-3
    device: str = "cpu"


class ResidualLandingMLP(nn.Module):
    def __init__(self, state_dim: int = 141, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        raw = self.net(state)
        direction = torch.tanh(raw[..., 0:3])
        speed = torch.tanh(raw[..., 3:4]) * 0.25
        yaw = torch.tanh(raw[..., 4:5]) * 0.25
        return torch.cat([direction, speed, yaw], dim=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum-manifest", type=Path, default=DEFAULT_MODEL_ROOT / "artifacts" / "01_env_and_curriculum" / "curriculum_manifest.json")
    parser.add_argument("--expert-config", type=Path, default=DEFAULT_MODEL_ROOT / "artifacts" / "02_privileged_expert" / "expert_config.json")
    parser.add_argument("--student-checkpoint", type=Path, default=DEFAULT_MODEL_ROOT / "artifacts" / "06_dagger" / "dagger_latest_checkpoint.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_ROOT / "artifacts" / "08_residual_landing")
    parser.add_argument("--episodes-per-stage", type=int, default=4)
    parser.add_argument("--stage-name", action="append", default=None, help="Optional repeatable stage filter for residual collection.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional per-episode step cap for smoke runs.")
    parser.add_argument(
        "--near-goal-distance-m",
        type=float,
        default=3.0,
        help="Collect residual supervision whenever the platform is within this radius.",
    )
    parser.add_argument(
        "--collection-policy",
        type=str,
        choices=("student_then_expert", "student_only", "expert_only"),
        default="student_then_expert",
        help="Which controller drives the environment during residual data collection.",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--random-seed", type=int, default=23)
    return parser.parse_args()


def resolve_student_checkpoint(path: Path) -> Path:
    if path.suffix == ".json":
        payload = load_json(path)
        return Path(payload["checkpoint_path"])
    return path


def collect_residual_samples(
    *,
    manifest: dict,
    expert: PrivilegedExpertPolicy,
    student: StudentInferencePolicy,
    config: ResidualLandingConfig,
    episodes_per_stage: int,
    stage_filter: set[str] | None,
    max_steps: int | None,
    control_policy: str,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, int | str | float]]:
    states: list[np.ndarray] = []
    residual_targets: list[np.ndarray] = []
    rollout_count = 0
    near_goal_hits = 0
    max_distance_seen = 0.0
    min_distance_seen = float("inf")
    collection_mode = "student" if control_policy == "student_only" else "expert"

    for stage in manifest["stages"]:
        if stage_filter is not None and stage["stage_name"] not in stage_filter:
            continue
        for payload in stage["splits"]["train"][:episodes_per_stage]:
            task = task_from_payload(payload)
            env = make_training_env(task, gui=False, privileged=True)
            try:
                observation, info = env.reset(seed=task.map_seed)
                student.reset()
                expert.reset()
                done = False
                steps = 0
                rollout_count += 1
                while not done:
                    if max_steps is not None and steps >= max_steps:
                        break
                    distance_to_platform = float(np.linalg.norm(info["privileged"]["relative_platform"]))
                    max_distance_seen = max(max_distance_seen, distance_to_platform)
                    min_distance_seen = min(min_distance_seen, distance_to_platform)
                    student_action = np.asarray(student.act(observation, info), dtype=np.float32)
                    expert_action = np.asarray(expert.act(observation, info), dtype=np.float32)
                    if distance_to_platform <= config.near_goal_distance_m:
                        near_goal_hits += 1
                        states.append(np.asarray(observation["state"], dtype=np.float32))
                        residual_targets.append(np.clip(expert_action - student_action, -0.25, 0.25))
                    action = student_action if collection_mode == "student" else expert_action
                    action = np.asarray(action, dtype=np.float32).reshape(env.action_space.shape)
                    observation, _, terminated, truncated, info = env.step(action)
                    done = bool(terminated or truncated)
                    steps += 1
            finally:
                env.close()

    summary = {
        "collection_mode": collection_mode,
        "rollout_count": rollout_count,
        "near_goal_hits": near_goal_hits,
        "max_distance_seen": max_distance_seen,
        "min_distance_seen": 0.0 if min_distance_seen == float("inf") else min_distance_seen,
    }
    return states, residual_targets, summary


def main() -> None:
    args = parse_args()
    seed_everything(args.random_seed)
    ensure_dir(args.output_dir)

    config = ResidualLandingConfig(
        near_goal_distance_m=args.near_goal_distance_m,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    manifest = load_json(args.curriculum_manifest)
    expert = PrivilegedExpertPolicy(load_expert_config(args.expert_config))
    student_model, _ = load_checkpoint(resolve_student_checkpoint(args.student_checkpoint), map_location=args.device)
    student = StudentInferencePolicy(student_model, device=args.device)
    stage_filter = set(args.stage_name) if args.stage_name else None

    collection_attempts: list[dict[str, int | str | float]] = []
    states: list[np.ndarray] = []
    residual_targets: list[np.ndarray] = []
    collection_modes: list[str]
    if args.collection_policy == "student_then_expert":
        collection_modes = ["student_only", "expert_only"]
    else:
        collection_modes = [args.collection_policy]

    for mode in collection_modes:
        states, residual_targets, summary = collect_residual_samples(
            manifest=manifest,
            expert=expert,
            student=student,
            config=config,
            episodes_per_stage=args.episodes_per_stage,
            stage_filter=stage_filter,
            max_steps=args.max_steps,
            control_policy=mode,
        )
        summary["requested_mode"] = mode
        collection_attempts.append(summary)
        if states:
            break

    if not states:
        model = ResidualLandingMLP(state_dim=141, hidden_dim=config.hidden_dim).to(config.device)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": asdict(config),
                "note": "No-op residual landing head. No near-goal states were collected.",
                "collection_attempts": collection_attempts,
            },
            args.output_dir / "residual_landing.pt",
        )
        save_json(args.output_dir / "residual_landing_config.json", asdict(config))
        save_json(
            args.output_dir / "residual_landing_summary.json",
            {
                "status": "no_data",
                "collected_states": 0,
                "collection_attempts": collection_attempts,
                "near_goal_distance_m": config.near_goal_distance_m,
                "stage_filter": sorted(stage_filter) if stage_filter else None,
            },
        )
        print(f"Saved no-op residual landing head to {args.output_dir / 'residual_landing.pt'}")
        return

    model = ResidualLandingMLP(state_dim=len(states[0]), hidden_dim=config.hidden_dim).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    state_tensor = torch.as_tensor(np.stack(states), dtype=torch.float32, device=config.device)
    target_tensor = torch.as_tensor(np.stack(residual_targets), dtype=torch.float32, device=config.device)

    for _ in range(config.epochs):
        permutation = torch.randperm(state_tensor.shape[0], device=config.device)
        for start in range(0, state_tensor.shape[0], config.batch_size):
            idx = permutation[start : start + config.batch_size]
            pred = model(state_tensor[idx])
            loss = torch.mean((pred - target_tensor[idx]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(config),
            "note": "Residual landing head trained on expert-student action deltas near the goal.",
        },
        args.output_dir / "residual_landing.pt",
    )
    summary = {
        "status": "trained",
        "collected_states": len(states),
        "near_goal_distance_m": config.near_goal_distance_m,
        "state_dim": len(states[0]),
        "residual_target_mean_abs": float(np.mean(np.abs(np.stack(residual_targets)))),
        "residual_target_max_abs": float(np.max(np.abs(np.stack(residual_targets)))),
        "collection_attempts": collection_attempts,
        "stage_filter": sorted(stage_filter) if stage_filter else None,
    }
    save_json(args.output_dir / "residual_landing_config.json", asdict(config))
    save_json(args.output_dir / "residual_landing_summary.json", summary)
    print(f"Saved residual landing head to {args.output_dir / 'residual_landing.pt'}")


if __name__ == "__main__":
    main()
