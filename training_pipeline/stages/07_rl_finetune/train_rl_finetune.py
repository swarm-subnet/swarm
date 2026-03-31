"""Fine-tune the student with recurrent PPO.

This script provides a baseline recurrent PPO path that has a realistic chance
to run with `sb3-contrib`. The asymmetric-critic upgrade is left as an explicit
extension point because it requires a custom policy/loss implementation.
"""

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

import gymnasium as gym

from training_env import make_training_env, task_from_payload
from training_lib.common import (
    ensure_dir,
    load_json,
    rollout_episode,
    save_json,
    seed_everything,
    summary_as_dict,
)
from training_lib.models import StudentInferencePolicy, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum-manifest", type=Path, default=DEFAULT_MODEL_ROOT / "artifacts" / "01_env_and_curriculum" / "curriculum_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_ROOT / "artifacts" / "07_rl_finetune")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--stage-name", type=str, default="benchmark_like")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument(
        "--trainer",
        choices=["sb3_recurrent_ppo", "custom_asymmetric_ppo", "smoke_only"],
        default="sb3_recurrent_ppo",
    )
    parser.add_argument("--student-checkpoint", type=Path, default=DEFAULT_MODEL_ROOT / "artifacts" / "05_behavior_cloning" / "best_student.pt")
    parser.add_argument("--model-config-path", type=Path, default=DEFAULT_MODEL_ROOT / "artifacts" / "04_student_model" / "student_model_config.json")
    parser.add_argument("--smoke-episodes", type=int, default=2)
    parser.add_argument("--smoke-max-steps", type=int, default=24)
    parser.add_argument("--random-seed", type=int, default=19)
    return parser.parse_args()


class ZeroPolicy:
    """Cheap deployable-only fallback policy for smoke runs."""

    def reset(self) -> None:
        self.last_metadata: dict[str, str] = {"source": "zero_policy"}

    def get_last_metadata(self) -> dict[str, str]:
        return dict(self.last_metadata)

    def act(self, observation, info):
        return [0.0, 0.0, 0.0, 0.0, 0.0]


def load_fallback_policy(args: argparse.Namespace):
    checkpoint_path = Path(args.student_checkpoint)
    config_path = Path(args.model_config_path)
    if checkpoint_path.exists() and config_path.exists():
        model, _ = load_checkpoint(checkpoint_path, map_location="cpu")
        return StudentInferencePolicy(model, device="cpu"), "student"
    policy = ZeroPolicy()
    policy.reset()
    return policy, "zero"


class CurriculumSamplerEnv(gym.Env):
    """Resettable env that rotates through manifest tasks."""

    metadata = {"render_modes": []}

    def __init__(self, task_payloads: list[dict]):
        super().__init__()
        self.task_payloads = list(task_payloads)
        self.cursor = 0
        self.inner_env = None
        self.observation_space = None
        self.action_space = None
        self._build_env()

    def _build_env(self):
        if self.inner_env is not None:
            self.inner_env.close()
        payload = self.task_payloads[self.cursor % len(self.task_payloads)]
        self.cursor += 1
        task = task_from_payload(payload)
        self.inner_env = make_training_env(task, gui=False, privileged=False)
        self.observation_space = self.inner_env.observation_space
        self.action_space = self.inner_env.action_space

    def reset(self, *, seed=None, options=None):
        self._build_env()
        return self.inner_env.reset(seed=seed)

    def step(self, action):
        return self.inner_env.step(action)

    def close(self):
        if self.inner_env is not None:
            self.inner_env.close()
            self.inner_env = None


def run_fallback_smoke(args: argparse.Namespace, manifest: dict) -> dict[str, object]:
    policy, policy_name = load_fallback_policy(args)
    stage = next(stage for stage in manifest["stages"] if stage["stage_name"] == args.stage_name)
    tasks = stage["splits"][args.split][: max(1, args.smoke_episodes)]
    rows = []
    for index, payload in enumerate(tasks):
        task = task_from_payload(payload)
        env = make_training_env(task, gui=False, privileged=False)
        try:
            _, summary = rollout_episode(
                env,
                policy,
                episode_index=index,
                seed=task.map_seed,
                max_steps=args.smoke_max_steps,
                record_steps=False,
            )
            row = summary_as_dict(summary)
            row["stage_name"] = stage["stage_name"]
            row["split"] = args.split
            row["policy"] = policy_name
            rows.append(row)
        finally:
            env.close()

    save_json(args.output_dir / "rl_fallback_manifest.json", rows)
    save_json(
        args.output_dir / "rl_fallback_summary.json",
        {
            "trainer": args.trainer,
            "policy": policy_name,
            "stage_name": args.stage_name,
            "split": args.split,
            "smoke_episodes": len(rows),
            "smoke_max_steps": args.smoke_max_steps,
            "note": "Fallback path executed because recurrent PPO was unavailable or explicitly skipped.",
        },
    )
    return {
        "mode": "fallback",
        "policy": policy_name,
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.random_seed)
    ensure_dir(args.output_dir)

    if args.trainer == "smoke_only":
        manifest = load_json(args.curriculum_manifest)
        result = run_fallback_smoke(args, manifest)
        print(json.dumps({"mode": result["mode"], "policy": result["policy"], "rows": len(result["rows"])}, indent=2))
        return

    if args.trainer == "custom_asymmetric_ppo":
        raise NotImplementedError(
            "The custom asymmetric-critic PPO path is the right long-term direction, "
            "but it still needs a dedicated implementation around the StudentPolicy "
            "and AsymmetricCritic modules."
        )

    try:
        from sb3_contrib import RecurrentPPO
    except ImportError as exc:
        manifest = load_json(args.curriculum_manifest)
        result = run_fallback_smoke(args, manifest)
        print(json.dumps({"mode": result["mode"], "policy": result["policy"], "rows": len(result["rows"])}, indent=2))
        return

    manifest = load_json(args.curriculum_manifest)
    stage = next(stage for stage in manifest["stages"] if stage["stage_name"] == args.stage_name)
    env = CurriculumSamplerEnv(stage["splits"][args.split])
    model = RecurrentPPO(
        policy="MultiInputLstmPolicy",
        env=env,
        verbose=1,
        tensorboard_log=str(args.output_dir / "tensorboard"),
        seed=args.random_seed,
    )
    model.learn(total_timesteps=args.total_timesteps)
    model.save(str(args.output_dir / "recurrent_ppo_policy"))
    save_json(
        args.output_dir / "rl_config.json",
        {
            "trainer": args.trainer,
            "stage_name": args.stage_name,
            "split": args.split,
            "total_timesteps": args.total_timesteps,
            "note": "This baseline uses recurrent PPO over deployable observations only. "
            "Upgrade to a custom asymmetric critic when the rest of the pipeline is stable.",
        },
    )
    env.close()


if __name__ == "__main__":
    main()
