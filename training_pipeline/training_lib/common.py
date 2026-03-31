"""Common filesystem, rollout, and reproducibility helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import random
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional during non-training usage
    torch = None


DEFAULT_MODEL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DEFAULT_MODEL_ROOT.parent
ARTIFACTS_ROOT = DEFAULT_MODEL_ROOT / "artifacts"


class PolicyProtocol(Protocol):
    """Simple rollout protocol used by the stage scripts."""

    def reset(self) -> None: ...

    def act(self, observation: dict[str, Any], info: dict[str, Any]) -> np.ndarray: ...

    def get_last_metadata(self) -> dict[str, Any]: ...


@dataclass
class EpisodeSummary:
    """Compact episode summary saved next to rollout artifacts."""

    seed: int | None
    episode_index: int
    total_reward: float
    steps: int
    success: bool
    collision: bool
    score: float
    challenge_type: int
    moving_platform: bool
    wall_time_sec: float


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def rollout_episode(
    env,
    policy: PolicyProtocol,
    *,
    episode_index: int,
    seed: int | None = None,
    max_steps: int | None = None,
    record_steps: bool = False,
) -> tuple[list[dict[str, Any]], EpisodeSummary]:
    """Roll out a single episode.

    The saved step record intentionally contains both deployable observations
    and training-time info so later stages can rebuild either view.
    """

    wall_start = time.perf_counter()
    observation, info = env.reset(seed=seed)
    policy.reset()

    step_records: list[dict[str, Any]] = []
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        if max_steps is not None and steps >= max_steps:
            break

        action = np.asarray(policy.act(observation, info), dtype=np.float32)
        expected_shape = tuple(getattr(env.action_space, "shape", ()))
        if expected_shape:
            if action.shape != expected_shape:
                if action.size != int(np.prod(expected_shape)):
                    raise ValueError(
                        f"Policy action shape {action.shape} is incompatible with env action shape {expected_shape}"
                    )
                action = action.reshape(expected_shape)
        next_observation, reward, terminated, truncated, next_info = env.step(action)
        total_reward += float(reward)

        if record_steps:
            step_records.append(
                {
                    "observation": observation,
                    "info": info,
                    "action": action,
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "metadata": dict(policy.get_last_metadata()),
                }
            )

        observation = next_observation
        info = next_info
        steps += 1

    privileged = info.get("privileged", {}) if isinstance(info, dict) else {}
    wall_time_sec = time.perf_counter() - wall_start
    summary = EpisodeSummary(
        seed=seed,
        episode_index=episode_index,
        total_reward=total_reward,
        steps=steps,
        success=bool(privileged.get("success", info.get("success", False))),
        collision=bool(privileged.get("collision", info.get("collision", False))),
        score=float(info.get("score", 0.0)),
        challenge_type=int(privileged.get("challenge_type", getattr(env.task, "challenge_type", -1))),
        moving_platform=bool(privileged.get("moving_platform", getattr(env.task, "moving_platform", False))),
        wall_time_sec=wall_time_sec,
    )
    return step_records, summary


def summary_as_dict(summary: EpisodeSummary) -> dict[str, Any]:
    return asdict(summary)
