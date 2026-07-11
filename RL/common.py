"""Shared training harness for the per-family baseline starters.

Each RL/<family_id>/train.py calls train_family(): a small PPO baseline is
trained on real generator tasks, exported to ONNX, packaged into a
validator-ready model_graph.zip, and smoke-tested against the family's
graph contract.

The policy sees a downsampled depth frame plus the full state vector; rgb is
never requested. Multi-drone families train one shared policy by exposing each
drone as one slot of a vectorized environment over a single shared simulation.
"""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from swarm.constants import SIM_DT, SWARM_MAX_DRONES, SWARM_MIN_DRONES
from swarm.domain_model import get_policy_interface_contract
from swarm.policy_interface import resolve_policy_interface_version, smoke_test_policy_package
from swarm.utils.env_factory import make_env_with_initial_obs
from swarm.validator.task_gen import random_task

from export_onnx import POLICY_DEPTH_SIZE, export_policy
DEFAULT_TIMESTEPS = 50_000
DEFAULT_SWARM_DRONES = 4
_MAX_TASK_RESAMPLES = 200


def _policy_view(depth: np.ndarray, state: np.ndarray) -> dict[str, np.ndarray]:
    step = max(1, depth.shape[0] // POLICY_DEPTH_SIZE)
    return {
        "depth": np.ascontiguousarray(depth[::step, ::step, :], dtype=np.float32),
        "state": np.asarray(state, dtype=np.float32),
    }


class FamilyVecEnv(VecEnv):
    """One shared simulation exposed as NUM_DRONES single-drone slots.

    Every episode samples a fresh task from the real generator, so the policy
    trains across maps and seeds instead of memorizing one layout.
    """

    def __init__(self, family_id: str, *, seed: int, n_drones: int | None = None):
        self._family_id = family_id
        self._rng = random.Random(seed)
        self._forced_drones = n_drones
        self._env, first_obs = self._build_episode()

        state_dim = int(self._env.observation_space["state"].shape[-1])
        observation_space = spaces.Dict(
            {
                "depth": spaces.Box(
                    0.0, 1.0, (POLICY_DEPTH_SIZE, POLICY_DEPTH_SIZE, 1), np.float32
                ),
                "state": spaces.Box(-np.inf, np.inf, (state_dim,), np.float32),
            }
        )
        contract = get_policy_interface_contract(
            family_id, resolve_policy_interface_version(family_id, None)
        )
        action_space = spaces.Box(
            low=np.asarray(contract["action_space"]["lower_bound"], dtype=np.float32),
            high=np.asarray(contract["action_space"]["upper_bound"], dtype=np.float32),
            dtype=np.float32,
        )
        super().__init__(self._env.NUM_DRONES, observation_space, action_space)
        self._last_obs = first_obs
        self._pending_actions: np.ndarray | None = None

    def _build_episode(self):
        for _ in range(_MAX_TASK_RESAMPLES):
            task = random_task(
                sim_dt=SIM_DT,
                seed=self._rng.randrange(1, 2**31),
                family_id=self._family_id,
            )
            if self._forced_drones is None:
                break
            if int(getattr(task, "num_drones", 1) or 1) == self._forced_drones:
                break
        else:
            raise RuntimeError(
                f"Could not sample a {self._forced_drones}-drone task for {self._family_id}"
            )
        return make_env_with_initial_obs(task)

    def _slot_obs(self, obs: dict[str, np.ndarray], slot: int) -> dict[str, np.ndarray]:
        if self.num_envs == 1:
            return _policy_view(obs["depth"], obs["state"])
        return _policy_view(obs["depth"][slot], obs["state"][slot])

    def _stack_obs(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        slots = [self._slot_obs(obs, i) for i in range(self.num_envs)]
        return {
            key: np.stack([slot[key] for slot in slots])
            for key in ("depth", "state")
        }

    def reset(self):
        return self._stack_obs(self._last_obs)

    def step_async(self, actions: np.ndarray) -> None:
        self._pending_actions = np.asarray(actions, dtype=np.float32)

    def step_wait(self):
        env_action = self._pending_actions
        if self.num_envs == 1:
            env_action = env_action.reshape(1, -1)
        obs, reward, terminated, truncated, _info = self._env.step(env_action)
        done = bool(terminated or truncated)

        rewards = np.full(self.num_envs, float(reward), dtype=np.float32)
        dones = np.full(self.num_envs, done, dtype=bool)
        infos = [{} for _ in range(self.num_envs)]

        if done:
            terminal = self._stack_obs(obs)
            for slot in range(self.num_envs):
                infos[slot]["terminal_observation"] = {
                    key: terminal[key][slot] for key in terminal
                }
                infos[slot]["TimeLimit.truncated"] = bool(truncated and not terminated)
            self._env.close()
            self._env, self._last_obs = self._build_episode()
        else:
            self._last_obs = obs

        return self._stack_obs(self._last_obs), rewards, dones, infos

    def close(self) -> None:
        self._env.close()

    def get_attr(self, attr_name, indices=None):
        return [getattr(self._env, attr_name)] * self.num_envs

    def set_attr(self, attr_name, value, indices=None) -> None:
        setattr(self._env, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [getattr(self._env, method_name)(*args, **kwargs)] * self.num_envs

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def seed(self, seed=None):
        if seed is not None:
            self._rng = random.Random(seed)
        return [seed] * self.num_envs


def _package_submission(model: PPO, family_id: str, out_dir: Path) -> Path:
    source_dir = out_dir / "graph"
    if source_dir.exists():
        shutil.rmtree(source_dir)
    export_policy(model, family_id, source_dir)

    artifact_zip = out_dir / "model_graph.zip"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "swarm.cli",
            "model",
            "package",
            "--source",
            str(source_dir),
            "--family-id",
            family_id,
            "--output",
            str(artifact_zip),
            "--overwrite",
        ],
        check=True,
    )
    return artifact_zip


def train_family(family_id: str, *, supports_drone_count: bool = False) -> None:
    parser = argparse.ArgumentParser(
        description=f"Train a baseline PPO model for {family_id} and package it."
    )
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=1)
    if supports_drone_count:
        parser.add_argument(
            "--drones",
            type=int,
            default=DEFAULT_SWARM_DRONES,
            help="Fixed drone count during training (evaluation still varies 2-8).",
        )
    args = parser.parse_args()

    drones = getattr(args, "drones", None)
    if drones is not None and not SWARM_MIN_DRONES <= drones <= SWARM_MAX_DRONES:
        parser.error(f"--drones must be between {SWARM_MIN_DRONES} and {SWARM_MAX_DRONES}")

    out_dir = Path(__file__).resolve().parent / family_id / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = FamilyVecEnv(family_id, seed=args.seed, n_drones=drones)
    model = PPO("MultiInputPolicy", env, n_steps=512, seed=args.seed, verbose=1)
    model.learn(total_timesteps=args.timesteps)

    model.save(str(out_dir / "ppo_policy.zip"))
    env.close()

    artifact_zip = _package_submission(model, family_id, out_dir)
    smoke_ok, smoke_reason = smoke_test_policy_package(artifact_zip)
    if not smoke_ok:
        raise RuntimeError(f"Packaged submission failed the contract smoke test: {smoke_reason}")

    print(f"\nSubmission ready: {artifact_zip}")
    print("Test it like a validator:")
    print(f"  python3 RL/test_RL.py --model {artifact_zip} --family_id {family_id}")
