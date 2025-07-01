#!/usr/bin/env python
"""
test_RL.py ────────────────────────────────────────────────────────────────
Quick sanity test for the new *policy*‑based pipeline.

• Trains a tiny PPO agent on randomly generated `MapTask`s.
• Wraps the trained `stable_baselines3` model in the *Pilot* API.
• Replays 100 evaluation episodes with the same code‑path the validator uses.
• Prints per‑episode telemetry and aggregate stats.
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    from loguru import logger
except ImportError:  # pragma: no cover
    import logging as logger

    logger.basicConfig(level=logger.INFO)

# ---------------- Swarm imports ------------------------------------------
from swarm.validator.task_gen import random_task
from swarm.validator.reward import flight_reward
from swarm.utils.env_factory import make_env
from swarm.protocol import MapTask, ValidationResult

# Simulation constants reused from validator
from swarm.constants import SIM_DT, HORIZON_SEC

# ---------------- RL toolkit ---------------------------------------------
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from gymnasium import Env
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Stable‑Baselines3 not found. Install with:\n"
        "   pip install 'stable-baselines3[extra]'"
    ) from e


# =========================================================================
# 1.  Environment wrapper (one MapTask per reset)
# =========================================================================
class SwarmTaskEnv(Env):
    """
    A lightweight Gymnasium wrapper that, on every `reset()`, builds a fresh
    `MapTask` and underlying Bullet world via `make_env`.

    Observation space and action space are inherited from the low‑level env.

    Reward shaping (simple):
        +1   every simulation step *after* the goal is reached (hover bonus)
        -1e‑4 × energy  per step  (penalise wasted RPM)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, gui: bool = False):
        super().__init__()
        self.gui = gui
        self._build_task()

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            np.random.seed(seed)
        if hasattr(self, "env"):  # close previous world
            self.env.close()

        self._build_task()
        obs = self.env.reset()
        info = {"task": self.task}
        return obs, info

    def step(self, action):
        obs, _, done, info = self.env.step(action)
        # --- simple shaped reward -------------------------------------
        rpm_vec = np.asarray(action, dtype=float).flatten()
        energy_penalty = (np.square(rpm_vec).sum()) * SIM_DT * 1e-4
        reward = 1.0 - energy_penalty
        if done and info.get("success", False):
            reward += 10.0  # goal bonus
        return obs, float(reward), done, False, info

    def render(self):
        # Everything rendered by PyBullet already
        pass

    def close(self):
        if hasattr(self, "env"):
            self.env.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _build_task(self):
        self.task: MapTask = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC)
        # raw_rpm=True ➜ action = np.ndarray(4,)
        self.env = make_env(self.task, gui=self.gui, raw_rpm=True, randomise=True)
        # Expose gym‑style spaces
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space


# =========================================================================
# 2.  Pilot wrapper expected by validators
# =========================================================================
class PilotRL:
    """Adapter around a `stable_baselines3` policy."""

    def __init__(self, model: PPO):
        self.model = model

    def reset(self, task: MapTask):  # noqa: D401
        # Policy here is task‑agnostic; nothing to do.
        pass

    def act(self, obs: np.ndarray, t: float) -> np.ndarray:  # noqa: D401
        action, _ = self.model.predict(obs, deterministic=True)
        return action.astype(float)


# =========================================================================
# 3.  Cheap training routine
# =========================================================================
def train_policy(total_steps: int = 10_000, gui: bool = False) -> PPO:
    """Train a tiny PPO agent; keep CPU time reasonable."""
    logger.info(f"Training PPO for {total_steps} steps …")
    env = DummyVecEnv([lambda: SwarmTaskEnv(gui=gui)])
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=0,
        n_steps=256,
        batch_size=64,
        gamma=0.99,
        learning_rate=3e-4,
    )
    model.learn(total_timesteps=total_steps, progress_bar=False)
    logger.info("Training finished.")
    return model


# =========================================================================
# 4.  Validation helpers (mirrors validator logic)
# =========================================================================
def run_episode(task: MapTask, pilot: PilotRL, gui: bool = False) -> ValidationResult:
    env = make_env(task, gui=gui, raw_rpm=True, randomise=True)
    obs = env.reset()
    pilot.reset(task)

    t_sim, energy, success = 0.0, 0.0, False
    while t_sim < task.horizon:
        rpm = pilot.act(obs, t_sim)
        obs, _, done, info = env.step(rpm[None, :])
        energy += np.abs(rpm).sum() * SIM_DT
        t_sim += SIM_DT
        if done:
            success = info.get("success", False)
            break

    env.close()
    score = flight_reward(success, t_sim, energy, task.horizon)
    return ValidationResult(-1, success, t_sim, energy, score)


def make_task() -> MapTask:
    return random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC)


# =========================================================================
# 5.  Entry‑points
# =========================================================================
def demo(sim_gui: bool = False, train_steps: int = 10_000):
    """
    Train + run 100 evaluation episodes; print stats.

    Parameters
    ----------
    sim_gui
        Whether to open a PyBullet viewer **for the first eval episode only**.
    train_steps
        PPO timesteps; raise for better scores (and longer runtimes).
    """
    t0 = time.time()
    model = train_policy(train_steps, gui=False)
    pilot = PilotRL(model)
    logger.info(f"Training took {time.time() - t0:.1f} s")

    results: List[ValidationResult] = []

    # -- first episode with optional GUI --------------------------------
    task0 = make_task()
    res0 = run_episode(task0, pilot, gui=sim_gui)
    results.append(res0)
    logger.info(
        f"[Episode 1] success={res0.success}  time={res0.time_sec:.2f}  "
        f"energy={res0.energy:.2f}  score={res0.score:.3f}"
    )

    # -- remaining 99 headless episodes ---------------------------------
    for i in range(2, 101):
        res = run_episode(make_task(), pilot, gui=False)
        results.append(res)
        logger.info(
            f"[Episode {i:3d}] success={res.success}  "
            f"time={res.time_sec:.2f}  energy={res.energy:.2f}  "
            f"score={res.score:.3f}"
        )

    # -- aggregate stats -------------------------------------------------
    successes = sum(r.success for r in results)
    times = [r.time_sec for r in results]
    energies = [r.energy for r in results]
    scores = [r.score for r in results]

    logger.info("\n═══════════ PPO Pilot Statistics (100 evals) ═══════════")
    logger.info(f"Success rate : {successes}/100  =  {successes/100:.1%}")
    logger.info(
        f"Time (s)     : mean={statistics.mean(times):6.2f}, "
        f"min={min(times):6.2f}, max={max(times):6.2f}"
    )
    logger.info(
        f"Energy       : mean={statistics.mean(energies):6.2f}, "
        f"min={min(energies):6.2f}, max={max(energies):6.2f}"
    )
    logger.info(
        f"Score        : mean={statistics.mean(scores):6.3f}, "
        f"min={min(scores):6.3f}, max={max(scores):6.3f}"
    )
    logger.info("════════════════════════════════════════════════════════\n")
    return results


# ---------------- pytest style smoke‑test ---------------------------------
def test_rl_roundtrip():  # pragma: no cover
    """FastCI check – trains 2 k steps, evaluates 5 tasks."""
    model = train_policy(total_steps=2_000, gui=False)
    pilot = PilotRL(model)
    for _ in range(5):
        res = run_episode(make_task(), pilot, gui=False)
        assert res.score >= 0.0


# ---------------- CLI -----------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Train a PPO pilot and validate it on 100 random tasks."
    )
    ap.add_argument(
        "--gui",
        action="store_true",
        help="Render the FIRST evaluation episode in a 3‑D PyBullet viewer",
    )
    ap.add_argument(
        "--steps",
        type=int,
        default=10_000,
        help="Total PPO timesteps for training (default: 10 000)",
    )
    args = ap.parse_args()

    demo(sim_gui=args.gui, train_steps=args.steps)
