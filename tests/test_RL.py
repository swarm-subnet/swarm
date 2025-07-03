#!/usr/bin/env python
"""
tests/test_RL.py ───────────────────────────────────────────────────────────
Minimal PPO pipeline for Swarm‑validator tasks **with step‑level debugging**

Key changes
-----------
✓ Reward shaping: alive bonus + progress per step + big terminal bonus  
✓ Fixed `last_goal_dist` initialisation  
✓ Evaluation uses the new `flight_reward` signature
"""
from __future__ import annotations

import argparse
import math
import statistics
import time
from typing import List

import numpy as np
from loguru import logger

# ── Swarm imports ─────────────────────────────────────────────────────────
from swarm.validator.task_gen import random_task
from swarm.validator.reward import flight_reward           # ← NEW signature
from swarm.utils.env_factory import make_env
from swarm.protocol import MapTask, ValidationResult
from swarm.constants import SIM_DT, HORIZON_SEC

# ── RL toolkit ────────────────────────────────────────────────────────────
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from gymnasium import Env

# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────
def _match_shape(a: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Return *a* reshaped or squeezed so that *.shape == shape*."""
    a = np.asarray(a, dtype=float)
    if a.shape == shape:
        return a
    if len(a.shape) == len(shape) + 1 and a.shape[0] == 1:
        return a.reshape(shape)
    if len(a.shape) == len(shape) - 1 and shape[0] == 1:
        return a.reshape(shape)
    return a.reshape(shape)


# ──────────────────────────────────────────────────────────────────────────
# 1.  Gymnasium wrapper around a single MapTask
# ──────────────────────────────────────────────────────────────────────────
class SwarmTaskEnv(Env):
    """Light wrapper that turns `make_env` into a Gymnasium‑compliant env."""
    metadata = {"render_modes": []}

    # ------------------------------------------------------------------
    # Construction / reset
    # ------------------------------------------------------------------
    def __init__(self, gui: bool = False, debug: bool = False):
        super().__init__()
        self.gui = gui
        self.debug = debug
        self._build_task()

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            np.random.seed(seed)

        if hasattr(self, "env"):
            self.env.close()

        self._build_task()
        obs = self.env.reset()
        obs, info = (obs, {}) if not isinstance(obs, tuple) else obs
        # Keep the very first distance for progress rewards
        self.last_goal_dist: float | None = info.get("goal_dist")
        return obs, info

    # ------------------------------------------------------------------
    # Step – per‑step reward shaping
    # ------------------------------------------------------------------
    def step(self, action):
        """
        Reward =   alive_bonus
                 + progress_bonus
                 + terminal_bonus  (at the very end)
        """
        action = _match_shape(action, self.env.action_space.shape)
        obs, _, terminated, truncated, info = self._gym_step(action)

        # ① alive bonus – survive every tick
        alive_bonus = 0.15                       # ~15 pts over 30 s

        # ② progress bonus – reward reduction in goal distance
        dist_now = info.get("goal_dist")
        progress_bonus = 0.0
        if dist_now is not None and self.last_goal_dist is not None:
            progress_bonus = 1 * (self.last_goal_dist - dist_now)
        self.last_goal_dist = dist_now

        # ③ cheap energy penalty (optional – keeps actions bounded)
        rpm_vec = action.astype(float).flatten()
        energy_penalty = 1e-11 * (rpm_vec ** 2).sum() * SIM_DT

        # ④ terminal bonus / penalty
        terminal_bonus = 0.0
        if terminated and info.get("success", False):
            terminal_bonus = 20.0
        elif terminated:
            terminal_bonus = -8.0

        reward = (
            alive_bonus +
            progress_bonus +
            terminal_bonus -
            energy_penalty
        )

        # Debug print
        # logger.debug(
        #     f"[{self.sim_time:6.2f}s] "
        #     f"r={reward:+.3f}  alive={alive_bonus:+.2f}  "
        #     f"prog={progress_bonus:+.2f}  term={terminal_bonus:+.1f}  "
        #     f"energy={-energy_penalty:+.2f}"
        # )

        self.sim_time += SIM_DT
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Boilerplate
    # ------------------------------------------------------------------
    def render(self):  # GUI is handled by the wrapped env
        pass

    def close(self):
        if hasattr(self, "env"):
            self.env.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _build_task(self):
        self.task: MapTask = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC)
        self.env = make_env(self.task, gui=self.gui, raw_rpm=False)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.sim_time = 0.0

    def _gym_step(self, action):
        """Handles (Gym‑v26 vs Gymnasium) return‑value quirks."""
        out = self.env.step(action)
        if len(out) == 5:                 # Gymnasium
            return out
        obs, _, done, info = out          # old Gym
        terminated, truncated = done, False
        return obs, None, terminated, truncated, info


# ──────────────────────────────────────────────────────────────────────────
# 2.  Pilot wrapper (SB3 PPO → Swarm validator)
# ──────────────────────────────────────────────────────────────────────────
class PilotRL:
    def __init__(self, model: PPO):
        self.model = model

    def reset(self, task: MapTask):       # noqa: D401
        pass  # policy is task‑agnostic

    def act(self, obs: np.ndarray, t: float) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=False)
        return np.squeeze(action).astype(float)


# ──────────────────────────────────────────────────────────────────────────
# 3.  Training
# ──────────────────────────────────────────────────────────────────────────
def train_policy(total_steps: int = 20_000) -> PPO:
    logger.info(f"Training PPO for {total_steps:,} steps …")
    N_ENVS = 8
    env = SubprocVecEnv([lambda: SwarmTaskEnv(gui=False) for _ in range(N_ENVS)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=2048 // N_ENVS,    # keep rollout length ≈ 2 k
        batch_size=64,
        gamma=0.995,               # a bit longer horizon helps with sparse reward
        learning_rate=1e-4,
        ent_coef=0.05,             # encourage exploration
        verbose=0,
        device="cpu",
        policy_kwargs=dict(net_arch=[dict(pi=[256,128], vf=[256,128])])
    )
    model.learn(total_timesteps=total_steps, progress_bar=True)
    logger.info("Training finished.")
    return model


# ──────────────────────────────────────────────────────────────────────────
# 4.  Evaluation helper (mirrors validator logic)
# ──────────────────────────────────────────────────────────────────────────
def _euclid(a, b):
    return math.dist(a, b)

def run_episode(
    task: MapTask,
    pilot: PilotRL,
    *,
    gui: bool = False,
    debug: bool = False,
) -> ValidationResult:
    """Run one episode on the *raw* low‑level env (Gym or Gymnasium)."""
    env = make_env(task, gui=gui, raw_rpm=False)

    obs_t = env.reset()
    obs = obs_t[0] if isinstance(obs_t, tuple) else obs_t

    pilot.reset(task)

    t_sim, energy, success = 0.0, 0.0, False
    step, last_goal_dist = 0, _euclid(task.start, task.goal)

    while t_sim < task.horizon:
        rpm = _match_shape(pilot.act(obs, t_sim), env.action_space.shape)
        obs, _, terminated, truncated, info = _unpack_step(env.step(rpm))
        done = terminated or truncated

        if debug:
            logger.debug(
                f"[STEP {step:03d}] t={t_sim:6.2f} | rpm={rpm} | done={done} "
                f"| info={info}"
            )
        step += 1

        energy += np.abs(rpm).sum() * SIM_DT
        t_sim += SIM_DT
        if done:
            success = info.get("success", False)
            last_goal_dist = info.get("goal_dist", last_goal_dist)
            break

    env.close()
    score = flight_reward(
        success=success,
        t_alive=t_sim,
        d_start=_euclid(task.start, task.goal),
        d_final=last_goal_dist,
        horizon=task.horizon,
    )
    return ValidationResult(-1, success, t_sim, energy, score)


def _unpack_step(out):
    """Handle both Gymnasium and old Gym step signatures."""
    if len(out) == 5:
        return out
    obs, _, done, info = out
    return obs, None, done, False, info


# ──────────────────────────────────────────────────────────────────────────
# 5.  Demo / CLI
# ──────────────────────────────────────────────────────────────────────────
def demo(sim_gui: bool, train_steps: int, debug: bool):
    start = time.time()
    model = train_policy(train_steps)
    pilot = PilotRL(model)
    logger.info(f"Training took {time.time() - start:.1f} s")

    # First episode (optionally GUI/debug) --------------------------
    task0 = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC)
    results: List[ValidationResult] = [
        run_episode(task0, pilot, gui=sim_gui, debug=debug)
    ]

    # Remaining 99 headless episodes -------------------------------
    for _ in range(99):
        results.append(
            run_episode(
                random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC), pilot, debug=False
            )
        )

    # Stats ---------------------------------------------------------
    successes = sum(r.success for r in results)
    times = [r.time_sec for r in results]
    energies = [r.energy for r in results]
    scores = [r.score for r in results]

    logger.info("\n═══════════ PPO Pilot Stats (100 evals) ═══════════")
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
    logger.info("════════════════════════════════════════════════════\n")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a PPO pilot and validate it on 100 random tasks."
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Render the FIRST evaluation episode in a PyBullet viewer",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10_000,
        help="Total PPO timesteps for training (default: 10 000)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print every observation/action/reward for the FIRST episode",
    )
    args = parser.parse_args()

    demo(sim_gui=args.gui, train_steps=args.steps, debug=args.debug)
