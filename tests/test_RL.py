#!/usr/bin/env python
"""
tests/test_RL.py ────────────────────────────────────────────────────────────
Minimal PPO pipeline for Swarm‑validator tasks **with step‑level debugging**.

Usage
-----

Normal run (headless):
    $ python -m tests.test_RL

Render the first episode in the PyBullet GUI:
    $ python -m tests.test_RL --gui

Add very verbose prints for the first episode:
    $ python -m tests.test_RL --debug
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import List

import numpy as np

try:
    from loguru import logger
except ImportError:  # pragma: no cover
    import logging as logger

    logger.basicConfig(level=logging.INFO)

# ── Swarm imports ──────────────────────────────────────────────────────────
from swarm.validator.task_gen import random_task
from swarm.validator.reward import flight_reward
from swarm.utils.env_factory import make_env
from swarm.protocol import MapTask, ValidationResult
from swarm.constants import SIM_DT, HORIZON_SEC

# ── RL toolkit ─────────────────────────────────────────────────────────────
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from gymnasium import Env
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "stable‑baselines3 not found – install with "
        "`pip install 'stable-baselines3[extra]'`"
    ) from e

# ── Optional: keep PyBullet from seg‑faulting on exit ──────────────────────
try:
    import atexit, pybullet as p

    atexit.register(lambda: p.disconnect())
except Exception:  # import failure on headless systems is fine
    pass


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────
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


# ────────────────────────────────────────────────────────────────────────────
# 1.  Gymnasium wrapper around a single MapTask
# ────────────────────────────────────────────────────────────────────────────
class SwarmTaskEnv(Env):
    """
    Light wrapper that turns `make_env` into a Gymnasium‑compliant env:

        reset -> (obs, info)
        step  -> (obs, reward, terminated, truncated, info)
    """

    metadata = {"render_modes": []}

    def __init__(self, gui: bool = False, debug: bool = False):
        super().__init__()
        self.gui = gui
        self.debug = debug
        self._build_task()

    # Gymnasium API -------------------------------------------------------
    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            np.random.seed(seed)

        if hasattr(self, "env"):
            self.env.close()

        self._build_task()
        obs = self.env.reset()
        obs, info = (obs, {}) if not isinstance(obs, tuple) else obs
        return obs, info

    def step(self, action):
        """Per-step reward shaping.

        •  tiny *alive* bonus – staying in the air beats crashing  
        •  *progress* reward – proportional to how much we reduce goal distance  
        •  quadratic *energy* penalty – spinning fast costs more  
        •  large terminal bonus/penalty on (success / failure)
        """
        action = _match_shape(action, self.env.action_space.shape)
        out = self.env.step(action)

        # unpack Gymnasium vs legacy Gym output --------------------------
        if len(out) == 5:
            obs, _, terminated, truncated, info = out
        else:
            obs, _, done, info = out
            terminated, truncated = done, False

        # ── ① alive bonus ───────────────────────────────────────────────
        alive_bonus = 0.05            # 0.05 per 40 ms → ~1 pt for full 20 s

        # ── ② progress toward goal ─────────────────────────────────────
        prog_reward = 0.0
        dist_now = info.get("goal_dist")
        if dist_now is not None and self.last_goal_dist is not None:
            prog_reward = 0.4 * (self.last_goal_dist - dist_now)
        self.last_goal_dist = dist_now

        # ── ③ energy penalty ───────────────────────────────────────────
        rpm_vec = action.astype(float).flatten()
        energy_penalty = (rpm_vec ** 2).sum() * SIM_DT * 2e-10

        # ── ④ terminal bonus / penalty ─────────────────────────────────
        term_bonus = 0.0
        if terminated and info.get("success", False):
            term_bonus = 15.0          # big positive spike on success
        elif terminated:               # crash / OOB / timeout
            term_bonus = -7.0          # discourage reckless failures

        reward = alive_bonus + prog_reward + term_bonus - energy_penalty

        # optional debug print ------------------------------------------
        
        logger.debug(
                f"[{self.sim_time:6.2f}s] r={reward:+.3f} "
                f"(alive {alive_bonus:+.2f}  prog {prog_reward:+.2f}  "
                f"term {term_bonus:+.1f}  energy {-energy_penalty:+.2f})"
            )

        self.sim_time += SIM_DT
        return obs, float(reward), terminated, truncated, info


    def render(self):  # GUI is handled by the wrapped env
        pass

    def close(self):
        if hasattr(self, "env"):
            self.env.close()

    # Internals -----------------------------------------------------------
    def _build_task(self):
        self.task: MapTask = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC)
        # NOTE raw_rpm=False  → action_space is Box(-1, +1)
        self.env = make_env(self.task, gui=self.gui, raw_rpm=False)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.sim_time = 0.0


# ────────────────────────────────────────────────────────────────────────────
# 2.  Pilot wrapper (SB3 PPO → Swarm validator)
# ────────────────────────────────────────────────────────────────────────────
class PilotRL:
    def __init__(self, model: PPO):
        self.model = model

    def reset(self, task: MapTask):  # noqa: D401
        pass  # policy is task‑agnostic

    def act(self, obs: np.ndarray, t: float) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=True)
        return np.squeeze(action).astype(float)


# ────────────────────────────────────────────────────────────────────────────
# 3.  Training
# ────────────────────────────────────────────────────────────────────────────
def train_policy(total_steps: int = 20_000) -> PPO:
    logger.info(f"Training PPO for {total_steps:,} steps …")
    env = DummyVecEnv([lambda: SwarmTaskEnv(gui=False)])
    model = PPO(
        policy="MlpPolicy",
        env=env,
        device="cpu",
        verbose=0,
        n_steps=256,
        batch_size=64,
        gamma=0.99,
        learning_rate=3e-4,
    )
    model.learn(total_timesteps=total_steps, progress_bar=False)
    logger.info("Training finished.")
    return model


# ────────────────────────────────────────────────────────────────────────────
# 4.  Evaluation helper (mirrors validator logic)
# ────────────────────────────────────────────────────────────────────────────
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
    step = 0
    while t_sim < task.horizon:
        rpm = _match_shape(pilot.act(obs, t_sim), env.action_space.shape)
        out = env.step(rpm)

        if len(out) == 5:
            obs, _, terminated, truncated, info = out
            done = terminated or truncated
        else:
            obs, _, done, info = out

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
            break

    env.close()
    score = flight_reward(success, t_sim, energy, task.horizon)
    return ValidationResult(-1, success, t_sim, energy, score)


# ────────────────────────────────────────────────────────────────────────────
# 5.  Demo / CLI
# ────────────────────────────────────────────────────────────────────────────
def demo(sim_gui: bool, train_steps: int, debug: bool):
    start = time.time()
    model = train_policy(train_steps)
    pilot = PilotRL(model)
    logger.info(f"Training took {time.time() - start:.1f} s")

    # First episode ----------------------------------------------
    task0 = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC)
    results: List[ValidationResult] = [
        run_episode(task0, pilot, gui=sim_gui, debug=debug)
    ]

    # Remaining 99 headless episodes ------------------------------
    for _ in range(99):
        results.append(
            run_episode(
                random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC), pilot, debug=False
            )
        )

    # Stats -------------------------------------------------------
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


# ── CLI ────────────────────────────────────────────────────────────────────
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
