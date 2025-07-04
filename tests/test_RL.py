#!/usr/bin/env python3
"""
ppo_demo_debug.py  –  Same task, but with exhaustive logging hooks
────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import argparse, logging, csv, time, os
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from rich.logging import RichHandler
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from tensorboardX import SummaryWriter

# ── project imports (unchanged) ─────────────────────────────────────
from swarm.utils.env_factory import make_env
from swarm.validator.task_gen import random_task
from swarm.validator.forward import SIM_DT, HORIZON_SEC
from swarm.constants import GOAL_TOL, HOVER_SEC
from swarm.validator.reward import flight_reward

# ═══════════════════════════════════════════════════════════════════
# 0.  Global logger setup
# ═══════════════════════════════════════════════════════════════════
LOG_LEVEL = logging.DEBUG               # DEBUG | INFO | WARNING | ...
LOG_FMT   = "%(message)s"
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FMT,
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
log = logging.getLogger("ppo_debug")

# ═══════════════════════════════════════════════════════════════════
# 1.  Debug wrappers
# ═══════════════════════════════════════════════════════════════════
class StepLoggerWrapper(gym.Wrapper):
    """
    Dumps obs, action, reward, info every step.  Disable at runtime by
    setting self.enabled = False (useful after N steps).
    """
    def __init__(self, env: gym.Env,
                 csv_path: Path | None = None,
                 print_every: int = 1):
        super().__init__(env)
        self.enabled      = True
        self.print_every  = print_every
        self._step_count  = 0

        # optional CSV for post‑mortem analysis
        self._csv_file = None
        self._csv_writer: csv.writer | None = None
        if csv_path is not None:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file   = csv_path.open("w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(
                ["episode", "step", "terminated",
                 *[f"obs_{i}" for i in range(np.prod(env.observation_space.shape))],
                 *[f"act_{i}" for i in range(np.prod(env.action_space.shape))],
                 "reward", "score", "success"]
            )

        self._episode = 0

    # ----------------------------------------------------------------
    def reset(self, **kwargs):
        if self._step_count:    # just finished an episode
            self._episode += 1
            self._step_count = 0
        obs, info = self.env.reset(**kwargs)
        if self.enabled:
            log.debug(f"[E{self._episode:04d}] reset ⇒ obs={obs}")
        return obs, info

    # ----------------------------------------------------------------
    def step(self, action):
        self._step_count += 1
        obs, reward, term, trunc, info = self.env.step(action)

        if self.enabled and self._step_count % self.print_every == 0:
            log.debug(
                f"[E{self._episode:04d} | t={self._step_count:04d}] "
                f"act={np.asarray(action).round(3)}  "
                f"r={reward:+.3f}  "
                f"score={info.get('score', np.nan):.3f}  "
                f"succ={info.get('success', False)}"
            )
        if hasattr(self.env.unwrapped._env, "last_rpm"):
            log.debug(f"rpm={self.env.unwrapped._env.last_rpm[0].round(0)}")

        if self._csv_writer is not None:
            row = [
                self._episode, self._step_count, int(term or trunc),
                *np.asarray(obs).flatten().tolist(),
                *np.asarray(action).flatten().tolist(),
                float(reward), info.get("score", np.nan),
                int(info.get("success", False))
            ]
            self._csv_writer.writerow(row)

        return obs, reward, term, trunc, info

    # ----------------------------------------------------------------
    def close(self):
        super().close()
        if self._csv_file is not None:
            self._csv_file.close()


# ═══════════════════════════════════════════════════════════════════
# 2.  Stable‑Baselines3 callback for *network‑side* debugging
# ═══════════════════════════════════════════════════════════════════
class TensorDebugCallback(BaseCallback):
    """
    Adds TensorBoard scalars & histograms:
        • raw observations (first env)
        • actions chosen
        • rewards
        • policy entropy & value loss
        • gradient norms every optimisation epoch
    """

    def __init__(self, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.tb = SummaryWriter(log_dir)
        self._grad_step = 0

    # called at every environment step *before* the action is applied
    def _on_step(self) -> bool:
        # observations & rewards are VecEnv‑shaped (n_env, obs_dim)
        obs      = self.locals["new_obs"][0]      # first env
        reward   = self.locals["rewards"][0]
        action   = self.locals["actions"][0]

        step     = self.num_timesteps
        self.tb.add_histogram("obs/raw", obs, step)
        self.tb.add_histogram("action/raw", action, step)
        self.tb.add_scalar("reward/step", reward, step)
        return True

    # called after every optimisation epoch (when gradients are available)
    def _on_rollout_end(self) -> None:
        model: PPO = self.model                # type: ignore
        step = self.num_timesteps

        # policy loss, value loss, entropy (already tracked by SB3’s own logger)
        for key in ("policy_loss", "value_loss", "entropy"):
            if key in model.logger.name_to_value:
                self.tb.add_scalar(f"loss/{key}",
                                   model.logger.name_to_value[key][-1], step)

        # gradient norms per parameter group
        total_norm = 0.0
        for p in model.policy.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
        total_norm **= 0.5
        self.tb.add_scalar("grad/global_norm", total_norm, step)
        self._grad_step += 1


# ═══════════════════════════════════════════════════════════════════
# 3.  Vec‑env factory with our new wrapper
# ═══════════════════════════════════════════════════════════════════
def make_vec_env(task,
                 gui: bool = False,
                 csv_log: bool = False,
                 disable_after: int | None = None):
    """
    If `csv_log` is True, dump *every* step to runs/debug_steps.csv.
    If `disable_after` (steps) is set, step logging stops afterwards
    to avoid giant files once you’ve seen enough.
    """
    csv_path = Path("runs/debug_steps.csv") if csv_log else None

    def _factory():
        env = RLTaskEnv(task, gui=gui)
        env = Monitor(env, info_keywords=("score", "success"))

        step_logger = StepLoggerWrapper(env,
                                        csv_path=csv_path,
                                        print_every=1)
        if disable_after is not None:
            # monkey‑patch to auto‑disable verbose log
            def _auto_disable_step(action, _orig=step_logger.step):
                if step_logger._step_count >= disable_after:
                    step_logger.enabled = False
                return _orig(action)
            step_logger.step = _auto_disable_step  # type: ignore
        return step_logger

    venv = DummyVecEnv([_factory])
    venv = VecNormalize(venv, norm_obs=True,
                        norm_reward=False,
                        gamma=0.99,
                        training=True)
    return venv


# ═══════════════════════════════════════════════════════════════════
# 4.  Training & evaluation – unchanged, but pass the callback
# ═══════════════════════════════════════════════════════════════════
def train_ppo(vec_env, timesteps: int):
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=1e-4,
        n_steps=2048,
        gamma=0.95,
        gae_lambda=0.95,
        batch_size=128,
        n_epochs=10,
        verbose=1,
        device="auto",
        tensorboard_log="runs/ppo_fly2goal",
    )
    tb_cb = TensorDebugCallback(log_dir="runs/tensor_debug")
    model.learn(total_timesteps=timesteps,
                progress_bar=True,
                callback=tb_cb)
    return model


# ═══════════════════════════════════════════════════════════════════
# 5.  RLTaskEnv (identical to yours – copied for completeness)
# ═══════════════════════════════════════════════════════════════════
class RLTaskEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, task, gui: bool = False):
        super().__init__()
        self.task = task
        self._env = make_env(task, gui=gui, raw_rpm=False)
        self.action_space       = self._env.action_space
        self.observation_space  = self._env.observation_space

        self._time        = 0.0
        self._hover_sec   = 0.0
        self._d_start     = 1.0
        self._prev_score  = 0.0
        self._success     = False

    def reset(self, *, seed=None, options=None):
        obs, *_ = self._env.reset(seed=seed)
        self._time = self._hover_sec = 0.0
        self._success = False
        pos_start = obs[0, :3]
        self._d_start = float(np.linalg.norm(pos_start - self.task.goal))
        if self._d_start <= 0.0:
            self._d_start = 1e-9
        self._prev_score = flight_reward(False, 0.0,
                                         self._d_start, self._d_start,
                                         self.task.horizon)
        return obs, {}

    def step(self, action):
        base = self._env.step(action)
        if len(base) == 5:
            obs, _, terminated_env, truncated_env, info = base
        else:
            obs, _, terminated_env, info = base
            truncated_env = False

        self._time += self.task.sim_dt
        pos  = obs[0, :3]
        dist = float(np.linalg.norm(pos - self.task.goal))
        reached = dist < GOAL_TOL
        if reached:
            self._hover_sec += self.task.sim_dt
            if self._hover_sec >= HOVER_SEC:
                self._success = True
        else:
            self._hover_sec = 0.0
        score = flight_reward(self._success, self._time,
                              self._d_start, dist, self.task.horizon)
        shaped_reward = score - self._prev_score
        self._prev_score = score

        terminated = truncated = False
        if terminated_env:
            terminated = True
        elif self._time >= self.task.horizon:
            terminated = True
        elif self._success:
            terminated = True

        info["score"]   = score
        info["success"] = self._success
        return obs, shaped_reward, terminated, truncated_env, info

    def render(self):  # pragma: no cover
        pass
    def close(self):
        self._env.close()


# ═══════════════════════════════════════════════════════════════════
# 6.  Evaluation helper (unchanged)
# ═══════════════════════════════════════════════════════════════════
def evaluate(model, vec_env, sim_dt: float, horizon: float):
    vec_env.training = False
    obs = vec_env.reset()
    done = False
    steps = 0
    final_score = 0.0
    success = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = vec_env.step(action)
        steps += 1
        if "score" in info[0]:
            final_score = info[0]["score"]
            success = info[0]["success"]
        if steps * sim_dt > horizon * 1.5:
            break
    return success, steps, final_score


# ═══════════════════════════════════════════════════════════════════
# 7.  Main CLI
# ═══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser("PPO demo (debug build)")
    ap.add_argument("--timesteps", type=float, default=50_000)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--csv", action="store_true",
                    help="write runs/debug_steps.csv")
    ap.add_argument("--disable-step-log-after", type=int, default=10_000,
                    help="auto‑silence per‑step logging after N steps")
    args = ap.parse_args()

    task = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC, seed=1)
    log.info(f"Task  start={np.round(task.start,2)} "
             f"goal={np.round(task.goal,2)} "
             f"horizon={task.horizon}s")

    venv = make_vec_env(task,
                        gui=args.gui,
                        csv_log=args.csv,
                        disable_after=args.disable_step_log_after)
    print(f"Timesteps: {args.timesteps}")
    model = train_ppo(venv, int(args.timesteps))

    success, steps, score = evaluate(model, venv, SIM_DT, task.horizon)
    venv.close()
    sim_time = steps * SIM_DT

    log.info("\n══════  Evaluation  ══════")
    log.info(f"success        : {success}")
    log.info(f"final score    : {score:.3f}")
    log.info(f"simulated time : {sim_time:.2f}s "
             f"({sim_time / task.horizon:.1%} of horizon)")
    log.info("═══════════════════════════")


if __name__ == "__main__":
    main()
