#!/usr/bin/env python3
"""
ppo_drone_train_eval.py
───────────────────────
* Parallel PPO training (head‑less, 8 envs)
* Deterministic evaluation afterwards
  • head‑less by default
  • GUI if --gui is passed
"""

from __future__ import annotations
import argparse
import csv
import logging
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import torch as th
from rich.logging import RichHandler
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env as sb3_make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from tensorboardX import SummaryWriter

# ── project imports ──────────────────────────────────────────────────────────
from swarm.utils.env_factory  import make_env
from swarm.validator.task_gen import random_task
from swarm.validator.forward  import SIM_DT, HORIZON_SEC
from swarm.validator.reward   import flight_reward  # noqa: F401  (log helper)

# ═════════════════════════════════════════════════════════════════════════════
# Logging
# ═════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(message)s",
    datefmt = "[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("ppo_demo")

# ═════════════════════════════════════════════════════════════════════════════
# Per‑step logger (optional CSV)
# ═════════════════════════════════════════════════════════════════════════════
class StepLoggerWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        *,
        csv_path: Optional[Path] = None,
    ) -> None:
        super().__init__(env)
        self._ep, self._step = 0, 0
        self._csv_file = None
        if csv_path is not None:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = csv_path.open("w", newline="")
            self._writer   = csv.writer(self._csv_file)
            self._writer.writerow(
                ["ep", "t", "term", *[f"obs{i}" for i in range(np.prod(env.observation_space.shape))],
                 *[f"act{i}" for i in range(np.prod(env.action_space.shape))],
                 "rew", "score", "succ"]
            )

    def reset(self, **kwargs):
        if self._step:
            self._ep += 1
            self._step = 0
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        self._step += 1
        obs, r, ter, tru, info = self.env.step(action)
        if self._csv_file is not None:
            self._writer.writerow(
                [self._ep, self._step, int(ter or tru),
                 *np.asarray(obs).flatten().tolist(),
                 *np.asarray(action).flatten().tolist(),
                 float(r), info.get("score", np.nan), int(info.get("success", False))]
            )
        return obs, r, ter, tru, info

    def close(self):
        super().close()
        if self._csv_file is not None:
            self._csv_file.close()

# ═════════════════════════════════════════════════════════════════════════════
# TensorBoard callback (quick)
# ═════════════════════════════════════════════════════════════════════════════
class TBCallback(BaseCallback):
    def __init__(self, log_dir: str):
        super().__init__()
        self.tb = SummaryWriter(log_dir)

    def _on_step(self) -> bool:
        self.tb.add_scalar("reward/step", self.locals["rewards"][0], self.num_timesteps)
        return True

# ═════════════════════════════════════════════════════════════════════════════
# Vec‑Env builder
# ═════════════════════════════════════════════════════════════════════════════
def build_vec_env(task, *, gui: bool, n_envs: int, csv: bool):
    csv_path = Path("runs/debug_steps.csv") if csv else None
    env_fn   = lambda **_: make_env(task, gui=gui)              # noqa: E731
    venv     = sb3_make_vec_env(
        env_fn,
        n_envs         = n_envs,
        seed           = 0,
        monitor_kwargs = {"info_keywords": ("score", "success")},
        wrapper_class  = StepLoggerWrapper,
        wrapper_kwargs = {"csv_path": csv_path},
    )
    return VecNormalize(venv, norm_obs=True, norm_reward=False, gamma=0.99)

# ═════════════════════════════════════════════════════════════════════════════
# PPO helpers
# ═════════════════════════════════════════════════════════════════════════════
def make_model(vec_env):
    policy_kwargs = dict(
        activation_fn = th.nn.ReLU,
        net_arch      = dict(pi=[256, 256], vf=[256, 256]),
    )
    return PPO(
        "MlpPolicy",
        vec_env,
        n_steps         = 2048,
        batch_size      = 2048,
        learning_rate   = 3e-4,
        ent_coef        = 0.01,
        target_kl       = 0.03,
        use_sde         = True,
        sde_sample_freq = 4,
        clip_range      = 0.1,
        device          = "auto",
        verbose         = 1,
        tensorboard_log = "runs/ppo_fly2goal",
        policy_kwargs   = policy_kwargs,
    )

def evaluate(model, vec_env, sim_dt: float, horizon: float):
    vec_env.training = False
    obs     = vec_env.reset()
    done    = False
    step    = 0
    ret     = 0.0
    score   = 0.0
    succ    = False
    while not done:
        act, _ = model.predict(obs, deterministic=True)
        obs, r, done_v, info = vec_env.step(act)
        ret  += r[0]
        done  = bool(done_v[0])
        step += 1
        if "score" in info[0]:
            score, succ = info[0]["score"], info[0]["success"]
        if step * sim_dt > 1.5 * horizon:      # safety break
            break
    return ret, succ, step, score

# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=500_000,
                   help="train steps (0 → skip training and load saved model)")
    p.add_argument("--gui", action="store_true",
                   help="show PyBullet window during evaluation")
    p.add_argument("--csv", action="store_true",
                   help="write runs/debug_steps.csv while training")
    args = p.parse_args()

    # same task for training & evaluation
    task = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC, seed=1)
    log.info(f"Task start={np.round(task.start,2)} goal={np.round(task.goal,2)} horizon={task.horizon}s")

    # ── Train (if requested) ─────────────────────────────────────────────────
    model_path = Path("runs/ppo_policy.zip")
    stats_path = Path("runs/venv_stats.pkl")

    if args.timesteps > 0:
        train_env = build_vec_env(task, gui=False, n_envs=8, csv=args.csv)
        model     = make_model(train_env)
        log.info(f"Training for {args.timesteps:,} steps …")
        model.learn(args.timesteps, progress_bar=True, callback=TBCallback("runs/tensor_debug"))
        train_env.save(stats_path)
        train_env.close()
        model.save(model_path)
        log.info("✔ training finished")
    else:
        if not model_path.exists():
            raise FileNotFoundError("No trained policy found; run with --timesteps > 0 first.")
        model = PPO.load(model_path)

    # ── Evaluation (always) ─────────────────────────────────────────────────
    eval_env = build_vec_env(
        task,
        gui   = args.gui,
        n_envs= 1,
        csv   = False,
    )
    # load normalisation (if present)
    if stats_path.exists():
        eval_env = VecNormalize.load(stats_path, eval_env)
        eval_env.training = False

    ret, succ, steps, score = evaluate(model, eval_env, SIM_DT, task.horizon)
    sim_time = steps * SIM_DT
    eval_env.close()

    log.info("\n══════  Evaluation  ══════")
    log.info(f"success        : {succ}")
    log.info(f"final score    : {score:.3f}")
    log.info(f"episode return : {ret:.3f}")
    log.info(f"sim time       : {sim_time:.2f}s  ({sim_time/task.horizon:.1%} of horizon)")
    log.info("═══════════════════════════")

# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
