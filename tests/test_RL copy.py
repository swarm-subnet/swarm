#!/usr/bin/env python3
"""
test_RL.py – PPO demo (single‑drone) that *directly* uses the reward produced by
             `MovingDroneAviary`.

This version relies on **stable_baselines3.common.env_util.make_vec_env**
instead of an in‑file re‑implementation.
"""

from __future__ import annotations
import torch as th
import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import torch  # noqa: F401  (SB3 may place tensors on GPU/CPU automatically)
from rich.logging import RichHandler
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env as sb3_make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
from tensorboardX import SummaryWriter

# ── project imports ──────────────────────────────────────────────────────────
from swarm.utils.env_factory import make_env                # your one‑stop env builder
from swarm.validator.task_gen import random_task
from swarm.validator.forward import SIM_DT, HORIZON_SEC

# *Only* needed for logging/inspection – does **not** affect the reward
from swarm.validator.reward import flight_reward  # noqa: F401

# ═════════════════════════════════════════════════════════════════════════════
# 0. Logging
# ═════════════════════════════════════════════════════════════════════════════
LOG_LEVEL = logging.DEBUG
LOG_FMT = "%(message)s"
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FMT,
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("ppo_debug")

# ═════════════════════════════════════════════════════════════════════════════
# 1.  Per‑step debug wrapper
# ═════════════════════════════════════════════════════════════════════════════
class StepLoggerWrapper(gym.Wrapper):
    """
    Print (and optionally CSV‑log) every observation/action/reward.

    Toggle on/off by changing ``self.enabled`` at runtime.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        csv_path: Optional[Path] = None,
        print_every: int = 1,
    ) -> None:
        super().__init__(env)
        self.enabled = True
        self.print_every = print_every
        self._step_count = 0
        self._episode = 0

        # Optional CSV for later post‑mortem analysis
        self._csv_file = None
        self._csv_writer = None
        if csv_path is not None:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = csv_path.open("w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(
                [
                    "episode",
                    "step",
                    "terminated",
                    *[f"obs_{i}" for i in range(np.prod(env.observation_space.shape))],
                    *[f"act_{i}" for i in range(np.prod(env.action_space.shape))],
                    "reward",
                    "score",
                    "success",
                ]
            )

    # --------------------------------------------------------------------- #
    def reset(self, **kwargs):
        if self._step_count:  # just finished an episode
            self._episode += 1
            self._step_count = 0
        obs, info = self.env.reset(**kwargs)  # gymnasium two‑return‑value API
        # if self.enabled:
        #     log.debug(f"[E{self._episode:04d}] reset ⇒ obs={obs}")
        return obs, info

    # --------------------------------------------------------------------- #
    def step(self, action):
        self._step_count += 1
        obs, reward, term, trunc, info = self.env.step(action)

        # Console print
        # if self.enabled and self._step_count % self.print_every == 0:
        #     log.debug(
        #         f"[E{self._episode:04d} | t={self._step_count:04d}] "
        #         f"act={np.asarray(action).round(3)} "
        #         f"r={reward:+.3f} "
        #         f"score={info.get('score', np.nan):.3f} "
        #         f"succ={info.get('success', False)}"
        #     )

        # CSV
        if self._csv_writer is not None:
            self._csv_writer.writerow(
                [
                    self._episode,
                    self._step_count,
                    int(term or trunc),
                    *np.asarray(obs).flatten().tolist(),
                    *np.asarray(action).flatten().tolist(),
                    float(reward),
                    info.get("score", np.nan),
                    int(info.get("success", False)),
                ]
            )
        return obs, reward, term, trunc, info

    # --------------------------------------------------------------------- #
    def close(self):
        super().close()
        if self._csv_file is not None:
            self._csv_file.close()


# ═════════════════════════════════════════════════════════════════════════════
# 2.  TensorBoard debug callback
# ═════════════════════════════════════════════════════════════════════════════
class TensorDebugCallback(BaseCallback):
    """Dumps raw observations, actions, rewards and gradient norms to TensorBoard."""

    def __init__(self, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.tb = SummaryWriter(log_dir)
        self._grad_step = 0

    def _on_step(self) -> bool:
        obs = self.locals["new_obs"][0]
        reward = self.locals["rewards"][0]
        action = self.locals["actions"][0]
        step = self.num_timesteps

        self.tb.add_histogram("obs/raw", obs, step)
        self.tb.add_histogram("action/raw", action, step)
        self.tb.add_scalar("reward/step", reward, step)
        return True

    def _on_rollout_end(self) -> None:
        model: PPO = self.model  # type: ignore
        step = self.num_timesteps

        for key in ("policy_loss", "value_loss", "entropy"):
            if key in model.logger.name_to_value:
                self.tb.add_scalar(f"loss/{key}", model.logger.name_to_value[key][-1], step)

        # Global L2 norm of all gradients
        total = 0.0
        for p in model.policy.parameters():
            if p.grad is not None:
                total += p.grad.data.norm(2).item() ** 2
        self.tb.add_scalar("grad/global_norm", total ** 0.5, step)
        self._grad_step += 1


# ═════════════════════════════════════════════════════════════════════════════
# 3.  Vec‑env constructor  (⚠️ renamed – *no* shadowing)
# ═════════════════════════════════════════════════════════════════════════════
def build_vec_env(
    task,
    *,
    csv_log: bool = False,
    disable_after: Optional[int] = None,
):
    """
    Create a single‑environment ``VecNormalize`` wrapped in ``DummyVecEnv`` using
    Stable‑Baselines 3's official helper.  Adds an optional per‑step logger.
    """
    csv_path = Path("runs/debug_steps.csv") if csv_log else None

    # 1️⃣  A callable that spawns a *fresh* environment every time SB3 asks.
    env_fn = lambda **_: make_env(task)  # noqa: E731

    # 2️⃣  Ask SB3 to vectorise it and to decorate it with our thin logger.
    venv = sb3_make_vec_env(
        env_fn,
        n_envs=16,
        seed=0,
        monitor_kwargs={"info_keywords": ("score", "success")},
        wrapper_class=StepLoggerWrapper,
        wrapper_kwargs={"csv_path": csv_path, "print_every": 1},
    )

    # 3️⃣  Optionally silence the console spam after N steps
    if disable_after is not None:
        def _find_logger(w):
            while isinstance(w, gym.Wrapper):
                if isinstance(w, StepLoggerWrapper):
                    return w
                w = w.env
            return None

        for e in venv.envs:  # VecEnv → list of base envs
            logger = _find_logger(e)
            if logger is None:
                continue

            def _auto_disable_step(action, _orig=logger.step):
                if logger._step_count >= disable_after:
                    logger.enabled = False
                return _orig(action)

            logger.step = _auto_disable_step  # type: ignore[attr-defined]

    venv = VecNormalize(
        venv,
        norm_obs=True,                 # z‑score normalise observations
        norm_reward=False,              # running mean/std on Δ‑rewards
        gamma=0.99,                    # keep in sync with PPO’s discount
    )

    return venv


# ═════════════════════════════════════════════════════════════════════════════
# 4.  PPO training
# ═════════════════════════════════════════════════════════════════════════════
def train_ppo(vec_env, timesteps: int):
    policy_kwargs = dict(
        activation_fn = th.nn.ReLU,    # good for large nets
        net_arch = {
            "pi": [1024, 1024, 512, 256],   # actor pathway
            "vf": [1024, 1024, 512, 256],   # critic pathway
        },
        ortho_init = False,            # disable SB3’s default orthogonal init
    )
    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps        = 2048,        # 8 envs × 1024 = 8192 samples/update
        batch_size     = 2048,
        learning_rate  = 3e-4,
        ent_coef       = 0.01,
        target_kl      = 0.03,
        use_sde        = True,        # state-dependent exploration
        sde_sample_freq= 4,
        device         = "auto",      # pick GPU if available
        clip_range= 0.1,
        verbose        = 1,
        policy_kwargs  = policy_kwargs,
        tensorboard_log= "runs/ppo_fly2goal",
    )
    model.learn(
        total_timesteps=timesteps,
        progress_bar=True,
        callback=TensorDebugCallback(log_dir="runs/tensor_debug"),
    )
    return model


# ═════════════════════════════════════════════════════════════════════════════
# 5.  Deterministic evaluation
# ═════════════════════════════════════════════════════════════════════════════
def evaluate(model, vec_env, sim_dt: float, horizon: float):
    vec_env.training = False  # freeze running means/vars in VecNormalize
    obs = vec_env.reset()  # SB3 wrapper converts (obs, info) → obs
    done = False
    steps = 0
    final_score = 0.0
    success = False
    episode_return = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done_vec, infos = vec_env.step(action)
        episode_return += reward[0]   # raw increment
        done = bool(done_vec[0])
        steps += 1
        if "score" in infos[0]:
            final_score = infos[0]["score"]
            success = infos[0]["success"]

        # emergency brake – should never trigger
        if steps * sim_dt > horizon * 1.5:
            break

    return episode_return, success, steps, final_score


# ═════════════════════════════════════════════════════════════════════════════
# 6.  CLI
# ═════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser("PPO demo – no duplicate reward shaping")
    ap.add_argument("--timesteps", type=float, default=50_000)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--csv", action="store_true", help="write runs/debug_steps.csv")
    ap.add_argument(
        "--disable-step-log-after",
        type=int,
        default=10_000,
        help="auto‑silence per‑step logging after N steps",
    )
    args = ap.parse_args()

    task = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC, seed=1)
    log.info(
        f"Task  start={np.round(task.start,2)}  "
        f"goal={np.round(task.goal,2)}  "
        f"horizon={task.horizon}s"
    )

    venv = build_vec_env(
        task,
        gui=args.gui,
        csv_log=args.csv,
        disable_after=args.disable_step_log_after,
    )

    log.info(f"Training for {args.timesteps:,} timesteps …")
    model = train_ppo(venv, int(args.timesteps))

    episode_return, success, steps, score = evaluate(model, venv, SIM_DT, task.horizon)
    venv.close()
    sim_time = steps * SIM_DT

    log.info("\n══════  Evaluation  ══════")
    log.info(f"success        : {success}")
    log.info(f"final score    : {score:.3f}")
    log.info(f"episode    : {episode_return:.3f}")
    log.info(
        f"simulated time : {sim_time:.2f}s "
        f"({sim_time / task.horizon:.1%} of horizon)"
    )
    log.info("═══════════════════════════")


if __name__ == "__main__":
    main()
