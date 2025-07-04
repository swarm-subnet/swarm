#!/usr/bin/env python3
"""
ppo_demo.py ────────────────────────────────────────────────────────────
End‑to‑end PPO training + evaluation of a single‑drone “fly‑to‑goal”
task using the HoverAviary environment supplied by
`swarm.utils.env_factory`.

Key changes
-----------
1. **Reward shaping**  
   The environment now keeps an incremental view of

       score_t = flight_reward(success_t, t_alive, d_start, d_t, horizon)

   and returns  
   `r_t = score_t − score_{t‑1}`.  
   Thus:

       Σ r_t  ≡  flight_reward(...)      # at the end of the episode

2. **Crash detection & survival reward**  
   – If the underlying PyBullet aviary reports `terminated=True`
     (e.g. after a hard collision) the episode ends and `success=False`.  
   – Staying alive keeps the `alive_term` of `flight_reward`
     growing smoothly toward 1.

3. **Progress reward**  
   Getting closer to the goal increases the `progress_term`
   and therefore the shaped reward returned each step.

4. **Success criterion**  
   The run is marked successful once the drone has hovered
   within `GOAL_TOL` of the goal for `HOVER_SEC` consecutive seconds.

Everything else – CLI, normalisation, PPO hyper‑parameters – is
unchanged.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ── project imports ───────────────────────────────────────────────────
from swarm.utils.env_factory import make_env
from swarm.validator.task_gen import random_task
from swarm.validator.forward import SIM_DT, HORIZON_SEC
from swarm.constants import GOAL_TOL, HOVER_SEC
from swarm.validator.reward import flight_reward          # << your scoring fn
from stable_baselines3.common.monitor import Monitor


# ──────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════
# 1. Gym‑compatible thin wrapper around make_env
# ══════════════════════════════════════════════════════════════════════
class RLTaskEnv(gym.Env):
    """
    Exposes the standard Gymnasium API while delegating physics to
    the PyBullet environment created by make_env().
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    # ────────────────────────────────────────────────────────────────
    # construction / reset
    # ────────────────────────────────────────────────────────────────
    def __init__(self, task, gui: bool = False):
        super().__init__()
        self.task = task
        self._env = make_env(task, gui=gui, raw_rpm=False)
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

        # mutable episode state
        self._time: float = 0.0               # [s] wall‑clock in env
        self._hover_sec: float = 0.0          # [s] time spent inside GOAL_TOL
        self._d_start: float = 1.0            # initial distance to goal
        self._prev_score: float = 0.0         # last call to flight_reward
        self._success: bool = False

    # Gymnasium reset --------------------------------------------------
    def reset(self, *, seed: int | None = None, options=None):
        obs, *_ = self._env.reset(seed=seed)
        self._time = 0.0
        self._hover_sec = 0.0
        self._success = False

        # starting distance to goal
        pos_start = obs[0, :3]
        self._d_start = float(np.linalg.norm(pos_start - self.task.goal))
        # guard against degenerate tasks
        if self._d_start <= 0.0:
            self._d_start = 1e-9

        # initial score (t_alive = 0, d_final = d_start, success = False)
        self._prev_score = flight_reward(False, 0.0,
                                         self._d_start, self._d_start,
                                         self.task.horizon)
        return obs, {}

    # Gymnasium step ---------------------------------------------------
    def step(self, action):
        """Return (obs, shaped_reward, terminated, truncated, info)."""
        # Unpack underlying env result, handling both Gym & Gymnasium
        base = self._env.step(action)
        if len(base) == 5:
            obs, _, terminated_env, truncated_env, info = base
        else:                     # legacy (Gym 0.26 style)
            obs, _, terminated_env, info = base
            truncated_env = False

        # advance simulated clock
        self._time += self.task.sim_dt

        # ------------------------------------------------------------------
        # 1) progress & success bookkeeping
        # ------------------------------------------------------------------
        pos = obs[0, :3]
        dist = float(np.linalg.norm(pos - self.task.goal))

        # within goal tolerance?
        reached = dist < GOAL_TOL
        if reached:
            self._hover_sec += self.task.sim_dt
            if self._hover_sec >= HOVER_SEC:
                self._success = True
        else:
            self._hover_sec = 0.0

        # ------------------------------------------------------------------
        # 2) construct current score (0‥1) via flight_reward
        # ------------------------------------------------------------------
        score = flight_reward(self._success,
                              t_alive=self._time,
                              d_start=self._d_start,
                              d_final=dist,
                              horizon=self.task.horizon)

        # shaped incremental reward
        shaped_reward = score - self._prev_score
        self._prev_score = score

        # ------------------------------------------------------------------
        # 3) episode termination logic
        # ------------------------------------------------------------------
        terminated = False
        truncated = False

        # (a) underlying env signalled a fatal event
        if terminated_env:
            terminated = True
            self._success = False     # guarantee

        # (b) task horizon exceeded
        elif self._time >= self.task.horizon:
            terminated = True

        # (c) mission accomplished (hovered long enough)
        elif self._success:
            terminated = True

        # Return Gymnasium‑style 5‑tuple
        info["score"] = score
        info["success"] = self._success
        return obs, shaped_reward, terminated, truncated, info

    # render() – rely on PyBullet’s own GUI
    def render(self):             # pragma: no cover
        pass

    def close(self):
        self._env.close()


# ══════════════════════════════════════════════════════════════════════
# 2.   Training / evaluation helpers
# ══════════════════════════════════════════════════════════════════════
def make_vec_env(task, gui=False):
    def _factory():
        env = RLTaskEnv(task, gui=gui)
        # Include our custom keys so that Monitor logs them
        env = Monitor(env,               # <- NEW
                    filename=None,     # stdout & TensorBoard only
                    info_keywords=("score", "success"))
        return env

    venv = DummyVecEnv([_factory])
    venv = VecNormalize(venv,
                        norm_obs=True,
                        norm_reward=False,
                        gamma=0.99,
                        training=True)
    return venv


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
    model.learn(total_timesteps=timesteps,
                progress_bar=True)
    return model


def evaluate(model, vec_env, sim_dt: float, horizon: float):
    vec_env.training = False         # freeze running mean/var
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

        # safety stop
        if steps * sim_dt > horizon * 1.5:
            break

    return success, steps, final_score


# ══════════════════════════════════════════════════════════════════════
# 3.  Main CLI
# ══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="PPO demo – fly to a goal")
    ap.add_argument("--timesteps", type=float, default=50_000,
                    help="training timesteps (default: 50 000)")
    ap.add_argument("--gui", action="store_true",
                    help="launch PyBullet GUI for both train & eval")
    args = ap.parse_args()

    # 1) generate a random single‑goal MapTask
    task = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC)
    print("Generated task:",
          f"start={np.round(task.start, 2)}",
          f"goal={np.round(task.goal, 2)}",
          f"horizon={task.horizon}s")

    # 2) environment & training
    vec_env = make_vec_env(task, gui=args.gui)
    model = train_ppo(vec_env, int(args.timesteps))

    # 3) deterministic evaluation
    success, steps, score = evaluate(model, vec_env, SIM_DT, task.horizon)
    vec_env.close()

    # 4) results
    sim_time = steps * SIM_DT
    print("\n══════  Evaluation  ══════")
    print(f"success        : {success}")
    print(f"final score    : {score:.3f}  (0‥1 by design)")
    print(f"simulated time : {sim_time:.2f}s  "
          f"({sim_time / task.horizon:.1%} of horizon)")
    print("═══════════════════════════")


if __name__ == "__main__":   # pragma: no cover
    main()
