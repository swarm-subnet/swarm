"""
swarm.validator.replay
──────────────────────
Deterministic *re‑execution* of a **trained Stable‑Baselines3 policy** on a
given `MapTask`, optionally with a PyBullet GUI.  The episode terminates when
either the environment signals `done` or a configurable safety horizon is
exceeded.

Returns
-------
success : bool
    Taken from the environment's ``info["success"]`` flag (if present) or
    from the first element of the "terminated" signal.
t_sim   : float
    Simulated time in seconds.
energy  : float
    Very coarse energy proxy ‑ Σ‖action‖²·dt·KF / η.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch as th
from stable_baselines3 import PPO

from swarm.utils.env_factory import make_env
from swarm.utils.gui_isolation import run_isolated
from swarm.constants import PROP_EFF        # propeller efficiency constant

# ──────────────────────────────────────────────────────────────────────
# Public façade
# ──────────────────────────────────────────────────────────────────────
def replay_model(
    task,
    *,
    model_path: Path,
    sim_dt: float,
    horizon_sec: float,
    gui: bool = False,
) -> Tuple[bool, float, float]:

    return run_isolated(
        _replay_impl,
        task,
        model_path,
        sim_dt,
        horizon_sec,
        gui=gui,          # <- argument captured by run_isolated
    )


# ──────────────────────────────────────────────────────────────────────
# Implementation (runs in the *current* process OR a child, depending
# on the gui_isolation wrapper).
# ──────────────────────────────────────────────────────────────────────
def _replay_impl(
    task,
    model_path: Path,
    sim_dt: float,
    horizon_sec: float,
    *,
    gui: bool = False,
):
    # 1 ─ environment ---------------------------------------------------
    env = make_env(task, gui=gui)
    model = PPO.load(model_path, device=th.device("cpu"))
    model.set_env(env)

    # 2 ─ episode loop --------------------------------------------------
    step    = 0
    done    = False
    success = False
    energy  = 0.0
    obs, _  = env.reset()

    while not done:
        act, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, info = env.step(act)
        done = bool(terminated[0])

        # success flag (if env provides it)
        if "success" in info[0]:
            success = bool(info[0]["success"])

        # very coarse energy estimate Σ‖u‖²·dt·KF / η
        energy += (np.square(act).sum() * env.KF / PROP_EFF) * sim_dt

        step += 1
        if step * sim_dt > 1.5 * horizon_sec:   # safety break
            break

        if gui:
            time.sleep(sim_dt)

    if not gui:
        env.close()

    return success, step * sim_dt, energy
