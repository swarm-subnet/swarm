# The MIT License (MIT)
# Copyright © 2026 Swarm

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# swarm/utils/env_factory.py
"""
Centralised creation of a fully‑initialised single‑drone PyBullet environment
using MovingDroneAviary
The function returns a *fully reset* environment with the world already built
according to the supplied MapTask, so it can be used immediately.
"""
from __future__ import annotations

import contextlib
import io
import time

import numpy as np
import pybullet as p
import pybullet_data
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# ─── project‑level imports ────────────────────────────────────────────────────
from swarm.challenge_families import runtime_profile_for_task
from swarm.core.moving_drone       import MovingDroneAviary
from swarm.protocol                import MapTask
from swarm.constants               import (
    SPEED_LIMIT, MAX_YAW_RATE, SOLVER_ITERATIONS, SOLVER_MIN_ISLAND_SIZE,
    INTERCEPTOR_MINER_SPEED, OFFICE_RC_SPEED, OFFICE_RC_YAW_RATE,
)

# ──────────────────────────────────────────────────────────────────────────────


@contextlib.contextmanager
def _hide_gui_rendering(cli: int, enabled: bool):
    if not enabled:
        yield
        return

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=cli)
    try:
        yield
    finally:
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=cli)


def _configure_gui_visualizer(cli: int) -> None:
    """Disable PyBullet GUI overlays that make procedural scenes look noisy."""
    for name in (
        "COV_ENABLE_SHADOWS",
        "COV_ENABLE_GUI",
        "COV_ENABLE_RGB_BUFFER_PREVIEW",
        "COV_ENABLE_DEPTH_BUFFER_PREVIEW",
        "COV_ENABLE_SEGMENTATION_MARK_PREVIEW",
        "COV_ENABLE_WIREFRAME",
    ):
        flag = getattr(p, name, None)
        if flag is None:
            continue
        p.configureDebugVisualizer(flag, 0, physicsClientId=cli)
        time.sleep(0.05)


def make_env(
    task: MapTask,
    *,
    gui: bool = False,
) -> MovingDroneAviary:
    """
    Create and fully‑initialised single‑drone PyBullet Crazyflie environment.

    Parameters
    ----------
    task     : MapTask   • scenario description (start, goal, map seed, dt, …)
    gui      : bool      • enable/disable PyBullet viewer (default False)
    Returns
    -------
    env : MovingDroneAviary
        A ready‑to‑use environment that has already been reset and whose world
        (obstacles, safe zone, goal beacon, …) has been spawned.
    """
    env, _obs = make_env_with_initial_obs(task, gui=gui)
    return env


def make_env_with_initial_obs(
    task: MapTask,
    *,
    gui: bool = False,
) -> tuple[MovingDroneAviary, dict[str, np.ndarray]]:
    """Create an env and return the observation produced by its initial reset."""
    ctrl_freq = int(round(1.0 / task.sim_dt))
    runtime_profile = runtime_profile_for_task(task)
    # Office physics ticks at 250 Hz: a 20 ms tick cannot resolve motor lag (~1% cost).
    pyb_mult = 5 if getattr(task, "family_id", "") == "cf_interceptor_office" else 1
    common_kwargs = dict(
        gui=gui,
        record=False,
        obs=ObservationType.RGB,
        ctrl_freq=ctrl_freq,
        pyb_freq=ctrl_freq * pyb_mult,
        **dict(runtime_profile.env_bootstrap),
    )

    with contextlib.redirect_stdout(io.StringIO()):
        env = MovingDroneAviary(
            task,
            act=ActionType.VEL,
            **common_kwargs,
        )

    family_id = getattr(task, "family_id", "")
    if family_id == "cf_interceptor":
        env.SPEED_LIMIT = INTERCEPTOR_MINER_SPEED
    elif family_id == "cf_interceptor_office":
        env.SPEED_LIMIT = OFFICE_RC_SPEED
    else:
        env.SPEED_LIMIT = SPEED_LIMIT
    env.MAX_YAW_RATE = (
        OFFICE_RC_YAW_RATE if family_id == "cf_interceptor_office" else MAX_YAW_RATE
    )
    env.ACT_TYPE = ActionType.VEL

    cli = env.getPyBulletClient()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    if gui:
        _configure_gui_visualizer(cli)

    with _hide_gui_rendering(cli, gui):
        with contextlib.redirect_stdout(io.StringIO()):
            obs, _ = env.reset(seed=task.map_seed)

    p.setPhysicsEngineParameter(
        numSolverIterations=SOLVER_ITERATIONS,
        minimumSolverIslandSize=SOLVER_MIN_ISLAND_SIZE,
        physicsClientId=cli,
    )

    return env, obs
