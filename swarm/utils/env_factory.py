# swarm/utils/env_factory.py
"""
Centralised creation of a fully‑initialised single‑drone PyBullet environment
using MovingDroneAviary
The function returns a *fully reset* environment with the world already built
according to the supplied MapTask, so it can be used immediately.
"""
from __future__ import annotations

import io
import time
import contextlib

import pybullet as p
import pybullet_data
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# ─── project‑level imports ────────────────────────────────────────────────────
from swarm.core.moving_drone       import MovingDroneAviary
from swarm.protocol                import MapTask
from swarm.constants               import SPEED_LIMIT

# ──────────────────────────────────────────────────────────────────────────────
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
    ctrl_freq = int(round(1.0 / task.sim_dt))
    common_kwargs = dict(
        gui=gui,
        record=False,
        obs=ObservationType.RGB,
        ctrl_freq=ctrl_freq,
        pyb_freq=ctrl_freq,
    )

    # Silence the copious PyBullet stdout spam when instantiating the env
    with contextlib.redirect_stdout(io.StringIO()):
        env = MovingDroneAviary(
            task,
            act=ActionType.VEL,
            **common_kwargs,
        )

    # Override parent class speed limit (0.25 m/s → 3.0 m/s)
    env.SPEED_LIMIT = SPEED_LIMIT
    env.ACT_TYPE = ActionType.VEL

    cli = env.getPyBulletClient()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    if gui:
        for flag in (p.COV_ENABLE_SHADOWS, p.COV_ENABLE_GUI):
            p.configureDebugVisualizer(flag, 0, physicsClientId=cli)
            time.sleep(0.1)

    with contextlib.redirect_stdout(io.StringIO()):
        env.reset(seed=task.map_seed)

    return env
