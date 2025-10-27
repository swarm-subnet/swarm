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
from typing import Union

import numpy as np
import pybullet as p
import pybullet_data
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# ─── project‑level imports ────────────────────────────────────────────────────
from swarm.core.moving_drone       import MovingDroneAviary
from swarm.core.env_builder        import build_world
from swarm.protocol                import MapTask
from swarm.constants               import SAFE_Z, SPEED_LIMIT

# ──────────────────────────────────────────────────────────────────────────────
def make_env(
    task: MapTask,
    *,
    gui: bool = False,
) -> Union[MovingDroneAviary]:
    """
    Create and fully‑initialise a single‑drone PyBullet Crazyflie environment.

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
    # 1 ─ choose environment class and common kwargs --------------------------
    ctrl_freq = int(round(1.0 / task.sim_dt))
    common_kwargs = dict(
        gui=gui,
        record=False,
        obs=ObservationType.KIN,
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

    # 2 ─ generic PyBullet plumbing ------------------------------------------
    cli = env.getPyBulletClient()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    if gui:
        # Hide debug GUI elements & shadows for clearer visuals
        for flag in (p.COV_ENABLE_SHADOWS, p.COV_ENABLE_GUI):
            p.configureDebugVisualizer(flag, 0, physicsClientId=cli)
            time.sleep(0.1)

    # 3 ─ deterministic reset & world build ----------------------------------
    with contextlib.redirect_stdout(io.StringIO()):
        env.reset(seed=task.map_seed)

    platform_support_uid, landing_surface_uid = build_world(
        seed=task.map_seed,
        cli=cli,
        start=task.start,
        goal=task.goal,
        challenge_type=task.challenge_type,
    )
    
    env._platform_support_uid = platform_support_uid
    env._landing_surface_uid = landing_surface_uid

    # 4 ─ spawn drone at the requested start pose ----------------------------
    start_xyz  = np.asarray(task.start, dtype=float)
    start_quat = p.getQuaternionFromEuler([0.0, 0.0, 0.0])

    p.resetBasePositionAndOrientation(
        env.DRONE_IDS[0],
        start_xyz,
        start_quat,
        physicsClientId=cli,
    )

    return env
