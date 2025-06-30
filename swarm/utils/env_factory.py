# swarm/utils/env_factory.py
"""
Centralised creation of a fully‑initialised single‑drone HoverAviary
environment, ready for immediate use by miners (PID control) *and*
the validator (raw‑RPM replay).
"""
from __future__ import annotations

import io
import time
import contextlib
from typing import Union

import numpy as np
import pybullet as p
import pybullet_data
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

from swarm.core.HoverAviaryRawRPM import HoverAviaryRawRPM
from swarm.core.env_builder import build_world
from swarm.protocol import MapTask
from swarm.constants import SAFE_Z

# ---------------------------------------------------------------------
def make_env(
    task: MapTask,
    *,
    gui: bool = False,
    raw_rpm: bool = False,
) -> Union[HoverAviary, HoverAviaryRawRPM]:
    """
    Create and fully‑initialise a PyBullet Crazyflie environment.

    Parameters
    ----------
    task     : MapTask   • scenario description (start, goal, map seed, dt …)
    gui      : bool      • enable/disable PyBullet viewer (default False)
    raw_rpm  : bool      • True  ⇒ action space = raw motor RPM
                          • False ⇒ action space = PID target position
    """
    # 1 ─ choose environment class -------------------------------------
    ctrl_freq = int(round(1.0 / task.sim_dt))
    kwargs = dict(
        gui=gui,
        record=False,
        obs=ObservationType.KIN,
        ctrl_freq=ctrl_freq,
        pyb_freq=ctrl_freq,
    )

    with contextlib.redirect_stdout(io.StringIO()):
        env = (
            HoverAviaryRawRPM(act=ActionType.RPM, **kwargs)
            if raw_rpm
            else HoverAviary(act=ActionType.PID, **kwargs)
        )

    # 2 ─ generic PyBullet plumbing ------------------------------------
    cli = env.getPyBulletClient()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    if gui:
        for flag in (p.COV_ENABLE_SHADOWS, p.COV_ENABLE_GUI):
            p.configureDebugVisualizer(flag, 0, physicsClientId=cli)
            time.sleep(0.1)

    # 3 ─ deterministic reset & world build ----------------------------
    with contextlib.redirect_stdout(io.StringIO()):
        env.reset(seed=task.map_seed)

    # ⬇⬇⬇  NEW: pass *start* to the world builder for safe‑zone logic
    build_world(
        seed=task.map_seed,
        cli=cli,
        start=task.start,
        goal=task.goal,
    )

    # 4 ─ spawn drone at requested start pose --------------------------
    start_xyz = np.asarray(task.start, dtype=float)
    start_quat = p.getQuaternionFromEuler([0.0, 0.0, 0.0])

    p.resetBasePositionAndOrientation(
        env.DRONE_IDS[0],
        start_xyz,
        start_quat,
        physicsClientId=cli,
    )

    return env
