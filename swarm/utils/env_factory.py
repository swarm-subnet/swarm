"""
swarm.utils.env_factory
───────────────────────
Centralised creation of a fully‑initialised single‑drone HoverAviary
environment, ready for immediate use by miners (PID control) *and*
the validator (raw‑RPM replay).

Import once and reuse everywhere:

    from swarm.utils.env_factory import make_env
    env = make_env(task, gui=False, raw_rpm=True)
"""
from __future__ import annotations

from typing import Union

import numpy as np
import pybullet as p
import pybullet_data
import time
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

    Returns
    -------
    env : HoverAviary | HoverAviaryRawRPM
        The initialised environment.  World, obstacles and drone pose are
        ready; just call `env.step()` in your control loop.
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

    if raw_rpm:
        env = HoverAviaryRawRPM(act=ActionType.RPM, **kwargs)
    else:
        env = HoverAviary(act=ActionType.PID, **kwargs)

    # 2 ─ generic PyBullet plumbing ------------------------------------
    cli = env.getPyBulletClient()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    if gui:
        # Hide default shadows & widgets for a cleaner viewer.
        for flag in (p.COV_ENABLE_SHADOWS, p.COV_ENABLE_GUI):
            p.configureDebugVisualizer(flag, 0, physicsClientId=cli)
            time.sleep(20)

    # 3 ─ deterministic reset & world build ----------------------------
    env.reset(seed=task.map_seed)
    build_world(task.map_seed, cli, task.goal)

    # 4 ─ spawn drone at requested start pose --------------------------
    start_xyz = np.asarray(task.start, dtype=float)
    # start_xyz[2] = max(start_xyz[2], SAFE_Z)   # never below safety floor
    start_quat = p.getQuaternionFromEuler([0.0, 0.0, 0.0])

    p.resetBasePositionAndOrientation(
        env.DRONE_IDS[0],      # one‑drone scenario
        start_xyz,
        start_quat,
        physicsClientId=cli,
    )

    return env
