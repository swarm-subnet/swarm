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
from swarm.core.env_builder import build_world, BirdSystem, WindSystem
from swarm.protocol import MapTask
from swarm.constants import SAFE_Z, ENABLE_BIRDS, ENABLE_WIND, ENABLE_MOVING_PLATFORM

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
    result = build_world(
        seed=task.map_seed,
        cli=cli,
        start=task.start,
        goal=task.goal,
    )
    
    bird_ids, obstacles, platform_uids = result if result else ([], [], [])

    # 4 ─ spawn drone at requested start pose --------------------------
    start_xyz = np.asarray(task.start, dtype=float)
    start_quat = p.getQuaternionFromEuler([0.0, 0.0, 0.0])

    p.resetBasePositionAndOrientation(
        env.DRONE_IDS[0],
        start_xyz,
        start_quat,
        physicsClientId=cli,
    )

    # 5 ─ Initialize avian simulation system if enabled ----------------------------
    if ENABLE_BIRDS and bird_ids:
        import random
        rng = random.Random(task.map_seed)
        env._bird_system = BirdSystem(cli, bird_ids, obstacles, rng, task.start, task.goal)

    # 6 ─ Initialize atmospheric wind simulation system if enabled ----------------------------
    if ENABLE_WIND:
        import random
        rng = random.Random(task.map_seed)
        env._wind_system = WindSystem(rng)

    # 7 ─ Initialize moving platform system if enabled ----------------------------
    if ENABLE_MOVING_PLATFORM and platform_uids and len(platform_uids) > 0:
        import random
        from swarm.core.env_builder import _MovingPlatform
        from swarm.constants import (
            PLATFORM_MOTION_TYPE, PLATFORM_MOTION_SPEED, 
            PLATFORM_MOTION_RADIUS, PLATFORM_PATH_LENGTH
        )
        
        # Create seeded random number generator for deterministic motion
        platform_rng = random.Random(task.map_seed)
        
        # Create moving platform system using the same seed as other systems
        moving_platform_system = _MovingPlatform(
            cli=cli,
            platform_uids=platform_uids,
            motion_type=PLATFORM_MOTION_TYPE,
            speed=PLATFORM_MOTION_SPEED,
            radius=PLATFORM_MOTION_RADIUS,
            path_length=PLATFORM_PATH_LENGTH,
            center=np.array(task.goal, dtype=np.float32),
            rng=platform_rng
        )
        
        env._moving_platform_system = moving_platform_system

    return env
