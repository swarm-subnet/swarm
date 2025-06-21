"""
neurons.miner – flying_strategy(task)
─────────────────────────────────────
Generate an open‑loop list of rotor‑RPM commands for one Crazyflie.

Changes vs original
-------------------
* Adds a **5 s hover requirement** (HOVER_SEC) over the final waypoint.
* Passes the task goal to *build_world* so the visual marker appears during
  planning runs.
"""
from __future__ import annotations

import time
from typing import List, Sequence, Tuple

import numpy as np
import pybullet as p
import pybullet_data
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

from swarm.utils.gui_isolation import run_isolated
from swarm.validator.env_builder import build_world
from swarm.protocol import MapTask, RPMCmd             
from swarm.utils.drone import track_drone  # late import        

# ───────── parameters & constants ─────────
from swarm.constants import (SAFE_Z,       
    GOAL_TOL,      
    HOVER_SEC,     
    CAM_HZ)
# ───────────────────────────────────────────


# ---------- public API ---------------------------------------------------
def flying_strategy(task: MapTask, *, gui: bool = False) -> List[RPMCmd]:
    """Thin wrapper that delegates to the real body through run_isolated."""
    return run_isolated(_flying_strategy_impl, task, gui=gui)


# ---------- implementation ----------------------------------------------
#Default, hardcoded miner -> Goes from point A to point B, then hovers for 5 seconds. Miners should implement their own logic here.
def _flying_strategy_impl(task: MapTask, *, gui: bool = False) -> List[RPMCmd]:
    # 1 ─ environment ----------------------------------------------------
    ctrl_freq = int(round(1.0 / task.sim_dt))
    pyb_freq  = ctrl_freq
    env = HoverAviary(gui=gui,
                  record=False,
                  obs=ObservationType.KIN,
                  act=ActionType.PID,       
                  ctrl_freq=ctrl_freq,
                  pyb_freq=pyb_freq)
    cli = env.getPyBulletClient()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Tidy viewer
    if gui:
        for flag in (p.COV_ENABLE_SHADOWS, p.COV_ENABLE_GUI):
            p.configureDebugVisualizer(flag, 0, physicsClientId=cli)

    # 2 ─ reset then build world ----------------------------------------
    env.reset(seed=task.map_seed)
    build_world(task.map_seed, cli, task.goal)           # ← pass goal
    # 4 ─ drone initial pose -------------------------------------------
    start_xyz = np.array(task.start, dtype=float)
    start_quat = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
    p.resetBasePositionAndOrientation(
        env.DRONE_IDS[0],
        start_xyz,
        start_quat,
        physicsClientId=cli,
    )

    # 5 ─ way‑points ----------------------------------------------------
    gx, gy, gz = task.goal
    safe_z = max(SAFE_Z, start_xyz[2], gz)
    wps = [
        np.array([*start_xyz[:2], safe_z]),
        np.array([gx, gy, safe_z]),
        np.array([gx, gy, gz]),   # final
    ]
    wp_idx = 0

    # camera bookkeeping
    if gui:
        frames_per_cam = max(1, int(round(1.0 / (task.sim_dt * CAM_HZ))))
        step_counter = 0

    # 6 ─ control loop --------------------------------------------------
    t_sim = 0.0
    hover_elapsed = 0.0      # NEW
    extra_counter = 0
    rpm_log: List[RPMCmd] = []

    while t_sim < task.horizon:
        target = wps[wp_idx]

        # physics + PID
        obs, *_ = env.step(target.reshape(1, 3))
        pos = obs[0, :3]

        # camera follow
        if gui and step_counter % frames_per_cam == 0:
            
            track_drone(
                cli=cli,
                drone_id=env.DRONE_IDS[0],
                frames_per_cam=frames_per_cam,
                cam_hz=CAM_HZ,
            )

        # log motor command
        # print(f"Last clipped action: {env.last_clipped_action[0]})  # debug")
        _record_cmd(rpm_log, env.last_clipped_action[0], t_sim)

        # waypoint / hover logic
        dist = np.linalg.norm(pos - target)
        if wp_idx < len(wps) - 1:
            # Normal waypoint switching
            if dist < GOAL_TOL:
                wp_idx += 1
        else:
            # Final waypoint – enforce 5 s hover
            if dist < GOAL_TOL:
                hover_elapsed += task.sim_dt
                if hover_elapsed >= HOVER_SEC+2:
                    extra_counter += 1
                    if extra_counter >= int(1.0 / task.sim_dt):   # 1 extra second
                        break
            else:
                hover_elapsed = 0.0    # drifted out – reset timer

        # bookkeeping
        t_sim += task.sim_dt
        if gui:
            time.sleep(task.sim_dt)
            step_counter += 1

    # 7 ─ clean‑up ------------------------------------------------------
    if not gui:                 # head‑less – safe to close Bullet
        env.close()

    # (In GUI mode we purposely leave PyBullet open – the subprocess dies
    # immediately after return, so resources are reclaimed by the OS.)
    return rpm_log


# ---------- helpers ------------------------------------------------------
def _record_cmd(buffer: List[RPMCmd], rpm_vec: Sequence[float], t: float) -> None:
    """Convert the 4‑element vector into an RPMCmd dataclass entry."""
    rpm_tuple: Tuple[float, float, float, float] = tuple(float(x) for x in rpm_vec)  # type: ignore[arg-type]
    buffer.append(RPMCmd(t=t, rpm=rpm_tuple))
