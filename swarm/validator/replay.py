"""
swarm.validator.replay
──────────────────────
PID‑based validation replay for a single drone.

Changes vs original
-------------------
* Success now requires **hovering 5 s** inside the goal tolerance.
* World builder receives the task goal so the marker is visible in replay.
"""
from __future__ import annotations

import math
import time
from typing import List, Tuple

import numpy as np
import pybullet as p
import pybullet_data
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

from swarm.utils.gui_isolation import run_isolated
from swarm.validator.env_builder import build_world
from swarm.protocol import FlightPlan, MapTask
from swarm.utils.drone import track_drone  # reused helper

# ─────────── constants ───────────
PROP_EFF = 0.60
WAYPOINT_TOL = 0.20
SAFE_ASCENT = 3.0
MAX_STEPS = 60_000
CAM_HZ = 60
HOVER_SEC = 5.0           # NEW – time to hover at goal (s)
# ─────────────────────────────────


# ---------- public façade -----------------------------------------------
def replay_once(
    task: MapTask,
    plan: FlightPlan,  # kept for API compatibility
    *,
    gui: bool = False,
) -> Tuple[bool, float, float]:
    """Wrapper that executes the body in an isolated process when needed."""
    return run_isolated(_replay_once_impl, task, plan, gui=gui)


# ---------- implementation ----------------------------------------------
def _replay_once_impl(
    task: MapTask,
    plan: FlightPlan,  # unused (open‑loop strategy, but we keep the API)
    *,
    gui: bool = False,
) -> Tuple[bool, float, float]:
    # 1 ─ environment ---------------------------------------------------
    frames_per_cam = int(round(1.0 / (task.sim_dt * CAM_HZ)))
    env = HoverAviary(gui=gui, obs=ObservationType.KIN, act=ActionType.PID)
    cli = env.getPyBulletClient()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Tidy viewer
    if gui:
        for flag in (p.COV_ENABLE_SHADOWS, p.COV_ENABLE_GUI):
            p.configureDebugVisualizer(flag, 0, physicsClientId=cli)

    env.reset(seed=task.map_seed)
    build_world(task.map_seed, cli, task.goal)          # ← pass goal

    # 2 ─ way‑points ----------------------------------------------------
    wps = _waypoints(task)
    wp_idx = 0
    energy = 0.0
    success = False
    hover_elapsed = 0.0        # NEW
    max_sim_steps = int(math.ceil(task.horizon / env.CTRL_TIMESTEP))

    # 3 ─ main loop -----------------------------------------------------
    for step in range(min(max_sim_steps, MAX_STEPS)):
        t_sim = step * env.CTRL_TIMESTEP
        if gui:
            time.sleep(env.CTRL_TIMESTEP)

        if gui and step % frames_per_cam == 0:
            track_drone(
                cli=env.getPyBulletClient(),
                drone_id=env.DRONE_IDS[0],
                frames_per_cam=frames_per_cam,
                cam_hz=CAM_HZ,
            )

        # current target
        target = wps[wp_idx]
        obs = env.step(target.reshape(1, 3))[0]
        pos = _extract_pos(obs)

        dist = np.linalg.norm(pos - target)

        # waypoint / hover logic
        if wp_idx < len(wps) - 1:
            if dist < WAYPOINT_TOL:
                wp_idx += 1
        else:   # final waypoint
            if dist < WAYPOINT_TOL:
                hover_elapsed += env.CTRL_TIMESTEP
                if hover_elapsed >= HOVER_SEC:
                    success = True
                    break
            else:
                hover_elapsed = 0.0

        # horizon reached?
        if t_sim >= task.horizon:
            break

        # energy bookkeeping
        thrusts = np.square(env.last_clipped_action[0]) * env.KF
        energy += (thrusts.sum() / PROP_EFF) * env.CTRL_TIMESTEP

    # 4 ─ clean‑up ------------------------------------------------------
    if not gui:
        env.close()

    return success, t_sim, energy


# ---------- helpers -----------------------------------------------------
def _extract_pos(obs: np.ndarray) -> np.ndarray:
    """Extract x, y, z from HoverAviary observation."""
    return obs[0, :3] if obs.ndim == 2 else obs[:3]


def _waypoints(task: MapTask) -> List[np.ndarray]:
    """Three‑segment trajectory [take‑off, cruise, land]."""
    start, goal = np.array(task.start), np.array(task.goal)
    safe_z = max(SAFE_ASCENT, start[2], goal[2])
    return [
        np.array([start[0], start[1], safe_z]),
        np.array([goal[0], goal[1], safe_z]),
        goal,
    ]
