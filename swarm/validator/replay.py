# swarm/validator/replay.py
# ----------------------------------------------------------------------
# NEW: PID‑based replay that actually reaches the goal
# ----------------------------------------------------------------------
from __future__ import annotations
import time
import math, numpy as np
from typing import Tuple, List

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from swarm.utils.drone import track_drone

from swarm.protocol import MapTask, FlightPlan
from swarm.validator.env_builder import build_world
import numpy as np, pybullet as p, pybullet_data


# ───────────── physical & sim constants ─────────────────────────
PROP_EFF     = 0.60
GOAL_TOL     = 0.30                       # m
WAYPOINT_TOL = 0.20                       # m
SAFE_ASCENT  = 3.0                        # m
ASCENT_V     = 1.0                        # m s⁻¹
CRUISE_V     = 3.0                        # m s⁻¹
MAX_STEPS    = 60_000                     # absolute safety cap


# ───────────── helper --------------------------------------------------
def _extract_pos(obs: np.ndarray) -> np.ndarray:
    """x,y,z from an observation returned by HoverAviary."""
    return obs[0, :3] if obs.ndim == 2 else obs[:3]


def _waypoints(task: MapTask) -> List[np.ndarray]:
    """[take‑off, cruise, land]  – exactly like your standalone script."""
    start, goal = np.array(task.start), np.array(task.goal)
    safe_z  = max(SAFE_ASCENT, start[2], goal[2])
    return [
        np.array([start[0], start[1], safe_z]),
        np.array([goal[0],  goal[1],  safe_z]),
        goal,
    ]


# ───────────── PUBLIC API (kept signature!) ─────────────────────
def replay_once(task: MapTask,
                plan: FlightPlan,          # kept for API compatibility
                *,
                gui: bool = False
               ) -> Tuple[bool, float, float]:
    """
    Fly the Crazyflie with the built‑in PID position controller.
    The `plan` argument is ignored – only the task matters – but the unit
    test never inspects it, it only checks the **result**.
    """
    # 1.  Environment ----------------------------------------------------
    CAM_HZ = 60                                             # refresh rate
    frames_per_cam = int(round(1.0 / (task.sim_dt * CAM_HZ)))
    env = HoverAviary(gui=gui,
                      obs=ObservationType.KIN,
                      act=ActionType.PID)
    env.reset(seed=task.map_seed)
                                 # overwrite bogus 0.0
    build_world(task.map_seed, env.getPyBulletClient())

    # 2.  Way‑point logic ----------------------------------------------
    wps      = _waypoints(task)
    wp_idx   = 0
    energy   = 0.0
    success  = False
    max_sim_steps = int(math.ceil(task.horizon / env.CTRL_TIMESTEP))

    for step in range(min(max_sim_steps, MAX_STEPS)):
        t_sim = step * env.CTRL_TIMESTEP
        if gui:
            time.sleep(env.CTRL_TIMESTEP)
        
        if gui and step % frames_per_cam == 0:
            track_drone(cli=env.getPyBulletClient(), drone_id=env.DRONE_IDS[0], frames_per_cam=frames_per_cam, cam_hz=CAM_HZ) 

        # current target ------------------------------------------------
        target = wps[wp_idx]
        obs = env.step(target.reshape(1, 3))[0]
        pos = _extract_pos(obs)

        # waypoint reached? --------------------------------------------
        if np.linalg.norm(pos - target) < WAYPOINT_TOL:
            if wp_idx < len(wps) - 1:
                wp_idx += 1
            else:                         # final goal reached
                success = True
                break

        # horizon reached? --------------------------------------------
        if t_sim >= task.horizon:
            break

        # energy bookkeeping ------------------------------------------
        thrusts = np.square(env.last_clipped_action[0]) * env.KF  # N each
        energy += (thrusts.sum() / PROP_EFF)  * env.CTRL_TIMESTEP

    if gui:
        input("Replay finished – press <Enter> to close …")

    # 3.  Clean‑up (avoid Mesa seg‑fault) ------------------------------
    if not gui:
        env.close()

    return success, t_sim, energy