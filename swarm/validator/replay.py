"""
swarm.validator.replay
──────────────────────
*Deterministic* re‑execution of a miner‑supplied FlightPlan.

Key changes vs the previous version
-----------------------------------
1.  **The FlightPlan is now applied verbatim**:
        • Environment uses ActionType.RPM  
        • One‑to‑one mapping “timestamp → sim‑step” (no interpolation)
2.  Way‑point logic is gone – success is evaluated purely from the flown path.
3.  Energy accounting unchanged.

This guarantees the miner and validator experience the *identical* physics
history – if they still diverge you’ll know it is a true determinism issue.
"""
from __future__ import annotations

import math
import time
from typing import Tuple, List

import numpy as np
import pybullet as p
import pybullet_data
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

from swarm.utils.gui_isolation import run_isolated
from swarm.utils.drone import track_drone
from swarm.validator.env_builder import build_world
from swarm.protocol import MapTask, FlightPlan, RPMCmd

# ───────── constants ─────────
WAYPOINT_TOL = 0.20      # success sphere
HOVER_SEC    = 5.0
CAM_HZ       = 60
PROP_EFF     = 0.60
# ─────────────────────────────


# ───────────────── public façade ─────────────────
def replay_once(task: MapTask,
                plan: FlightPlan,
                *,
                gui: bool = False
                ) -> Tuple[bool, float, float]:
    """Run in an isolated subprocess when required."""
    return run_isolated(_replay_once_impl, task, plan, gui=gui)


# ───────────────── implementation ─────────────────
def _replay_once_impl(task: MapTask,
                      plan: FlightPlan,
                      *,
                      gui: bool = False
                      ) -> Tuple[bool, float, float]:

    # 1 ─ environment ---------------------------------------------------
    env = HoverAviary(gui=gui,
                      obs=ObservationType.KIN,
                      act=ActionType.RPM)          # ← we feed raw RPM
    cli = env.getPyBulletClient()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Cleaner viewer
    if gui:
        for flag in (p.COV_ENABLE_SHADOWS, p.COV_ENABLE_GUI):
            p.configureDebugVisualizer(flag, 0, physicsClientId=cli)

    # Sync time‑step with the task
    env.CTRL_TIMESTEP = task.sim_dt
    env.CTRL_FREQ     = int(round(1.0 / task.sim_dt))

    env.reset(seed=task.map_seed)
    build_world(task.map_seed, cli, task.goal)

    # 2 ─ turn the FlightPlan into a step‑indexed RPM table -------------
    max_steps = int(math.ceil(task.horizon / task.sim_dt))
    rpm_table = _plan_to_table(plan.commands, max_steps, task.sim_dt)

    # 3 ─ main replay loop ---------------------------------------------
    frames_per_cam = max(1, int(round(1.0 / (task.sim_dt * CAM_HZ))))
    hover_elapsed  = 0.0
    energy         = 0.0
    success        = False
    goal           = np.asarray(task.goal, dtype=float)

    for k in range(max_steps):
        t_sim   = k * task.sim_dt
        rpm_vec = rpm_table[k]

        obs, *_ = env.step(rpm_vec[None, :])       # shape (1,4)
        pos     = obs[0, :3]

        # camera follow
        if gui and k % frames_per_cam == 0:
            track_drone(cli, env.DRONE_IDS[0], frames_per_cam, CAM_HZ)

        # success logic
        if np.linalg.norm(pos - goal) < WAYPOINT_TOL:
            hover_elapsed += task.sim_dt
            if hover_elapsed >= HOVER_SEC:
                success = True
                break
        else:
            hover_elapsed = 0.0

        # energy bookkeeping (same formula as before)
        energy += (np.square(rpm_vec).sum() * env.KF / PROP_EFF) * task.sim_dt

        if gui:
            time.sleep(task.sim_dt)

    if not gui:
        env.close()

    return success, t_sim, energy


# ───────────────── helpers ───────────────────────
def _plan_to_table(cmds: List[RPMCmd],
                   max_steps: int,
                   sim_dt: float
                   ) -> np.ndarray:
    """
    Convert the ragged list of (t, rpm) commands into a fully populated
    (max_steps × 4) numpy array, holding the last known RPM once the plan ends.
    """
    table = np.zeros((max_steps, 4), dtype=float)
    last  = np.zeros(4, dtype=float)
    idx   = 0

    for cmd in cmds:
        k = int(round(cmd.t / sim_dt))
        k = max(0, min(k, max_steps - 1))          # clip to valid range

        # fill gap up to (but not including) k
        if k > idx:
            table[idx:k, :] = last
        # new rpm at k
        last = np.asarray(cmd.rpm, dtype=float)
        table[k, :] = last
        idx = k + 1

    # pad remaining steps
    if idx < max_steps:
        table[idx:, :] = last

    return table
