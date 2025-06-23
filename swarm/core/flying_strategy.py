"""
neurons.miner – flying_strategy(task)
─────────────────────────────────────
Generate an open‑loop list of rotor‑RPM commands for one Crazyflie.

Key points
----------
* 5 s hover over the final waypoint (HOVER_SEC).
* Uses the new `make_env` factory to avoid boilerplate duplication.
"""
from __future__ import annotations

import time
from typing import List, Sequence, Tuple

import numpy as np
import pybullet as p
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

from swarm.utils.gui_isolation import run_isolated
from swarm.core.env_builder import build_world
from swarm.utils.env_factory import make_env       # ← NEW
from swarm.protocol import MapTask, RPMCmd
from swarm.core.drone import track_drone

# ───────── parameters & constants ─────────
from swarm.constants import (
    SAFE_Z,
    GOAL_TOL,
    HOVER_SEC,
    CAM_HZ,
)
# ───────────────────────────────────────────


# ---------- public API ---------------------------------------------------
def flying_strategy(task: MapTask, *, gui: bool = False) -> List[RPMCmd]:
    """Thin wrapper that delegates to the real body through run_isolated."""
    return run_isolated(_flying_strategy_impl, task, gui=gui)


# ---------- implementation ----------------------------------------------
def _flying_strategy_impl(task: MapTask, *, gui: bool = False) -> List[RPMCmd]:
    # 1 ─ environment ----------------------------------------------------
    env = make_env(task, gui=gui, raw_rpm=False)   # ← one‑liner factory
    cli = env.getPyBulletClient()

    # 2 ─ way‑points -----------------------------------------------------
    start_xyz = np.array(task.start, dtype=float)
    gx, gy, gz = task.goal
    safe_z = max(SAFE_Z, start_xyz[2], gz)

    wps = [
        np.array([*start_xyz[:2], safe_z]),
        np.array([gx, gy, safe_z]),
        np.array([gx, gy, gz]),  # final
    ]
    wp_idx = 0

    # camera bookkeeping
    if gui:
        frames_per_cam = max(1, int(round(1.0 / (task.sim_dt * CAM_HZ))))
        step_counter = 0

    # 3 ─ control loop ---------------------------------------------------
    t_sim = 0.0
    hover_elapsed = 0.0       # NEW
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
                drone_id=env.DRONE_IDS[0]
            )

        # log motor command
        _record_cmd(rpm_log, env.last_clipped_action[0], t_sim)

        # waypoint / hover logic
        dist = np.linalg.norm(pos - target)
        if wp_idx < len(wps) - 1:
            if dist < GOAL_TOL:
                wp_idx += 1
        else:
            if dist < GOAL_TOL:
                hover_elapsed += task.sim_dt
                if hover_elapsed >= HOVER_SEC + 2:
                    extra_counter += 1
                    if extra_counter >= int(1.0 / task.sim_dt):  # 1 extra second
                        break
            else:
                hover_elapsed = 0.0  # drifted out – reset timer

        # bookkeeping
        t_sim += task.sim_dt
        if gui:
            time.sleep(task.sim_dt)
            step_counter += 1

    # 4 ─ clean‑up -------------------------------------------------------
    if not gui:  # head‑less – safe to close Bullet
        env.close()

    return rpm_log


# ---------- helpers ------------------------------------------------------
def _record_cmd(buffer: List[RPMCmd], rpm_vec: Sequence[float], t: float) -> None:
    """Convert the 4‑element vector into an RPMCmd dataclass entry."""
    rpm_tuple: Tuple[float, float, float, float] = tuple(float(x) for x in rpm_vec)
    buffer.append(RPMCmd(t=t, rpm=rpm_tuple))
