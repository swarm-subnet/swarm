"""
flying_strategy(task)
─────────────────────
Empirically generate a complete list of rotor‑RPM commands that will
pilot a single drone through the map described by `MapTask`, **using the
same three‑way‑point logic** we analysed earlier (take‑off → flat cruise
→ descent).  The list is returned as `List[RPMCmd]` so that
`swarm.validator.replay_once` can replay the exact same manoeuvre during
validation.

Changes vs. previous draft
──────────────────────────
* **No manual obstacle spawning.**  We now recreate the world with the
  validator’s canonical helper: `build_world(task.map_seed, cli)`.
* **Respect `task.start`, `task.goal`, `task.sim_dt`.**  The drone takes
  off from `task.start`, the controller period matches `task.sim_dt`,
  and we record RPM commands at that cadence.
* **Fix field names in `RPMCmd`.**  The dataclass has `t`, not `time`:
  `_record_cmd()` now uses `t=` and converts floats → ints as required
  by the type annotation `Tuple[int, int, int, int]`.
"""

from __future__ import annotations

import time
from typing import List, Sequence, Tuple

import numpy as np
import pybullet as p
import pybullet_data
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from swarm.utils.drone import track_drone, safe_disconnect_gui

from swarm.validator.env_builder import build_world  # authoritative map builder
from swarm.protocol import MapTask, RPMCmd  # type: ignore
import numpy as np, pybullet as p, pybullet_data
# ────────────────────── parameters & constants ──────────────────────
SAFE_Z: float   = 2.0  # cruise altitude (m)
GOAL_TOL: float = 0.2   # waypoint acceptance sphere (m)
CAM_HZ:  int    = 60
# -------------------------------------------------------------------
# main API
# -------------------------------------------------------------------

def flying_strategy(task: MapTask, *, gui: bool = False) -> List[RPMCmd]:
    """Generate an open-loop list of RPM commands for one drone.

    The drone follows three way-points (vertical climb → flat cruise →
    descent) under the built-in PID position controller.  Every control
    tick happens exactly at `task.sim_dt`, and we log the 4-motor RPM
    vector so that the validator can replay them verbatim.

    When `gui=True`, the PyBullet camera tracks the drone at 60 Hz,
    mirroring `replay_once`.
    """
    # 1 ─ initialise simulation --------------------------------------------
    env = HoverAviary(gui=gui,
                      record=False,
                      obs=ObservationType.KIN,
                      act=ActionType.PID)
    cli = env.getPyBulletClient()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    if gui:                                              # tidy viewer
        for flag in (p.COV_ENABLE_SHADOWS, p.COV_ENABLE_GUI):
            p.configureDebugVisualizer(flag, 0, physicsClientId=cli)

    # 2 ─ reset environment *first* ----------------------------------------
    _ = env.reset(seed=task.map_seed)

    # 3 ─ then build deterministic obstacles -------------------------------
    build_world(task.map_seed, cli)

    # 4 ─ sync controller period with the task -----------------------------
    env.CTRL_TIMESTEP = task.sim_dt
    env.CTRL_FREQ     = int(round(1.0 / task.sim_dt))

    # 5 ─ place drone at the requested take-off pose -----------------------
    start_xyz = np.array(task.start, dtype=float)
    start_quat = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
    p.resetBasePositionAndOrientation(env.DRONE_IDS[0],         # type: ignore[attr-defined]
                                      start_xyz,
                                      start_quat,
                                      physicsClientId=cli)

    # 6 ─ construct way-points --------------------------------------------
    gx, gy, gz = task.goal
    safe_z = max(SAFE_Z, start_xyz[2], gz)        # never descend below ends
    wps = [np.array([*start_xyz[:2], safe_z]),    # climb
           np.array([gx, gy, safe_z ]),           # cruise
           np.array([gx, gy, gz     ])]           # final descent
    wp_idx = 0

    # ── camera bookkeeping ------------------------------------------------
    if gui:
        frames_per_cam = max(1, int(round(1.0 / (task.sim_dt * CAM_HZ))))
        step_counter   = 0

    # 7 ─ main loop --------------------------------------------------------
    t_sim   = 0.0
    rpm_log: List[RPMCmd] = []

    while t_sim < task.horizon:
        target = wps[wp_idx]

        # physics step + low-level PID
        obs, *_ = env.step(target.reshape(1, 3))
        pos = obs[0, :3]

        # camera follow
        if gui and step_counter % frames_per_cam == 0:
            track_drone(cli=cli,
                        drone_id=env.DRONE_IDS[0],
                        frames_per_cam=frames_per_cam,
                        cam_hz=CAM_HZ)

        # log applied motor command
        _record_cmd(rpm_log, env.last_clipped_action[0], t_sim)

        # waypoint switching
        if np.linalg.norm(pos - target) < GOAL_TOL:
            if wp_idx < len(wps) - 1:
                wp_idx += 1
            else:                                   # mission accomplished
                break

        # bookkeeping
        t_sim += task.sim_dt
        if gui:
            time.sleep(task.sim_dt)
            step_counter += 1
    if gui:
        safe_disconnect_gui(cli)
    else:
        env.close()
    return rpm_log

# -------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------

def _record_cmd(buffer: List[RPMCmd], rpm_vec: Sequence[float], t: float) -> None:
    """Convert the 4‑element motor‑RPM array into an `RPMCmd` dataclass.

    The user’s schema is `RPMCmd(t: float, rpm: Tuple[int,int,int,int])`.
    We therefore:
    1. **Round** each float RPM to the nearest integer (type safety).
    2. Store them as a 4‑tuple under the field `rpm`.
    """

    rpm_tuple: Tuple[int, int, int, int] = tuple(int(round(x)) for x in rpm_vec)  # type: ignore[arg-type]
    buffer.append(RPMCmd(t=t, rpm=rpm_tuple))