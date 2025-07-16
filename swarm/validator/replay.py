# swarm/validator/replay.py
"""
swarm.validator.replay
──────────────────────
Deterministic re‑execution of a miner‑supplied FlightPlan.

• Any physical contact between the drone and another object is considered a
  collision ⇒ the episode is flagged as a failure ⇒ flight_reward() returns 0.
"""
from __future__ import annotations

import time
from typing import Tuple, List

import numpy as np
import pybullet as p

from gym_pybullet_drones.utils.enums import ObservationType, ActionType

from swarm.utils.env_factory import make_env
from swarm.utils.gui_isolation import run_isolated
from swarm.core.drone import track_drone
from swarm.protocol import MapTask, FlightPlan, RPMCmd

# ───────── constants ─────────
from swarm.constants import (
    CAM_HZ,          # camera follow rate
    PROP_EFF,        # propeller efficiency
    WAYPOINT_TOL,    # way-point success tolerance
    LANDING_PLATFORM_RADIUS as _PR,  # platform radius constant
    STABLE_LANDING_SEC,  # required stable landing duration
    PLATFORM,        # platform mode toggle
    HOVER_SEC,       # legacy hover time for visual-only mode
)
# ─────────────────────────────


# ───────────────── public façade ─────────────────
def replay_once(
    task: MapTask,
    plan: FlightPlan,
    *,
    gui: bool = False,
) -> Tuple[bool, float, float]:
    """Run in an isolated subprocess when required."""
    return run_isolated(_replay_once_impl, task, plan, gui=gui)


# ───────────────── implementation ─────────────────
def _replay_once_impl(
    task: MapTask,
    plan: FlightPlan,
    *,
    gui: bool = False,
) -> Tuple[bool, float, float]:

    # 1 ─ environment ---------------------------------------------------
    env = make_env(task, gui=gui, raw_rpm=True)   # RPM‑controlled
    cli = env.getPyBulletClient()                 # physicsClientId (int)
    
    # Retrieve avian simulation system if enabled
    bird_system = getattr(env, '_bird_system', None)
    
    # Retrieve atmospheric wind simulation system if enabled
    wind_system = getattr(env, '_wind_system', None)
    
    # Retrieve moving platform system if enabled
    moving_platform_system = getattr(env, '_moving_platform_system', None)

    # 2 ─ turn the FlightPlan into a step‑indexed RPM table -------------
    last_t = plan.commands[-1].t
    max_steps = int(round(last_t / task.sim_dt)) + 1
    rpm_table = _plan_to_table(plan.commands, max_steps, task.sim_dt)

    # 3 ─ main replay loop ---------------------------------------------
    frames_per_cam = max(1, int(round(1.0 / (task.sim_dt * CAM_HZ))))
    energy = 0.0
    success = False
    collided = False
    stable_landing_time = 0.0  # accumulated stable landing time
    hover_elapsed = 0.0        # for legacy hover mode
    goal = np.array(task.goal, dtype=np.float32)
    drone_id = env.DRONE_IDS[0]

    for k in range(max_steps):
        t_sim = k * task.sim_dt
        rpm_vec = rpm_table[k]
        obs, *_ = env.step(rpm_vec[None, :])          # shape (1,4)
        pos = obs[0, :3]
        
        # Update avian behavioral states
        if bird_system:
            bird_system.update(task.sim_dt)

        # Update wind system states
        if wind_system:
            wind_system.update(task.sim_dt)
            
            # Apply wind force to drone
            wind_force = wind_system.get_wind_force(pos)
            if np.linalg.norm(wind_force) > 0.01:  # Only apply if force is significant
                # Apply wind force as external force to drone
                p.applyExternalForce(
                    drone_id,
                    -1,  # Link index (-1 for base)
                    wind_force.tolist(),
                    pos.tolist(),
                    p.WORLD_FRAME,
                    physicsClientId=cli
                )

        # Update moving platform system
        if moving_platform_system:
            moving_platform_system.step(task.sim_dt)
            # Update goal position to track moving platform
            goal[:] = moving_platform_system.pos

        # camera follow
        if gui and k % frames_per_cam == 0:
            track_drone(cli, drone_id)

        # energy bookkeeping
        energy += (np.square(rpm_vec).sum() * env.KF / PROP_EFF) * task.sim_dt

        # -------------------------------------------------------------
        # Success & collision handling (platform vs legacy modes)
        # -------------------------------------------------------------
        if PLATFORM:
            # ─── collision check (ignore contacts on the landing platform) ───
            if not collided:
                contacts = p.getContactPoints(bodyA=drone_id, physicsClientId=cli)
                if contacts:
                    bird_collision = False
                    allowed_contact = True  # assume contact is allowed until proven otherwise

                    for cp in contacts:
                        body_b = cp[2]

                        # Detect avian collision events with drone
                        if bird_system and body_b in bird_system.bird_ids:
                            bird_collision = True
                            bird_system.handle_bird_collision(body_b)
                            break

                        # Evaluate contact relative to platform only if not avian
                        if not bird_collision:
                            cpos = cp[5]
                            if isinstance(cpos, (list, tuple)) and len(cpos) >= 3:
                                cx, cy, cz = cpos[:3]
                                # Use circular platform collision detection - back to original
                                platform_radius = _PR * 0.9  # Platform is now circular and smaller
                                horiz = np.linalg.norm([cx - goal[0], cy - goal[1]])
                                vert = abs(cz - goal[2])
                                # Contacts within platform radius & low vertical offset are allowed
                                if horiz < platform_radius + 0.05 and vert < 0.3:
                                    continue
                                # Any other contact is disallowed
                                allowed_contact = False
                            break

                    if bird_collision or not allowed_contact:
                        collided = True
                        break  # stop episode early

            # ─── landing success logic on platform ───
            horizontal_distance = np.linalg.norm(pos[:2] - goal[:2])
            vertical_distance = abs(pos[2] - goal[2])

            # Use circular platform detection - back to original circular design
            platform_radius = _PR * 0.9  # Platform is now circular and smaller
            landing_radius = platform_radius * 0.8 * 0.9  # Landing area radius (80% of platform * 90% for TAO logo)
            on_tao_logo = (
                horizontal_distance < landing_radius
                and vertical_distance < 0.3
                and pos[2] >= goal[2] - 0.1
            )
        
            if on_tao_logo:
                stable_landing_time += task.sim_dt
                if stable_landing_time >= STABLE_LANDING_SEC:
                    success = True
                    break
            else:
                stable_landing_time = 0.0

        else:
            # ─── legacy visual-only mode (hover) ───
            if not collided:
                contacts = p.getContactPoints(bodyA=drone_id, physicsClientId=cli)
                if contacts:
                    # Treat ANY contact as failure in legacy mode (except avian entities)
                    bird_collision = False
                    for cp in contacts:
                        body_b = cp[2]
                        if bird_system and body_b in bird_system.bird_ids:
                            bird_collision = True
                            bird_system.handle_bird_collision(body_b)
                            break
                    collided = True
                    if bird_collision:
                        break

            # Hover success condition near goal
            if np.linalg.norm(pos - goal) < WAYPOINT_TOL:
                hover_elapsed += task.sim_dt
                if hover_elapsed >= HOVER_SEC:
                    success = True
                    break
            else:
                hover_elapsed = 0.0

        if gui:
            time.sleep(task.sim_dt)

    if not gui:
        env.close()

    # Any collision ⇒ failure (success = False)
    if collided:
        success = False

    return success, t_sim, energy


# ───────────────── helpers ───────────────────────
def _plan_to_table(
    cmds: List[RPMCmd],
    max_steps: int,
    sim_dt: float,
) -> np.ndarray:
    """
    Convert the ragged list of (t, rpm) commands into a fully populated
    (max_steps × 4) numpy array, holding the last known RPM once the plan ends.
    """
    table = np.zeros((max_steps, 4), dtype=float)
    last = np.zeros(4, dtype=float)
    idx = 0

    for cmd in cmds:
        k = int(cmd.t / sim_dt + 1e-9)
        k = max(0, min(k, max_steps - 1))  # clip

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
