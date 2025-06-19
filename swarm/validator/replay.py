# swarm/validator/replay.py
from __future__ import annotations
import numpy as np
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from swarm.protocol import MapTask, FlightPlan, RPMCmd
from swarm.validator.env_builder import build_world

KF               = 1.91e-6
PROP_EFF         = 0.60
MAX_STEPS        = 60_000

def _inject_cmd(env: HoverAviary, cmd: RPMCmd):
    env.last_clipped_action = np.array([cmd.rpm], dtype=float)

def replay_once(task: MapTask, plan: FlightPlan):
    env = HoverAviary(gui=False, obs=ObservationType.KIN, act=ActionType.RPM)
    build_world(task.map_seed, env.getPyBulletClient())

    next_i, energy, success = 0, 0.0, False
    goal = np.array(task.goal)

    for step in range(MAX_STEPS):
        t_sim = step * env.CTRL_TIMESTEP
        while next_i < len(plan.commands) and plan.commands[next_i].t <= t_sim:
            _inject_cmd(env, plan.commands[next_i]); next_i += 1
        obs, _ = env.step(np.zeros((1,3)))
        pos     = obs[0,:3]

        thrusts = env.last_clipped_action[0]**2 * KF
        energy += thrusts.sum()/PROP_EFF * env.CTRL_TIMESTEP
        if np.linalg.norm(pos - goal) < 0.3:
            success = True; break
        if t_sim >= task.horizon: break
    env.close()
    return success, t_sim, energy
