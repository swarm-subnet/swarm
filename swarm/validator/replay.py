"""
swarm.validator.replay
──────────────────────
Deterministic *re‑execution* of a **trained Stable‑Baselines3 policy** on a
given `MapTask`, optionally with a PyBullet GUI.  The episode terminates when
either the environment signals `done` or a configurable safety horizon is
exceeded.

Returns
-------
success : bool
    Taken from the environment's ``info["success"]`` flag (if present) or
    from the first element of the "terminated" signal.
t_sim   : float
    Simulated time in seconds.
energy  : float
    Physics-based energy model: mass^1.5 × altitude + speed² + acceleration effects.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch as th
from stable_baselines3 import PPO

from swarm.utils.env_factory import make_env
from swarm.utils.gui_isolation import run_isolated
from swarm.constants import (
    DRONE_MASS, ENERGY_ALPHA, ENERGY_BETA, 
    ENERGY_DELTA, ENERGY_EFFICIENCY, ALTITUDE_SCALE
)

# ──────────────────────────────────────────────────────────────────────
# Public façade
# ──────────────────────────────────────────────────────────────────────
def replay_model(
    task,
    *,
    model_path: Path,
    sim_dt: float,
    horizon_sec: float,
    gui: bool = False,
) -> Tuple[bool, float, float]:

    return run_isolated(
        _replay_impl,
        task,
        model_path,
        sim_dt,
        horizon_sec,
        gui=gui,          # <- argument captured by run_isolated
    )


def calculate_enhanced_energy(obs, prev_obs, mass, dt):
    """
    Calculate energy using physics-based formula:
    E = [α×m^1.5×(1 + h/10000) + β×v² + δ×m×a] × dt / efficiency
    
    Args:
        obs: current observation
        prev_obs: previous observation (for acceleration calculation)
        mass: drone mass in kg
        dt: time step in seconds
    
    Returns:
        energy: energy consumed in Joules
    """
    # Physics-based energy model coefficients
    alpha = ENERGY_ALPHA       # hover power coefficient (W/kg^1.5)
    beta = ENERGY_BETA         # speed penalty coefficient (W·s²/m²)  
    delta = ENERGY_DELTA        # acceleration penalty coefficient (W·s²/m·kg)
    efficiency = ENERGY_EFFICIENCY # total system efficiency (motor×prop×battery)
    
    # Extract position and velocity from observation
    altitude = obs[2]  # z-position
    
    # Extract velocity components (adjust indices based on actual obs structure)
    # Assuming velocity is in obs[10:13] - adjust if different
    try:
        vx, vy, vz = obs[10], obs[11], obs[12]
    except IndexError:
        if prev_obs is not None:
            position = obs[0:3]
            prev_position = prev_obs[0:3]
            velocity = (position - prev_position) / dt
            vx, vy, vz = velocity[0], velocity[1], velocity[2]
        else:
            vx, vy, vz = 0.0, 0.0, 0.0
    
    # Calculate velocity magnitude squared (for speed term)
    v_squared = vx**2 + vy**2 + vz**2
    
    # Calculate acceleration magnitude
    if prev_obs is not None:
        try:
            # Try to get previous velocity from observation
            vx_prev, vy_prev, vz_prev = prev_obs[10], prev_obs[11], prev_obs[12]
        except IndexError:
            # Fallback: calculate from position
            prev_position = prev_obs[0:3]
            position = obs[0:3]
            prev_prev_position = prev_position - (position - prev_position)  # estimate
            prev_velocity = (prev_position - prev_prev_position) / dt
            vx_prev, vy_prev, vz_prev = prev_velocity[0], prev_velocity[1], prev_velocity[2]
        
        # Calculate acceleration components
        ax = (vx - vx_prev) / dt
        ay = (vy - vy_prev) / dt
        az = (vz - vz_prev) / dt
        
        # Acceleration magnitude
        a_magnitude = np.sqrt(ax**2 + ay**2 + az**2)
    else:
        a_magnitude = 0.0  # first step, no previous observation
    
    # Apply enhanced energy formula
    altitude_factor = 1.0 + altitude / ALTITUDE_SCALE  # thin air effect
    
    # Power components (in Watts)
    hover_power = alpha * mass**1.5 * altitude_factor
    speed_power = beta * v_squared
    accel_power = delta * mass * a_magnitude
    
    total_power = hover_power + speed_power + accel_power
    
    # Convert power to energy (Joules) accounting for efficiency
    energy = (total_power * dt) / efficiency
    
    return energy


# ──────────────────────────────────────────────────────────────────────
# Implementation (runs in the *current* process OR a child, depending
# on the gui_isolation wrapper).
# ──────────────────────────────────────────────────────────────────────
def _replay_impl(
    task,
    model_path: Path,
    sim_dt: float,
    horizon_sec: float,
    *,
    gui: bool = False,
):
    # 1 ─ environment ---------------------------------------------------
    env = make_env(task, gui=gui)
    model = PPO.load(model_path, device=th.device("cpu"))
    model.set_env(env)

    # 2 ─ episode loop --------------------------------------------------
    step    = 0
    done    = False
    success = False
    energy  = 0.0
    obs, _  = env.reset()
    
    # Enhanced energy calculation variables
    prev_obs = None
    drone_mass = DRONE_MASS  # kg - adjust based on actual drone specifications

    while not done:
        act, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, info = env.step(act)
        done = bool(terminated[0])

        # success flag (if env provides it)
        if "success" in info[0]:
            success = bool(info[0]["success"])

        # Enhanced physics-based energy calculation
        energy += calculate_enhanced_energy(obs, prev_obs, drone_mass, sim_dt)
        
        # Store current observation for next step
        prev_obs = obs.copy()

        step += 1
        if step * sim_dt > 1.5 * horizon_sec:   # safety break
            break

        if gui:
            time.sleep(sim_dt)

    if not gui:
        env.close()

    return success, step * sim_dt, energy
