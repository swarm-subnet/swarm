# The MIT License (MIT)
# Copyright © 2026 Swarm

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Dock take-off, return home and landing (task 9): where the dock stands, and the flights it makes by itself.

The dock owns the flight phase. While it flies the drone (take-off, the flight home, the landing) its setpoint
replaces the model's; once the drone is up, the model flies until it asks to come home.

Each seed stands the Dock 3 (task 3's model) on its own open spot, found in the built world with rays: inside the
fence by DJI's take-off margin, clear of everything standing by DJI's installation distance, on ground its concrete
base can level. Take-off opens the lids, then climbs straight up to the patrol height. Return home is DJI's dock
return: its height is fixed when it is pressed, the drone reaches it first, flies straight to the point above the
pad, and descends onto the pad.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import numpy as np
import pybullet as p

from . import dock3, park
from .contract import (
    FLIGHT_PHASES,
    MAX_CLIMB_MPS,
    MAX_DESCENT_MPS,
    MAX_HORIZONTAL_MPS,
    PATROL_HEIGHT_M,
    STATE_SLICES,
    Command,
    put,
)
from .episode import Setpoint, SolarEpisode
from .fixed_order import norm

DRONE_REST_M = 0.0                     # dock_position is the aircraft's own resting point on the pad
ARRIVE_M = 0.5                         # close enough to a target height, and DJI's landing start above the pad
NEAR_M = 5.5                           # FlightHub 2: nearer the dock than this, a return keeps any height above its own
LANDED_SPEED_MPS = 0.3                 # slower than this on the pad counts as landed
TOUCHDOWN_MPS = 0.4                    # the slowest descent, so the last centimetres still close
GAIN_PER_S = 1.0                       # speed asked for per metre still to go
CENTRED_M = 0.3                        # the descent pauses while the aircraft is further than this off the pad
BRAKE_MPS2 = 1.5                       # approach no faster than a stop at this deceleration allows, inside the setpoint's own 2 m/s2

SITE_SEED_STREAM = 0xD0C9              # the dock's own stream, so its spot never moves another part's draws
FENCE_MARGIN_M = 10.0                  # DJI refuses take-off this near the flight area's edge, which never lies inside the fence
CLEAR_M = 3.4                          # DJI: anything under 5 m tall stands over 2.5 m from the dock, whose open lids reach 0.88 m
INSTALL_M = 1.5                        # the ground the slope is read over: DJI's 2.6 x 3 m installation area
MAX_SLOPE_DEG = 20.0                   # steeper ground than any open patch inside the park, where the base would not stand
COARSE_STEP_M = 1.0                    # a first sparse look that turns most spots away cheaply
RAY_STEP_M = 0.15                      # rays this close cannot pass either side of a 0.26 m trunk
RAY_REACH_M = 1000.0
SITE_BATCH = 256
SITE_BATCHES = 80


def _disc(step: float) -> np.ndarray:
    """Offsets on a square grid of the given step that lie within CLEAR_M of the centre."""
    r = np.arange(-CLEAR_M, CLEAR_M + 1e-9, step)
    u, v = np.meshgrid(r, r)
    keep = u * u + v * v <= CLEAR_M * CLEAR_M
    return np.column_stack([u[keep], v[keep]])


_COARSE = _disc(COARSE_STEP_M)
_FINE = _disc(RAY_STEP_M)


def _inside(polygon: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Which points lie inside the closed polygon, by counting the sides a ray from each crosses."""
    a, b = polygon, np.roll(polygon, -1, axis=0)
    x, y = points[:, :1], points[:, 1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        crosses = ((a[:, 1] > y) != (b[:, 1] > y)) & (x < a[:, 0] + (y - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1]))
    return crosses.sum(axis=1) % 2 == 1


def _edge_distance(polygon: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest side of the closed polygon."""
    a = polygon
    ab = np.roll(polygon, -1, axis=0) - a
    rel = points[:, None, :] - a[None, :, :]
    t = np.clip((rel * ab).sum(axis=2) / (ab * ab).sum(axis=1), 0.0, 1.0)
    return np.min(np.linalg.norm(rel - t[..., None] * ab, axis=2), axis=1)


def _ground(cli: int, spot: np.ndarray, offsets: np.ndarray, terrain_uids: frozenset) -> Optional[np.ndarray]:
    """The ground points under rays dropped around the spot, or None when any ray meets something else first."""
    xy = spot + offsets
    hits = p.rayTestBatch([[x, y, RAY_REACH_M] for x, y in xy], [[x, y, -RAY_REACH_M] for x, y in xy],
                          physicsClientId=cli)
    if any(hit[0] not in terrain_uids for hit in hits):
        return None
    return np.array([hit[3] for hit in hits], dtype=float)


def _slope_deg(ground: np.ndarray, spot: np.ndarray) -> float:
    """Tilt of the plane fitted through the ground points within INSTALL_M of the spot."""
    near = ground[np.hypot(*(ground[:, :2] - spot).T) <= INSTALL_M]
    # Least squares from exactly rounded sums about the points' centre, not LAPACK, whose rounding follows the CPU.
    u, v, w = ((x - math.fsum(x) / len(x)).tolist() for x in (near[:, 0] - spot[0], near[:, 1] - spot[1], near[:, 2]))
    uu, uv, vv = math.fsum(x * x for x in u), math.fsum(x * y for x, y in zip(u, v)), math.fsum(y * y for y in v)
    uw, vw = math.fsum(x * z for x, z in zip(u, w)), math.fsum(y * z for y, z in zip(v, w))
    det = uu * vv - uv * uv
    a, b = (uw * vv - vw * uv) / det, (vw * uu - uw * uv) / det
    return math.degrees(math.atan(math.hypot(a, b)))


def find_site(cli: int, seed: int, fence: np.ndarray, terrain_uids: frozenset,
              passable: np.ndarray) -> Tuple[float, float]:
    """A spot for the dock drawn from the seed's own stream: FENCE_MARGIN_M inside the fence, CLEAR_M from every
    body standing and from every passable outline, on ground no steeper than MAX_SLOPE_DEG.

    Only the terrain may meet the rays dropped around the spot, so the check reads whatever the world holds this
    seed, wherever the park put it, and the column the drone climbs through is clear with it.
    """
    rng = np.random.default_rng([SITE_SEED_STREAM, int(seed)])
    low, high = fence.min(axis=0) + FENCE_MARGIN_M, fence.max(axis=0) - FENCE_MARGIN_M
    for _ in range(SITE_BATCHES):
        spots = rng.uniform(low, high, size=(SITE_BATCH, 2))
        spots = spots[_inside(fence, spots)]
        spots = spots[_edge_distance(fence, spots) >= FENCE_MARGIN_M]
        for spot in spots:
            if np.any(np.hypot(*(passable[:, :2] - spot).T) < passable[:, 2] + CLEAR_M):
                continue
            if _ground(cli, spot, _COARSE, terrain_uids) is None:
                continue
            ground = _ground(cli, spot, _FINE, terrain_uids)
            if ground is not None and _slope_deg(ground, spot) <= MAX_SLOPE_DEG:
                return float(spot[0]), float(spot[1])
    raise RuntimeError(f"no open ground for the dock in seed {seed}")


def reset(env: Any, ep: SolarEpisode) -> None:
    """Stand the dock on this seed's spot, set the drone down on its pad, and tell the environment the pad is a
    landing."""
    cli = env.CLIENT
    x, y = find_site(cli, ep.seed, ep.fence, ep.terrain_uids, park.passable_outlines(ep))
    model = dock3.spawn(env, x, y, yaw=0.0)
    pad, yaw = dock3.rest_pose(model)
    ep.dock_uid = model.uid
    ep.dock_position = pad
    ep.dock_yaw = yaw
    ep.phase = "docked"
    ep.dock = {"model": model, "return_z": 0.0, "at_height": False}
    drone = int(env.DRONE_IDS[0])
    p.resetBasePositionAndOrientation(drone, (pad + [0.0, 0.0, DRONE_REST_M]).tolist(),
                                      p.getQuaternionFromEuler([0.0, 0.0, ep.dock_yaw]), physicsClientId=cli)
    p.resetBaseVelocity(drone, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], physicsClientId=cli)
    # Only the pad takes a landing; the body and the lids are a solid the aircraft can hit.
    env._end_platform_uids = [model.pad_uid]
    env.task.start = tuple(float(v) for v in pad)
    env.task.goal = tuple(float(v) for v in pad)
    env.GOAL_POS = pad.copy()


def _return_height(ep: SolarEpisode, pos: np.ndarray) -> float:
    """DJI's return height, fixed when return home is pressed: the patrol height, or nearer than NEAR_M to the dock
    the current height when that is higher."""
    patrol_z = float(ep.dock_position[2]) + PATROL_HEIGHT_M
    if math.hypot(*(ep.dock_position[:2] - pos[:2])) > NEAR_M:
        return patrol_z
    return max(patrol_z, float(pos[2]))


def command(env: Any, ep: SolarEpisode, cmd: Command) -> None:
    """Act on the take-off, return home and cancel buttons, each only in the phase where the real dock takes it."""
    if cmd.take_off and ep.phase == "docked":
        ep.phase = "taking_off"
        ep.outcome.took_off = True
    elif cmd.return_home and ep.phase == "flying":
        ep.phase = "returning"
        ep.outcome.returned_by_model = True
        ep.dock["return_z"] = _return_height(ep, np.asarray(env.pos[0], dtype=float))
        ep.dock["at_height"] = False
    elif cmd.cancel_return and ep.phase in ("returning", "landing"):
        ep.phase = "flying"
        ep.outcome.returned_by_model = False


def _towards(delta: np.ndarray, limit: float) -> np.ndarray:
    """A velocity along delta, proportional to its length, capped at limit and slow enough to stop in time."""
    distance = norm(delta)
    if distance < 1e-6:
        return np.zeros_like(delta)
    return delta / distance * min(limit, GAIN_PER_S * distance, math.sqrt(2.0 * BRAKE_MPS2 * distance))


def _climb(rise: float) -> float:
    """A vertical speed towards a height rise metres above, inside the climb and descent limits."""
    return float(np.clip(GAIN_PER_S * rise, -MAX_DESCENT_MPS, MAX_CLIMB_MPS))


def autopilot(env: Any, ep: SolarEpisode) -> Optional[Setpoint]:
    """The dock's own setpoint in the phases it flies, or None while the model has the drone."""
    if ep.phase in ("docked", "landed"):
        return Setpoint(motors_on=False)
    if ep.phase == "flying":
        return None
    if ep.phase == "taking_off" and ep.dock["model"].opening < 1.0:
        return Setpoint(motors_on=False)
    pos = np.asarray(env.pos[0], dtype=float)
    east, north = (float(v) for v in _towards(ep.dock_position[:2] - pos[:2], MAX_HORIZONTAL_MPS))
    if ep.phase == "taking_off":
        rise = float(ep.dock_position[2]) + PATROL_HEIGHT_M - pos[2]
        if abs(rise) < ARRIVE_M:
            ep.phase = "flying"
        return Setpoint((east, north, _climb(rise)))
    if ep.phase == "returning":
        rise = ep.dock["return_z"] - pos[2]
        if not ep.dock["at_height"] and abs(rise) >= ARRIVE_M:
            return Setpoint((0.0, 0.0, _climb(rise)))
        ep.dock["at_height"] = True
        if math.hypot(*(ep.dock_position[:2] - pos[:2])) < ARRIVE_M:
            ep.phase = "landing"
        return Setpoint((east, north, _climb(rise)))
    height = pos[2] - DRONE_REST_M - ep.dock_position[2]
    vz = -min(MAX_DESCENT_MPS, max(TOUCHDOWN_MPS, GAIN_PER_S * height))
    if math.hypot(*(ep.dock_position[:2] - pos[:2])) > CENTRED_M:
        vz = 0.0
    return Setpoint((east, north, vz))


def update(env: Any, ep: SolarEpisode) -> None:
    """Open the lids for the flight, shut them at rest, hold a resting aircraft still on the pad, and close the patrol
    once a landing drone rests on the pad."""
    model = ep.dock["model"]
    dock3.move_lids(env, model, ep.phase not in ("docked", "landed"), float(env.CTRL_TIMESTEP))
    if ep.phase == "docked" or (ep.phase == "taking_off" and model.opening < 1.0):
        _hold(env, ep)
    if ep.phase != "landing" or not env._platform_hit:
        return
    if norm(env.vel[0]) < LANDED_SPEED_MPS:
        ep.phase = "landed"
        ep.outcome.landed_in_dock = True
        ep.end("landed")


def _hold(env: Any, ep: SolarEpisode) -> None:
    """Keep the aircraft on its resting pose while its motors are off: the pad's V holds the real one still, where
    four small feet in the pad's V slowly creep at the patrol's physics step."""
    drone = int(env.DRONE_IDS[0])
    orn = p.getQuaternionFromEuler([0.0, 0.0, ep.dock_yaw])
    p.resetBasePositionAndOrientation(drone, ep.dock_position.tolist(), orn, physicsClientId=env.CLIENT)
    p.resetBaseVelocity(drone, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], physicsClientId=env.CLIENT)


def observe(env: Any, ep: SolarEpisode, state: np.ndarray) -> None:
    """The flight phase, as the dock reports it."""
    put(state, STATE_SLICES, "flight_phase", FLIGHT_PHASES.index(ep.phase))
