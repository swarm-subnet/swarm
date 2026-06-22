from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import numpy as np

from swarm.constants import (
    INTERCEPTOR_DEPTH_RES,
    MAX_RAY_DISTANCE,
    SAR_DEPTH_RES,
    SAR_RGB_RES,
    SWARM_NEIGHBOR_K,
)


def action_buffer_size(ctrl_freq: int) -> int:
    """Action-buffer length used by the RL aviary base (ctrl_freq // 2)."""
    return int(ctrl_freq) // 2


@dataclass(frozen=True)
class SensorChannel:
    """One named input the simulator can expose to a miner."""

    channel_id: str
    semantic_label: str
    kind: str  # "vector" or "image"
    compute: Callable[[Any, np.ndarray, dict], np.ndarray]
    env_dim: Optional[Callable[[Any], int]] = None
    param_dim: Optional[Callable[[int, int], int]] = None
    image_shape: Optional[Callable[[Any], Tuple[int, ...]]] = None
    param_image_shape: Optional[Tuple[int, ...]] = None


def _position(env, sv, ctx):
    return np.asarray(sv[0:3], dtype=np.float32)


def _orientation(env, sv, ctx):
    return np.asarray(sv[7:10], dtype=np.float32)


def _linear_velocity(env, sv, ctx):
    return np.asarray(sv[10:13], dtype=np.float32)


def _angular_velocity(env, sv, ctx):
    return np.asarray(sv[13:16], dtype=np.float32)


def _action_history(env, sv, ctx):
    n = int(getattr(env, "ACTION_BUFFER_SIZE", 0))
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)
    d = int(ctx.get("self_index", 0))
    return np.concatenate(
        [np.asarray(env.action_buffer[i][d, :], dtype=np.float32) for i in range(n)]
    )


def _altitude_norm(env, sv, ctx):
    d = int(ctx.get("self_index", 0))
    return np.asarray([env._get_altitude_distance(d) / MAX_RAY_DISTANCE], dtype=np.float32)


def _goal_offset(env, sv, ctx):
    goal = getattr(env, "_search_area_center", None)
    if goal is None:
        goal = env.GOAL_POS
    return np.asarray(goal - sv[0:3], dtype=np.float32)


def _teammate_state(env, sv, ctx):
    """The K nearest teammates, each [rel_pos(3), rel_vel(3), present(1)].

    Fixed width (K*7) regardless of how many drones exist this episode: nearest
    first (ties broken by drone index for determinism), empty slots zero-padded
    with present=0 so a policy can tell a real neighbour from padding.
    """
    width = SWARM_NEIGHBOR_K * 7
    team = ctx.get("team_states")
    if team is None:
        return np.zeros((width,), dtype=np.float32)
    i = int(ctx.get("self_index", 0))
    me = np.asarray(team[i], dtype=np.float64)
    neighbours = []
    for j in range(len(team)):
        if j == i:
            continue
        other = np.asarray(team[j], dtype=np.float64)
        rel_pos = other[0:3] - me[0:3]
        rel_vel = other[3:6] - me[3:6]
        d2 = float(rel_pos[0] ** 2 + rel_pos[1] ** 2 + rel_pos[2] ** 2)
        neighbours.append((d2, j, rel_pos, rel_vel))
    neighbours.sort(key=lambda nb: (nb[0], nb[1]))
    slots = []
    for k in range(SWARM_NEIGHBOR_K):
        if k < len(neighbours):
            _, _, rel_pos, rel_vel = neighbours[k]
            slots.append(np.concatenate([rel_pos, rel_vel, [1.0]]))
        else:
            slots.append(np.zeros(7, dtype=np.float64))
    return np.concatenate(slots).astype(np.float32)


def _search_clue_offset(env, sv, ctx):
    return np.asarray((env._search_area_center - sv[0:3])[:2], dtype=np.float32)


def _depth_camera(env, sv, ctx):
    return ctx["depth"]


def _rgb_camera(env, sv, ctx):
    """On-demand RGB for SAR: the drone's frame if it requested one this step (and is under
    its budget), otherwise a zero frame. Populated by the env before the obs is assembled."""
    d = int(ctx.get("self_index", 0))
    return np.asarray(env._rgb_buffer[d], dtype=np.float32)


def _action_dim(env) -> int:
    return int(env.action_space.shape[-1])


OBSERVATION_CHANNELS = {
    "position": SensorChannel(
        "position", "position_xyz", "vector", _position,
        env_dim=lambda e: 3, param_dim=lambda cf, ad: 3,
    ),
    "orientation": SensorChannel(
        "orientation", "orientation_rpy", "vector", _orientation,
        env_dim=lambda e: 3, param_dim=lambda cf, ad: 3,
    ),
    "linear_velocity": SensorChannel(
        "linear_velocity", "linear_velocity_xyz", "vector", _linear_velocity,
        env_dim=lambda e: 3, param_dim=lambda cf, ad: 3,
    ),
    "angular_velocity": SensorChannel(
        "angular_velocity", "angular_velocity_xyz", "vector", _angular_velocity,
        env_dim=lambda e: 3, param_dim=lambda cf, ad: 3,
    ),
    "action_history": SensorChannel(
        "action_history", "action_history", "vector", _action_history,
        env_dim=lambda e: int(e.ACTION_BUFFER_SIZE) * _action_dim(e),
        param_dim=lambda cf, ad: action_buffer_size(cf) * ad,
    ),
    "altitude_norm": SensorChannel(
        "altitude_norm", "altitude_norm", "vector", _altitude_norm,
        env_dim=lambda e: 1, param_dim=lambda cf, ad: 1,
    ),
    "goal_offset": SensorChannel(
        "goal_offset", "goal_offset_xyz", "vector", _goal_offset,
        env_dim=lambda e: 3, param_dim=lambda cf, ad: 3,
    ),
    "teammate_state": SensorChannel(
        "teammate_state", "teammate_rel_state", "vector", _teammate_state,
        env_dim=lambda e: SWARM_NEIGHBOR_K * 7,
        param_dim=lambda cf, ad: SWARM_NEIGHBOR_K * 7,
    ),
    "search_clue_offset": SensorChannel(
        "search_clue_offset", "search_clue_offset_xy", "vector", _search_clue_offset,
        env_dim=lambda e: 2, param_dim=lambda cf, ad: 2,
    ),
    "depth_camera": SensorChannel(
        "depth_camera", "depth_camera", "image", _depth_camera,
        image_shape=lambda e: (int(e.IMG_RES[1]), int(e.IMG_RES[0]), 1),
        param_image_shape=(128, 128, 1),
    ),
    # higher-resolution depth for cf_interceptor (a 36 cm target must be visible at range);
    # live shape still follows env.IMG_RES, only the contract/smoke shape differs.
    "depth_camera_hd": SensorChannel(
        "depth_camera_hd", "depth_camera", "image", _depth_camera,
        image_shape=lambda e: (int(e.IMG_RES[1]), int(e.IMG_RES[0]), 1),
        param_image_shape=(INTERCEPTOR_DEPTH_RES, INTERCEPTOR_DEPTH_RES, 1),
    ),
    # 256 px depth for the SAR families (a victim is a recognizable shape, not a blob);
    # live shape still follows env.IMG_RES, only the contract/smoke shape differs.
    "depth_camera_md": SensorChannel(
        "depth_camera_md", "depth_camera", "image", _depth_camera,
        image_shape=lambda e: (int(e.IMG_RES[1]), int(e.IMG_RES[0]), 1),
        param_image_shape=(SAR_DEPTH_RES, SAR_DEPTH_RES, 1),
    ),
    # On-demand colour frame for SAR; zeros unless the drone requested one this step.
    "rgb_camera": SensorChannel(
        "rgb_camera", "rgb_camera", "image", _rgb_camera,
        image_shape=lambda e: (SAR_RGB_RES, SAR_RGB_RES, 3),
        param_image_shape=(SAR_RGB_RES, SAR_RGB_RES, 3),
    ),
}
