from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import numpy as np

from swarm.constants import MAX_RAY_DISTANCE


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
    return np.concatenate(
        [np.asarray(env.action_buffer[i][0, :], dtype=np.float32) for i in range(n)]
    )


def _altitude_norm(env, sv, ctx):
    return np.asarray([env._get_altitude_distance() / MAX_RAY_DISTANCE], dtype=np.float32)


def _goal_offset(env, sv, ctx):
    return np.asarray(env.GOAL_POS - sv[0:3], dtype=np.float32)


def _search_clue_offset(env, sv, ctx):
    return np.asarray((env._search_area_center - sv[0:3])[:2], dtype=np.float32)


def _depth_camera(env, sv, ctx):
    return ctx["depth"]


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
    "search_clue_offset": SensorChannel(
        "search_clue_offset", "search_clue_offset_xy", "vector", _search_clue_offset,
        env_dim=lambda e: 2, param_dim=lambda cf, ad: 2,
    ),
    "depth_camera": SensorChannel(
        "depth_camera", "depth_camera", "image", _depth_camera,
        image_shape=lambda e: (int(e.IMG_RES[1]), int(e.IMG_RES[0]), 1),
        param_image_shape=(128, 128, 1),
    ),
}
