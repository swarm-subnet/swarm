from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import gymnasium.spaces as spaces
import numpy as np

from .channels import OBSERVATION_CHANNELS

ObservationLayout = Mapping[str, Sequence[str]]


class UnknownSensorChannelError(KeyError):
    """Raised when a layout names a channel that is not registered."""


def _channels_for(channel_ids: Sequence[str]):
    resolved = []
    for channel_id in channel_ids:
        channel = OBSERVATION_CHANNELS.get(channel_id)
        if channel is None:
            raise UnknownSensorChannelError(channel_id)
        resolved.append(channel)
    return resolved


def _is_image(channels) -> bool:
    return any(channel.kind == "image" for channel in channels)


def _image_channel(channels, key: str):
    if len(channels) != 1:
        raise ValueError(f"image key '{key}' must declare exactly one channel")
    return channels[0]


def assemble(layout: ObservationLayout, env: Any, state_vec: np.ndarray, ctx: dict) -> dict:
    """Build the observation dict for a layout from the live simulator state."""
    obs: dict = {}
    for key, channel_ids in layout.items():
        channels = _channels_for(channel_ids)
        if _is_image(channels):
            channel = _image_channel(channels, key)
            obs[key] = np.asarray(channel.compute(env, state_vec, ctx), dtype=np.float32)
        else:
            parts = [
                np.asarray(channel.compute(env, state_vec, ctx), dtype=np.float32).reshape(-1)
                for channel in channels
            ]
            joined = np.concatenate(parts) if parts else np.zeros((0,), dtype=np.float32)
            obs[key] = joined.astype(np.float32)
    return obs


def observation_space(layout: ObservationLayout, env: Any) -> spaces.Dict:
    """Build the gym observation space for a layout using live env parameters."""
    fields: dict = {}
    for key, channel_ids in layout.items():
        channels = _channels_for(channel_ids)
        if _is_image(channels):
            shape = tuple(_image_channel(channels, key).image_shape(env))
            fields[key] = spaces.Box(low=0.0, high=1.0, shape=shape, dtype=np.float32)
        else:
            dim = int(sum(channel.env_dim(env) for channel in channels))
            fields[key] = spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)
    return spaces.Dict(fields)


def observation_vector_dim(layout: ObservationLayout, env: Any, key: str = "state") -> Optional[int]:
    """Length of a flat vector key for the env, or None if the key is absent."""
    if key not in layout:
        return None
    channels = _channels_for(layout[key])
    if _is_image(channels):
        return None
    return int(sum(channel.env_dim(env) for channel in channels))


def smoke_observation(
    layout: ObservationLayout,
    *,
    ctrl_freq: int,
    action_dim: int,
    fills: Optional[Mapping[str, float]] = None,
) -> dict:
    """Build a synthetic observation matching a layout without a live env."""
    fills = fills or {}
    obs: dict = {}
    for key, channel_ids in layout.items():
        channels = _channels_for(channel_ids)
        fill = float(fills.get(key, 0.0))
        if _is_image(channels):
            obs[key] = np.full(tuple(_image_channel(channels, key).param_image_shape), fill, dtype=np.float32)
        else:
            dim = int(sum(channel.param_dim(ctrl_freq, action_dim) for channel in channels))
            obs[key] = np.full((dim,), fill, dtype=np.float32)
    return obs
