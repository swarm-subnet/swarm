"""The sky computed from the seeded sun in the swarm-bullet3 wheel: the frames are pinned to committed
hashes on both colour paths and every thread count, so a validator anywhere paints the same sky."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys

import numpy as np
import pybullet as p
import pytest

from swarm.core.daylight import seeded_sun, sun_render_kwargs
from validator.tests.test_render_backend import SIZE, _build_world, _camera

_needs_wheel = pytest.mark.skipif(
    not hasattr(p, "ER_SWARM_SKY_SUN"), reason="swarm-bullet3 wheel without the sun sky"
)
SEED = 101
# A seed whose sun stands high, so the sky it paints is blue on average rather than a sunset glow.
HIGH_SUN_SEED = 7
EYE, TARGET = (3.0, -4.0, 2.5), (0.0, 0.5, 4.0)
# SHA-256 of the colour frame under the seed's sun and clouds, rasterised and ray cast. Every machine
# and thread count must reproduce them; a renderer change that moves one pixel, or a change to the sun
# a seed draws, updates them on purpose.
TINY_SKY_SHA256 = "fce1fc542f02989f0099ae43f962f1276baff6795105889116bfdf926f8fcc7e"
RAYCAST_SKY_SHA256 = "e8f727f4015e72e5cf64a0105853398c499dcac84d70daf6b261b2a333320721"


def _sky_frame(cli, view, proj, flags, seed=SEED):
    """RGB of one frame lit by the seed's sun with the sun sky and its clouds painted."""
    sun = seeded_sun(seed)
    image = p.getCameraImage(
        SIZE, SIZE, viewMatrix=view, projectionMatrix=proj, renderer=p.ER_TINY_RENDERER, shadow=0,
        lightDirection=list(sun.direction), flags=flags | p.ER_NO_SEGMENTATION_MASK | p.ER_SWARM_SKY_SUN,
        skyCloudSeed=seed, physicsClientId=cli, **sun_render_kwargs(sun),
    )
    return np.asarray(image[2], dtype=np.uint8).reshape(SIZE, SIZE, 4)[:, :, :3]


def _sky_hashes():
    """SHA-256 of the rasterised and the ray-cast sun sky frame of the test world."""
    cli = p.connect(p.DIRECT)
    try:
        _build_world(cli)
        view, proj = _camera(cli, EYE, TARGET)
        tiny = _sky_frame(cli, view, proj, 0)
        raycast = _sky_frame(cli, view, proj, p.ER_SWARM_RAYCAST)
        return hashlib.sha256(tiny.tobytes()).hexdigest(), hashlib.sha256(raycast.tobytes()).hexdigest()
    finally:
        p.disconnect(cli)


@_needs_wheel
def test_sun_sky_replaces_the_white_background():
    """With the flag the empty pixels carry a sky, without it they are white, and depth is the same."""
    cli = p.connect(p.DIRECT)
    try:
        _build_world(cli)
        view, proj = _camera(cli, EYE, TARGET)
        plain = p.getCameraImage(SIZE, SIZE, viewMatrix=view, projectionMatrix=proj, renderer=p.ER_TINY_RENDERER,
                                 flags=p.ER_NO_SEGMENTATION_MASK, shadow=0, physicsClientId=cli)
        rgb = np.asarray(plain[2], dtype=np.uint8).reshape(SIZE, SIZE, 4)[:, :, :3]
        empty = np.all(rgb == 255, axis=2)
        assert empty.mean() > 0.3
        sky = _sky_frame(cli, view, proj, 0)
        assert not np.all(sky[empty] == 255)
        # A high sun gives a blue sky on average; the glow around the sun itself is warmer.
        assert seeded_sun(HIGH_SUN_SEED).elevation_deg > 45
        blue_sky = _sky_frame(cli, view, proj, 0, HIGH_SUN_SEED)
        assert blue_sky[empty][:, 2].mean() > blue_sky[empty][:, 0].mean() + 10
    finally:
        p.disconnect(cli)


@_needs_wheel
@pytest.mark.parametrize("threads", ["1", "2", "4"])
def test_sun_sky_frames_are_identical_for_every_thread_count(threads):
    """Every thread count gives the committed bytes on both paths; the count is read once per process."""
    env = dict(os.environ, SWARM_RENDER_THREADS=threads)
    result = subprocess.run(
        [sys.executable, "-m", "validator.tests.test_sky_sun"],
        capture_output=True, text=True, env=env, check=True,
    )
    tiny, raycast = result.stdout.strip().splitlines()[-1].split()
    assert tiny == TINY_SKY_SHA256
    assert raycast == RAYCAST_SKY_SHA256


if __name__ == "__main__":
    print(*_sky_hashes())
