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

"""Tests for the ray-cast depth backend that sits beside TinyRenderer in the swarm-bullet3 wheel."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from importlib import resources

import numpy as np
import pybullet as p
import pybullet_data
import pytest

import swarm.challenge_families  # noqa: F401  registers every family runtime class
from swarm.challenge_families.base import ChallengeFamilyRuntime
from swarm.constants import SIM_DT
from swarm.utils.env_factory import make_env_with_initial_obs
from swarm.validator.task_gen import task_for_seed_and_type

SIZE = 128
NEAR = 0.05
FAR = 30.0
_BOX_OBJ = os.path.join(
    os.path.dirname(swarm.__file__), "assets", "maps", "kenney", "kenney_conveyor-kit",
    "Models", "OBJ format", "box-small.obj",
)
_needs_wheel = pytest.mark.skipif(
    not hasattr(p, "ER_SWARM_RAYCAST"), reason="swarm-bullet3 wheel without the ray-cast backend"
)


def _every_family_class(cls=ChallengeFamilyRuntime):
    """Yield every runtime family class registered under the base class."""
    for sub in cls.__subclasses__():
        yield sub
        yield from _every_family_class(sub)


def test_every_family_defaults_to_tiny_renderer():
    """No family opts into the ray caster on its own; the switch is explicit per family."""
    assert ChallengeFamilyRuntime.render_backend == "tiny"
    for family in _every_family_class():
        assert family.render_backend == "tiny", family.__name__


def _static(cli, shape, position, **kwargs):
    """Add a mass-less body with one visual shape of the given type."""
    visual = p.createVisualShape(shape, physicsClientId=cli, **kwargs)
    return p.createMultiBody(baseVisualShapeIndex=visual, basePosition=position, physicsClientId=cli)


def _build_world(cli):
    """A ground plane, the four primitives, an OBJ mesh, a quad facing away and a Crazyflie."""
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cli)
    p.loadURDF("plane.urdf", physicsClientId=cli)
    _static(cli, p.GEOM_BOX, [-1.5, -1.0, 0.5], halfExtents=[0.5, 0.7, 0.5])
    _static(cli, p.GEOM_SPHERE, [-0.8, 1.0, 0.6], radius=0.6)
    _static(cli, p.GEOM_CYLINDER, [0.8, 1.0, 0.7], radius=0.55, length=1.4)
    _static(cli, p.GEOM_CAPSULE, [1.6, -0.2, 0.7], radius=0.25, length=0.8)
    _static(cli, p.GEOM_MESH, [2.0, 1.0, 0.2], fileName=_BOX_OBJ, meshScale=[1, 1, 1])
    # One quad whose winding faces the floor: single-sided, so the camera above must not see it.
    _static(
        cli, p.GEOM_MESH, [0.0, -1.2, 1.5],
        vertices=[[-0.6, -0.6, 0], [0.6, 0.6, 0], [0.6, -0.6, 0], [-0.6, 0.6, 0]],
        indices=[0, 1, 2, 0, 3, 1],
    )
    assets = resources.files("gym_pybullet_drones").joinpath("assets")
    p.setAdditionalSearchPath(str(assets), physicsClientId=cli)
    p.loadURDF("cf2x.urdf", [0.3, 0.3, 2.0], physicsClientId=cli)


def _camera(cli, eye=(0.0, 0.0, 8.0), target=(0.0, 0.0, 0.0)):
    """View and projection matrices of a camera looking from eye to target."""
    view = p.computeViewMatrix(eye, target, [0.0, 1.0, 0.0], physicsClientId=cli)
    proj = p.computeProjectionMatrixFOV(55.0, 1.0, NEAR, FAR, physicsClientId=cli)
    return view, proj


def _render(cli, view, proj, flags):
    """Depth and segmentation buffers of one getCameraImage call."""
    image = p.getCameraImage(
        SIZE, SIZE, viewMatrix=view, projectionMatrix=proj, renderer=p.ER_TINY_RENDERER,
        flags=flags, shadow=0, physicsClientId=cli,
    )
    depth = np.asarray(image[3], dtype=np.float32).reshape(SIZE, SIZE)
    seg = np.asarray(image[4]).reshape(SIZE, SIZE) if image[4] is not None else None
    return depth, seg


def _linear(depth):
    """Metres along the view axis from a normalised depth buffer."""
    return FAR * NEAR / (FAR - (FAR - NEAR) * depth)


def _agreement(tiny, raycast):
    """Fraction of pixels where both hit or both miss, and where common hits agree within 1 percent."""
    hits = (tiny < 1.0) == (raycast < 1.0)
    both = (tiny < 1.0) & (raycast < 1.0)
    close = np.abs(_linear(tiny[both]) - _linear(raycast[both])) < 0.01 * _linear(tiny[both])
    return float(np.mean(hits)), float(np.mean(close)) if both.any() else 1.0


@_needs_wheel
@pytest.mark.parametrize("eye,target", [((0.0, 0.0, 8.0), (0.0, 0.0, 0.0)), ((3.0, -4.0, 2.5), (0.0, 0.5, 0.8))])
def test_raycast_depth_agrees_with_tiny_renderer(eye, target):
    """Both eyes see the same primitives, mesh, back-facing quad and drone from two viewpoints."""
    cli = p.connect(p.DIRECT)
    try:
        _build_world(cli)
        view, proj = _camera(cli, eye, target)
        tiny, _ = _render(cli, view, proj, p.ER_NO_SEGMENTATION_MASK | p.ER_DEPTH_ONLY)
        raycast, _ = _render(cli, view, proj, p.ER_NO_SEGMENTATION_MASK | p.ER_DEPTH_ONLY | p.ER_SWARM_RAYCAST)
        hit_agreement, depth_agreement = _agreement(tiny, raycast)
        assert np.count_nonzero(tiny < 1.0) > SIZE * SIZE * 0.3
        assert hit_agreement >= 0.99
        assert depth_agreement >= 0.99
    finally:
        p.disconnect(cli)


@_needs_wheel
def test_raycast_segmentation_matches_tiny_renderer():
    """The object and link ids per pixel are the same ids TinyRenderer writes."""
    cli = p.connect(p.DIRECT)
    try:
        _build_world(cli)
        view, proj = _camera(cli)
        _, tiny_seg = _render(cli, view, proj, p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
        _, raycast_seg = _render(cli, view, proj, p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX | p.ER_SWARM_RAYCAST)
        assert len(np.unique(tiny_seg)) >= 6
        assert np.mean(tiny_seg == raycast_seg) >= 0.99
    finally:
        p.disconnect(cli)


@_needs_wheel
def test_back_facing_quad_is_invisible_on_both_paths():
    """A single-sided face seen from behind is culled by TinyRenderer, so the ray caster must miss it too."""
    cli = p.connect(p.DIRECT)
    try:
        quad = _static(
            cli, p.GEOM_MESH, [0.0, 0.0, 1.5],
            vertices=[[-0.6, -0.6, 0], [0.6, 0.6, 0], [0.6, -0.6, 0], [-0.6, 0.6, 0]],
            indices=[0, 1, 2, 0, 3, 1],
        )
        view, proj = _camera(cli)
        for flags in (p.ER_DEPTH_ONLY, p.ER_DEPTH_ONLY | p.ER_SWARM_RAYCAST):
            depth, _ = _render(cli, view, proj, p.ER_NO_SEGMENTATION_MASK | flags)
            assert np.all(depth == np.float32(1.0)), flags
        p.changeVisualShape(quad, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED_MULTIBODY, physicsClientId=cli)
        for flags in (p.ER_DEPTH_ONLY, p.ER_DEPTH_ONLY | p.ER_SWARM_RAYCAST):
            depth, _ = _render(cli, view, proj, p.ER_NO_SEGMENTATION_MASK | flags)
            assert np.any(depth < 1.0), flags
    finally:
        p.disconnect(cli)


@_needs_wheel
def test_moved_body_is_seen_at_its_new_place():
    """A body teleported after the first frame renders at the new position without a rebuild."""
    cli = p.connect(p.DIRECT)
    try:
        box = _static(cli, p.GEOM_BOX, [0.0, 0.0, 0.75], halfExtents=[0.75, 0.75, 0.75])
        view, proj = _camera(cli)
        flags = p.ER_NO_SEGMENTATION_MASK | p.ER_DEPTH_ONLY | p.ER_SWARM_RAYCAST
        before, _ = _render(cli, view, proj, flags)
        p.resetBasePositionAndOrientation(box, [0.0, 0.0, 2.75], [0, 0, 0, 1], physicsClientId=cli)
        after, _ = _render(cli, view, proj, flags)
        centre = (SIZE // 2, SIZE // 2)
        assert _linear(after[centre]) < _linear(before[centre]) - 1.9
    finally:
        p.disconnect(cli)


def _depth_hash():
    """SHA-256 of a ray-cast depth frame of the test world, for the thread-count check."""
    cli = p.connect(p.DIRECT)
    try:
        _build_world(cli)
        view, proj = _camera(cli, (3.0, -4.0, 2.5), (0.0, 0.5, 0.8))
        depth, _ = _render(cli, view, proj, p.ER_NO_SEGMENTATION_MASK | p.ER_DEPTH_ONLY | p.ER_SWARM_RAYCAST)
        return hashlib.sha256(depth.tobytes()).hexdigest()
    finally:
        p.disconnect(cli)


@_needs_wheel
def test_raycast_frame_is_identical_for_every_thread_count():
    """The render thread count is read once per process, so each count renders in its own process."""
    hashes = set()
    for threads in ("1", "2", "4"):
        env = dict(os.environ, SWARM_RENDER_THREADS=threads)
        result = subprocess.run(
            [sys.executable, "-m", "validator.tests.test_render_backend"],
            capture_output=True, text=True, env=env, check=True,
        )
        hashes.add(result.stdout.strip().splitlines()[-1])
    assert len(hashes) == 1


def _autopilot_obs(monkeypatch, backend):
    """First observation of an open-map autopilot episode on the given backend."""
    if backend is None:
        monkeypatch.delenv("SWARM_RENDER_BACKEND", raising=False)
    else:
        monkeypatch.setenv("SWARM_RENDER_BACKEND", backend)
    task = task_for_seed_and_type(sim_dt=SIM_DT, seed=123, challenge_type=1, family_id="cf_autopilot")
    env, obs = make_env_with_initial_obs(task, gui=False)
    try:
        for _ in range(2):
            obs, _reward, _terminated, _truncated, _info = env.step(np.zeros((1, 5), dtype=np.float32))
        return env._raycast_enabled, obs["depth"].copy()
    finally:
        env.close()


@_needs_wheel
def test_env_switch_renders_the_autopilot_depth_on_the_ray_caster(monkeypatch):
    """The family switch, overridden by the environment variable, changes the backend and keeps the contract."""
    enabled, tiny = _autopilot_obs(monkeypatch, None)
    assert not enabled
    enabled, raycast = _autopilot_obs(monkeypatch, "raycast")
    assert enabled
    assert raycast.shape == tiny.shape and raycast.dtype == tiny.dtype
    assert float(np.mean(np.abs(raycast - tiny))) < 0.02


@_needs_wheel
def test_office_family_stays_on_tiny_renderer(monkeypatch):
    """The colour-observing office family ignores the switch: only TinyRenderer shades colour."""
    monkeypatch.setenv("SWARM_RENDER_BACKEND", "raycast")
    task = task_for_seed_and_type(sim_dt=SIM_DT, seed=7, challenge_type=7, family_id="cf_interceptor_office")
    env, _obs = make_env_with_initial_obs(task, gui=False)
    try:
        assert not env._raycast_enabled
    finally:
        env.close()


if __name__ == "__main__":
    print(_depth_hash())
