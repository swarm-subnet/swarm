"""Tests for the distance cull: what it hides, what it isolates, and which backend it applies to."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pybullet as p
import pybullet_data
import pytest

from swarm.core import moving_drone as moving_drone_mod

_needs_wheel = pytest.mark.skipif(
    not hasattr(p, "ER_SWARM_RAYCAST"), reason="swarm-bullet3 wheel without the ray-cast backend"
)


def test_cull_reenable_keeps_static_bodies_isolated(monkeypatch) -> None:
    """A body whose collisions come back after the cull is still isolated from the other static bodies."""
    cli = p.connect(p.DIRECT)
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cli)
        plane_id = p.loadURDF("plane.urdf", physicsClientId=cli)
        collision = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[4, 4, 1], physicsClientId=cli
        )
        mesh_path = Path(
            "swarm/assets/maps/kenney/kenney_conveyor-kit/Models/OBJ format/structure-doorway-wide.obj"
        ).resolve()
        visual = p.createVisualShape(
            p.GEOM_MESH, fileName=str(mesh_path), physicsClientId=cli
        )
        box_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[0, 0, 0],
            physicsClientId=cli,
        )
        drone_collision = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1], physicsClientId=cli
        )
        drone_id = p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=drone_collision,
            basePosition=[moving_drone_mod.CULL_PHYSICS_RADIUS + 10, 0, 2],
            physicsClientId=cli,
        )

        env = object.__new__(moving_drone_mod.MovingDroneAviary)
        env.CLIENT = cli
        env.DRONE_IDS = [drone_id]
        env.NUM_DRONES = 1
        env.PLANE_ID = plane_id
        env.GUI = False
        env.family_runtime = SimpleNamespace(protected_body_uids=lambda _: [])
        monkeypatch.setattr(
            moving_drone_mod,
            "CULL_MIN_TOTAL_FACES",
            moving_drone_mod.CULL_MIN_FACES,
        )

        env._isolate_static_collisions()
        env._build_cull_targets()
        assert [target[0] for target in env._cull_targets] == [box_id]

        for _ in range(moving_drone_mod.CULL_INTERVAL_STEPS):
            env._apply_distance_cull()
        assert box_id in env._cull_phys_disabled

        p.resetBasePositionAndOrientation(
            drone_id, [0, 0, 2], [0, 0, 0, 1], physicsClientId=cli
        )
        for _ in range(moving_drone_mod.CULL_INTERVAL_STEPS):
            env._apply_distance_cull()

        p.stepSimulation(physicsClientId=cli)
        assert not p.getContactPoints(
            bodyA=plane_id, bodyB=box_id, physicsClientId=cli
        )
    finally:
        p.disconnect(cli)


def _body_pixels(cli, eye, target, uid, flags):
    """Number of pixels whose segmentation id is uid in a 128 px frame from eye towards target."""
    view = p.computeViewMatrix(eye, target, [0, 0, 1], physicsClientId=cli)
    proj = p.computeProjectionMatrixFOV(90.0, 1.0, 0.05, 110.0, physicsClientId=cli)
    image = p.getCameraImage(
        128, 128, viewMatrix=view, projectionMatrix=proj, renderer=p.ER_TINY_RENDERER,
        flags=flags, shadow=0, physicsClientId=cli,
    )
    return int(np.count_nonzero(np.asarray(image[4]) == uid))


def _cull_world(cli, monkeypatch, raycast_enabled):
    """A large mesh body at the origin with a drone just past the visual radius, and an env whose single cull target is that body."""
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cli)
    plane_id = p.loadURDF("plane.urdf", physicsClientId=cli)
    collision = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[4, 4, 4], physicsClientId=cli
    )
    mesh_path = Path(
        "swarm/assets/maps/kenney/kenney_conveyor-kit/Models/OBJ format/structure-doorway-wide.obj"
    ).resolve()
    visual = p.createVisualShape(
        p.GEOM_MESH, fileName=str(mesh_path), meshScale=[8, 8, 8], physicsClientId=cli
    )
    box_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=[0, 0, 0],
        physicsClientId=cli,
    )
    drone_collision = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1], physicsClientId=cli
    )
    eye = [moving_drone_mod.CULL_VISUAL_RADIUS + 10, 0, 2]
    drone_id = p.createMultiBody(
        baseMass=1, baseCollisionShapeIndex=drone_collision, basePosition=eye, physicsClientId=cli,
    )

    env = object.__new__(moving_drone_mod.MovingDroneAviary)
    env.CLIENT = cli
    env.DRONE_IDS = [drone_id]
    env.NUM_DRONES = 1
    env.PLANE_ID = plane_id
    env.GUI = False
    env._raycast_enabled = raycast_enabled
    env.family_runtime = SimpleNamespace(protected_body_uids=lambda _: [])
    monkeypatch.setattr(
        moving_drone_mod, "CULL_MIN_TOTAL_FACES", moving_drone_mod.CULL_MIN_FACES
    )
    env._build_cull_targets()
    assert [target[0] for target in env._cull_targets] == [box_id]
    return env, box_id, drone_id, eye


@pytest.mark.parametrize("raycast_enabled, hidden", [(True, False), (False, True)])
def test_visual_cull_only_hides_far_bodies_on_tiny_renderer(monkeypatch, raycast_enabled, hidden) -> None:
    """Past the visual radius the cull drops a large body's alpha to 0 on TinyRenderer and leaves it alone on the ray caster, and neither loses its collisions before the physics radius."""
    cli = p.connect(p.DIRECT)
    try:
        env, box_id, drone_id, _eye = _cull_world(cli, monkeypatch, raycast_enabled)

        for _ in range(moving_drone_mod.CULL_INTERVAL_STEPS):
            env._apply_distance_cull()
        assert (box_id in env._cull_vis_hidden) == hidden
        alpha = p.getVisualShapeData(box_id, physicsClientId=cli)[0][7][3]
        assert (alpha == 0.0) == hidden
        assert box_id not in env._cull_phys_disabled

        p.resetBasePositionAndOrientation(
            drone_id, [moving_drone_mod.CULL_PHYSICS_RADIUS + 10, 0, 2], [0, 0, 0, 1], physicsClientId=cli
        )
        for _ in range(moving_drone_mod.CULL_INTERVAL_STEPS):
            env._apply_distance_cull()
        assert box_id in env._cull_phys_disabled
    finally:
        p.disconnect(cli)


@pytest.mark.parametrize(
    "raycast_enabled, still_visible",
    [pytest.param(True, True, marks=_needs_wheel), (False, False)],
)
def test_the_far_body_keeps_its_pixels_only_on_the_ray_cast_path(monkeypatch, raycast_enabled, still_visible) -> None:
    """What the cull does to the picture: after the tick the body is gone from a TinyRenderer frame and still drawn in a ray-cast one."""
    cli = p.connect(p.DIRECT)
    try:
        env, box_id, _drone_id, eye = _cull_world(cli, monkeypatch, raycast_enabled)
        flags = p.ER_SWARM_RAYCAST if raycast_enabled else 0
        assert _body_pixels(cli, eye, [0, 0, 2], box_id, flags) > 0

        for _ in range(moving_drone_mod.CULL_INTERVAL_STEPS):
            env._apply_distance_cull()
        assert (_body_pixels(cli, eye, [0, 0, 2], box_id, flags) > 0) == still_visible
    finally:
        p.disconnect(cli)
