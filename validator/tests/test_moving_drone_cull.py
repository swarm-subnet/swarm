from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pybullet as p
import pybullet_data

from swarm.core import moving_drone as moving_drone_mod


def test_cull_reenable_keeps_static_bodies_isolated(monkeypatch) -> None:
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
