# swarm/miner/env_builder.py
import math, random
import pybullet as p

WORLD_RANGE = 40.0

def _add_box(cli, pos, size, yaw):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in size],
                                 physicsClientId=cli)
    quat = p.getQuaternionFromEuler([0,0,yaw])
    p.createMultiBody(0, col, basePosition=pos, baseOrientation=quat,
                      physicsClientId=cli)

def build_world(seed: int, cli: int) -> None:
    rng = random.Random(seed)
    for _ in range(120):
        kind = rng.choice(["wall","pillar","box"])
        x,y = rng.uniform(-WORLD_RANGE,WORLD_RANGE), rng.uniform(-WORLD_RANGE,WORLD_RANGE)
        if math.hypot(x,y) < 2: continue
        yaw = rng.uniform(0, math.pi)
        if kind=="box":
            _add_box(cli, [x,y,1.0], [rng.uniform(1,4) for _ in range(3)], yaw)
        elif kind=="wall":
            _add_box(cli, [x,y,rng.uniform(1,3)],
                     [rng.uniform(5,15),0.3,rng.uniform(2,5)], yaw)
        else:   # pillar
            r,h = rng.uniform(0.3,0.6), rng.uniform(2,7)
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=h,
                                         physicsClientId=cli)
            p.createMultiBody(0, col, basePosition=[x,y,h/2], physicsClientId=cli)
