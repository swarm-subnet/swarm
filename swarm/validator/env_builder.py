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

HEIGHT_SCALE = 0.20          # keep just 20 % of the original height

def build_world(seed: int, cli: int) -> None:
    rng = random.Random(seed)

    for _ in range(120):
        kind = rng.choice(["wall", "pillar", "box"])
        x, y = rng.uniform(-WORLD_RANGE, WORLD_RANGE), rng.uniform(-WORLD_RANGE, WORLD_RANGE)
        if math.hypot(x, y) < 2:                # keep spawn zone clear
            continue

        yaw = rng.uniform(0, math.pi)

        # ───────────────────────────────────────────────────────── box ─
        if kind == "box":
            sx, sy, sz = (rng.uniform(1, 4) for _ in range(3))
            sz *= HEIGHT_SCALE                            # 80 % shorter
            _add_box(cli,
                     pos=[x, y, sz / 2],             # centre at mid-height
                     size=[sx, sy, sz],
                     yaw=yaw)

        # ───────────────────────────────────────────────────────── wall ─
        elif kind == "wall":
            length = rng.uniform(5, 15)
            height = rng.uniform(2, 5) * HEIGHT_SCALE
            _add_box(cli,
                     pos=[x, y, height / 2],
                     size=[length, 0.3, height],
                     yaw=yaw)

        # ─────────────────────────────────────────────────────── pillar ─
        else:  # pillar
            r  = rng.uniform(0.3, 0.6)
            h  = rng.uniform(2, 7) * HEIGHT_SCALE
            col = p.createCollisionShape(p.GEOM_CYLINDER,
                                         radius=r, height=h,
                                         physicsClientId=cli)
            p.createMultiBody(0, col,
                              basePosition=[x, y, h / 2],
                              physicsClientId=cli)
