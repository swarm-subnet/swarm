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

"""Builder for the solar park map.

The park is a real site, so its rows, fence, road and terrain never move. The map is a manifest of reusable pieces
with their placements; this builder loads what the manifest lists and lets the seed decide only what may vary:
how much of the vegetation outside the fence stands, and how the movers cross it. Sun, sky and wind belong to the
family's options.

Adding a piece to the map is a new file and a manifest entry, never a change here. Changing how the seed varies
the world is a number in CONFIG.
"""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p
import swarm_worlds

SOLAR_ASSET_DIR = os.path.join(swarm_worlds.maps_dir(), "custom", "solar")

CONFIG: Dict[str, Any] = {
    "seed_offset": 0x501A2,                      # own stream: the map must not ride the other seed draws
    "density": {"near": (0.6, 1.0),              # share of the trees by the fence and the road a seed keeps
                "mid": (0.5, 1.0),               # share of the trees on the slopes a seed keeps
                "far": (1.0, 1.0),               # the forest mass on the far hills always stands
                "grass": (0.5, 1.0)},            # share of the grass tufts a seed keeps, a dry year against a green one
    "cylinder_radius_m": 0.13,                   # trunk and post stand-in for the drone to hit
    "cylinder_height_share": 0.6,                # of the piece's height, so a crown is flown through, not into
    "forest_trunk_min_m": 2.0,                   # a near tree at least this tall gets a trunk for the drone to hit
    "forest_trunk_step_m": 0.25,                 # trunk heights snap to this, so few distinct cylinders are made
    "forest_trunks_per_body": 16,                # trunks one body holds, the engine's compound limit
    "forest_trunk_cell_m": 10.0,                 # trunks are grouped by this grid, so a body's trunks are neighbours
    "mover_seed_offset": 0x4D0FE,                # movers draw from their own stream, so a new density tier cannot move the truck
    "step_hz": 50,                               # the rate every mover table is written at, one row per simulator step
    "pickup_delay_s": (0.0, 40.0),               # window the truck's start is drawn from, so it passes at a different moment
    "pickup_parked_share": 0.2,                  # share of seeds that leave the truck parked at the first row all episode
    "bird_phase_share": (0.0, 1.0),              # where on its loop the bird starts, as a share of the whole loop
    "goat_offset_m": 3.0,                        # how far within the meadow the herd's start is moved, each axis
    "goat_heading_rad": (-0.7, 0.7),             # heading a seed draws per goat, about the direction the herd was posed in
    "goat_turn_rad_frame": (0.010, 0.022),       # turn a seed draws per goat, radians per pose frame, the sign alternating
    "goat_pace": (0.90, 1.10),                   # share of the walk's own speed a seed draws per goat
    "goat_ray_steps": 1,                         # steps between two ground rays under a goat, measured at 4 us a ray
    "goat_ray_window_m": (2.0, 5.0),             # how far above and below the last ground height that ray runs
    "goat_mesh_steps": 1,                        # steps between two mesh rewrites of a goat, the one lever on their cost
}

# tree_cache is left out on purpose: measured, a cached piece leaves the one static tree and every warm frame pays for it.
_FLAG_BITS = {"double_sided": "VISUAL_SHAPE_DOUBLE_SIDED_MULTIBODY", "glass": "VISUAL_SHAPE_GLASS",
              "materials_from_mtl": "VISUAL_SHAPE_MATERIALS_FROM_MTL"}


@lru_cache(maxsize=2)
def solar_manifest(asset_dir: str = SOLAR_ASSET_DIR) -> Dict[str, Any]:
    """The map's manifest: every item with its files and flags, and every placement with its tags."""
    path = os.path.join(asset_dir, "manifest.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"solar map manifest missing: {path}")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=8)
def _mover_table(path: str) -> Dict[str, Any]:
    """One mover's table, read once per file and shared by every body that follows it."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"solar mover table missing: {path}")
    with open(path, encoding="utf-8") as handle:
        table = json.load(handle)
    if int(table.get("hz", CONFIG["step_hz"])) != CONFIG["step_hz"]:
        raise ValueError(f"solar mover table {path} is written at {table['hz']} Hz, not {CONFIG['step_hz']}")
    return table


@lru_cache(maxsize=4)
def _mover_poses(path: str) -> Dict[str, Any]:
    """One mover's pose table, as the arrays its npz holds."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"solar mover poses missing: {path}")
    with np.load(path) as loaded:
        return {name: loaded[name] for name in loaded.files}


def solar_densities(seed: int) -> Dict[str, float]:
    """The share of each vegetation tier this seed keeps, drawn from the map's own random stream."""
    rng = random.Random((int(seed) ^ CONFIG["seed_offset"]) & 0xFFFFFFFF)
    return {tier: rng.uniform(low, high) for tier, (low, high) in CONFIG["density"].items()}


def solar_mover_rules(seed: int) -> Dict[str, Any]:
    """What this seed decides about the movers: when the truck leaves, where the bird is on its loop, how the herd walks.

    The truck is parked for a share of seeds, which is the episode where the road stays empty; the rest of the draw
    happens either way so that one branch cannot shift the stream the bird and the goats read.
    """
    rng = random.Random((int(seed) ^ CONFIG["mover_seed_offset"]) & 0xFFFFFFFF)
    parked = rng.random() < CONFIG["pickup_parked_share"]
    delay = rng.uniform(*CONFIG["pickup_delay_s"])
    phase = rng.uniform(*CONFIG["bird_phase_share"])
    offset = [rng.uniform(-CONFIG["goat_offset_m"], CONFIG["goat_offset_m"]) for _ in range(2)]
    return {"pickup_parked": parked, "pickup_delay_s": 0.0 if parked else delay, "bird_phase_share": phase,
            "goat_offset_m": offset, "goat_seed": rng.getrandbits(32)}


def _goat_plans(stride: Dict[str, float], count: int, rules: Dict[str, Any]) -> List[Dict[str, float]]:
    """Each goat's heading, turn, pace and stride phase for this seed, drawn the way the herd's own export draws them.

    Speed and pose rate follow from the pace and the stride the walk was measured at, so the hooves cannot skate
    however the seed moves the animal, and a rebuilt walk of another length or another rate carries itself here.
    """
    rng = random.Random(rules["goat_seed"])
    plans = []
    for index in range(count):
        pace = rng.uniform(*CONFIG["goat_pace"])
        turn = rng.uniform(*CONFIG["goat_turn_rad_frame"]) * (1.0 if index % 2 else -1.0)
        plans.append({"heading_rad": rng.uniform(*CONFIG["goat_heading_rad"]), "turn_rad_s": turn * stride["fps"],
                      "phase": float(rng.randrange(stride["frames"])), "pose_fps": stride["fps"] * pace,
                      "speed_m_s": stride["stride_m"] * pace * stride["fps"] / stride["frames"]})
    return plans


def _stride_pose(poses: np.ndarray, phase: float) -> np.ndarray:
    """The shape a stride holds at a continuous phase: the two frames it falls between, blended in proportion.

    A 50 Hz step against a table authored in the twenties lands between frames far more often than on one, and
    holding the nearer frame instead makes the animal repeat a shape and then jump, which reads as a limp.
    """
    frames = len(poses)
    first = int(math.floor(phase))
    share = phase - first
    return poses[first % frames] * (1.0 - share) + poses[(first + 1) % frames] * share


def _stands(place: Dict[str, Any], densities: Dict[str, float]) -> bool:
    """Whether a placement is part of this seed's world: fixed pieces always, vegetation by its rank.

    A merged piece carries the lowest rank of the vegetation inside it as band_low, so a whole band stands or
    falls together; a single plant carries its own rank.
    """
    tier = place.get("tier")
    if tier is None:
        return True
    return place.get("band_low", place.get("rank", 0.0)) < densities.get(tier, 1.0)


class _Shapes:
    """Visual and collision shapes of the map's items, created once per item and scale and shared by its bodies."""

    def __init__(self, cli: int, asset_dir: str, items: Dict[str, Any]):
        """Remember where the files are; nothing is loaded until a placement asks."""
        self.cli = cli
        self.asset_dir = asset_dir
        self.items = items
        self.cache: Dict[Tuple[str, Tuple[float, ...]], Tuple[int, int]] = {}

    def get(self, name: str, scale: List[float]) -> Tuple[int, int]:
        """The visual and collision shape ids of an item at a scale."""
        key = (name, tuple(round(float(s), 4) for s in scale))
        if key not in self.cache:
            item = self.items[name]
            path = os.path.join(self.asset_dir, item["folder"], item["obj"])
            if not os.path.exists(path):
                raise FileNotFoundError(f"solar map asset missing: {path}")
            self.cache[key] = (self._visual(item, path, scale), self._collision(item, path, scale))
        return self.cache[key]

    def _visual(self, item: Dict[str, Any], path: str, scale: List[float]) -> int:
        """The item's render shape with the flags its record asks for; a matte piece returns no sheen."""
        flags = 0
        for flag in item.get("flags", []):
            flags |= getattr(p, _FLAG_BITS.get(flag, ""), 0)
        return p.createVisualShape(p.GEOM_MESH, fileName=path, meshScale=list(scale), flags=flags,
                                   specularColor=[float(item.get("specular", 0.0))] * 3, physicsClientId=self.cli)

    def _collision(self, item: Dict[str, Any], path: str, scale: List[float]) -> int:
        """What the drone can hit: the mesh itself, its bounding box, a trunk-sized cylinder, or nothing."""
        kind = item.get("collision", "none")
        low = [a * s for a, s in zip(item["bounds_min"], scale)]
        high = [a * s for a, s in zip(item["bounds_max"], scale)]
        if kind == "mesh":
            flags = p.GEOM_FORCE_CONCAVE_TRIMESH | getattr(p, "GEOM_CONCAVE_BVH_CACHE", 0)
            return p.createCollisionShape(p.GEOM_MESH, fileName=path, meshScale=list(scale), flags=flags,
                                          physicsClientId=self.cli)
        if kind == "box":
            half = [max((top - bottom) / 2.0, 0.01) for bottom, top in zip(low, high)]
            centre = [(top + bottom) / 2.0 for bottom, top in zip(low, high)]
            return p.createCollisionShape(p.GEOM_BOX, halfExtents=half, collisionFramePosition=centre,
                                          physicsClientId=self.cli)
        if kind == "cylinder":
            height = (high[2] - low[2]) * CONFIG["cylinder_height_share"]
            return p.createCollisionShape(p.GEOM_CYLINDER, radius=CONFIG["cylinder_radius_m"], height=height,
                                          collisionFramePosition=[0.0, 0.0, low[2] + height / 2.0],
                                          physicsClientId=self.cli)
        return -1


class SolarMovers:
    """The map's movers, driven a step at a time from the tables the export wrote.

    The pickup follows its table from the step this seed lets it leave and holds the last row at the dead end, the
    bird loops its carrier path with the wingbeat composed on top, and each goat walks an arc of its own with its
    foot dropped onto the terrain and its mesh rewritten from the stride at a phase that never stops moving.

    The stride is written in the goat's own frame, so the vertices go in as they are and the body carries them:
    turning them first would pay for the transform twice and put the animal in the wrong place.
    """

    def __init__(self, cli: int, asset_dir: str, manifest: Dict[str, Any], placed: Sequence[Tuple[Dict[str, Any], int]],
                 rules: Dict[str, Any], terrain: Sequence[int]):
        """Sort the placed mover bodies by what drives them and load the tables each of them reads."""
        self.cli = cli
        self.rules = rules
        self.step = -1
        self.terrain = frozenset(int(uid) for uid in terrain)
        self.bodies = [int(body) for _, body in placed]
        self._pickup: List[Tuple[int, List[float], List[float], Optional[List[float]]]] = []
        self._bird: List[Tuple[int, int]] = []
        self._goats: List[Dict[str, Any]] = []
        self._pickup_rows: List[List[float]] = []
        self._bird_rows: List[List[float]] = []
        self._bird_poses = np.zeros((0, 0, 7), dtype=np.float32)
        self._goat_poses = np.zeros((0, 0, 3), dtype=np.float32)
        self.stride: Dict[str, float] = {}
        self.goat_stride = False
        self.goat_mesh = False
        self._load(asset_dir, manifest, placed)
        # A family may look at the world before it steps it, so everything stands at step zero from the start.
        self._advance_pickup(0)
        self._advance_bird(0)
        self._advance_goats(0)

    @property
    def body_uids(self) -> frozenset:
        """Every mover body, for a family to keep out of the tagging, the obstacle cull and the clearance metric."""
        return frozenset(self.bodies)

    def advance(self, step: Optional[int] = None) -> None:
        """Move every mover to the world it holds at that simulator step, or at the next one when none is given."""
        self.step = int(step) if step is not None else self.step + 1
        self._advance_pickup(self.step)
        self._advance_bird(self.step)
        self._advance_goats(self.step)

    def _load(self, asset_dir: str, manifest: Dict[str, Any], placed: Sequence[Tuple[Dict[str, Any], int]]) -> None:
        """Read each mover's table and remember what every body needs to be placed from it."""
        for place, body in placed:
            folder = os.path.join(asset_dir, manifest["items"][place["item"]]["folder"])
            kind = place["mover"]
            if kind == "pickup":
                self._pickup_rows = _mover_table(os.path.join(folder, place["path"]))["rows"]
                self._pickup.append((int(body), list(place["local_position"]), list(place["local_quaternion"]),
                                     place.get("spin_axis")))
            elif kind == "bird":
                self._bird_rows = _mover_table(os.path.join(folder, place["path"]))["rows"]
                poses = _mover_poses(os.path.join(folder, place["poses"]))
                self._bird_poses = poses["poses"]
                self._bird.append((int(body), [str(name) for name in poses["parts"]].index(place["item"])))
            elif kind == "goat":
                self._load_goat(folder, manifest, place, int(body))
        self._pickup_delay_steps = int(round(self.rules["pickup_delay_s"] * CONFIG["step_hz"]))
        self._bird_offset = int(self.rules["bird_phase_share"] * max(len(self._bird_rows), 1))
        if self._goats:
            self._probe_goat_mesh()

    def _load_goat(self, folder: str, manifest: Dict[str, Any], place: Dict[str, Any], body: int) -> None:
        """Give one goat this seed's path and stride, from the herd table and the offset the seed drew."""
        herd = _mover_table(os.path.join(folder, place["path"]))
        self.goat_stride = bool(herd.get("poses"))
        if not self.goat_stride:
            # No stride shipped with the herd, so the animals stand where the manifest put them rather than skate.
            return
        if not len(self._goat_poses):
            table = _mover_poses(os.path.join(folder, herd["poses"]))
            self._goat_poses = table["poses"]
            # The pose table is the walk: its own length and rate answer for it, whatever the herd file remembers.
            self.stride = {"frames": len(table["poses"]),
                           "fps": float(table.get("hz", herd["cycle_fps"])),
                           "stride_m": float(table.get("stride_m", herd["stride_m"]))}
        plan = _goat_plans(self.stride, len(herd["goats"]), self.rules)[int(place["herd"])]
        start = herd["goats"][int(place["herd"])]["start"]
        self._goats.append(dict(plan, body=body, ground=float(start[2]),
                                foot=float(manifest["items"][place["item"]]["bounds_min"][2]),
                                start=[float(start[0]) + self.rules["goat_offset_m"][0],
                                       float(start[1]) + self.rules["goat_offset_m"][1]]))

    def _probe_goat_mesh(self) -> None:
        """Try the stride against the engine once; one that will not rewrite a rigid mesh still gets the herd, stiffly."""
        try:
            p.resetMeshData(self._goats[0]["body"], self._goat_poses[0].tolist(), physicsClientId=self.cli)
            self.goat_mesh = True
        except p.error:
            self.goat_mesh = False

    def _advance_pickup(self, step: int) -> None:
        """Set the truck's body, glass and wheels to the row this step reads, holding the last row at the dead end."""
        if not self._pickup:
            return
        index = 0 if self.rules["pickup_parked"] else min(max(step - self._pickup_delay_steps, 0),
                                                          len(self._pickup_rows) - 1)
        row = self._pickup_rows[index]
        for body, local_position, local_quaternion, axis in self._pickup:
            turn = local_quaternion
            if axis is not None:
                half = float(row[7]) / 2.0
                spin = [axis[0] * math.sin(half), axis[1] * math.sin(half), axis[2] * math.sin(half), math.cos(half)]
                turn = p.multiplyTransforms([0.0, 0.0, 0.0], spin, [0.0, 0.0, 0.0], turn)[1]
            position, orientation = p.multiplyTransforms(row[:3], row[3:7], local_position, turn)
            p.resetBasePositionAndOrientation(body, position, orientation, physicsClientId=self.cli)

    def _advance_bird(self, step: int) -> None:
        """Set every bird part to the carrier row this step reads and the wingbeat pose that belongs to it."""
        if not self._bird:
            return
        index = (step + self._bird_offset) % len(self._bird_rows)
        row = self._bird_rows[index]
        for body, part in self._bird:
            local = self._bird_poses[part, index]
            position, orientation = p.multiplyTransforms(row[:3], row[3:7], local[:3].tolist(), local[3:7].tolist())
            p.resetBasePositionAndOrientation(body, position, orientation, physicsClientId=self.cli)

    def _advance_goats(self, step: int) -> None:
        """Carry each goat along its arc, drop its foot on the terrain and rewrite its mesh from the stride."""
        seconds = step / float(CONFIG["step_hz"])
        for goat in self._goats:
            half = goat["turn_rad_s"] * seconds / 2.0
            heading = goat["heading_rad"] + 2.0 * half
            # The heading turns at a constant rate, so the track is the arc that rate draws, not a run of chords.
            gone = goat["speed_m_s"] * seconds * (math.sin(half) / half if half else 1.0)
            x = goat["start"][0] + gone * math.sin(goat["heading_rad"] + half)
            y = goat["start"][1] - gone * math.cos(goat["heading_rad"] + half)
            if step % CONFIG["goat_ray_steps"] == 0:
                goat["ground"] = self._ground(x, y, goat["ground"])
            # The goat is modelled facing its own -y, which a yaw of heading turns onto the track's direction.
            p.resetBasePositionAndOrientation(goat["body"], [x, y, goat["ground"] - goat["foot"]],
                                              [0.0, 0.0, math.sin(heading / 2.0), math.cos(heading / 2.0)],
                                              physicsClientId=self.cli)
            if self.goat_mesh and step % CONFIG["goat_mesh_steps"] == 0:
                pose = _stride_pose(self._goat_poses, seconds * goat["pose_fps"] + goat["phase"])
                p.resetMeshData(goat["body"], pose.tolist(), physicsClientId=self.cli)

    def _ground(self, x: float, y: float, last: float) -> float:
        """Terrain height under a point from a short downward ray, holding the last height where the ray misses it."""
        up, down = CONFIG["goat_ray_window_m"]
        hit = p.rayTest([x, y, last + up], [x, y, last - down], physicsClientId=self.cli)[0]
        return float(hit[3][2]) if int(hit[0]) in self.terrain else last


@lru_cache(maxsize=2)
def _forest_table(path: str) -> Dict[str, np.ndarray]:
    """Every tree of the forest: its mesh, position, yaw, scale, zone and rank, read once per process."""
    with np.load(path) as table:
        return {key: table[key] for key in table.files}


def _stand_forest(cli: int, asset_dir: str, forest: Dict[str, Any], densities: Dict[str, float]) -> Tuple[List[int], int]:
    """Stand this seed's share of the forest as one body, with a collision trunk for every near tree a drone meets.

    A tree stands when its rank is below the seed's density for its zone, the rule a single placement follows. The
    trees that stand become a forest file the renderer draws as one instanced batch per mesh. Returns the bodies and
    the number of trees standing.
    """
    instanced = getattr(p, "VISUAL_SHAPE_RENDER_INSTANCED", 0)
    if not instanced:
        raise RuntimeError("the solar forest needs a swarm-bullet3 wheel with VISUAL_SHAPE_RENDER_INSTANCED")
    folder = os.path.join(asset_dir, forest["folder"])
    table = _forest_table(os.path.join(folder, forest["table"]))
    limits = np.array([densities.get(name, 1.0) for name in forest["tiers"]])
    keep = table["rank"] < limits[table["tier"]]
    position, yaw, scale = table["position"][keep], table["yaw"][keep], table["scale"][keep]
    half = yaw * 0.5
    rows = np.column_stack([table["mesh"][keep], position, np.zeros(len(yaw)), np.zeros(len(yaw)), np.sin(half),
                            np.cos(half), scale])
    handle, path = tempfile.mkstemp(suffix=".fst")
    try:
        with os.fdopen(handle, "w") as out:
            out.write("".join(f"mesh {os.path.join(folder, name)}\n" for name in forest["meshes"]))
            np.savetxt(out, rows, fmt="%d " + " ".join(["%.4f"] * 10))
        flags = instanced | p.VISUAL_SHAPE_DOUBLE_SIDED_MULTIBODY | p.VISUAL_SHAPE_MATERIALS_FROM_MTL
        visual = p.createVisualShape(p.GEOM_MESH, fileName=path, flags=flags, specularColor=[0, 0, 0], physicsClientId=cli)
        bodies = [p.createMultiBody(0, -1, visual, physicsClientId=cli)]
    finally:
        os.remove(path)
    near = forest["tiers"].index("near")
    reach = (table["tier"][keep] == near) & (scale[:, 2] >= CONFIG["forest_trunk_min_m"])
    step = CONFIG["forest_trunk_step_m"]
    tall = np.maximum(step, np.round(scale[reach, 2] * CONFIG["cylinder_height_share"] / step) * step)
    # A body without a visual is drawn from its collision shape, which the renderer would still have to trace past;
    # a trunk's visual is one clear sliver under the ground instead, so the cylinder exists for collision alone.
    sliver = p.createVisualShape(p.GEOM_MESH, vertices=[[0, 0, -1.0], [0.001, 0, -1.0], [0, 0.001, -1.0]], indices=[0, 1, 2],
                                 rgbaColor=[1, 1, 1, 0], physicsClientId=cli)
    # The renderer pays for every body each frame, so neighbouring trunks share a body, as many as a compound holds.
    centres = position[reach] + np.column_stack([np.zeros((len(tall), 2)), tall / 2.0])
    cells = np.floor(centres[:, :2] / CONFIG["forest_trunk_cell_m"]).astype(np.int64)
    order = np.lexsort((cells[:, 1], cells[:, 0]))
    per = CONFIG["forest_trunks_per_body"]
    for start in range(0, len(order), per):
        group = order[start:start + per]
        base = centres[group].mean(0)
        shape = p.createCollisionShapeArray([p.GEOM_CYLINDER] * len(group), radii=[CONFIG["cylinder_radius_m"]] * len(group),
                                            lengths=tall[group].tolist(), collisionFramePositions=(centres[group] - base).tolist(),
                                            physicsClientId=cli)
        bodies.append(p.createMultiBody(0, shape, sliver, base.tolist(), physicsClientId=cli))
    return bodies, int(keep.sum())


def build_solar_map(seed: int = 0, cli: int = 0, asset_dir: Optional[str] = None,
                    groups: Optional[Tuple[str, ...]] = None) -> Dict[str, Any]:
    """Build the solar park inside an existing PyBullet world.

    Returns the body ids per group, the placement and body of every mover for the runtime to drive, the densities
    the seed drew, and the counts of what was placed. Naming groups builds only those, which is how a check loads
    the terrain on its own.
    """
    asset_dir = asset_dir or SOLAR_ASSET_DIR
    manifest = solar_manifest(asset_dir)
    shapes = _Shapes(cli, asset_dir, manifest["items"])
    densities = solar_densities(seed)
    bodies: Dict[str, List[int]] = {}
    movers: List[Tuple[Dict[str, Any], int]] = []
    triangles = 0
    for place in manifest["placements"]:
        item = manifest["items"][place["item"]]
        if groups is not None and item["group"] not in groups:
            continue
        if not _stands(place, densities):
            continue
        visual, collision = shapes.get(place["item"], place["scale"])
        body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision, baseVisualShapeIndex=visual,
                                 basePosition=place["position"], baseOrientation=place["quaternion"],
                                 physicsClientId=cli)
        p.changeVisualShape(body, -1, rgbaColor=[1, 1, 1, 1], specularColor=[float(item.get("specular", 0.0))] * 3,
                            physicsClientId=cli)
        bodies.setdefault(item["group"], []).append(body)
        if "mover" in place:
            movers.append((place, body))
        triangles += item["triangles"]
    trees = 0
    forest = manifest.get("forest")
    if forest and (groups is None or "plants" in groups):
        standing, trees = _stand_forest(cli, asset_dir, forest, densities)
        bodies.setdefault("plants", []).extend(standing)
    return {"bodies": bodies, "movers": movers, "densities": densities, "triangles": triangles, "trees": trees,
            "asset_dir": asset_dir, "body_count": sum(len(ids) for ids in bodies.values())}


def build_solar_movers(world: Dict[str, Any], seed: int = 0, cli: int = 0) -> SolarMovers:
    """The runtime that drives the movers build_solar_map placed, with the rules this seed drew for them."""
    asset_dir = world["asset_dir"]
    return SolarMovers(cli, asset_dir, solar_manifest(asset_dir), world["movers"], solar_mover_rules(seed),
                       world["bodies"].get("terrain", ()))
