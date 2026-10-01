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

The park is a real site, so its layout, fence, road and terrain never move. The map is a manifest of reusable pieces
with their placements; this builder loads what the manifest lists and lets the seed decide only what may vary:
how much of the vegetation outside the fence stands, how the movers cross it, and a small shift of every row of
tables, building and tree inside it, so no seed stands the park exactly where the survey does. Sun, sky and wind
belong to the family's options.

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
    "pickup_delay_s": (40.0, 365.0),             # the truck leaves after take-off and reaches the dead end before 390 s
    "bird_phase_share": (0.0, 1.0),              # where on its loop the bird starts, as a share of the whole loop
    "movers_left_out": ("goat",),                # the herd the manifest still lists; dogs replace it inside the fence
    "shift_seed_offset": 0x5B1F7,                # the park's shifts draw from their own stream
    "shifts": {                                  # per kind: move along and across its own axis (m), turn (deg), size, darkening
        "table": {"move_m": (0.3, 0.6), "turn_deg": 1.0, "scale": 0.03, "tint": 0.08},
        "building": {"move_m": (1.0, 1.0), "turn_deg": 5.0, "scale": 0.05, "tint": 0.10},
        "tree": {"move_m": (1.5, 1.5), "turn_deg": 180.0, "scale": 0.15, "tint": 0.10},
    },
    "shift_units": {"building": (("white_unit_north", "white_unit_south"), ("service_object",), ("slab",)),
                    "tree": (("olive_bark", "olive_leaves_0", "olive_leaves_1"),)},
    "table_legs": ("pile",),                     # items standing under a table, which move with the table over them
    "shift_scale_step": 0.01,                    # sizes snap to this, so a seed loads few distinct meshes
    "shift_tries": 8,                            # draws a unit gets to land clear before it stays where the survey stands it
    "fence_clear_m": 1.0,                        # a moved piece keeps this far inside the fence, or its surveyed distance if nearer
    "piece_gap_m": 1.0,                          # moved units keep this far apart, or their surveyed gap if nearer
    "shift_reach_m": 10.0,                       # units further apart than this in the survey can never meet
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


@lru_cache(maxsize=2)
def solar_fence(asset_dir: str = SOLAR_ASSET_DIR) -> np.ndarray:
    """The fence posts where the survey stands them, world metres east and north, in the order the fence joins them."""
    posts = np.array([place["position"][:2] for place in solar_manifest(asset_dir)["placements"]
                      if place["item"] == "fence_post"], dtype=float).reshape(-1, 2)
    order, left = [0] if len(posts) else [], set(range(1, len(posts)))
    while left:
        last = posts[order[-1]]
        nearest = min(left, key=lambda k: (float(np.hypot(*(posts[k] - last))), k))
        order.append(nearest)
        left.remove(nearest)
    ring = posts[order]
    ring.flags.writeable = False
    return ring


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
    """What this seed decides about the movers: when the truck leaves and where the bird is on its loop."""
    rng = random.Random((int(seed) ^ CONFIG["mover_seed_offset"]) & 0xFFFFFFFF)
    delay = rng.uniform(*CONFIG["pickup_delay_s"])
    return {"pickup_delay_s": delay, "bird_phase_share": rng.uniform(*CONFIG["bird_phase_share"])}


def _stands(place: Dict[str, Any], densities: Dict[str, float]) -> bool:
    """Whether a placement is part of this seed's world: fixed pieces always, vegetation by its rank, and no mover
    the map leaves out.

    A merged piece carries the lowest rank of the vegetation inside it as band_low, so a whole band stands or
    falls together; a single plant carries its own rank.
    """
    if place.get("mover") in CONFIG["movers_left_out"]:
        return False
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

    The pickup waits at the first row until the step this seed lets it leave, follows its table and holds the last
    row at the dead end; the bird loops its carrier path with the wingbeat composed on top.
    """

    def __init__(self, cli: int, asset_dir: str, manifest: Dict[str, Any], placed: Sequence[Tuple[Dict[str, Any], int]],
                 rules: Dict[str, Any]):
        """Sort the placed mover bodies by what drives them and load the tables each of them reads."""
        self.cli = cli
        self.rules = rules
        self.step = -1
        self.bodies = [int(body) for _, body in placed]
        self._pickup: List[Tuple[int, List[float], List[float], Optional[List[float]]]] = []
        self._bird: List[Tuple[int, int]] = []
        self._pickup_rows: List[List[float]] = []
        self._bird_rows: List[List[float]] = []
        self._bird_poses = np.zeros((0, 0, 7), dtype=np.float32)
        self._pickup_index = -1
        self._load(asset_dir, manifest, placed)
        # The bird's poses repeat with its loop, so each row's world poses are composed once and kept.
        self._bird_world = np.zeros((len(self._bird_rows), len(self._bird), 7))
        self._bird_known = np.zeros(len(self._bird_rows), dtype=bool)
        # A family may look at the world before it steps it, so everything stands at step zero from the start.
        self._advance_pickup(0)
        self._advance_bird(0)

    @property
    def body_uids(self) -> frozenset:
        """Every mover body, for a family to keep out of the tagging, the obstacle cull and the clearance metric."""
        return frozenset(self.bodies)

    def advance(self, step: Optional[int] = None) -> None:
        """Move every mover to the world it holds at that simulator step, or at the next one when none is given."""
        self.step = int(step) if step is not None else self.step + 1
        self._advance_pickup(self.step)
        self._advance_bird(self.step)

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
        self._pickup_delay_steps = int(round(self.rules["pickup_delay_s"] * CONFIG["step_hz"]))
        self._bird_offset = int(self.rules["bird_phase_share"] * max(len(self._bird_rows), 1))

    def _advance_pickup(self, step: int) -> None:
        """Set the truck's body, glass and wheels to the row this step reads, holding the last row at the dead end."""
        if not self._pickup:
            return
        index = min(max(step - self._pickup_delay_steps, 0), len(self._pickup_rows) - 1)
        # The truck holds its row while it waits and once it parks, and writing a pose a body holds changes nothing.
        if index == self._pickup_index:
            return
        self._pickup_index = index
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
        if not self._bird_known[index]:
            row = self._bird_rows[index]
            for slot, (_, part) in enumerate(self._bird):
                local = self._bird_poses[part, index]
                position, orientation = p.multiplyTransforms(row[:3], row[3:7], local[:3].tolist(), local[3:7].tolist())
                self._bird_world[index, slot] = [*position, *orientation]
            self._bird_known[index] = True
        for (body, _), pose in zip(self._bird, self._bird_world[index].tolist()):
            p.resetBasePositionAndOrientation(body, pose[:3], pose[3:], physicsClientId=self.cli)


def _fixed_body(cli: int, collision: int, visual: int, position: Sequence[float], orientation: Sequence[float]) -> int:
    """A piece nothing but a reset moves, as a plain static object instead of a jointed body the physics step solves.

    It stays awake, so every step re-measures its bounds as it did for the jointed body and contacts are found in the
    same order. A jointed body keeps its rotation as the quaternion of its matrix, so the piece is set to that
    quaternion read back from its own matrix and stands to the last bit where a jointed body would.
    """
    body = p.createMultiBody(0, collision, visual, position, orientation, useMaximalCoordinates=True, physicsClientId=cli)
    turned = p.getBasePositionAndOrientation(body, physicsClientId=cli)[1]
    p.resetBasePositionAndOrientation(body, position, turned, physicsClientId=cli)
    return body


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
        bodies.append(_fixed_body(cli, shape, sliver, base.tolist(), [0.0, 0.0, 0.0, 1.0]))
    return bodies, int(keep.sum())


def _outline(item: Dict[str, Any], place: Dict[str, Any]) -> Tuple[float, float, float]:
    """The circle a placed piece covers seen from above: the centre of its bounds in the world and their larger half
    width, which for a tree crown is its radius."""
    low = np.array(item["bounds_min"][:2]) * place["scale"][:2]
    high = np.array(item["bounds_max"][:2]) * place["scale"][:2]
    x, y, z, w = place["quaternion"]
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    cx, cy = (low + high) / 2.0
    c, s = math.cos(yaw), math.sin(yaw)
    return (float(place["position"][0] + c * cx - s * cy), float(place["position"][1] + s * cx + c * cy),
            float(np.max(high - low) / 2.0))


def _yaw(quaternion: Sequence[float]) -> float:
    """The heading of a rotation about the vertical, radians."""
    x, y, z, w = quaternion
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _turned(quaternion: Sequence[float], angle: float) -> List[float]:
    """A rotation turned further about the world vertical by angle radians, which keeps any tilt it carries."""
    x, y, z, w = quaternion
    s, c = math.sin(angle / 2.0), math.cos(angle / 2.0)
    return [c * x - s * y, c * y + s * x, c * z + s * w, c * w - s * z]


def _corners(piece: Dict[str, Any], anchor: np.ndarray, yaw: float, scale: float) -> np.ndarray:
    """The four corners of a piece's outline seen from above, standing on anchor at a heading and a size."""
    (x0, y0), (x1, y1) = piece["low"] * scale, piece["high"] * scale
    c, s = math.cos(yaw), math.sin(yaw)
    local = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    return local @ np.array([[c, s], [-s, c]]) + anchor


def _gap(a: np.ndarray, b: np.ndarray) -> float:
    """The distance between two convex outlines, 0 when they overlap."""
    for poly in (a, b):
        edges = np.roll(poly, -1, axis=0) - poly
        normals = np.column_stack([-edges[:, 1], edges[:, 0]])
        pa, pb = a @ normals.T, b @ normals.T
        if np.any((pa.max(axis=0) < pb.min(axis=0)) | (pb.max(axis=0) < pa.min(axis=0))):
            return float(min(_point_gap(a, b), _point_gap(b, a)))
    return 0.0


def _point_gap(points: np.ndarray, ring: np.ndarray) -> float:
    """The smallest distance from any of the points to the sides of a closed ring."""
    side = np.roll(ring, -1, axis=0) - ring
    rel = points[:, None, :] - ring[None, :, :]
    t = np.clip((rel * side).sum(axis=2) / np.maximum((side * side).sum(axis=1), 1e-12), 0.0, 1.0)
    return float(np.min(np.linalg.norm(rel - t[..., None] * side, axis=2)))


def _inside(ring: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Which points lie inside the closed ring, by counting the sides a ray from each crosses."""
    a, b = ring, np.roll(ring, -1, axis=0)
    x, y = points[:, :1], points[:, 1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        crosses = ((a[:, 1] > y) != (b[:, 1] > y)) & (x < a[:, 0] + (y - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1]))
    return crosses.sum(axis=1) % 2 == 1


def _fence_gap(outline: np.ndarray, ring: np.ndarray) -> float:
    """How far an outline stands from the fence: positive inside it, negative outside it, 0 when the fence crosses it."""
    inside = _inside(ring, outline)
    if np.any(_inside(outline, ring)) or inside.any() != inside.all():
        return 0.0
    gap = min(_point_gap(outline, ring), _point_gap(ring, outline))
    return gap if inside.all() else -gap


def _local(piece: Dict[str, Any], xy: Sequence[float]) -> np.ndarray:
    """A world point in the piece's own axes, about its anchor: along its long side, then across it."""
    c, s = math.cos(piece["yaw"]), math.sin(piece["yaw"])
    rel = np.asarray(xy[:2], dtype=float) - piece["anchor"]
    return np.array([c * rel[0] + s * rel[1], -s * rel[0] + c * rel[1]])


def _piece(placements: Sequence[Dict[str, Any]], items: Dict[str, Any], indices: List[int]) -> Dict[str, Any]:
    """The placements standing on one spot as one piece: where it stands, how it turns, the outline its parts cover
    seen from above, and the height its size is taken from, the lowest point of its parts."""
    first = placements[indices[0]]
    low = np.min([np.array(items[placements[i]["item"]]["bounds_min"]) * placements[i]["scale"] for i in indices], axis=0)
    high = np.max([np.array(items[placements[i]["item"]]["bounds_max"]) * placements[i]["scale"] for i in indices], axis=0)
    return {"parts": list(indices), "legs": [], "anchor": np.array(first["position"][:2], dtype=float),
            "yaw": _yaw(first["quaternion"]), "low": low[:2], "high": high[:2], "base": float(first["position"][2] + low[2])}


@lru_cache(maxsize=2)
def _shift_units(asset_dir: str) -> Tuple[Dict[str, Any], ...]:
    """Every part of the park that shifts as one: each row of tables with the legs under it, each building and the tree.

    A row moves as one straight string, as the tables in it stand end to end. Each unit keeps the outlines of its
    pieces as surveyed, its distance to the fence, and its gap to every unit near enough to meet it.
    """
    manifest = solar_manifest(asset_dir)
    placements, items = manifest["placements"], manifest["items"]
    ring = solar_fence(asset_dir)
    if len(ring) < 3:
        # A map without a fence has no line to keep a shift inside, so nothing on it moves.
        return ()
    spots: Dict[Tuple[str, Any, Tuple[float, ...]], List[int]] = {}
    for index, place in enumerate(placements):
        if "row" in place:
            spots.setdefault(("table", place["row"], tuple(place["position"])), []).append(index)
    for kind, groups in CONFIG["shift_units"].items():
        for number, names in enumerate(groups):
            for index, place in enumerate(placements):
                if place["item"] in names:
                    spots.setdefault((kind, number, tuple(place["position"])), []).append(index)
    grouped: Dict[Tuple[str, Any], List[Dict[str, Any]]] = {}
    for (kind, number, _), indices in spots.items():
        grouped.setdefault((kind, number), []).append(_piece(placements, items, indices))
    tables = [piece for (kind, _), pieces in grouped.items() if kind == "table" for piece in pieces]
    for index, place in enumerate(placements):
        if place["item"] in CONFIG["table_legs"] and tables:
            # The table whose outline the leg stands deepest inside carries it.
            outside = [float(np.max(np.abs(_local(table, place["position"]) - (table["low"] + table["high"]) / 2.0)
                                    - (table["high"] - table["low"]) / 2.0)) for table in tables]
            tables[int(np.argmin(outside))]["legs"].append(index)
    kinds = list(CONFIG["shifts"])
    units = []
    for (kind, _), pieces in sorted(grouped.items(), key=lambda entry: (kinds.index(entry[0][0]), entry[0][1])):
        outlines = [_corners(piece, piece["anchor"], piece["yaw"], 1.0) for piece in pieces]
        fence = [_fence_gap(outline, ring) for outline in outlines]
        # A unit keeps the side of the fence it stands on; one the fence runs through is built into it and stays.
        side = 1.0 if min(fence) > 0.0 else -1.0 if max(fence) < 0.0 else 0.0
        units.append({"kind": kind, "pieces": pieces, "outlines": outlines,
                      "pivot": np.mean([piece["anchor"] for piece in pieces], axis=0), "yaw": pieces[0]["yaw"],
                      "fence_side": side, "fence_m": min(abs(gap) for gap in fence)})
    for unit in units:
        unit["gaps"] = {}
        for other_number, other in enumerate(units):
            if other is not unit:
                gap = min(_gap(a, b) for a in unit["outlines"] for b in other["outlines"])
                if gap < CONFIG["shift_reach_m"]:
                    unit["gaps"][other_number] = gap
    return tuple(units)


def _pose(unit: Dict[str, Any], draw: Tuple[float, float, float, float], piece: Dict[str, Any]) -> Tuple[np.ndarray, float]:
    """Where a piece of the unit stands and its heading once the unit takes a draw: moved along and across the
    unit's own axis, turned about its centre and sized about it."""
    along, across, turn, scale = draw
    c, s = math.cos(unit["yaw"]), math.sin(unit["yaw"])
    move = np.array([c * along - s * across, s * along + c * across])
    ct, st = math.cos(turn), math.sin(turn)
    rel = (piece["anchor"] - unit["pivot"]) * scale
    return unit["pivot"] + move + np.array([ct * rel[0] - st * rel[1], st * rel[0] + ct * rel[1]]), piece["yaw"] + turn


def _clear(number: int, units: Sequence[Dict[str, Any]], outlines: List[np.ndarray],
           standing: Dict[int, List[np.ndarray]], ring: np.ndarray) -> bool:
    """Whether a unit's outlines keep to their side of the fence and clear of it and of every unit near it, each by
    the configured distance or by what the survey leaves when that is less."""
    unit = units[number]
    fence_m = min(CONFIG["fence_clear_m"], unit["fence_m"])
    if any(_fence_gap(outline, ring) * unit["fence_side"] < fence_m for outline in outlines):
        return False
    for other, surveyed in unit["gaps"].items():
        need = min(CONFIG["piece_gap_m"], surveyed)
        if any(_gap(a, b) < need for a in outlines for b in standing[other]):
            return False
    return True


def solar_shifts(seed: int, asset_dir: str = SOLAR_ASSET_DIR) -> Dict[int, Dict[str, Any]]:
    """Where this seed stands every placement that shifts: its moved position, rotation and size, its tint, and the
    spots its piece reads the ground at where the survey stands it and where the seed does.

    Each unit draws a move along and across its own axis, a turn about its centre and a size, and keeps the first
    draw that leaves it inside the fence and clear of every other unit as it stands so far; a unit no draw clears
    stays where the survey stands it, which every earlier unit was already kept clear of. Heights are the survey's,
    sized about the piece's lowest point; the builder drops each piece by the ground it moved over. Each piece
    draws its own tint, a little darker than the survey's in each colour, as a brighter one would overflow the
    renderer.
    """
    manifest = solar_manifest(asset_dir)
    placements = manifest["placements"]
    units = _shift_units(asset_dir)
    ring = solar_fence(asset_dir)
    rng = random.Random((int(seed) ^ CONFIG["shift_seed_offset"]) & 0xFFFFFFFF)
    step = CONFIG["shift_scale_step"]
    standing = {number: list(unit["outlines"]) for number, unit in enumerate(units)}
    shifts: Dict[int, Dict[str, Any]] = {}
    for number, unit in enumerate(units):
        spec = CONFIG["shifts"][unit["kind"]]
        chosen = (0.0, 0.0, 0.0, 1.0)
        for _ in range(CONFIG["shift_tries"] if unit["fence_side"] else 0):
            draw = (rng.uniform(-spec["move_m"][0], spec["move_m"][0]), rng.uniform(-spec["move_m"][1], spec["move_m"][1]),
                    math.radians(rng.uniform(-spec["turn_deg"], spec["turn_deg"])),
                    1.0 + step * round(rng.uniform(-spec["scale"], spec["scale"]) / step))
            poses = [_pose(unit, draw, piece) for piece in unit["pieces"]]
            outlines = [_corners(piece, anchor, yaw, draw[3]) for piece, (anchor, yaw) in zip(unit["pieces"], poses)]
            if _clear(number, units, outlines, standing, ring):
                chosen, standing[number] = draw, outlines
                break
        turn, scale = chosen[2], chosen[3]
        ct, st = math.cos(turn), math.sin(turn)
        for piece in unit["pieces"]:
            anchor, _ = _pose(unit, chosen, piece)
            tint = [round(1.0 - rng.uniform(0.0, spec["tint"]), 4) for _ in range(3)]
            for index in piece["parts"] + piece["legs"]:
                place = placements[index]
                rel = np.array(place["position"][:2]) - piece["anchor"]
                x, y = anchor + scale * np.array([ct * rel[0] - st * rel[1], st * rel[0] + ct * rel[1]])
                z = piece["base"] + scale * (place["position"][2] - piece["base"])
                size = place["scale"] if index in piece["legs"] else [scale * v for v in place["scale"]]
                shifts[index] = {"position": [round(float(x), 4), round(float(y), 4), round(float(z), 4)],
                                 "quaternion": _turned(place["quaternion"], turn), "scale": size, "tint": tint,
                                 "ground": (tuple(map(float, piece["anchor"])), tuple(map(float, anchor)))}
    return shifts


def _ground_rise(cli: int, terrain: frozenset, spots: Sequence[Tuple[np.ndarray, np.ndarray]]) -> List[float]:
    """How much higher the terrain lies under each moved spot than under the same piece where the survey stands it;
    0 where either ray meets anything but the terrain."""
    if not terrain or not spots:
        return [0.0] * len(spots)
    points = [xy for pair in spots for xy in pair]
    hits = p.rayTestBatch([[float(x), float(y), 1000.0] for x, y in points],
                          [[float(x), float(y), -1000.0] for x, y in points], physicsClientId=cli)
    rise = []
    for before, after in zip(hits[0::2], hits[1::2]):
        on_terrain = int(before[0]) in terrain and int(after[0]) in terrain
        rise.append(float(after[3][2]) - float(before[3][2]) if on_terrain else 0.0)
    return rise


def build_solar_map(seed: int = 0, cli: int = 0, asset_dir: Optional[str] = None,
                    groups: Optional[Tuple[str, ...]] = None) -> Dict[str, Any]:
    """Build the solar park inside an existing PyBullet world.

    Returns the body ids per group, the placement and body of every mover for the runtime to drive, the outline of
    every standing park piece the simulator lets a drone pass through (a tree's crown), the glass of the panel tables,
    the densities the seed drew, and the counts of what was placed. Naming groups builds only those, which is how a
    check loads the terrain on its own.

    The pieces this seed shifts stand last, once the terrain is in, so each is dropped by the ground it moved over.
    """
    asset_dir = asset_dir or SOLAR_ASSET_DIR
    manifest = solar_manifest(asset_dir)
    shapes = _Shapes(cli, asset_dir, manifest["items"])
    densities = solar_densities(seed)
    shifts = solar_shifts(seed, asset_dir)
    bodies: Dict[str, List[int]] = {}
    movers: List[Tuple[Dict[str, Any], int]] = []
    passable: List[Tuple[float, float, float]] = []
    glass: List[int] = []
    triangles = 0

    def stand(place: Dict[str, Any], item: Dict[str, Any], tint: Sequence[float]) -> None:
        """Create one placed piece in the world and file its body where the caller will look for it."""
        nonlocal triangles
        visual, collision = shapes.get(place["item"], place["scale"])
        body = _fixed_body(cli, collision, visual, place["position"], place["quaternion"])
        p.changeVisualShape(body, -1, rgbaColor=[*tint, 1], specularColor=[float(item.get("specular", 0.0))] * 3,
                            physicsClientId=cli)
        bodies.setdefault(item["group"], []).append(body)
        if "mover" in place:
            movers.append((place, body))
        if item["group"] == "park" and item.get("collision", "none") == "none":
            passable.append(_outline(item, place))
        if item["group"] == "park" and "glass" in item.get("flags", ()):
            glass.append(body)
        triangles += item["triangles"]

    shifted = []
    for index, place in enumerate(manifest["placements"]):
        item = manifest["items"][place["item"]]
        if groups is not None and item["group"] not in groups:
            continue
        if not _stands(place, densities):
            continue
        if index in shifts:
            shifted.append((place, item, shifts[index]))
            continue
        stand(place, item, (1.0, 1.0, 1.0))
    spots = list(dict.fromkeys(shift["ground"] for _, _, shift in shifted))
    rise = dict(zip(spots, _ground_rise(cli, frozenset(bodies.get("terrain", ())), spots)))
    for place, item, shift in shifted:
        x, y, z = shift["position"]
        moved = dict(place, position=[x, y, z + rise[shift["ground"]]], quaternion=shift["quaternion"],
                     scale=shift["scale"])
        stand(moved, item, shift["tint"])
    trees = 0
    forest = manifest.get("forest")
    if forest and (groups is None or "plants" in groups):
        standing, trees = _stand_forest(cli, asset_dir, forest, densities)
        bodies.setdefault("plants", []).extend(standing)
    return {"bodies": bodies, "movers": movers, "passable": passable, "glass": glass, "densities": densities,
            "triangles": triangles, "trees": trees, "asset_dir": asset_dir,
            "body_count": sum(len(ids) for ids in bodies.values())}


def build_solar_movers(world: Dict[str, Any], seed: int = 0, cli: int = 0) -> SolarMovers:
    """The runtime that drives the movers build_solar_map placed, with the rules this seed drew for them."""
    asset_dir = world["asset_dir"]
    return SolarMovers(cli, asset_dir, solar_manifest(asset_dir), world["movers"], solar_mover_rules(seed))
