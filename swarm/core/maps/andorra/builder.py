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

"""Load an area into a client: its ground, trees, rocks, houses, water and the mountains around it.

Trees and rocks are drawn as one instanced body per group and grass tufts and far trees as instanced fields, each from
a forest file written to a scratch folder. Every other piece is its own body, each terrain tile with a concave collider
in collision group 2. The solid pieces that are not terrain are joined with their neighbours on the same 32 m tile into
compound colliders of at most sixteen children in group 1, tree trunks as cheap prisms, so a ray masked to group 2
reads the ground alone. Placements the converter marks "collision": "none" (pebbles) are drawn and never collide.

Two optional boxes, (west, south, east, north) in area metres, cut the load. keep_xy builds only the terrain tiles,
placements and tufts whose footprint reaches into it; the horizon and the far trees always stay, so the skyline is
whole. collide_xy gives solid shapes only to pieces whose footprint reaches into it. Every terrain tile that is built
collides, since a ground ray may fall anywhere the cameras see.

The tufts can be cut closer and stood in pieces: tuft_xy keeps only the tufts inside its own box, and tuft_cell_m
stands them as one body a square cell of that side, on a grid fixed in area metres, so a caller can draw the grass of
one cell without the rest.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pybullet as p

from swarm.core.maps.solar.builder import _four_decimals

from .rendering import LIGHT_GRADE

CONFIG = {"core_m": 400, "tile_m": 32, "compound_children": 16}
TUFT_REACH = 0.7072                    # a unit card's corner from its middle, the widest any tuft or grit card reaches
TERRAIN_GROUP = 2                      # the collision group a ground ray masks for
SOLID_GROUP = 1
ALWAYS_KEPT = ("horizon",)             # groups no load box cuts
RAY_REACH_M = 1000.0                   # above and below every area's ground
GROUND_RAYS = 16

Box = Tuple[float, float, float, float]


class Patch(NamedTuple):
    """One cell of the tufts stood as a body of its own, and the box its tufts fill: (west, south, bottom) and (east,
    north, top) in area metres."""

    body: int
    low: Tuple[float, float, float]
    high: Tuple[float, float, float]


class LoadedArea(NamedTuple):
    """What a load built: the manifest it read, every drawn body, the terrain tiles among them, the colliders, the
    bodies of each group drawn one placement to a body (terrain, buildings, water and the like), the tufts' patches
    when they were stood by the cell, and the one instanced body of each group drawn as a forest (trees, rocks)."""

    manifest: Dict[str, Any]
    bodies: List[int]
    terrain: frozenset
    colliders: List[int]
    groups: Optional[Dict[str, frozenset]] = None
    patches: Tuple[Patch, ...] = ()
    forests: Optional[Dict[str, int]] = None


def asset_path(root: Path, manifest: Dict[str, Any], item: Dict[str, Any], name: str) -> Path:
    """Path of one of an item's files: in the area folder, or in the shared folder the manifest names for items marked
    shared."""
    base = Path(root)
    if item.get("root") == "shared":
        base = base / manifest["andorra"]["shared_root"]
    # Tidied, so a file every area shares is one path to the engine whichever area names it, links or none.
    return Path(os.path.normpath(base / item["folder"] / name))


def field_mesh(root: Path, manifest: Dict[str, Any], field: Dict[str, Any], name: str) -> Path:
    """Path of one unit mesh of an instanced field, resolved through its manifest item when the manifest lists it."""
    item = manifest["items"].get(name)
    return asset_path(root, manifest, item, item["obj"]) if item else Path(root) / field["folder"] / (name + ".obj")


def visual_path(root: Path, manifest: Dict[str, Any], item: Dict[str, Any]) -> Path:
    """The mesh an item is drawn with: its own visual mesh when it has one, otherwise the mesh it collides with."""
    return asset_path(root, manifest, item, item.get("visual", item["obj"]))


def validate_manifest(manifest: Dict[str, Any], root: Path) -> None:
    """Fail before building when a referenced mesh, texture or placement table is absent, or the area was converted
    under another light grade."""
    for item in manifest["items"].values():
        for key in ("obj", "texture", "visual"):
            if key not in item:
                continue
            path = asset_path(root, manifest, item, item[key])
            if not path.is_file():
                raise FileNotFoundError(f"area asset missing: {path}")
    for key in ("tuft_field", "far_trees"):
        field = manifest.get(key)
        if field:
            for path in [Path(root) / field["folder"] / field["table"], *(field_mesh(root, manifest, field, name) for name in field["items"])]:
                if not path.is_file():
                    raise FileNotFoundError(f"area asset missing: {path}")
    light = manifest.get("andorra", {}).get("light")
    # Every area is converted under one light grade; a manifest from another grade would render wrongly here.
    if light is not None and any(abs(float(light[key]) - value) > 1e-9 for key, value in LIGHT_GRADE.items() if key in light):
        raise ValueError(f"area light grade {light} differs from the renderer's {LIGHT_GRADE}")


def footprints(manifest: Dict[str, Any], places: Sequence[Dict[str, Any]]) -> np.ndarray:
    """(west, south, east, north) of each placement's item box as it stands, turned and scaled, in area metres; just
    its position for an item with no box."""
    count = len(places)
    low, high = np.zeros((count, 3)), np.zeros((count, 3))
    for n, place in enumerate(places):
        item = manifest["items"][place["item"]]
        if "bounds_min" in item:
            low[n] = np.asarray(item["bounds_min"], dtype=float) * place["scale"]
            high[n] = np.asarray(item["bounds_max"], dtype=float) * place["scale"]
    position = np.array([place["position"] for place in places], dtype=float).reshape(count, 3)
    x, y, z, w = np.array([place["quaternion"] for place in places], dtype=float).reshape(count, 4).T
    # The two horizontal rows of each placement's rotation matrix.
    rows = np.stack([np.column_stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)]),
                     np.column_stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)])], axis=1)
    centre = position[:, :2] + np.einsum("nij,nj->ni", rows, (low + high) / 2.0)
    reach = np.einsum("nij,nj->ni", np.abs(rows), (high - low) / 2.0)
    return np.column_stack([centre - reach, centre + reach])


def overlaps(boxes: np.ndarray, box: Optional[Box]) -> np.ndarray:
    """Which (west, south, east, north) boxes reach into a box; all of them when no box is given."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if box is None:
        return np.ones(len(boxes), dtype=bool)
    west, south, east, north = box
    return (boxes[:, 0] <= east) & (boxes[:, 2] >= west) & (boxes[:, 1] <= north) & (boxes[:, 3] >= south)


def field_placements(table: np.ndarray) -> np.ndarray:
    """Forest-file rows from a field table: item, position, a yaw about z as XYZW quaternion, and the (width, width,
    height) scale."""
    item, x, y, z, yaw, width, height = (table[:, i] for i in range(7))
    half = yaw * 0.5
    zeros = np.zeros(len(table))
    return np.column_stack([item, x, y, z, zeros, zeros, np.sin(half), np.cos(half), width, width, height])


def rotate(quaternion: Sequence[float], vector: Sequence[float]) -> List[float]:
    """Rotate a vector by an XYZW quaternion with the engine's own matrix."""
    return (np.array(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3) @ np.asarray(vector, float)).tolist()


def tile_of(position: Sequence[float]) -> Tuple[int, int]:
    """Index of the 32 m ground tile a placement stands on, clamped to the core."""
    count = -(-int(CONFIG["core_m"]) // int(CONFIG["tile_m"]))
    return tuple(min(count - 1, max(0, int((c + CONFIG["core_m"] / 2) // CONFIG["tile_m"]))) for c in position[:2])


def tile_colliders(children: Dict[Tuple[int, int], List[Dict[str, Any]]], cli: int) -> List[int]:
    """Join neighbouring solid pieces into static compounds of at most sixteen children."""
    sliver = p.createVisualShape(p.GEOM_MESH, vertices=[[0, 0, -1.0], [0.001, 0, -1.0], [0, 0.001, -1.0]], indices=[0, 1, 2],
                                 rgbaColor=[1, 1, 1, 0], physicsClientId=cli)
    per = CONFIG["compound_children"]
    uids = []
    for key in sorted(children):
        group = children[key]
        for start in range(0, len(group), per):
            part = group[start:start + per]
            columns = {name: [child[name] for child in part] for name in part[0]}
            shape = p.createCollisionShapeArray(
                shapeTypes=columns["type"], radii=columns["radius"], halfExtents=columns["half"], lengths=columns["length"],
                fileNames=columns["file"], meshScales=columns["scale"], flags=columns["flags"],
                collisionFramePositions=columns["position"], collisionFrameOrientations=columns["quaternion"], physicsClientId=cli)
            if shape < 0:
                raise RuntimeError("compound collision shape was refused by the engine")
            uid = p.createMultiBody(0, shape, sliver, [0, 0, 0], useMaximalCoordinates=True, physicsClientId=cli)
            # Group 1 keeps the ground height ray, which masks for group 2, from hitting a tree or a rock.
            p.setCollisionFilterGroupMask(uid, -1, SOLID_GROUP, -1, physicsClientId=cli)
            uids.append(uid)
    return uids


def collider_child(item: Dict[str, Any], place: Dict[str, Any], filename: Path, prism: Path) -> Optional[Dict[str, Any]]:
    """Describe one placed piece as a child of its tile's compound collider, or None when it does not collide."""
    # The converter marks pebbles too small to matter on the placement itself; thousands of them would make every step slow.
    if place.get("collision") == "none":
        return None
    low, high = np.array(item["bounds_min"]) * place["scale"], np.array(item["bounds_max"]) * place["scale"]
    child = {"type": p.GEOM_MESH, "radius": 0.5, "half": [1, 1, 1], "length": 1, "file": str(filename), "scale": place["scale"],
             "flags": (p.GEOM_FORCE_CONCAVE_TRIMESH | getattr(p, "GEOM_CONCAVE_BVH_CACHE", 0)), "position": place["position"],
             "quaternion": place["quaternion"]}
    if item["collision"] == "box":
        offset = rotate(place["quaternion"], (low + high) / 2)
        child.update(type=p.GEOM_BOX, half=np.maximum((high - low) / 2, 0.01).tolist(), file="", scale=[1, 1, 1], flags=0,
                     position=(np.array(place["position"]) + offset).tolist())
    elif item["collision"] == "cylinder":
        # A scaled prism mesh steps far cheaper than a cylinder primitive inside a compound.
        height = max(0.1, float(high[2] - low[2]) * 0.6)
        offset = rotate(place["quaternion"], [0, 0, float(low[2])])
        child.update(file=str(prism), scale=[0.13, 0.13, height], position=(np.array(place["position"]) + offset).tolist())
    elif item["collision"] != "mesh":
        return None
    return child


def field_flags() -> int:
    """How an instanced field of cards is drawn: instanced, its materials from its own files, both sides."""
    return p.VISUAL_SHAPE_RENDER_INSTANCED | p.VISUAL_SHAPE_MATERIALS_FROM_MTL | p.VISUAL_SHAPE_DOUBLE_SIDED_MULTIBODY


def write_forest(path: Path, meshes: Sequence[Path], rows: np.ndarray) -> None:
    """Write binary placements with exactly the numbers parsed from the original four-decimal forest text."""
    header = "".join(f"mesh {mesh}\n" for mesh in meshes)
    if getattr(p, "FOREST_FILE_BINARY", 0):
        rows = np.asarray(rows, dtype=float).copy()
        rows[:, 0] = np.trunc(rows[:, 0])
        rows[:, 1:] = _four_decimals(rows[:, 1:])
        with path.open("wb") as out:
            out.write(header.encode())
            out.write(f"binary {len(rows)}\n".encode())
            out.write(np.ascontiguousarray(rows, dtype="<f8").tobytes())
    else:
        with path.open("w") as out:
            out.write(header)
            np.savetxt(out, rows, fmt="%d " + " ".join(["%.4f"] * 10))


def stand_field(root: Path, cli: int, field: Dict[str, Any], work: Path, manifest: Dict[str, Any],
                keep_xy: Optional[Box] = None) -> Optional[int]:
    """Draw a field (tufts, far crowns) as one instanced body: a small library of unit meshes placed from a compact
    table, only its rows inside keep_xy when given; None when no row is left."""
    table = np.load(root / field["folder"] / field["table"])
    if keep_xy is not None:
        table = table[overlaps(np.column_stack([table[:, 1:3], table[:, 1:3]]), keep_xy)]
    if not len(table):
        return None
    path = work / (Path(field["table"]).stem + ".fst")
    write_forest(path, [field_mesh(root, manifest, field, name) for name in field["items"]], field_placements(table))
    visual = p.createVisualShape(p.GEOM_MESH, fileName=str(path), flags=field_flags(), specularColor=[0, 0, 0], physicsClientId=cli)
    return p.createMultiBody(0, -1, visual, useMaximalCoordinates=True, physicsClientId=cli)


def stand_patches(root: Path, cli: int, field: Dict[str, Any], work: Path, manifest: Dict[str, Any], keep_xy: Optional[Box],
                  cell_m: float) -> List[Patch]:
    """Draw a field's rows inside keep_xy as one instanced body a square cell of cell_m, the cells on a grid fixed in
    area metres and stood west to east, then south to north; each body lists only the meshes its own rows use."""
    table = np.load(root / field["folder"] / field["table"])
    if keep_xy is not None:
        table = table[overlaps(np.column_stack([table[:, 1:3], table[:, 1:3]]), keep_xy)]
    cells = np.floor(table[:, 1:3] / cell_m).astype(np.int64)
    order = np.lexsort((cells[:, 1], cells[:, 0]))
    table, cells = table[order], cells[order]
    starts = np.flatnonzero(np.r_[True, np.any(cells[1:] != cells[:-1], axis=1)]) if len(table) else np.zeros(0, dtype=int)
    patches = []
    for first, last in zip(starts, np.r_[starts[1:], len(table)]):
        rows = table[first:last].copy()
        kinds, rows[:, 0] = np.unique(rows[:, 0].astype(np.int64), return_inverse=True)
        path = work / f"{Path(field['table']).stem}_{cells[first][0]}_{cells[first][1]}.fst"
        write_forest(path, [field_mesh(root, manifest, field, field["items"][kind]) for kind in kinds], field_placements(rows))
        visual = p.createVisualShape(p.GEOM_MESH, fileName=str(path), flags=field_flags(), specularColor=[0, 0, 0],
                                     physicsClientId=cli)
        body = p.createMultiBody(0, -1, visual, useMaximalCoordinates=True, physicsClientId=cli)
        reach = rows[:, 5] * TUFT_REACH
        low = (float(np.min(rows[:, 1] - reach)), float(np.min(rows[:, 2] - reach)), float(np.min(rows[:, 3] - rows[:, 6] * 0.5)))
        high = (float(np.max(rows[:, 1] + reach)), float(np.max(rows[:, 2] + reach)), float(np.max(rows[:, 3] + rows[:, 6])))
        patches.append(Patch(body, low, high))
    return patches


def trunk_prism(work: Path) -> Path:
    """Write a six-sided unit prism, radius 1 and height 1, whose scaled copies stand in for tree trunks as cheap meshes."""
    path = work / "trunk_prism.obj"
    angles = [math.pi / 3 * k for k in range(6)]
    lines = [f"v {math.cos(a):.4f} {math.sin(a):.4f} {z}" for z in (0, 1) for a in angles]
    faces = [f"f {k + 1} {(k + 1) % 6 + 1} {(k + 1) % 6 + 7}\nf {k + 1} {(k + 1) % 6 + 7} {k + 7}" for k in range(6)]
    faces += ["f 1 3 2", "f 1 4 3", "f 1 5 4", "f 1 6 5", "f 7 8 9", "f 7 9 10", "f 7 10 11", "f 7 11 12"]
    path.write_text("\n".join(lines + faces) + "\n")
    return path


def stand_forest(root: Path, cli: int, manifest: Dict[str, Any], group: str, places: Sequence[Dict[str, Any]],
                 work: Path) -> Optional[int]:
    """Draw every given placement of one group as a single instanced body, one mesh per distinct item; None when the
    group has none."""
    names = sorted({pl["item"] for pl in places if manifest["items"][pl["item"]]["group"] == group})
    if not names:
        return None
    slot = {name: i for i, name in enumerate(names)}
    rows = [[slot[pl["item"]], *pl["position"], *pl["quaternion"], *pl["scale"]] for pl in places if pl["item"] in slot]
    path = work / f"{group}.fst"
    write_forest(path, [visual_path(root, manifest, manifest["items"][name]) for name in names], np.asarray(rows))
    flags = p.VISUAL_SHAPE_RENDER_INSTANCED | p.VISUAL_SHAPE_MATERIALS_FROM_MTL
    # One forest body draws the whole group, so a single leaf-card piece (a bush's visual) makes it double-sided.
    if any("double_sided" in manifest["items"][name]["flags"] for name in names):
        flags |= p.VISUAL_SHAPE_DOUBLE_SIDED_MULTIBODY
    visual = p.createVisualShape(p.GEOM_MESH, fileName=str(path), flags=flags, specularColor=[0, 0, 0], physicsClientId=cli)
    return p.createMultiBody(0, -1, visual, useMaximalCoordinates=True, physicsClientId=cli)


def load_area(root: Path, cli: int, work: Optional[Path] = None, keep_xy: Optional[Box] = None,
              collide_xy: Optional[Box] = None, tuft_xy: Optional[Box] = None,
              tuft_cell_m: Optional[float] = None) -> LoadedArea:
    """Build an area into a client, cut to keep_xy for what is drawn and to collide_xy for what is solid when given;
    the tufts cut to tuft_xy instead when it is given, and stood as patches of tuft_cell_m when that is; forest files
    go to work, or to a scratch folder removed once the bodies are made."""
    root = Path(root)
    if work is None:
        with tempfile.TemporaryDirectory(prefix="swarm-area-") as scratch:
            return load_area(root, cli, Path(scratch), keep_xy, collide_xy, tuft_xy, tuft_cell_m)
    manifest = json.loads((root / "manifest.json").read_text())
    validate_manifest(manifest, root)
    items, work = manifest["items"], Path(work)
    shapes = footprints(manifest, manifest["placements"])
    always = np.array([items[pl["item"]]["group"] in ALWAYS_KEPT for pl in manifest["placements"]], dtype=bool)
    kept = overlaps(shapes, keep_xy) | always
    solid = overlaps(shapes, collide_xy)
    places = [(pl, bool(hard)) for pl, keep, hard in zip(manifest["placements"], kept, solid) if keep]

    cache, textures, placed, terrain, colliders, ground_shapes, groups, patches = {}, {}, [], set(), {}, {}, {}, []
    forests = [g for g in ("trees", "rocks") if any(m["group"] == g and "instanced" in m["flags"] for m in items.values())]
    forest_bodies = {}
    for group in forests:
        body = stand_forest(root, cli, manifest, group, [pl for pl, _hard in places], work)
        if body is not None:
            placed.append(body)
            forest_bodies[group] = body
    tuft_box = keep_xy if tuft_xy is None else tuft_xy
    if manifest.get("tuft_field") and tuft_cell_m:
        patches = stand_patches(root, cli, manifest["tuft_field"], work, manifest, tuft_box, tuft_cell_m)
        placed.extend(patch.body for patch in patches)
    elif manifest.get("tuft_field"):
        body = stand_field(root, cli, manifest["tuft_field"], work, manifest, tuft_box)
        if body is not None:
            placed.append(body)
    if manifest.get("far_trees"):
        placed.append(stand_field(root, cli, manifest["far_trees"], work, manifest))
    prism = trunk_prism(work)
    for place, hard in places:
        item = items[place["item"]]
        if "instanced" in item["flags"] and item["group"] in forests:
            child = collider_child(item, place, asset_path(root, manifest, item, item["obj"]), prism) if hard else None
            if child is not None:
                colliders.setdefault(tile_of(place["position"]), []).append(child)
            continue
        scale = place["scale"]
        key = (place["item"], tuple(scale))
        if key not in cache:
            filename = asset_path(root, manifest, item, item["obj"])
            texture = asset_path(root, manifest, item, item["texture"])
            flags = 0
            if "double_sided" in item["flags"]:
                flags |= p.VISUAL_SHAPE_DOUBLE_SIDED_MULTIBODY
            if "instanced" in item["flags"]:
                flags |= p.VISUAL_SHAPE_RENDER_INSTANCED
            visual = p.createVisualShape(p.GEOM_MESH, fileName=str(visual_path(root, manifest, item)), meshScale=scale, flags=flags,
                                         rgbaColor=[1, 1, 1, 1], specularColor=[0, 0, 0], physicsClientId=cli)
            if str(texture) not in textures:
                textures[str(texture)] = p.loadTexture(str(texture), physicsClientId=cli)
            cache[key] = (visual, textures[str(texture)], filename)
        visual, texture, filename = cache[key]
        is_terrain = item["group"] == "terrain"
        collision = -1
        if is_terrain:
            if key not in ground_shapes:
                ground_shapes[key] = p.createCollisionShape(p.GEOM_MESH, fileName=str(filename), meshScale=scale,
                                                            flags=(p.GEOM_FORCE_CONCAVE_TRIMESH | getattr(p, "GEOM_CONCAVE_BVH_CACHE", 0)),
                                                            physicsClientId=cli)
            collision = ground_shapes[key]
        uid = p.createMultiBody(0, collision, visual, place["position"], place["quaternion"], useMaximalCoordinates=True,
                                physicsClientId=cli)
        p.changeVisualShape(uid, -1, textureUniqueId=texture, rgbaColor=[1, 1, 1, 1], specularColor=[0, 0, 0], physicsClientId=cli)
        placed.append(uid)
        groups.setdefault(item["group"], set()).add(uid)
        if is_terrain:
            terrain.add(uid)
            p.setCollisionFilterGroupMask(uid, -1, TERRAIN_GROUP, -1, physicsClientId=cli)
        elif hard:
            child = collider_child(item, place, filename, prism)
            if child is not None:
                colliders.setdefault(tile_of(place["position"]), []).append(child)
    return LoadedArea(manifest, placed, frozenset(terrain), tile_colliders(colliders, cli),
                      {group: frozenset(uids) for group, uids in groups.items()}, tuple(patches), forest_bodies)


def ground_height(x: float, y: float, cli: int, terrain: frozenset) -> float:
    """World z of the terrain under a point, read past anything else a ray meets on the way down."""
    top = RAY_REACH_M
    for _ in range(GROUND_RAYS):
        hit = p.rayTest([x, y, top], [x, y, -RAY_REACH_M], collisionFilterMask=TERRAIN_GROUP, physicsClientId=cli)[0]
        if hit[0] < 0:
            break
        if hit[0] in terrain:
            return float(hit[3][2])
        top = float(hit[3][2]) - 0.01
    raise RuntimeError(f"no terrain under {x}, {y}")
