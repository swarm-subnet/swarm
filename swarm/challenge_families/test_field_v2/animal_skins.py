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

"""The animals' asset package: one table per kind, and per body one smooth skin with every pose of its points.

A kind (goats, dogs, birds, boars, rats) ships a folder under the worlds' `testsite` folder with a table,
`<kind>s.json`: the kind's
class, its heat range and emissivity, the spreads of size and pace a seed draws from, how long one clip takes to ease
into the next, `joins` (which clip may follow which; a table without it lets any clip follow any), and per body its
files and clips. A clip is a run of frames in the pose table with its frame rate, whether it loops, how fast the body
travels while it plays, and for a clip that ends, the clip that follows it.

A body's pose table holds every pose of the body's points (`poses`), the body point each drawn vertex shows
(`source`), each drawn vertex's place on the coat (`uv`) and the triangles over the drawn vertices (`faces`). The
body is built from its rest pose with shared vertices, so the engine shades it smooth, and a new pose is one upload of
the drawn vertices.
"""

from __future__ import annotations

import json
import math
import os
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import pybullet as p
import swarm_worlds

ANIMALS_DIR = os.path.join(swarm_worlds.maps_dir(), "custom", "testsite")
TABLES = {"goat": os.path.join("goats", "goats.json"), "dog": os.path.join("dogs", "dogs.json"),
          "bird": os.path.join("birds", "birds.json"), "boar": os.path.join("boars", "boars.json"),
          "rat": os.path.join("rats", "rats.json")}
# The bodies of a kind the field draws, so a table may hold others that are never read or asked for; a kind not listed
# draws every body its table names.
DRAWN = {"bird": frozenset({"chough", "crow", "magpie", "pigeon", "woodpigeon"}),
         "boar": frozenset({"boar", "piglet"}), "rat": frozenset({"mouse", "rat"})}
THERMAL = hasattr(p, "ER_SWARM_THERMAL")
REST_CLIP = "stand"                    # the body is built in this clip's first frame, or the table's first frame


@dataclass(frozen=True)
class Clip:
    """A run of frames in a body's pose table: where it starts, how long and fast it plays, how far it carries the
    body each second at size and pace 1, and what follows it when it ends."""

    start: int
    frames: int
    fps: float
    loop: bool
    speed_m_s: float
    then: Optional[str] = None


@dataclass(frozen=True)
class Body:
    """One body of a kind: its files, its poses and drawn skin, its clips, how far any point of it reaches across
    the ground, and in any direction, from its origin in any pose at size 1, and each pose's lowest point."""

    name: str
    folder: str
    coats: Dict[str, str]
    heat_map: str
    length_m: float
    poses: np.ndarray                  # frames x body points x 3, float32, metres in the body's own frame
    source: np.ndarray                 # the body point each drawn vertex shows
    uv: np.ndarray
    faces: np.ndarray                  # triangles over the drawn vertices
    clips: Dict[str, Clip]
    reach_m: float
    extent_m: float = 0.0
    lows: Optional[np.ndarray] = None  # each frame's lowest point over the body's own ground, metres at size 1

    def rest(self) -> int:
        """The frame the body is built in."""
        return self.clips[REST_CLIP].start if REST_CLIP in self.clips else 0


@dataclass(frozen=True)
class Kind:
    """One kind of animal as its table describes it."""

    name: str
    heat_range_c: Tuple[float, float]
    emissivity: float
    size_spread: Tuple[float, float]
    pace_spread: Tuple[float, float]
    blend_s: float
    joins: Optional[Dict[str, Tuple[str, ...]]]
    bodies: Dict[str, Body]


def table_path(kind: str) -> str:
    """Where a kind's table is read from."""
    return os.path.join(ANIMALS_DIR, TABLES[kind])


def shipped(kind: str) -> bool:
    """Whether a kind's table is installed."""
    return os.path.isfile(table_path(kind))


def drawn_bodies(kind: str, names) -> List[str]:
    """The names of a table's bodies the field draws, in order."""
    return sorted(name for name in names if kind not in DRAWN or name in DRAWN[kind])


def asset_files() -> List[str]:
    """Every file the animals read: each kind's table, and every pose table, coat and heat map of the bodies the field
    draws. A table that is missing or unreadable is listed alone, so an install check names it; so is a body the field
    draws that its table does not hold."""
    files: List[str] = []
    for kind in sorted(TABLES):
        path = table_path(kind)
        files.append(path)
        try:
            with open(path, encoding="utf-8") as handle:
                bodies = json.load(handle)["bodies"]
        except (OSError, ValueError, KeyError):
            continue
        folder = os.path.dirname(path)
        # A table that lacks a body the field draws is named too, as a file of that body's name that is not there.
        files += [os.path.join(folder, f"{name} in {os.path.basename(path)}") for name in sorted(DRAWN.get(kind, ())) if name not in bodies]
        for name in drawn_bodies(kind, bodies):
            body = bodies[name]
            files += [os.path.join(folder, leaf) for leaf in (body["poses"], body["heat_map"], *sorted(body["coats"].values()))]
    return files


def load(kind: str) -> Kind:
    """A kind's table and the bodies the field draws of it, read once per process and path."""
    return read(table_path(kind), DRAWN.get(kind))


@lru_cache(maxsize=8)
def read(path: str, drawn: Optional[frozenset] = None) -> Kind:
    """The kind whose table lies at a path, the poses of its bodies loaded: those named in drawn, or all of them."""
    folder = os.path.dirname(path)
    with open(path, encoding="utf-8") as handle:
        table = json.load(handle)
    joins = table.get("joins")
    names = sorted(name for name in table["bodies"] if drawn is None or name in drawn)
    bodies = {name: _body(folder, name, table["bodies"][name]) for name in names}
    return Kind(name=str(table["class"]), heat_range_c=tuple(float(c) for c in table["heat"]["range_c"]),
                emissivity=float(table["heat"]["emissivity"]),
                size_spread=tuple(float(s) for s in table["size_spread"]),
                pace_spread=tuple(float(s) for s in table["pace_spread"]), blend_s=float(table["blend_s"]),
                joins=None if joins is None else {clip: tuple(after) for clip, after in joins.items()}, bodies=bodies)


def _body(folder: str, name: str, spec: dict) -> Body:
    """One body of a table, with its pose table read."""
    with np.load(os.path.join(folder, spec["poses"])) as table:
        poses = table["poses"].astype(np.float32)
        source = table["source"].astype(np.int64)
        uv = table["uv"].astype(np.float32)
        faces = table["faces"].astype(np.int64).reshape(-1)
    clips = {clip: Clip(start=int(c["start"]), frames=int(c["frames"]), fps=float(c["fps"]), loop=bool(c["loop"]),
                        speed_m_s=float(c["speed_m_s"]), then=c.get("then")) for clip, c in spec["clips"].items()}
    wide = poses.astype(np.float64)
    reach = float(np.sqrt((wide[..., 0] ** 2 + wide[..., 1] ** 2).max()))
    return Body(name=name, folder=folder, coats=dict(spec["coats"]), heat_map=str(spec["heat_map"]),
                length_m=float(spec["length_m"]), poses=poses, source=source, uv=uv, faces=faces, clips=clips,
                reach_m=reach, extent_m=float(np.sqrt((wide ** 2).sum(axis=-1).max())),
                lows=np.ascontiguousarray(poses[..., 2].min(axis=1)))


def follows(kind: Kind, before: str, after: str) -> bool:
    """Whether a clip may follow another directly; a clip always follows itself."""
    return before == after or kind.joins is None or after in kind.joins.get(before, ())


def route(kind: Kind, before: str, after: str) -> List[str]:
    """The clips to play to get from one clip to another by the table's joins, the last being the one asked for;
    just that one when it follows directly or no way leads there."""
    if follows(kind, before, after):
        return [after]
    came: Dict[str, Optional[str]] = {before: None}
    queue = deque([before])
    while queue:
        clip = queue.popleft()
        for nxt in sorted(kind.joins.get(clip, ())):
            if nxt in came:
                continue
            came[nxt] = clip
            if nxt == after:
                path = [after]
                while came[path[-1]] != before:
                    path.append(came[path[-1]])
                return path[::-1]
            queue.append(nxt)
    return [after]


def wrapped(clip: Clip, frame: float) -> float:
    """A frame within a clip: round a loop, held on the last frame of a clip that ends."""
    return frame % clip.frames if clip.loop else min(max(frame, 0.0), clip.frames - 1.0)


def points(body: Body, clip: str, frame: float) -> np.ndarray:
    """The body's points at a fractional frame of a clip, blended between the two frames round it."""
    spec = body.clips[clip]
    frame = wrapped(spec, frame)
    first = int(math.floor(frame))
    share = np.float32(frame - first)
    second = (first + 1) % spec.frames if spec.loop else min(first + 1, spec.frames - 1)
    return body.poses[spec.start + first] * (np.float32(1.0) - share) + body.poses[spec.start + second] * share


def low(body: Body, clip: str, frame: float) -> float:
    """The lowest point of the body at a fractional frame of a clip, blended as its points are. The blended pose
    reaches no lower: every point of it lies between its place in the two frames."""
    spec = body.clips[clip]
    frame = wrapped(spec, frame)
    first = int(math.floor(frame))
    share = np.float32(frame - first)
    second = (first + 1) % spec.frames if spec.loop else min(first + 1, spec.frames - 1)
    return float(body.lows[spec.start + first] * (np.float32(1.0) - share) + body.lows[spec.start + second] * share)


def drawn(body: Body, body_points: np.ndarray, size: float) -> np.ndarray:
    """The drawn vertices of a pose of the body's points at a size, as the engine takes them."""
    return np.ascontiguousarray(body_points[body.source] * np.float32(size))


def _normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted normal of every drawn vertex."""
    tris = faces.reshape(-1, 3)
    a, b, c = vertices[tris[:, 0]], vertices[tris[:, 1]], vertices[tris[:, 2]]
    out = np.zeros_like(vertices, dtype=np.float64)
    face = np.cross(b - a, c - a)
    for k in range(3):
        np.add.at(out, tris[:, k], face)
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)


def spawn(cli: int, kind: Kind, body: Body, coat: str, size: float, textures: Dict[str, int]) -> int:
    """Create one animal's body in its rest pose with its coat and its heat map, visual only, so nothing collides with
    it; textures already loaded on this client are taken from the map and new ones added to it."""
    rest = drawn(body, body.poses[body.rest()], size)
    shape = p.createVisualShape(p.GEOM_MESH, vertices=rest.tolist(), indices=body.faces.tolist(), uvs=body.uv.tolist(),
                                normals=_normals(rest, body.faces).tolist(), physicsClientId=cli)
    uid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=shape, physicsClientId=cli)
    # A coat does not shine: without this the sun paints a white sheen along the back.
    p.changeVisualShape(uid, -1, textureUniqueId=_texture(cli, os.path.join(body.folder, body.coats[coat]), textures),
                        rgbaColor=[1.0, 1.0, 1.0, 1.0], specularColor=[0.0, 0.0, 0.0], physicsClientId=cli)
    if THERMAL:
        p.changeVisualShape(uid, -1, thermalTextureUniqueId=_texture(cli, os.path.join(body.folder, body.heat_map), textures),
                            temperatureRange=list(kind.heat_range_c), emissivity=kind.emissivity, physicsClientId=cli)
    return int(uid)


def _texture(cli: int, path: str, textures: Dict[str, int]) -> int:
    """A texture's id on a client, loaded the first time it is asked for."""
    if path not in textures:
        textures[path] = int(p.loadTexture(path, physicsClientId=cli))
    return textures[path]
