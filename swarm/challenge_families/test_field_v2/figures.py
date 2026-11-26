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

"""The field's people: painted skin, eyes, hair and clothes on the bodies of the worlds' people package.

The package (maps/custom/testsite/people in swarm-worlds) holds the skeleton with every build's fitted bind pose, the
skin's points on every build, and pieces (eyes, hair, clothes, shoes), each with its points on every build, so nothing
is fitted here. Its table of contents names every piece's slot, who wears it, how warm it is, its pictures and the
palette each picture is tinted from, the skins and the builds.

A person is one visual body for the skin that shows and one for each piece, every one with its own picture; a picture
painted pale takes its colour from a tint. A frame of a move is posed by forward kinematics and every drawn point
blends its few joint transforms, summed in a fixed order with elementwise arithmetic, never through BLAS, so every
validator's CPU poses a person to the same bits. Skin that clothes cover is neither drawn nor posed.

Who a person is comes from the seed's generator alone: a build, a skin of that sex, eyes that go with the skin, hair
(or none), a top and a bottom fit for the air's temperature, shoes, and sometimes a safety vest.
"""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p
import swarm_worlds

from ..solar_patrol.intruders import _apply, _inverse, _product, _quat_to_mat, _to_zup

ASSET_DIR = os.path.join("custom", "testsite", "people")
PEOPLE_DIR = os.path.join(swarm_worlds.maps_dir(), ASSET_DIR)
THERMAL = hasattr(p, "ER_SWARM_THERMAL")
EMISSIVITY = 0.95                       # skin and cloth in the long-wave band
SPECULAR = [0.0, 0.0, 0.0]              # cloth and skin do not shine like plastic
HAIR_SHARE = {"f": 1.0, "m": 0.9}       # how many women and men have hair to draw; the others' is cut to the skin
VEST_SHARE = 0.12                       # how many wear a safety vest over their top
HAT_SHARE = 0.3                         # how many of those with short hair or none wear something on the head
PACK_SHARE = 0.3                        # how many carry a pack on the back
PLAIN, PLAIN_SHARE = "plain", 0.6       # a piece with a plain picture and patterned ones is plain this often
HAT_OVER = ("", "hair_short")           # the hair a hat fits over: none, or hair cut short
LIGHT_FROM_C = 12.0                     # light clothes (bare arms or legs) are worn from this air temperature up
WARM_UNTIL_C = 18.0                     # warm clothes (sweater, jacket, puffer) up to this one


class Wardrobe:
    """Everything people need from a package folder, read once per process."""

    def __init__(self, folder: str):
        """Read the table of contents, the skeleton and the drawn skin; garments are read when first worn."""
        self.folder = folder
        with open(os.path.join(folder, "people.json"), encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        rig = np.load(os.path.join(folder, "rig.npz"))
        self.joint_names = [str(n) for n in rig["joint_names"]]
        self.parents = rig["parents"].astype(np.int64)
        depth = np.array([self._depth(j) for j in range(len(self.parents))])
        self.levels = [np.flatnonzero(depth == d) for d in range(depth.max() + 1)]
        self.slot = np.array([self.joint_names.index(str(n)) for n in rig["soma30"]])
        self.relaxed = rig["relaxed_hands"].astype(np.float64)
        self.builds = [str(n) for n in rig["build_names"]]
        self.build_bind = rig["build_bind"].astype(np.float64)
        self.build_rest = rig["build_rest"].astype(np.float64)
        self.root_scale = rig["build_root_scale"].astype(np.float64)
        self.weight_joints = rig["weight_joints"].astype(np.int64)
        self.weights = rig["weights"].astype(np.float64)
        body = np.load(os.path.join(folder, "body.npz"))
        self.body_points = body["points"]                       # builds x points x 3, in the skeleton model's frame
        self.body_faces = body["faces"].astype(np.int64)        # triangles over the drawn skin vertices
        self.body_source = body["source"].astype(np.int64)      # the skin point each drawn vertex shows
        self.body_uv = body["uv"].astype(np.float64)
        self._pieces: Dict[str, Dict[str, np.ndarray]] = {}

    def _depth(self, j: int) -> int:
        """Number of ancestors of joint j."""
        n = 0
        while self.parents[j] >= 0:
            j, n = self.parents[j], n + 1
        return n

    def piece(self, name: str) -> Dict[str, np.ndarray]:
        """A garment's arrays, read the first time it is worn."""
        if name not in self._pieces:
            with np.load(os.path.join(self.folder, self.manifest["pieces"][name]["file"])) as data:
                self._pieces[name] = {key: data[key] for key in data.files}
        return self._pieces[name]

    def world(self, build: int, local30: np.ndarray, root: np.ndarray) -> np.ndarray:
        """World transforms (77, 4, 4) of one build's skeleton for 30 local rotations and the root, in the skeleton
        model's frame."""
        rest = self.build_rest[build]
        world = np.zeros((len(self.parents), 4, 4))
        world[:, :3, :3] = self.relaxed
        world[self.slot, :3, :3] = local30
        world[:, :3, 3] = rest - np.where(self.parents[:, None] >= 0, rest[self.parents], 0.0)
        world[:, 3, 3] = 1.0
        for joints in self.levels[1:]:
            world[joints] = _product(world[self.parents[joints]], world[joints])
        world[:, :3, 3] += root * self.root_scale[build]
        return world


@lru_cache(maxsize=4)
def wardrobe(folder: str) -> Wardrobe:
    """The package at a folder; raises FileNotFoundError when it is not installed."""
    if not os.path.isfile(os.path.join(folder, "people.json")):
        raise FileNotFoundError(f"people package missing: {folder}")
    return Wardrobe(folder)


def _normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted normal of every drawn vertex."""
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    out = np.zeros_like(vertices, dtype=np.float64)
    face = np.cross(b - a, c - a)
    for k in range(3):
        np.add.at(out, faces[:, k], face)
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)


class Figure:
    """One dressed person in the scene: a visual body for the skin and one for each garment, posed frame by frame."""

    def __init__(self, cat: Wardrobe, dress: Dict[str, Any], first: tuple, client: int = 0, size: float = 1.0,
                 textures: Optional[Dict[str, int]] = None):
        """Stack the skin that shows and the garments, and create the visual bodies in the first frame: (local
        rotations xyzw, root, place, yaw). dress names the build, the skin's picture and each garment with its picture;
        textures maps pictures already loaded on this client to their ids and takes the new ones."""
        self.cat, self.client, self.dress, self.size = cat, client, dress, float(size)
        self.textures = {} if textures is None else textures
        self.build = cat.builds.index(dress["build"])
        hidden = np.zeros(len(cat.body_faces), bool)
        rest, joints, weights, meshes = [], [], [], []
        for worn in dress["pieces"]:
            name, picture, tint = (list(worn) + [None])[:3]
            piece = cat.piece(name)
            hidden[piece["hides"]] = True
            rest.append(piece["points"][self.build].astype(np.float64))
            joints.append(piece["weight_joints"].astype(np.int64))
            weights.append(piece["weights"].astype(np.float64))
            meshes.append((piece["faces"].astype(np.int64), piece["source"].astype(np.int64), piece["uv"].astype(np.float64),
                           picture, cat.manifest["pieces"][name].get("temperature_c"), tint))
        # The skin that shows: its drawn triangles, and only the points they use.
        shown = cat.body_faces[~hidden]
        used = np.unique(shown)
        remap = np.full(len(cat.body_source), -1)
        remap[used] = np.arange(len(used))
        points = np.unique(cat.body_source[used])
        point_of = np.full(cat.body_points.shape[1], -1)
        point_of[points] = np.arange(len(points))
        rest.insert(0, cat.body_points[self.build][points].astype(np.float64))
        joints.insert(0, cat.weight_joints[points])
        weights.insert(0, cat.weights[points])
        meshes.insert(0, (remap[shown], point_of[cat.body_source[used]], cat.body_uv[used], dress["skin"],
                          cat.manifest["body"].get("temperature_c"), None))
        offsets = np.concatenate([[0], np.cumsum([len(r) for r in rest])])
        rest_all, joints_all, weights_all = np.concatenate(rest), np.concatenate(joints), np.concatenate(weights)
        dense = np.zeros((len(rest_all), len(cat.parents)))
        np.add.at(dense, (np.repeat(np.arange(len(rest_all)), joints_all.shape[1]), joints_all.ravel()), weights_all.ravel())
        self.bones = np.flatnonzero(dense.any(0))
        # Each point's weights summed slot by slot, most-weighted points first, so slot k is one slice.
        blend = dense[:, self.bones]
        count = (blend != 0).sum(1)
        order = np.argsort(-count, kind="stable")
        rows, cols = np.nonzero(blend[order])
        rank = np.arange(len(rows)) - np.searchsorted(rows, rows)
        self.slots = [(cols[rank == k], blend[order[rows[rank == k]], cols[rank == k]][:, None]) for k in range(int(count.max()))]
        self.unsort = np.argsort(order)
        self.rest = rest_all[order]
        self.inv_bind = _inverse(cat.build_bind[self.build])
        start = self.vertices(*first)
        self.meshes = []
        for group, (faces, source, uv, picture, celsius, tint) in enumerate(meshes):
            rows = offsets[group] + source
            mine = start[rows]
            shape = p.createVisualShape(p.GEOM_MESH, vertices=mine.tolist(), indices=faces.ravel().tolist(), uvs=uv.tolist(),
                                        normals=_normals(mine, faces).tolist(), physicsClientId=client)
            uid = int(p.createMultiBody(0, -1, shape, [0.0, 0.0, 0.0], physicsClientId=client))
            p.changeVisualShape(uid, -1, textureUniqueId=self._texture(picture), rgbaColor=list(tint or (1.0, 1.0, 1.0)) + [1.0],
                                specularColor=SPECULAR, physicsClientId=client)
            if THERMAL and celsius is not None:
                p.changeVisualShape(uid, -1, temperature=float(celsius), emissivity=EMISSIVITY, physicsClientId=client)
            self.meshes.append((uid, rows))

    def _texture(self, picture: str) -> int:
        """A picture's id on this client, loaded the first time it is asked for."""
        path = os.path.join(self.cat.folder, picture)
        if path not in self.textures:
            self.textures[path] = int(p.loadTexture(path, physicsClientId=self.client))
        return self.textures[path]

    @property
    def bodies(self) -> List[int]:
        """The visual body ids of this person."""
        return [uid for uid, _ in self.meshes]

    @property
    def vertex_count(self) -> int:
        """Points posed per frame."""
        return len(self.rest)

    def vertices(self, local_xyzw: np.ndarray, root: np.ndarray, place: Sequence[float], yaw: float) -> np.ndarray:
        """Every posed point for one move frame at the body's size, turned by yaw and standing with its lowest point on
        place (x, y, z)."""
        world = self.cat.world(self.build, _quat_to_mat(np.asarray(local_xyzw, dtype=np.float64)), np.asarray(root, dtype=np.float64))
        joint = _product(world[self.bones], self.inv_bind[self.bones])[:, :3].reshape(len(self.bones), 12)
        blended = np.zeros((len(self.rest), 12))
        for bones, weights in self.slots:
            blended[:len(bones)] += joint[bones] * weights
        v = _to_zup(_apply(blended.reshape(-1, 3, 4), self.rest)[self.unsort])
        if self.size != 1.0:
            v = v * self.size
        c, s = math.cos(yaw), math.sin(yaw)
        x, y = v[:, 0] * c - v[:, 1] * s, v[:, 0] * s + v[:, 1] * c
        return np.stack([x + place[0], y + place[1], v[:, 2] - v[:, 2].min() + place[2]], 1)

    def pose(self, local_xyzw: np.ndarray, root: np.ndarray, place: Sequence[float], yaw: float) -> None:
        """Rewrite every mesh for one move frame. An engine that cannot rewrite a mesh raises: someone left where he
        was built, or drawn in pieces, would be counted and never found."""
        v = self.vertices(local_xyzw, root, place, yaw)
        for uid, rows in self.meshes:
            p.resetMeshData(uid, v[rows], physicsClientId=self.client)

    def remove(self) -> None:
        """Take the person out of the scene."""
        for uid, _ in self.meshes:
            p.removeBody(uid, physicsClientId=self.client)
        self.meshes = []


def pictures(ep: Any) -> Dict[str, int]:
    """The pictures already loaded for a patrol's people, kept with the patrol's site so every person of a seed shares
    them and a new seed starts with none."""
    if ep.park is None:
        ep.park = {}
    return ep.park.setdefault("pictures", {})


def catalogue() -> Wardrobe:
    """The shipped people package; raises FileNotFoundError when the installed swarm-worlds predates it."""
    return wardrobe(PEOPLE_DIR)


def asset_files(folder: Optional[str] = None) -> List[str]:
    """Every file the people read: the table of contents, the skeleton, the skin, and every piece and picture the
    table names. A table that is missing or unreadable is listed with the skeleton and the skin alone, so an install
    check names it."""
    folder = PEOPLE_DIR if folder is None else folder
    files = [os.path.join(folder, leaf) for leaf in ("people.json", "rig.npz", "body.npz")]
    try:
        with open(files[0], encoding="utf-8") as handle:
            manifest = json.load(handle)
        leaves = sorted({skin["file"] for skin in manifest["skins"].values()}
                        | {piece["file"] for piece in manifest["pieces"].values()}
                        | {picture["file"] for piece in manifest["pieces"].values() for picture in piece["pictures"].values()})
    except (OSError, ValueError, KeyError, TypeError):
        return files
    return files + [os.path.join(folder, leaf) for leaf in leaves]


def problems(folder: Optional[str] = None) -> List[str]:
    """What is wrong with a people package whose files are all there, one line each: a table that cannot be read, a
    sex with no build or no skin, a skin tone with no eyes, a palette a picture names that is not there, or nothing
    to wear on top, below or on the feet in the cold or in the heat. Empty for a package people can be drawn from."""
    folder = PEOPLE_DIR if folder is None else folder
    try:
        with open(os.path.join(folder, "people.json"), encoding="utf-8") as handle:
            manifest = json.load(handle)
        found = []
        for sex in ("f", "m"):
            if not any(build["sex"] == sex for build in manifest["builds"].values()):
                found.append(f"no build of sex {sex}")
            if not any(skin["sex"] == sex for skin in manifest["skins"].values()):
                found.append(f"no skin of sex {sex}")
            for slot in ("top", "bottom", "shoes"):
                for air_c in (LIGHT_FROM_C - 1.0, WARM_UNTIL_C + 1.0):
                    if _worn(manifest, np.random.default_rng(0), slot, sex, air_c, "dark")[0] is None:
                        found.append(f"nothing for the slot {slot} of sex {sex} at {air_c:.0f} C")
        eyes = manifest["pieces"]["eyes"]["pictures"]
        for skin in manifest["skins"].values():
            if not all(eye in eyes for eye in manifest["eyes_of_tone"][skin["tone"]]) or not manifest["eyes_of_tone"][skin["tone"]]:
                found.append(f"no eyes for the skin tone {skin['tone']}")
            if f"hair_{skin['hair']}" not in manifest["palettes"]:
                found.append(f"no palette hair_{skin['hair']}")
        for name, piece in manifest["pieces"].items():
            for picture in piece["pictures"].values():
                palette = picture["palette"]
                if palette is not None and palette != "hair" and not manifest["palettes"].get(palette):
                    found.append(f"{name} names the palette {palette}, which is empty or missing")
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        return [f"the people package cannot be read: {type(error).__name__}: {error}"]
    return sorted(set(found))


def _one(rng: np.random.Generator, options: Sequence[Any]) -> Any:
    """One of a list of options, each as likely as the next."""
    return options[int(rng.integers(len(options)))]


def _style(rng: np.random.Generator, styles: Sequence[str]) -> str:
    """One of a piece's pictures from a single draw: its plain one in PLAIN_SHARE of the draws when it has others
    too, as most clothes are plain, the others evenly; evenly among all when none is plain or it is the only one."""
    u = float(rng.random())
    if PLAIN in styles and len(styles) > 1:
        others = [style for style in styles if style != PLAIN]
        return PLAIN if u < PLAIN_SHARE else others[min(int((u - PLAIN_SHARE) / (1.0 - PLAIN_SHARE) * len(others)), len(others) - 1)]
    return styles[min(int(u * len(styles)), len(styles) - 1)]


def _worn(manifest: Dict[str, Any], rng: np.random.Generator, slot: str, sex: str, air_c: Optional[float],
          hair: str, bulk: Optional[str] = None) -> Tuple[Optional[List[Any]], str]:
    """A piece of a slot that a person of a sex wears in the air's temperature, with one of its pictures and a tint
    from the picture's palette; and what it looks like from far: the colour it was given. With a bulk, only pieces
    cut for a top of that bulk. No piece when the package holds none that fits."""
    names = []
    for name in sorted(manifest["pieces"]):
        piece = manifest["pieces"][name]
        if piece["slot"] != slot or piece["sex"] not in ("any", sex):
            continue
        warmth = piece.get("warmth", "any")
        if air_c is not None and ((warmth == "light" and air_c < LIGHT_FROM_C) or (warmth == "warm" and air_c > WARM_UNTIL_C)):
            continue
        if bulk is not None and piece.get("bulk", "thin") != bulk:
            continue
        names.append(name)
    if not names:
        return None, ""
    name = _one(rng, names)
    pictures = manifest["pieces"][name]["pictures"]
    style = _style(rng, sorted(pictures))
    palette = pictures[style]["palette"]
    if palette is None:
        return [name, pictures[style]["file"], None], f"{name}:{style}"
    palette = f"hair_{hair}" if palette == "hair" else palette
    index = int(rng.integers(len(manifest["palettes"][palette])))
    return [name, pictures[style]["file"], [float(c) for c in manifest["palettes"][palette][index]]], f"{palette}:{index}"


def outfit(rng: np.random.Generator, air_c: Optional[float] = None, cat: Optional[Wardrobe] = None) -> Dict[str, Any]:
    """One person drawn from the seed's generator: build, skin, eyes, hair and clothes with their tints, and his
    look: what tells him apart from above (his top's colour, his head, his bottom's colour)."""
    manifest = (cat or catalogue()).manifest
    build = _one(rng, sorted(manifest["builds"]))
    sex = manifest["builds"][build]["sex"]
    skin_name = _one(rng, sorted(name for name, skin in manifest["skins"].items() if skin["sex"] == sex))
    skin = manifest["skins"][skin_name]
    eye = _one(rng, list(manifest["eyes_of_tone"][skin["tone"]]))
    pieces = [["eyes", manifest["pieces"]["eyes"]["pictures"][eye]["file"], None]]
    head = f"bare:{skin['hair']}"
    has_hair = float(rng.random()) < HAIR_SHARE[sex]
    hair, hair_look = _worn(manifest, rng, "hair", sex, None, skin["hair"])
    if has_hair:
        pieces.append(hair)
        head = f"{hair[0]}:{hair_look}"
    top, top_look = _worn(manifest, rng, "top", sex, air_c, skin["hair"])
    bottom, bottom_look = _worn(manifest, rng, "bottom", sex, air_c, skin["hair"])
    shoes, _shoe_look = _worn(manifest, rng, "shoes", sex, air_c, skin["hair"])
    pieces += [top, bottom, shoes]
    if bottom[2] is not None:
        # Trousers and a skirt of one colour look alike from above, whatever palette the colour came from.
        bottom_look = "bottom:" + ",".join(f"{c:.3f}" for c in bottom[2][:3])
    vest, vest_look = _worn(manifest, rng, "over", sex, None, skin["hair"])
    # A vest is worn over a thin top only: a thick one would cover it.
    if float(rng.random()) < VEST_SHARE and vest is not None and manifest["pieces"][top[0]].get("bulk", "thin") == "thin":
        pieces.append(vest)
        top_look = vest_look
    # A hat sits on short hair or a bare head; a pack is the one cut for the top's bulk.
    hat, hat_look = _worn(manifest, rng, "head", sex, air_c, skin["hair"])
    if float(rng.random()) < HAT_SHARE and hat is not None and (hair[0] if has_hair else "") in HAT_OVER:
        pieces.append(hat)
        head = f"{hat[0]}:{hat_look}"
    pack, _pack_look = _worn(manifest, rng, "back", sex, None, skin["hair"], manifest["pieces"][top[0]].get("bulk", "thin"))
    if float(rng.random()) < PACK_SHARE and pack is not None:
        pieces.append(pack)
    return {"build": build, "sex": sex, "skin": skin["file"], "pieces": pieces, "look": [top_look, head, bottom_look]}


def look(dress: Dict[str, Any]) -> Tuple[str, str, str]:
    """What tells a person apart from above: the colour of his top (or his vest), his head (hair and its colour, or
    none) and the colour of his trousers or skirt."""
    return tuple(dress["look"])


def height_m(dress: Dict[str, Any], size: float, cat: Optional[Wardrobe] = None) -> float:
    """How tall a person of a build and size stands, metres."""
    return float((cat or catalogue()).manifest["builds"][dress["build"]]["height_m"]) * size


def stride(dress: Dict[str, Any], size: float, cat: Optional[Wardrobe] = None) -> float:
    """How far a move carries a person of a build and size for each metre it carries the standard build."""
    cat = cat or catalogue()
    return float(cat.root_scale[cat.builds.index(dress["build"])]) * size
