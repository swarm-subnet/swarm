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

"""Intruders (task 4): the thieves' bodies, what they wear and carry, and how one frame of a move poses them.

The assets are maps/custom/solar/intruders in swarm-worlds: NVIDIA's SOMA body on its 77-joint skeleton in six male
builds (1.66 to 1.86 m), and 24 garments and tools made on it, each stored as one OBJ per colour with the bone
weights of its vertices beside it. Clothes follow the skin under them onto every build; rigid pieces (pack, bolt
cutters, cable coil) ride one bone.

A thief is one visual body per colour piece, built from vertex arrays so that resetMeshData can rewrite it every frame.
A frame of a Kimodo SOMA-30 move (30 local rotations and the root) is posed by forward kinematics on the skeleton, then
every drawn vertex blends its few joint transforms; skin the clothes cover is never posed. All of it is summed in a fixed
order with elementwise arithmetic, never through BLAS, so every validator's CPU poses a thief to the same bits.
Each piece carries a temperature for the thermal camera; tools and the pack stay on the engine's passive model.
"""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pybullet as p
import swarm_worlds

ASSET_DIR = os.path.join("custom", "solar", "intruders")
THERMAL = hasattr(p, "ER_SWARM_THERMAL")
EMISSIVITY = 0.95                       # skin and cloth in the long-wave band
SPECULAR = [0.03, 0.03, 0.03]

MASKED_SHARE = 0.7                      # decided: about 7 in 10 thieves cover their face
LONG_TOPS = {"hoodie", "hoodie_up", "work_jacket", "puffer"}
# Weighted choices per slot; masks, heads and what is carried follow the rules in outfit().
TOPS = {"hoodie": 0.30, "hoodie_up": 0.15, "work_jacket": 0.25, "puffer": 0.15, "tshirt": 0.15}
BOTTOMS = {"work_trousers": 0.35, "jeans": 0.35, "cargo": 0.30}
SHOES = {"work_boots": 0.55, "trainers": 0.45}
MASKS = {"balaclava": 0.45, "balaclava3": 0.25, "gaiter": 0.30}
HEADS = {"balaclava": {"": 0.60, "beanie": 0.25, "cap": 0.15}, "balaclava3": {"": 0.60, "beanie": 0.25, "cap": 0.15},
         "gaiter": {"beanie": 0.45, "cap": 0.30, "hair_short": 0.25}, "": {"hair_short": 0.50, "cap": 0.30, "beanie": 0.20}}
GLOVES_SHARE = 0.6
CUTTERS_SHARE = 0.5
BACK = {"backpack": 0.35, "cable_coil": 0.30, "": 0.35}
# The pack and the coil come in three fits, by how far the top stands off the body.
FIT = {"tshirt": "_thin", "puffer": "_puffer"}


def _to_zup(v: np.ndarray) -> np.ndarray:
    """SOMA's y-up, z-forward frame to the engine's z-up frame with the thief facing -y."""
    return np.stack([v[..., 0], -v[..., 2], v[..., 1]], -1)


def _to_yup(v: np.ndarray) -> np.ndarray:
    """The engine's z-up frame back to SOMA's y-up frame."""
    return np.stack([v[..., 0], v[..., 2], -v[..., 1]], -1)


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    """xyzw quaternions to 3x3 rotation matrices."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def _product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Products of stacked 4x4 transforms, summed in a fixed order with elementwise operations so that no BLAS
    kernel, and so no CPU, changes a bit."""
    return (a[..., :, 0:1] * b[..., 0:1, :] + a[..., :, 1:2] * b[..., 1:2, :] + a[..., :, 2:3] * b[..., 2:3, :]
            + a[..., :, 3:4] * b[..., 3:4, :])


def _inverse(m: np.ndarray) -> np.ndarray:
    """Inverses of stacked affine 4x4 transforms by cofactors, in a fixed order of elementwise operations."""
    r0, r1, r2 = m[..., 0, :3], m[..., 1, :3], m[..., 2, :3]
    cof = np.stack([np.cross(r1, r2), np.cross(r2, r0), np.cross(r0, r1)], -1)
    det = r0[..., 0] * cof[..., 0, 0] + r0[..., 1] * cof[..., 1, 0] + r0[..., 2] * cof[..., 2, 0]
    out = np.zeros(m.shape)
    out[..., :3, :3] = cof / det[..., None, None]
    out[..., :3, 3] = -_apply(out, m[..., :3, 3], translate=False)
    out[..., 3, 3] = 1.0
    return out


def _apply(m: np.ndarray, v: np.ndarray, translate: bool = True) -> np.ndarray:
    """Points v (..., 3) through affine transforms m (..., 3 or 4, 4), in a fixed order of elementwise operations."""
    out = m[..., :3, 0] * v[..., 0:1] + m[..., :3, 1] * v[..., 1:2] + m[..., :3, 2] * v[..., 2:3]
    return out + m[..., :3, 3] if translate else out


def _read_obj(path: str) -> tuple:
    """Vertices and triangles of one of the pieces' OBJ files, in file order."""
    verts, faces = [], []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(token.split("/")[0]) - 1 for token in line.split()[1:4]])
    return np.asarray(verts), np.asarray(faces, dtype=np.int64)


def _face_frames(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Per-face orthonormal frames (tangent, bitangent, normal) as rows."""
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    t = (b - a) / (np.linalg.norm(b - a, axis=1, keepdims=True) + 1e-12)
    n = np.cross(b - a, c - a)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return np.stack([t, np.cross(n, t), n], 1)


class Catalogue:
    """Everything the intruders need from swarm-worlds, loaded once per process."""

    def __init__(self, folder: str):
        """Read the manifest, the rig and every piece."""
        self.folder = folder
        with open(os.path.join(folder, "intruders.json"), encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        rig = np.load(os.path.join(folder, "rig.npz"))
        self.joint_names = [str(n) for n in rig["joint_names"]]
        self.parents = rig["parents"].astype(np.int64)
        depth = np.array([self._depth(j) for j in range(len(self.parents))])
        self.levels = [np.flatnonzero(depth == d) for d in range(depth.max() + 1)]
        self.slot = np.array([self.joint_names.index(str(n)) for n in rig["soma30"]])
        self.relaxed = rig["relaxed_hands"].astype(np.float64)
        self.builds = [str(n) for n in rig["build_names"]]
        self.build_verts = rig["build_verts"].astype(np.float64)
        self.build_bind = rig["build_bind"]
        self.build_rest = rig["build_rest"]
        self.root_scale = rig["build_root_scale"]
        self.faces = rig["faces"].astype(np.int64)
        self.weight_joints = rig["weight_joints"].astype(np.int64)
        self.weights = rig["weights"].astype(np.float64)
        self.dominant = np.array(self.joint_names)[rig["dominant"]]
        self.pieces: Dict[str, List[Dict[str, Any]]] = {}
        for name, garment in self.manifest["garments"].items():
            pieces = []
            for entry in garment["pieces"]:
                verts, faces = _read_obj(os.path.join(folder, entry["file"] + ".obj"))
                data = dict(np.load(os.path.join(folder, entry["file"] + ".npz")))
                pieces.append(dict(entry, verts=verts + np.asarray(entry["offset"]), faces=faces, **data))
            self.pieces[name] = pieces

    def _depth(self, j: int) -> int:
        """Number of ancestors of joint j."""
        n = 0
        while self.parents[j] >= 0:
            j, n = self.parents[j], n + 1
        return n

    def world(self, build: int, local30: np.ndarray, root: np.ndarray) -> np.ndarray:
        """World transforms (77, 4, 4) of one build's skeleton for 30 local rotations and the root, SOMA frame."""
        rest = self.build_rest[build]
        world = np.zeros((len(self.parents), 4, 4))
        world[:, :3, :3] = self.relaxed
        world[self.slot, :3, :3] = local30
        world[:, :3, 3] = rest - np.where(self.parents[:, None] >= 0, rest[self.parents], 0.0)
        world[:, 3, 3] = 1.0
        # Joints of one depth only hang on shallower ones, so each depth is one step from its parents' transforms.
        for joints in self.levels[1:]:
            world[joints] = _product(world[self.parents[joints]], world[joints])
        world[:, :3, 3] += root * self.root_scale[build]
        return world


@lru_cache(maxsize=1)
def catalogue() -> Catalogue:
    """The shipped intruder catalogue; raises FileNotFoundError when the installed swarm-worlds predates it."""
    folder = os.path.join(swarm_worlds.maps_dir(), ASSET_DIR)
    if not os.path.isfile(os.path.join(folder, "intruders.json")):
        raise FileNotFoundError(f"intruder assets missing: {folder}")
    return Catalogue(folder)


def _pick(rng: np.random.Generator, weights: Dict[str, float]) -> str:
    """One key drawn with its weight."""
    keys = list(weights)
    share = np.array([weights[k] for k in keys])
    return keys[int(rng.choice(len(keys), p=share / share.sum()))]


def outfit(rng: np.random.Generator) -> Dict[str, Any]:
    """One thief's build, garments, tools and colours, drawn from the seed's generator."""
    cat = catalogue()
    top, bottom, shoes = _pick(rng, TOPS), _pick(rng, BOTTOMS), _pick(rng, SHOES)
    mask = _pick(rng, MASKS) if rng.random() < MASKED_SHARE else ""
    # A raised hood covers the head, so nothing is worn on it but the mask.
    head = "" if top == "hoodie_up" else _pick(rng, HEADS[mask])
    back = _pick(rng, BACK)
    if top == "hoodie" and back:
        back = ""  # the lowered hood lies on the upper back where a pack or a coil would sit
    garments = [top, bottom, shoes] + [g for g in (mask, head) if g]
    if rng.random() < GLOVES_SHARE:
        garments.append("gloves")
    if rng.random() < CUTTERS_SHARE:
        garments.append("bolt_cutters")
    if back:
        garments.append(back + FIT.get(top, ""))
    colours = {}
    for name in garments:
        for piece in cat.pieces[name]:
            options = piece["colours"]
            colours[piece["file"]] = options[int(rng.integers(len(options)))]
    body = cat.manifest["body"]
    return {"build": cat.builds[int(rng.integers(len(cat.builds)))], "garments": garments, "colours": colours,
            "skin": body["skin_tones"][int(rng.integers(len(body["skin_tones"])))],
            "undershirt": body["undershirt_colours"][int(rng.integers(len(body["undershirt_colours"])))]}


class Intruder:
    """One dressed thief in the scene: a visual body per colour piece, posed frame by frame."""

    def __init__(self, dress: Dict[str, Any], first: tuple, client: int = 0):
        """Fit every piece to the build, hide the covered skin, stack the weights, and create the visual bodies in the
        first frame: (local rotations xyzw, root, place, yaw), the pose an engine without mesh rewriting keeps."""
        cat = catalogue()
        self.cat, self.client, self.dress = cat, client, dress
        self.build = cat.builds.index(dress["build"])
        body_yup = cat.build_verts[self.build]
        body_zup = _to_zup(body_yup)
        frames = _face_frames(body_zup, cat.faces)
        bind = cat.build_bind[self.build]
        standard = cat.build_bind[cat.builds.index("standard")]
        covered = np.zeros(len(cat.faces), bool)
        rest, joints, weights, parts = [], [], [], []
        for name in dress["garments"]:
            hides = cat.manifest["garments"][name]["hides"]
            if hides:
                covered |= np.isin(cat.dominant, hides)[cat.faces].all(1)
            for piece in cat.pieces[name]:
                tri = cat.faces[piece["body_face"]]
                point = (body_zup[tri] * piece["bary"][..., None]).sum(1)
                fitted = point + np.einsum("vji,vj->vi", frames[piece["body_face"]], piece["local"])
                rigid = np.flatnonzero(piece["rigid_joint"] >= 0)
                if len(rigid):
                    j = piece["rigid_joint"][rigid].astype(np.int64)
                    carry = _product(bind[j], _inverse(standard[j]))
                    fitted[rigid] = _to_zup(_apply(carry, _to_yup(piece["verts"][rigid])))
                covered[piece["body_face"][piece["outer"]]] = True
                rest.append(_to_yup(fitted))
                joints.append(piece["weight_joints"].astype(np.int64))
                weights.append(piece["weights"].astype(np.float64))
                parts.append((piece["faces"], dress["colours"][piece["file"]], piece["temperature_c"]))
        # A covered face stays drawn at the edge of the clothes, where the skin under a hem can still be seen.
        edge = np.zeros(len(body_yup), bool)
        edge[cat.faces[~covered].ravel()] = True
        hidden = covered & ~edge[cat.faces].any(1)
        visible = cat.faces[~hidden]
        used = np.unique(visible)
        remap = np.full(len(body_yup), -1)
        remap[used] = np.arange(len(used))
        undershirt = np.isin(cat.dominant, cat.manifest["body"]["undershirt_joints"])[visible].all(1)
        if not LONG_TOPS & set(dress["garments"]):
            undershirt[:] = False
        body = cat.manifest["body"]
        skin_parts = [(remap[visible[~undershirt]], dress["skin"], body["skin_temperature_c"])]
        if undershirt.any():
            skin_parts.append((remap[visible[undershirt]], dress["undershirt"], body["undershirt_temperature_c"]))
        self.body_vertices = used
        rest.insert(0, body_yup[used])
        joints.insert(0, cat.weight_joints[used])
        weights.insert(0, cat.weights[used])
        sizes = [len(r) for r in rest]
        offsets = np.concatenate([[0], np.cumsum(sizes)])
        rest_all, joints_all, weights_all = np.concatenate(rest), np.concatenate(joints), np.concatenate(weights)
        dense = np.zeros((len(rest_all), len(cat.parents)))
        np.add.at(dense, (np.repeat(np.arange(len(rest_all)), joints_all.shape[1]), joints_all.ravel()), weights_all.ravel())
        self.bones = np.flatnonzero(dense.any(0))
        # Each vertex's few weights, summed bone by bone in a fixed order: vertices with the most weights come first,
        # so the k-th weight of every vertex that has one is one slice. No BLAS, so the same bits on every CPU.
        blend = dense[:, self.bones]
        count = (blend != 0).sum(1)
        order = np.argsort(-count, kind="stable")
        rows, cols = np.nonzero(blend[order])
        rank = np.arange(len(rows)) - np.searchsorted(rows, rows)
        self.slots = [(cols[rank == k], blend[order[rows[rank == k]], cols[rank == k]][:, None])
                      for k in range(int(count.max()))]
        self.unsort = np.argsort(order)
        self.rest = rest_all[order]
        self.inv_bind = _inverse(bind)
        # One visual body per colour: (vertex range in the stacked arrays, faces local to that range).
        meshes = [(0, faces, colour, celsius) for faces, colour, celsius in skin_parts]
        meshes += [(k + 1, faces, colour, celsius) for k, (faces, colour, celsius) in enumerate(parts)]
        self.meshes = []
        start = self.vertices(*first)
        for group, faces, colour, celsius in meshes:
            lo, hi = offsets[group], offsets[group + 1]
            local_used = np.unique(faces)
            local = np.full(hi - lo, -1)
            local[local_used] = np.arange(len(local_used))
            rows = lo + local_used
            shape = p.createVisualShape(p.GEOM_MESH, vertices=start[rows].tolist(), indices=local[faces].ravel().tolist(),
                                        rgbaColor=list(colour) + [1.0], specularColor=SPECULAR,
                                        physicsClientId=client)
            uid = int(p.createMultiBody(0, -1, shape, [0.0, 0.0, 0.0], physicsClientId=client))
            if THERMAL and celsius is not None:
                p.changeVisualShape(uid, -1, temperature=float(celsius), emissivity=EMISSIVITY, physicsClientId=client)
            self.meshes.append((uid, rows))
        self.moving: Optional[bool] = None

    @property
    def bodies(self) -> List[int]:
        """The visual body ids of this thief."""
        return [uid for uid, _ in self.meshes]

    @property
    def vertex_count(self) -> int:
        """Vertices posed per frame."""
        return len(self.rest)

    def vertices(self, local_xyzw: np.ndarray, root: np.ndarray, place: Sequence[float], yaw: float) -> np.ndarray:
        """Every drawn vertex for one move frame, turned by yaw and standing with its lowest point on place (x, y, z)."""
        world = self.cat.world(self.build, _quat_to_mat(np.asarray(local_xyzw, dtype=np.float64)),
                               np.asarray(root, dtype=np.float64))
        joint = _product(world[self.bones], self.inv_bind[self.bones])[:, :3].reshape(len(self.bones), 12)
        blended = np.zeros((len(self.rest), 12))
        for bones, weights in self.slots:
            blended[:len(bones)] += joint[bones] * weights
        v = _to_zup(_apply(blended.reshape(-1, 3, 4), self.rest)[self.unsort])
        c, s = math.cos(yaw), math.sin(yaw)
        x, y = v[:, 0] * c - v[:, 1] * s, v[:, 0] * s + v[:, 1] * c
        return np.stack([x + place[0], y + place[1], v[:, 2] - v[:, 2].min() + place[2]], 1)

    def pose(self, local_xyzw: np.ndarray, root: np.ndarray, place: Sequence[float], yaw: float) -> bool:
        """Rewrite every piece for one move frame; False when the engine cannot rewrite a visual mesh in place."""
        if self.moving is False:
            return False
        v = self.vertices(local_xyzw, root, place, yaw)
        for uid, rows in self.meshes:
            try:
                p.resetMeshData(uid, v[rows].tolist(), physicsClientId=self.client)
            except p.error:
                # An engine before visual resetMeshData: the thief keeps the pose it was created in.
                self.moving = False
                return False
        self.moving = True
        return True

    def remove(self) -> None:
        """Take the thief out of the scene."""
        for uid, _ in self.meshes:
            p.removeBody(uid, physicsClientId=self.client)
        self.meshes = []
