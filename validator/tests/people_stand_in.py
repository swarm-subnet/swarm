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

"""A small generated people package in the people's package format, so the tests need no shipped people.

It stands on Sentinel's own skeleton, which the installed worlds carry: two of its builds, a woman's and a man's; a
skin of every other one of the body's triangles, so it fills its outline; and one piece for every slot (eyes, hair, a light top,
a warm top, trousers, a skirt, shoes, a safety vest), each a run of the skin's triangles lifted off it. Every picture
is four texels across.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from swarm.challenge_families.solar_patrol import intruders

BUILDS = {"woman": ("f", "slim", 1.66), "man": ("m", "tall", 1.86)}
SKIN_EVERY = 2                         # one of this many of the body's triangles is drawn as skin
# name: (slot, sex, warmth, which of the drawn skin triangles it lies on, pictures as name -> palette)
PIECES = {
    "eyes": ("eyes", "any", "any", (0, 120), {"brown": None, "blue": None}),
    "hair_short": ("hair", "any", "any", (120, 900), {"combed": "hair"}),
    "tee": ("top", "any", "light", (900, 5400), {"plain": "top", "striped": "top"}),
    "sweater": ("top", "any", "warm", (900, 6000), {"plain": "top"}),
    "jeans": ("bottom", "any", "any", (6000, 10800), {"denim": "bottom"}),
    "skirt_short": ("bottom", "f", "light", (6000, 9000), {"plain": "skirt"}),
    "sneakers": ("shoes", "any", "any", (10800, 12000), {"plain": "shoe"}),
    "vest": ("over", "any", "any", (1500, 4500), {"yellow": None}),
}
PALETTES = {"top": [[0.9, 0.9, 0.9], [0.7, 0.2, 0.2], [0.2, 0.3, 0.6], [0.2, 0.4, 0.3], [0.9, 0.7, 0.2]],
            "bottom": [[0.1, 0.1, 0.1], [0.3, 0.3, 0.5], [0.6, 0.5, 0.4], [0.4, 0.4, 0.4]],
            "skirt": [[0.7, 0.2, 0.2], [0.2, 0.3, 0.6], [0.9, 0.9, 0.9]], "shoe": [[0.9, 0.9, 0.9], [0.1, 0.1, 0.1]],
            "hair_dark": [[0.1, 0.1, 0.1], [0.3, 0.2, 0.1]], "hair_light": [[0.7, 0.6, 0.4], [0.6, 0.6, 0.6]]}


def _picture(path: Path, colour: tuple) -> None:
    """Write a picture of one colour, four texels across."""
    Image.fromarray(np.full((4, 4, 3), colour, dtype=np.uint8)).save(path)


def write_package(folder: Path) -> str:
    """Write the stand-in people package into a folder and return the folder."""
    folder = Path(folder)
    if (folder / "people.json").is_file():
        return str(folder)
    (folder / "pieces").mkdir(parents=True, exist_ok=True)
    (folder / "pictures").mkdir(exist_ok=True)
    cat = intruders.catalogue()
    which = [cat.builds.index(source) for _sex, source, _height in BUILDS.values()]
    np.savez(folder / "rig.npz", joint_names=np.array(cat.joint_names), parents=cat.parents.astype(np.int16),
             soma30=np.array([cat.joint_names[k] for k in cat.slot]), relaxed_hands=cat.relaxed,
             build_names=np.array(list(BUILDS)), build_bind=cat.build_bind[which], build_rest=cat.build_rest[which],
             build_root_scale=cat.root_scale[which], weight_joints=cat.weight_joints.astype(np.int16),
             weights=cat.weights.astype(np.float32))
    drawn = cat.faces[::SKIN_EVERY]
    source, faces = np.unique(drawn, return_inverse=True)
    faces = faces.reshape(-1, 3).astype(np.int32)
    uv = np.stack([(np.arange(len(source)) % 7) / 7.0, (np.arange(len(source)) % 5) / 5.0], 1).astype(np.float32)
    points = cat.build_verts[which].astype(np.float32)
    np.savez(folder / "body.npz", points=points, faces=faces, source=source.astype(np.int32), uv=uv, point_faces=drawn.astype(np.int32))
    manifest: Dict[str, object] = {
        "format": 1, "body": {"temperature_c": 33.0}, "palettes": PALETTES, "pieces": {}, "skins": {},
        "eyes_of_tone": {"light": ["brown", "blue"], "deep": ["brown"]},
        "builds": {name: {"sex": sex, "height_m": height} for name, (sex, _source, height) in BUILDS.items()}}
    for name, (slot, sex, warmth, (first, last), pictures) in PIECES.items():
        tris = drawn[first:last]
        own, piece_faces = np.unique(tris, return_inverse=True)
        lifted = points[:, own] * np.float32(1.01)
        np.savez(folder / "pieces" / f"{name}.npz", points=lifted, weight_joints=cat.weight_joints[own].astype(np.int16),
                 weights=cat.weights[own].astype(np.float32), faces=piece_faces.reshape(-1, 3).astype(np.int32),
                 source=np.arange(len(own), dtype=np.int32), uv=uv[:len(own)], hides=np.arange(first, last, dtype=np.int32))
        entry = {"file": f"pieces/{name}.npz", "slot": slot, "sex": sex, "warmth": warmth, "temperature_c": 27.0, "pictures": {}}
        for style, palette in pictures.items():
            leaf = f"pictures/{name}_{style}.png"
            _picture(folder / leaf, (230, 230, 230) if palette else (220, 230, 40))
            entry["pictures"][style] = {"file": leaf, "palette": palette}
        manifest["pieces"][name] = entry
    skins: List[tuple] = [("light", "dark", "f"), ("light", "light", "f"), ("deep", "dark", "f"), ("light", "dark", "m"),
                          ("light", "light", "m"), ("deep", "dark", "m")]
    for tone, hair, sex in skins:
        leaf = f"pictures/skin_{tone}_{hair}_{sex}.png"
        _picture(folder / leaf, (200, 160, 130) if tone == "light" else (100, 70, 50))
        manifest["skins"][f"{tone}_{hair}_{sex}"] = {"file": leaf, "tone": tone, "hair": hair, "sex": sex}
    (folder / "people.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return str(folder)
