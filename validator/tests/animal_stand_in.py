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

"""Small generated animals in the animals' package format, so the tests need no shipped goat, dog or bird.

Each body is a box a metre long on four short legs' worth of points, with a head point that rises and falls through
its frames, drawn with one seam so a body point shows on two drawn vertices; a bird's box is BIRD_SCALE of that, its
head point raised like a spread wing so the bird reaches further from its origin than its length. Each
clip is a few frames; the travelling clips carry the body at walking or flying speeds. The goat, dog and bird tables
carry the clips and joins their minds use.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

GOAT_CLIPS = {"walk": 0.8, "trot": 1.4, "graze_walk": 0.16, "head_down": 0.0, "graze": 0.0, "head_up": 0.0,
              "stand": 0.0, "look": 0.0, "lie_down": 0.0, "lying": 0.0, "get_up": 0.0}
GOAT_JOINS = {"stand": ["walk", "look", "head_down", "lie_down"], "walk": ["stand", "trot", "graze_walk"],
              "trot": ["walk"], "graze_walk": ["walk", "graze"], "head_down": ["graze"], "graze": ["head_up", "graze_walk"],
              "head_up": ["stand"], "look": ["stand"], "lie_down": ["lying"], "lying": ["get_up"], "get_up": ["stand"]}
DOG_CLIPS = {"walk": 0.75, "trot": 1.7, "run": 3.7, "stand": 0.0, "look": 0.0, "sniff": 0.0, "nose_down": 0.0,
             "sniff_walk": 0.45, "lie_down": 0.0, "lying": 0.0, "get_up": 0.0}
DOG_JOINS = {"stand": ["walk", "look", "sniff", "nose_down", "lie_down"], "walk": ["stand", "trot", "sniff_walk"],
             "trot": ["walk", "run"], "run": ["trot"], "sniff_walk": ["walk", "sniff"], "sniff": ["stand", "sniff_walk"],
             "nose_down": ["sniff"], "look": ["stand"], "lie_down": ["lying"], "lying": ["get_up"], "get_up": ["stand"]}
BIRD_CLIPS = {"walk": 0.4, "hop": 0.69, "stand": 0.0, "look": 0.0, "peck": 0.0, "take_off": 0.0, "fly": 9.0, "glide": 8.0,
              "land": 0.0}
BIRD_JOINS = {"stand": ["walk", "look", "peck", "take_off"], "walk": ["stand", "hop"], "hop": ["walk"], "look": ["stand"],
              "peck": ["stand"], "take_off": ["fly"], "fly": ["glide", "land"], "glide": ["fly", "land"], "land": ["stand"]}
RAT_CLIPS = {"walk": 0.3, "trot": 1.4, "stand": 0.0, "look": 0.0, "sniff": 0.0, "nose_down": 0.0, "sniff_walk": 0.12,
             "groom": 0.0}
RAT_JOINS = {"stand": ["walk", "look", "sniff", "nose_down", "groom"], "walk": ["stand", "trot", "sniff_walk"],
             "trot": ["walk"], "sniff_walk": ["walk", "sniff"], "sniff": ["stand", "sniff_walk"], "nose_down": ["sniff"],
             "look": ["stand"], "groom": ["stand"]}
RAT_SCALE = 0.3
BIRDS = ("chough", "crow", "magpie", "pigeon", "woodpigeon")
BIRD_SCALE = 0.4
BIRD_HEAD_RISE = 0.9
FPS = 8
CLIP_FPS = {"take_off": 4.4, "land": 3.2}   # a bird takes off and lands over about as long as the shipped birds do
KINDS = {"goat": (GOAT_CLIPS, GOAT_JOINS, 1.0, 0.0), "dog": (DOG_CLIPS, DOG_JOINS, 1.0, 0.0),
         "bird": (BIRD_CLIPS, BIRD_JOINS, BIRD_SCALE, BIRD_HEAD_RISE), "boar": (GOAT_CLIPS, GOAT_JOINS, 1.0, 0.0),
         "rat": (RAT_CLIPS, RAT_JOINS, RAT_SCALE, 0.0)}
ENDS = {"head_down": "graze", "head_up": "stand", "lie_down": "lying", "get_up": "stand", "peck": "stand",
        "take_off": "fly", "land": "stand"}
FRAMES = 4
_BOX = [(x, y, z) for x in (-0.5, 0.5) for y in (-0.15, 0.15) for z in (0.2, 0.6)]
_FACES = [(0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3), (0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
          (2, 3, 7), (2, 7, 6), (5, 8, 7), (4, 6, 8)]


def _poses(clips: Dict[str, float], scale: float = 1.0, rise: float = 0.0) -> np.ndarray:
    """Every frame of every clip: the box still, its head point raised and bobbing by the frame, all at a scale."""
    frames = []
    for _ in clips:
        for f in range(FRAMES):
            frames.append(_BOX + [(0.65, 0.0, 0.55 + rise + 0.05 * f)])
    return np.asarray(np.asarray(frames) * scale, dtype=np.float16)


def _image(path: Path, colour: Tuple[int, int, int]) -> str:
    """A 4 x 4 image of one colour; returns its file name."""
    Image.fromarray(np.full((4, 4, 3), colour, dtype=np.uint8)).save(path)
    return path.name


def write_kind(folder: Path, kind: str, bodies: List[str]) -> str:
    """Write a kind's table and its bodies into a folder; returns the table's path."""
    clips, joins, scale, rise = KINDS[kind]
    folder.mkdir(parents=True, exist_ok=True)
    source = np.array(list(range(9)) + [8], dtype=np.uint16)
    faces = np.array(_FACES[:-1] + [(4, 6, 9)], dtype=np.uint16)
    uv = np.linspace(0.0, 1.0, 2 * len(source), dtype=np.float32).reshape(-1, 2)
    table_bodies = {}
    for n, name in enumerate(bodies):
        np.savez(folder / f"{name}_poses.npz", poses=_poses(clips, scale, rise), source=source, uv=uv, faces=faces)
        coats = {coat: _image(folder / f"{name}_{coat}.png", (40 * k + 20 * n, 90, 60)) for k, coat in enumerate(("brown", "grey", "white"))}
        spec = {}
        for k, (clip, speed) in enumerate(clips.items()):
            spec[clip] = {"start": k * FRAMES, "frames": FRAMES, "fps": CLIP_FPS.get(clip, FPS) if kind == "bird" else FPS,
                          "loop": clip not in ENDS, "speed_m_s": speed}
            if clip in ENDS:
                spec[clip]["then"] = ENDS[clip]
        table_bodies[name] = {"obj": f"{name}.obj", "poses": f"{name}_poses.npz", "coats": coats,
                              "heat_map": _image(folder / f"{name}_heat.png", (200, 200, 200)),
                              "length_m": 1.15 * scale, "height_m": 0.6 * scale, "clips": spec}
    table = {"class": kind, "threat": False, "decoy": True, "heat": {"range_c": [20.0, 37.0], "emissivity": 0.98},
             "size_spread": [0.9, 1.1], "pace_spread": [0.9, 1.1], "blend_s": 0.25, "joins": joins, "bodies": table_bodies}
    path = folder / f"{kind}s.json"
    path.write_text(json.dumps(table))
    return str(path)


def write_package(root: Path) -> str:
    """Write a goat kind of one body, a dog kind of three, a bird kind of five, a boar kind of two and a rat kind of
    two under a root laid out as the worlds' `testsite` folder; returns the root."""
    write_kind(root / "goats", "goat", ["doe"])
    write_kind(root / "dogs", "dog", ["husky", "shepherd", "shiba"])
    write_kind(root / "birds", "bird", list(BIRDS))
    write_kind(root / "boars", "boar", ["boar", "piglet"])
    write_kind(root / "rats", "rat", ["mouse", "rat"])
    return str(root)
