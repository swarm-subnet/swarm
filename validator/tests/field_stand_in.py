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

"""A small stand-in test field for the Test Field tests: a flat field inside a square outline on a wide flat valley.

The real field's assets are not in the released swarm-worlds package the tests run on, so the family's tests build
this one instead, through the real test site builder: a terrain slab, the field on it, and the survey's site block.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from swarm.core.maps.testsite import builder

HALF_M = 30.0                          # the outline is a 60 m square around the map origin
OUTLINE = [[-HALF_M, -HALF_M], [HALF_M, -HALF_M], [HALF_M, HALF_M], [-HALF_M, HALF_M]]
ORIGIN_LAT_LON = [42.5, 1.5]          # a rounded point in Andorra, not the field: the stand-in only needs the latitude band

_BOX = """v -0.5 -0.5 0
v 0.5 -0.5 0
v 0.5 0.5 0
v -0.5 0.5 0
v -0.5 -0.5 1
v 0.5 -0.5 1
v 0.5 0.5 1
v -0.5 0.5 1
f 1 3 2
f 1 4 3
f 5 6 7
f 5 7 8
f 1 2 6
f 1 6 5
f 2 3 7
f 2 7 6
f 3 4 8
f 3 8 7
f 4 1 5
f 4 5 8
"""


def _item(name: str, group: str) -> dict:
    """A record for the unit box, as the export writes them."""
    return {"name": name, "obj": "box.obj", "texture": "", "triangles": 12, "group": group, "folder": "kit",
            "flags": [], "collision": "mesh", "specular": 0.0, "bounds_min": [-0.5, -0.5, 0.0], "bounds_max": [0.5, 0.5, 1.0]}


def write_site(folder: Path, buildings: list = (), passable: list = ()) -> str:
    """Write the stand-in field's assets into a folder and point the builder at it; returns the folder."""
    (folder / "kit").mkdir(parents=True, exist_ok=True)
    (folder / "kit" / "box.obj").write_text(_BOX)
    items = {"valley": _item("valley", "terrain"), "field": _item("field", "field")}
    placements = [{"item": "valley", "position": [0.0, 0.0, -1.0], "quaternion": [0, 0, 0, 1], "scale": [400.0, 400.0, 1.0]},
                  {"item": "field", "position": [0.0, 0.0, -0.5], "quaternion": [0, 0, 0, 1],
                   "scale": [2.0 * HALF_M + 4.0, 2.0 * HALF_M + 4.0, 0.5]}]
    site = {"origin_lat_lon": ORIGIN_LAT_LON, "field_outline": OUTLINE, "field_area_m2": (2.0 * HALF_M) ** 2}
    if len(buildings):
        site["buildings"] = [list(map(float, row)) for row in np.asarray(buildings, dtype=float)]
    if len(passable):
        site["passable"] = [list(map(float, row)) for row in np.asarray(passable, dtype=float)]
    (folder / "manifest.json").write_text(json.dumps({"version": 1, "items": items, "placements": placements, "site": site}))
    builder.testsite_manifest.cache_clear()
    return str(folder)
