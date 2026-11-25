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

"""Real Andorra areas for the simulator: where the package lies, its catalogue, and each area's folder.

Each area is a 400 m square of real ground with its trees, rocks, houses, water and the mountains around it. The
package root is SWARM_ANDORRA_ROOT when set, else swarm_worlds/maps/andorra. It holds areas.json
({"version", "areas": [...]}), a shared/ folder of pieces many areas use, and one folder per area named by its slug
(NN_name) with manifest.json (builder) and spots.npz (survey). A catalogue row gives the area's id, slug, place,
kind, parish and frame.

Heights: a manifest placement's z is already the world z. The survey's heights stand above its own base, so a world z
is the survey height plus Survey.z_shift(frame).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import swarm_worlds

ROOT_ENV = "SWARM_ANDORRA_ROOT"


def andorra_root() -> Path:
    """The package root: SWARM_ANDORRA_ROOT when set, else the installed swarm-worlds maps."""
    override = os.environ.get(ROOT_ENV)
    return Path(override) if override else Path(swarm_worlds.maps_dir()) / "andorra"


def catalog(root: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """Every area the package lists, by id, so the order never depends on how the file was written."""
    path = Path(root if root is not None else andorra_root()) / "areas.json"
    if not path.is_file():
        raise FileNotFoundError(f"area catalogue missing: {path}; install the worlds package or set {ROOT_ENV}")
    areas = sorted(json.loads(path.read_text())["areas"], key=lambda row: int(row["id"]))
    if not areas:
        raise ValueError(f"area catalogue is empty: {path}")
    return areas


def area_folder(area: Union[int, Dict[str, Any]], root: Optional[Union[str, Path]] = None) -> Path:
    """The folder of an area, given its catalogue row or its id."""
    root = Path(root if root is not None else andorra_root())
    if not isinstance(area, dict):
        row = next((row for row in catalog(root) if int(row["id"]) == int(area)), None)
        if row is None:
            raise KeyError(f"no area {int(area)} in {root / 'areas.json'}")
        area = row
    folder = root / area["slug"]
    if not (folder / "manifest.json").is_file():
        raise FileNotFoundError(f"area manifest missing: {folder / 'manifest.json'}")
    return folder
