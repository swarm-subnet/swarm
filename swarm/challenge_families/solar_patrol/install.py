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

"""What a machine needs installed to fly Swarm Sentinel as the rest of the fleet does (task 52).

Most pieces of the patrol fall back quietly when theirs is missing: no thief without the thieves' files, no farmers
without the road, a blank thermal frame on an engine without the flag, plain sky and ground on a CPU the ray caster
cannot run on. Each fallback still flies, only another patrol, so a validator with one of them scores differently and
nothing says so. Here every piece is checked up front, and a patrol on an install that lacks one is refused by name.
"""

from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version
from typing import List, Optional, Set

import pybullet as p
import swarm_worlds

from swarm.core.maps.solar import builder

from ..base import IncompleteInstall
from . import airframe, camera, decoys, dock3, intruders, theft_moves

# The engine flags the park and its pictures rest on: the ray caster with its thermal, low-light and daylight
# models, every picture flag a frame is drawn with, and the shape flags the park's meshes are loaded with.
ENGINE_FLAGS = tuple(dict.fromkeys((
    "ER_SWARM_RAYCAST", "ER_SWARM_THERMAL", "ER_SWARM_LOW_LIGHT", "ER_SWARM_NEAR_INFRARED", "ER_SWARM_SKY_SUN",
    "ER_SWARM_DAYLIGHT", "ER_SWARM_SHADOW_MAP", "ER_SWARM_MOVER_SHADOW", "ER_SWARM_LINEAR_LIGHT", "ER_EDGE_ANTIALIAS",
    "ER_ALPHA_CUTOUT", "ER_TEXTURE_FILTER", "ER_SPECULAR_GLINT", *camera.NEEDED_FLAGS,
    "VISUAL_SHAPE_DOUBLE_SIDED_MULTIBODY", "VISUAL_SHAPE_GLASS", "VISUAL_SHAPE_GLASS_BACKED",
    "VISUAL_SHAPE_RENDER_INSTANCED", "VISUAL_SHAPE_MATERIALS_FROM_MTL",
)))
# The ray caster's only code path; on a CPU without them the engine logs a warning and draws no scene.
CPU_FEATURES = ("avx2", "fma")
CPUINFO = "/proc/cpuinfo"


def asset_files() -> List[str]:
    """Every swarm-worlds file a patrol loads or falls back without: the park's manifest and the road its farmers
    drive, the thieves, their moves, the dogs, the aircraft and the dock."""
    maps, robots = swarm_worlds.maps_dir(), swarm_worlds.robots_dir()
    files = [
        os.path.join(builder.SOLAR_ASSET_DIR, "manifest.json"),
        os.path.join(maps, intruders.ASSET_DIR, "intruders.json"),
        os.path.join(maps, theft_moves.ASSET_DIR, "motions.json"),
        os.path.join(decoys.DOGS_DIR, "dogs.json"),
        *(os.path.join(robots, name) for name in (airframe.URDF, airframe.MOVING_URDF, dock3.DOCK_URDF, dock3.PAD_URDF)),
    ]
    return files + _road_files()


def _road_files() -> List[str]:
    """The road file beside every pickup the park places, where the farmers' trips are read from."""
    try:
        manifest = builder.solar_manifest(builder.SOLAR_ASSET_DIR)
    except FileNotFoundError:
        return []
    folders = {manifest["items"][place["item"]]["folder"] for place in manifest["placements"]
               if place.get("mover") == "pickup"}
    return [os.path.join(builder.SOLAR_ASSET_DIR, folder, builder.CONFIG["road_file"]) for folder in sorted(folders)]


def cpu_features() -> Optional[Set[str]]:
    """The feature flags this CPU reports, or None when they cannot be read."""
    try:
        with open(CPUINFO, encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("flags"):
                    return set(line.split(":", 1)[1].split())
    except OSError:
        return None
    return None


def _installed(name: str) -> str:
    """A distribution's installed version, or "unknown" when it is not installed as one."""
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def problems() -> List[str]:
    """Everything this install lacks to fly Swarm Sentinel, one line per piece; empty when it has it all."""
    found = []
    root = os.path.dirname(os.path.abspath(swarm_worlds.__file__))
    missing = [os.path.relpath(path, root) for path in asset_files() if not os.path.isfile(path)]
    if missing:
        found.append(f"swarm-worlds {_installed('swarm-worlds')} is missing {', '.join(missing)}")
    flags = [name for name in ENGINE_FLAGS if not hasattr(p, name)]
    if flags:
        found.append(f"swarm-bullet3 {_installed('swarm-bullet3')} has no {', '.join(flags)}")
    features = cpu_features()
    if features is None:
        found.append(f"the CPU's features cannot be read from {CPUINFO}; the ray caster needs {' and '.join(CPU_FEATURES)}")
    elif not features.issuperset(CPU_FEATURES):
        lacking = [name for name in CPU_FEATURES if name not in features]
        found.append(f"the CPU has no {' or '.join(lacking)}, which the ray caster needs")
    return found


def require() -> None:
    """Refuse to go on, naming every missing piece, when this install cannot fly Swarm Sentinel."""
    found = problems()
    if found:
        raise IncompleteInstall("Swarm Sentinel cannot run on this install: " + "; ".join(found))
