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

"""What a machine needs installed to fly Swarm Test Field as the rest of the fleet does.

The engine flags and CPU features are Sentinel's (solar's install module), since the field is drawn by the same
cameras. The assets are the field's own map, Sentinel's bodies and moves for its people, and the aircraft and dock.
"""

from __future__ import annotations

import os
from typing import List

import pybullet as p
import swarm_worlds

from ..base import IncompleteInstall
from ..solar_patrol import airframe, dock3, intruders, theft_moves
from ..solar_patrol import install as sentinel_install
from . import site


def asset_files() -> List[str]:
    """Every file a patrol of the field loads: the field's manifest, the people's bodies and moves, the aircraft and
    the dock."""
    maps, robots = swarm_worlds.maps_dir(), swarm_worlds.robots_dir()
    return [
        os.path.join(site.asset_dir(), "manifest.json"),
        os.path.join(maps, intruders.ASSET_DIR, "intruders.json"),
        os.path.join(maps, theft_moves.ASSET_DIR, "motions.json"),
        *(os.path.join(robots, name) for name in (airframe.URDF, airframe.MOVING_URDF, dock3.DOCK_URDF, dock3.PAD_URDF)),
    ]


def problems() -> List[str]:
    """Everything this install lacks to fly Swarm Test Field, one line per piece; empty when it has it all."""
    found = []
    missing = [path for path in asset_files() if not os.path.isfile(path)]
    if missing:
        found.append(f"swarm-worlds {sentinel_install._installed('swarm-worlds')} is missing {', '.join(missing)}")
    flags = [name for name in sentinel_install.ENGINE_FLAGS if not hasattr(p, name)]
    if flags:
        found.append(f"swarm-bullet3 {sentinel_install._installed('swarm-bullet3')} has no {', '.join(flags)}")
    features = sentinel_install.cpu_features()
    if features is None:
        found.append(f"the CPU's features cannot be read from {sentinel_install.CPUINFO}; the ray caster needs "
                     f"{' and '.join(sentinel_install.CPU_FEATURES)}")
    elif not features.issuperset(sentinel_install.CPU_FEATURES):
        lacking = [name for name in sentinel_install.CPU_FEATURES if name not in features]
        found.append(f"the CPU has no {' or '.join(lacking)}, which the ray caster needs")
    return found


def require() -> None:
    """Refuse to go on, naming every missing piece, when this install cannot fly Swarm Test Field."""
    found = problems()
    if found:
        raise IncompleteInstall("Swarm Test Field cannot run on this install: " + "; ".join(found))
