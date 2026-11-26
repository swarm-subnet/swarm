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

"""What a machine needs installed to fly Swarm Test Field v2 as the rest of the fleet does.

The engine flags and CPU features are Sentinel's (solar's install module), since the field is drawn by the same
cameras, and one flag more: the setting that tells the thermal camera how a real area's ground holds heat.
The assets are the area package the fields are cut from, the people package and the everyday motion library
for the people, and the animals, from swarm-worlds, and the aircraft and dock, from swarm-description.

The areas are too heavy to ship inside swarm-worlds: a release that carries them lists one archive per area and
fetches them beside the package (`python -m swarm_worlds.areas`). On such a release the check also asks that every
area was unpacked from the listed archive, so every validator builds its fields from the same bytes.
"""

from __future__ import annotations

import importlib
import json
import os
from typing import List

import pybullet as p
import swarm_description

from swarm.core.maps import andorra

from ..base import IncompleteInstall
from ..solar_patrol import airframe
from ..solar_patrol import install as sentinel_install
from . import animal_skins, figures, moves, site

# What this task asks of the engine beyond Sentinel's flags: the ground's own thermal read.
ENGINE_FLAGS = ("THERMAL_FIELD_PASSIVE",)


def asset_files() -> List[str]:
    """Every swarm-worlds file a patrol of the field loads: the area package's catalogue and every area's manifest
    and survey, the animals' tables, skins and coats, and the people's bodies, colours and moves."""
    return [
        *(str(path) for path in site.asset_files()),
        *animal_skins.asset_files(),
        *figures.asset_files(),
        *moves.asset_files(),
    ]


def area_problems() -> List[str]:
    """Check the active area's root against the archive list shipped by swarm-worlds."""
    package = f"swarm-worlds {sentinel_install._installed('swarm-worlds')}"
    try:
        fetched = importlib.import_module("swarm_worlds.areas")
        found = fetched.problems(root=andorra.andorra_root())
    except (ImportError, OSError, json.JSONDecodeError, UnicodeError, KeyError, TypeError) as error:
        return [f"{package}: cannot verify area archives ({error}); run `python -m swarm_worlds.areas`"]
    return [f"{package}: {line}" for line in found]


def problems() -> List[str]:
    """Everything this install lacks to fly Swarm Test Field v2, one line per piece; empty when it has it all."""
    found = area_problems()
    missing = [path for path in asset_files() if not os.path.isfile(path)]
    if missing:
        found.append(f"swarm-worlds {sentinel_install._installed('swarm-worlds')} is missing {', '.join(missing)}")
    else:
        # Files that are all there can still hold a table nobody can be drawn from.
        found.extend(f"swarm-worlds {sentinel_install._installed('swarm-worlds')}: {line}"
                     for line in figures.problems() + moves.problems())
    drones = os.path.dirname(swarm_description.drone_dir(airframe.DRONE))
    missing = [os.path.relpath(path, drones) for path in sentinel_install.description_files() if not os.path.isfile(path)]
    if missing:
        found.append(f"swarm-description {sentinel_install._installed('swarm-description')} is missing {', '.join(missing)}")
    flags = [name for name in (*sentinel_install.ENGINE_FLAGS, *ENGINE_FLAGS) if not hasattr(p, name)]
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
    """Refuse to go on, naming every missing piece, when this install cannot fly Swarm Test Field v2."""
    found = problems()
    if found:
        raise IncompleteInstall("Swarm Test Field v2 cannot run on this install: " + "; ".join(found))
