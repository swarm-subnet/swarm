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

"""The sky photographs a daylight family paints behind its colour camera.

The worlds package ships a pack of real skies, each with the heading and height of the sun in it.
A seed picks one whose sun stands about as high as the seed's own sun and turns it so both suns
share a heading; the renderer then lights the map from the seeded sun and paints, reflects and
lights from the photograph. Same seed, same photograph, same turn, on every validator.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from swarm.constants import DAYLIGHT_SKY_ELEVATION_TOLERANCE_DEG, DAYLIGHT_SKY_SEED_OFFSET
from swarm.core.daylight import SunLight


@dataclass(frozen=True)
class SkyPhoto:
    """One sky of the pack: its file, and where its sun stands in the picture."""

    name: str
    path: str
    sun_azimuth_deg: float
    sun_elevation_deg: float


def load_sky_pack(folder: Optional[str] = None) -> List[SkyPhoto]:
    """The pack from its manifest, in the manifest's order; the worlds package's own pack by default."""
    if folder is None:
        try:
            import swarm_worlds
        except ImportError as exc:
            raise RuntimeError("the daylight sky pack needs the swarm-worlds package with its skies folder") from exc
        folder = swarm_worlds.skies_dir()
    with open(os.path.join(folder, "skies.json"), "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return [
        SkyPhoto(
            name=str(entry["name"]),
            path=os.path.join(folder, str(entry["file"])),
            sun_azimuth_deg=float(entry["sun_azimuth_deg"]),
            sun_elevation_deg=float(entry["sun_elevation_deg"]),
        )
        for entry in manifest["skies"]
    ]


def sky_yaw_deg(photo: SkyPhoto, sun: SunLight) -> float:
    """The turn, in degrees about the up axis, that puts the photograph's sun on the seeded sun's heading."""
    return round((photo.sun_azimuth_deg - sun.azimuth_deg) % 360.0, 3)


def pick_sky(seed: int, sun: SunLight, pack: Sequence[SkyPhoto]) -> Tuple[SkyPhoto, float]:
    """The photograph and turn for a seed: drawn among the skies whose sun stands within the
    tolerance of the seed's sun, or among the three nearest in height when none does."""
    if not pack:
        raise ValueError("the sky pack is empty")
    by_height = sorted(pack, key=lambda photo: (abs(photo.sun_elevation_deg - sun.elevation_deg), photo.name))
    candidates = [photo for photo in by_height if abs(photo.sun_elevation_deg - sun.elevation_deg) <= DAYLIGHT_SKY_ELEVATION_TOLERANCE_DEG]
    if not candidates:
        candidates = by_height[:3]
    rng = random.Random((int(seed) ^ DAYLIGHT_SKY_SEED_OFFSET) & 0xFFFFFFFF)
    photo = rng.choice(sorted(candidates, key=lambda photo: photo.name))
    return photo, sky_yaw_deg(photo, sun)


__all__ = ["SkyPhoto", "load_sky_pack", "pick_sky", "sky_yaw_deg"]
