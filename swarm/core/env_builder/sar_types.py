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

from __future__ import annotations

import pickle
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

Vec3 = Tuple[float, float, float]


class BodyCategory(str, Enum):
    SUPPORT_TERRAIN = "SUPPORT_TERRAIN"
    SUPPORT_ROOFTOP = "SUPPORT_ROOFTOP"
    SUPPORT_FLOOR = "SUPPORT_FLOOR"
    SUPPORT_SLOPE = "SUPPORT_SLOPE"
    SUPPORT_WALKWAY = "SUPPORT_WALKWAY"
    VICTIM = "VICTIM"
    OBSTACLE_CANOPY = "OBSTACLE_CANOPY"
    OBSTACLE_BEAM = "OBSTACLE_BEAM"
    OBSTACLE_CLUTTER = "OBSTACLE_CLUTTER"
    OBSTACLE_OTHER = "OBSTACLE_OTHER"


SUPPORT_CATEGORIES = frozenset(
    {
        BodyCategory.SUPPORT_TERRAIN,
        BodyCategory.SUPPORT_ROOFTOP,
        BodyCategory.SUPPORT_FLOOR,
        BodyCategory.SUPPORT_SLOPE,
        BodyCategory.SUPPORT_WALKWAY,
    }
)


@dataclass
class SafetyPatch:
    support_uid: int
    xy: Tuple[float, float]
    surface_z: float
    radius: float = 2.5
    z_below: float = 0.25
    z_above: float = 1.35


@dataclass
class SARWorld:
    victim_uids: List[int]
    victim_aabb: Tuple[Vec3, Vec3]
    victim_centre: Vec3
    support_uid: int
    support_category: str
    surface_z: float
    safety_patch: SafetyPatch
    body_tags: Dict[int, str]
    adjusted_start: Optional[Vec3] = None
    search_centre: Optional[Tuple[float, float]] = None

    @property
    def victim_centre_xy(self) -> Tuple[float, float]:
        return (self.victim_centre[0], self.victim_centre[1])

    def to_bytes(self) -> bytes:
        return pickle.dumps(self)

    @staticmethod
    def from_bytes(blob: bytes) -> "SARWorld":
        return pickle.loads(blob)
