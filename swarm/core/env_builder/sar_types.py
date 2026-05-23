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
