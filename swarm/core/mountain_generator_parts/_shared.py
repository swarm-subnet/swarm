"""
OBJ mesh mountain generator for Type 3 challenge maps.
Ports PR #72 mountain scripts (load_mountains_only + load_ski_village) into a single V4-compatible module.

Subtypes:
    Mountains Only (75%) — procedural snow terrain with scattered peaks/hills
    Ski Village    (25%) — flat village with road grid, buildings, and mountain rings
"""

import math
import os
import random
import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import pybullet as p

from swarm.constants import (
    MOUNTAIN_SUBTYPE_DISTRIBUTION,
    TYPE_3_SCALE_MAX,
    TYPE_3_SCALE_MIN,
    TYPE_3_SCALE_SEED_OFFSET,
)

# ---------------------------------------------------------------------------
# SECTION 1: Constants & asset paths
# ---------------------------------------------------------------------------
ASSETS_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir,
        os.pardir,
        "assets",
        "maps",
    )
)
CUSTOM_DIR = os.path.join(ASSETS_DIR, "custom")
MOUNTAIN_DIR = os.path.join(CUSTOM_DIR, "mountains")
BUILDING_DIR = os.path.join(CUSTOM_DIR, "buildings")
KENNEY_DIR = os.path.join(ASSETS_DIR, "kenney")
ROAD_ASSET_DIR = os.path.join(KENNEY_DIR, "kenney_roads", "Models", "OBJ format")
SUBURBAN_DIR = os.path.join(KENNEY_DIR, "kenney_suburban")
HOLIDAY_DIR = os.path.join(KENNEY_DIR, "holiday")
CAR_ASSET_DIR = os.path.join(KENNEY_DIR, "kenney_car-kit", "Models", "OBJ format")

PEAK_OBJ = os.path.join(MOUNTAIN_DIR, "mountain_peak.obj")
PEAK_TEX = os.path.join(MOUNTAIN_DIR, "mountain_peak.png")
LANTERN_PATH = os.path.join(HOLIDAY_DIR, "lantern.obj")
LANTERN_ROOF_PATH = os.path.join(BUILDING_DIR, "SnowRoofs", "lantern_roof.obj")

SNOW = [0.98, 0.98, 1.0, 1]
ROAD_COLOR = [0.35, 0.35, 0.38, 1]

TERRAIN_RESOLUTION = 97
TERRAIN_N_OCTAVES = 4
TERRAIN_TILES = 4

VILLAGE_SIZE = 100.0
ROAD_WIDTH = 6.0
HOUSE_SCALE = 2.0
HOUSE_GAP = 0.3

CAR_ASSETS = [
    "sedan.obj",
    "hatchback-sports.obj",
    "taxi.obj",
    "police.obj",
    "suv.obj",
    "van.obj",
    "truck.obj",
]

ROAD_ASSETS = {
    "intersection": "road-crossroad.obj",
    "corner": "road-bend.obj",
    "t_junction": "road-intersection.obj",
    "straight_v": "road-straight.obj",
    "straight_h": "road-straight.obj",
    "crossing": "road-crossing.obj",
    "roundabout": "road-roundabout.obj",
    "dead_end": "road-end-round.obj",
}

HOUSE_SPECS = [
    ("building-type-a.obj", 6.50, 5.14),
    ("building-type-b.obj", 9.14, 5.70),
    ("building-type-c.obj", 6.44, 5.14),
    ("building-type-d.obj", 8.79, 5.14),
    ("building-type-e.obj", 6.50, 5.14),
    ("building-type-f.obj", 7.14, 7.03),
    ("building-type-g.obj", 7.25, 5.89),
    ("building-type-h.obj", 6.50, 4.58),
    ("building-type-i.obj", 6.44, 5.14),
    ("building-type-j.obj", 6.85, 4.58),
    ("building-type-k.obj", 4.61, 5.10),
    ("building-type-l.obj", 5.20, 5.12),
    ("building-type-m.obj", 7.14, 7.14),
    ("building-type-n.obj", 8.93, 6.89),
    ("building-type-o.obj", 6.35, 5.14),
    ("building-type-p.obj", 6.20, 4.95),
    ("building-type-q.obj", 6.20, 4.43),
    ("building-type-r.obj", 5.16, 5.12),
    ("building-type-s.obj", 7.03, 5.44),
    ("building-type-t.obj", 6.60, 7.05),
    ("building-type-u.obj", 7.14, 5.44),
]


# ---------------------------------------------------------------------------

__all__ = [name for name in globals() if not name.startswith("__")]
