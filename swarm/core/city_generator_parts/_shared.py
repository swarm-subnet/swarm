"""
OBJ mesh city generator for Type 1 challenge maps.
Ports PR #72 city_gen logic + run_city_sim renderer into a single V4-compatible module.
"""

import math
import os
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pybullet as p

from swarm.constants import CITY_VARIANT_DISTRIBUTION, LANDING_PLATFORM_RADIUS

# ---------------------------------------------------------------------------
# SECTION 1: Constants & asset paths
# ---------------------------------------------------------------------------
MAP_SIZE = 200
TILE_SIZE = 10
SCALE_FACTOR = 5.0
MODEL_BASE_SIZE = 1.8

ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir,
    os.pardir,
    "assets",
    "maps",
)
ASSETS_DIR = os.path.normpath(ASSETS_DIR)

KENNEY_DIR = os.path.join(ASSETS_DIR, "kenney")
OTHER_SOURCES_DIR = os.path.join(ASSETS_DIR, "other_sources")

COLORS = {
    "grass": [0.13, 0.55, 0.13, 1],
    "road": [0.3, 0.3, 0.3, 1],
}

ASSET_MAP = {
    "intersection": ["kenney_roads/Models/OBJ format/road-crossroad.obj"],
    "corner": ["kenney_roads/Models/OBJ format/road-bend.obj"],
    "t_junction": ["kenney_roads/Models/OBJ format/road-intersection.obj"],
    "straight": ["kenney_roads/Models/OBJ format/road-straight.obj"],
    "crossing": ["kenney_roads/Models/OBJ format/road-crossing.obj"],
    "roundabout": ["kenney_roads/Models/OBJ format/road-roundabout.obj"],
    "roundabout_arm": [],
    "dead_end": ["kenney_roads/Models/OBJ format/road-end-round.obj"],
    "streetlight": ["obj_converted/streetlight.obj"],
    "traffic_light": ["obj_converted/trafficlight_A.obj"],
    "tree": ["kenney_suburban/tree-large.obj", "kenney_suburban/tree-small.obj"],
    "house": [
        f"kenney_suburban/building-type-{c}.obj"
        for c in "abcdefghijklmnopqrstu"
    ],
    "apt": [
        f"kenney_commercial/building-{c}.obj"
        for c in "abcdefghijklmn"
    ],
    "tower": [
        f"kenney_commercial/building-skyscraper-{c}.obj"
        for c in "abcde"
    ],
    "sedan": ["kenney_car-kit/Models/OBJ format/sedan.obj"],
    "taxi": ["kenney_car-kit/Models/OBJ format/taxi.obj"],
    "police": ["kenney_car-kit/Models/OBJ format/police.obj"],
    "suv": ["kenney_car-kit/Models/OBJ format/suv.obj"],
    "truck": ["kenney_car-kit/Models/OBJ format/truck.obj"],
    "van": ["kenney_car-kit/Models/OBJ format/van.obj"],
    "hatchback-sports": ["kenney_car-kit/Models/OBJ format/hatchback-sports.obj"],
}

CAR_TYPES = ["sedan", "taxi", "police", "suv", "truck", "van", "hatchback-sports"]


# ---------------------------------------------------------------------------
# SECTION 2: City gen logic (ported from city_gen.py)

__all__ = [name for name in globals() if not name.startswith("__")]
