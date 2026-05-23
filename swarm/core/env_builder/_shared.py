from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import pybullet as p

from swarm.constants import (
    GOAL_AREA_CLEARANCE,
    SAFE_ZONE_RADIUS,
)
from swarm.core.maps.city import build_city as build_city_map
from swarm.core.maps.forest import build_forest_map
from swarm.core.maps.mountain import build_mountain_map, build_mountains
from swarm.core.maps.open import build_open_world
from swarm.core.maps.village import build_village_map
from swarm.core.maps.warehouse import build_warehouse_map

STATE_DIR = Path(__file__).resolve().parent.parent.parent / "state"
_tao_tex_id: dict[int, int] = {}

__all__ = [name for name in globals() if not name.startswith("__")]
