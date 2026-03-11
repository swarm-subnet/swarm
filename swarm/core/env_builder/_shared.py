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
    BENCHMARK_VERSION,
    GOAL_COLOR_PALETTE,
    LANDING_PLATFORM_RADIUS,
    MAP_CACHE_ENABLED,
    MAP_CACHE_SAVE_ON_BUILD,
    MAX_ATTEMPTS_PER_OBS,
    PLATFORM,
    START_PLATFORM,
    START_PLATFORM_HEIGHT,
    START_PLATFORM_RADIUS,
    START_PLATFORM_RANDOMIZE,
    START_PLATFORM_SURFACE_Z,
    START_PLATFORM_TAKEOFF_BUFFER,
    TYPE_1_H_MAX,
    TYPE_1_H_MIN,
    TYPE_1_WORLD_RANGE,
    TYPE_2_HEIGHT_SCALE,
    TYPE_2_N_OBSTACLES,
    TYPE_2_SAFE_ZONE,
    TYPE_2_WORLD_RANGE,
    TYPE_3_VILLAGE_RANGE,
    TYPE_4_H_MAX,
    TYPE_4_H_MIN,
    TYPE_4_MIN_PLATFORM_DISTANCE,
    TYPE_4_PLATFORM_CLEARANCE,
    TYPE_4_PLATFORM_MAX_ATTEMPTS,
    TYPE_4_WORLD_RANGE_X,
    TYPE_4_WORLD_RANGE_Y,
    TYPE_6_H_MAX,
    TYPE_6_H_MIN,
    TYPE_6_WORLD_RANGE,
)
from swarm.core.maps.city import build_city as build_city_map
from swarm.core.maps.forest import build_forest_map
from swarm.core.maps.mountain import build_mountain_map, build_mountains
from swarm.core.maps.open import build_open_world
from swarm.core.maps.village import build_village_map
from swarm.core.maps.warehouse import build_warehouse_map
from swarm.validator.task_gen import get_platform_height_for_seed

STATE_DIR = Path(__file__).resolve().parent.parent.parent / "state"
MAP_CACHE_DIR = STATE_DIR / "map_cache"

current_epoch_number: int | None = None
_tao_tex_id: dict[int, int] = {}

__all__ = [name for name in globals() if not name.startswith("__")]
