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

"""Shared imports, constants, and caches for warehouse helpers."""

import json
import math
import os
import random
import shutil

import pybullet as p

from swarm.core.warehouse.constants import (
    CONVEYOR_ASSETS,
    CONVEYOR_KIT_OBJ_DIR,
    CONVEYOR_KIT_TEXTURE,
    ENABLE_FORKLIFT_PARKING,
    ENABLE_LOADING_OPERATION_FORKLIFTS,
    ENABLE_LOADING_STAGING,
    ENABLE_LOADING_TRUCKS,
    ENABLE_MACHINING_CELL_LAYOUT,
    ENABLE_STORAGE_RACK_LAYOUT,
    FLOOR_INNER_MARGIN_TILES,
    FLOOR_SPAWN_SAFETY_MARGIN_M,
    FORKLIFT_MODEL_NAME,
    FORKLIFT_TEXTURE_NAME,
    HALF_X,
    HALF_Y,
    LOADING_KIT_DIR,
    LOADING_STAGING_MODELS,
    LOADING_TRUCK_EXTRA_GAP_CLOSED,
    LOADING_TRUCK_EXTRA_GAP_HALF,
    LOADING_TRUCK_MODELS,
    LOADING_TRUCK_SCALE_XYZ,
    MACHINING_FORCE_REFRESH_MTL_PROXY,
    STORAGE_RACK_MODEL_NAME,
    UNIFORM_SCALE,
    UNIFORM_SPECULAR_COLOR,
    VEHICLE_DIR,
    WAREHOUSE_BASE_SIZE_X,
    WAREHOUSE_BASE_SIZE_Y,
    WAREHOUSE_SHELL_DIR,
    WAREHOUSE_SHELL_FILES,
    WAREHOUSE_SIZE_X,
    WAREHOUSE_SIZE_Y,
)
from swarm.core.warehouse.shared import normalize_mtl_texture_paths

_TEXTURE_CACHE = {}
_OBJ_MTL_SPLIT_CACHE = {}
_OBJ_COLLISION_PROXY_CACHE = {}
_OBJ_MTL_VISUAL_PROXY_CACHE = {}
_OBJ_DOUBLE_SIDED_PROXY_CACHE = {}
_LOADING_TRUCK_ALONG_EXTENT_CACHE = {}
_NORMALIZED_MTL_DIRS = set()
_MESH_VISUAL_SHAPE_CACHE = {}
_MESH_COLLISION_SHAPE_CACHE = {}
_RESOLVED_MESH_PATH_CACHE = {}
_ORIENTED_XY_SIZE_CACHE = {}
_MODEL_BOUNDS_CACHE = {}

__all__ = [name for name in globals() if not name.startswith('__')]
