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

"""
Operational builders: forklifts, worker crew, parking, machining cell.
"""

import math
import os
import random
import shutil

from ..constants import (
    ENABLE_FORKLIFT_PARK_SLOT_LINES,
    ENABLE_FORKLIFT_PARKING,
    ENABLE_LOADING_OPERATION_FORKLIFTS,
    ENABLE_MACHINING_CELL_LAYOUT,
    ENABLE_WORKER_CREW,
    FORKLIFT_AREA_PREFERENCE,
    FORKLIFT_MODEL_NAME,
    FORKLIFT_PARK_GAP_M,
    FORKLIFT_PARK_LINE_CENTER_Z,
    FORKLIFT_PARK_LINE_HEIGHT_M,
    FORKLIFT_PARK_LINE_RGBA,
    FORKLIFT_PARK_LINE_WIDTH_M,
    FORKLIFT_PARK_SLOT_ALONG_PAD_M,
    FORKLIFT_PARK_SLOT_COUNT,
    FORKLIFT_PARK_SLOT_CROSS_PAD_M,
    FORKLIFT_PARK_SPAWN_MAX,
    FORKLIFT_PARK_SPAWN_MIN,
    FORKLIFT_PARK_YAW_EXTRA_DEG,
    FORKLIFT_SCALE_UNIFORM,
    FORKLIFT_WALL_BACK_CLEARANCE,
    LOADING_OPERATION_FORKLIFT_EMPTY_OFFSET_M,
    LOADING_OPERATION_FORKLIFT_TARGET_COUNT,
    LOADING_OPERATION_FORKLIFT_TRUCK_OFFSET_M,
    LOADING_OPERATION_TRUCK_KEEPOUT_ALONG_PAD_M,
    LOADING_OPERATION_TRUCK_KEEPOUT_CROSS_PAD_M,
    MACHINING_AISLE_WIDTH,
    MACHINING_CELL_AREA_NAME,
    MACHINING_EDGE_MARGIN,
    MACHINING_FORCE_REFRESH_MTL_PROXY,
    MACHINING_FORCE_SIMPLE_VISUALS,
    MACHINING_HEAVY_EXTRA_YAW_DEG,
    MACHINING_LATHE_MODEL_NAME,
    MACHINING_LATHE_SCALE_UNIFORM,
    MACHINING_MILL_MODEL_NAME,
    MACHINING_MILL_SCALE_UNIFORM,
    MACHINING_PENDING_RGBA,
    MACHINING_PENDING_SLOT_SIZE,
    MACHINING_SHOW_PENDING_MARKERS,
    MACHINING_SIMPLE_LATHE_RGBA,
    MACHINING_SIMPLE_MILL_RGBA,
    MACHINING_SLOT_TYPES,
    MACHINING_TABLE_RGBA,
    MACHINING_TABLE_SIZE,
    MACHINING_USE_NATIVE_MTL_VISUALS,
    MACHINING_USE_PART_TEXTURES,
    MACHINING_VISUAL_DOUBLE_SIDED,
    WALL_SLOTS,
    WORKER_COLOR_GAIN,
    WORKER_MIN_SPACING_M,
    WORKER_TARGET_COUNT,
    WORKER_TARGET_HEIGHT_M,
)
from ..helpers import (
    _OBJ_COLLISION_PROXY_CACHE,
    _OBJ_DOUBLE_SIDED_PROXY_CACHE,
    _OBJ_MTL_SPLIT_CACHE,
    _OBJ_MTL_VISUAL_PROXY_CACHE,
    _TEXTURE_CACHE,
    _attached_wall_from_area_bounds,
    _obj_collision_proxy_path,
    _obj_material_parts,
    _obj_mtl_visual_proxy_path,
    _safe_token_name,
    _spawn_box_primitive,
    _spawn_collision_only_with_anchor,
    _spawn_mesh_with_anchor,
    _spawn_native_mtl_visual_with_anchor,
    model_bounds_xyz,
    slot_point,
)
from ..loading import _spawn_obj_with_mtl_parts

__all__ = [name for name in globals() if not name.startswith("__")]
