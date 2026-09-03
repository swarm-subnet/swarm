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
Loading zone builders: trucks, overhead cranes, and staging props.
"""

import math
import os
import random

from ..constants import (
    ENABLE_LOADING_STAGING,
    ENABLE_LOADING_TRUCKS,
    ENABLE_OVERHEAD_CRANES,
    LOADING_BARREL_MAX_STACK_LAYERS,
    LOADING_BUNDLES_PER_TRUCK_MAX,
    LOADING_BUNDLES_PER_TRUCK_MIN,
    LOADING_CONTAINER_MODEL_NAME,
    LOADING_CONTAINER_SCALE_XYZ,
    LOADING_CONTAINER_STACK_ENABLED,
    LOADING_CONTAINER_STACK_VERTICAL_GAP_M,
    LOADING_EMPTY_PALLET_STACK_COUNT,
    LOADING_EMPTY_PALLET_STACK_MAX_LAYERS,
    LOADING_EMPTY_PALLET_STACK_MIN_LAYERS,
    LOADING_KIT_DIR,
    LOADING_LOADED_PALLET_STACK_MAX_LAYERS,
    LOADING_LOADED_PALLET_STACK_MIN_LAYERS,
    LOADING_SECTION_MIN_SPAN_M,
    LOADING_STAGING_EDGE_MARGIN_M,
    LOADING_STAGING_GOODS_BACK_EDGE_PAD_M,
    LOADING_STAGING_MAX_DEPTH_M,
    LOADING_STAGING_MODELS,
    LOADING_STAGING_PROP_GAP_M,
    LOADING_STAGING_SCALES,
    LOADING_STAGING_SUPPORT_BACK_BIAS,
    LOADING_STAGING_TRUCK_TAIL_CLEARANCE_M,
    LOADING_TRUCK_MODELS,
    LOADING_TRUCK_SCALE_XYZ,
    LOADING_TRUCK_WALL_GAP,
    OVERHEAD_CRANE_ATTACH_CLEARANCE_M,
    OVERHEAD_CRANE_COLOR_GAIN,
    OVERHEAD_CRANE_MIN_SPACING_M,
    OVERHEAD_CRANE_SCALE_UNIFORM,
    OVERHEAD_CRANE_TARGET_BY_ZONE,
    OVERHEAD_CRANE_TRUSS_TOUCH_EXTRA_M,
    OVERHEAD_CRANE_WITH_COLLISION,
    OVERHEAD_CRANE_YAW_EXTRA_DEG,
    OVERHEAD_CRANE_ZONE_EDGE_MARGIN_M,
    WALL_SLOTS,
    WAREHOUSE_BASE_SIZE_X,
    WAREHOUSE_SIZE_X,
    WAREHOUSE_SIZE_Y,
)
from ..helpers import (
    _obj_double_sided_proxy_path,
    _obj_material_parts,
    _spawn_collision_only_with_anchor,
    _spawn_mesh_with_anchor,
    _truck_extra_gap_for_gate_state,
    dock_inward_yaw_for_slot,
    model_bounds_xyz,
    slot_point,
)

__all__ = [name for name in globals() if not name.startswith("__")]
