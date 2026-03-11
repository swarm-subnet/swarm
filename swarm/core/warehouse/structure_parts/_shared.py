"""
Warehouse structural build stages: floor, walls, personnel floor lane,
curved roof, roof truss system, and corner columns.
"""

import math
import random

import pybullet as p

from ..constants import (
    AREA_LAYOUT_BLOCKS,
    AREA_LAYOUT_EDGE_MARGIN,
    AREA_LAYOUT_WALL_ATTACH_THICKNESS_FACTOR,
    CONVEYOR_ASSETS,
    DOCK_INWARD_NUDGE,
    ENABLE_PERSONNEL_FLOOR_LANE,
    ENABLE_ROOF_TRUSS_SYSTEM,
    FLOOR_UNIFORM_COLOR,
    LOADING_DOOR_CENTER_STEP_FRACTION,
    LOADING_INTER_GATE_WALL_STEPS,
    LOADING_SLOTS,
    LOADING_TRUCK_WALL_GAP,
    PERSONNEL_FLOOR_LANE_MODEL_CANDIDATES,
    PERSONNEL_FLOOR_LANE_Z_OFFSET,
    ROOF_UNIFORM_COLOR,
    TRUSS_UNIFORM_COLOR,
    TRUSS_WITH_COLLISION,
    UNIFORM_SCALE,
    UNIFORM_SPECULAR_COLOR,
    WALL_SLOTS,
    WALL_TIERS,
    WALL_UNIFORM_COLOR,
    WAREHOUSE_SIZE_X,
    WAREHOUSE_SIZE_Y,
)
from ..helpers import (
    _estimate_loading_truck_along_extent_m,
    _filter_mirrored_single_windows,
    _filter_mirrored_wide_windows,
    _first_existing_model_name,
    _floor_spawn_half_extents,
    _indices_blocked_by_doors,
    _loading_marker_xy_size,
    _merge_spans_1d,
    _orient_dims_long_side_on_wall,
    _shell_mesh_scale_xy,
    _spawn_generated_mesh,
    _spawn_mesh_with_anchor,
    _subtract_spans_1d,
    _wall_along_limits,
    dock_inward_yaw_for_slot,
    mirrored_wide_window_starts,
    mirrored_window_indices,
    oriented_xy_size,
    slot_point,
    tiled_centers,
    wall_yaw_for_slot,
)

__all__ = [name for name in globals() if not name.startswith("__")]
