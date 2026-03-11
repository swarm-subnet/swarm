"""Procedural world building and static cache helpers."""

from ._shared import MAP_CACHE_DIR, STATE_DIR
from .build import build_world
from .cache import (
    _build_static_world_cache_meta,
    _invalidate_static_world_cache,
    _normalize_xy,
    _read_static_world_cache_meta,
    _save_static_world_cache_from_client,
    _static_world_cache_file,
    _static_world_cache_meta_file,
    _try_load_static_world_cache,
    _write_static_world_cache_meta,
    cleanup_old_epoch_cache,
    prebuild_static_world_cache,
    set_map_cache_epoch,
)
from .generation import (
    _add_box,
    _build_static_world,
    _find_clear_platform_position,
    _get_tao_tex,
    _raycast_surface_z,
)

__all__ = [
    "STATE_DIR",
    "MAP_CACHE_DIR",
    "set_map_cache_epoch",
    "cleanup_old_epoch_cache",
    "prebuild_static_world_cache",
    "build_world",
    "_raycast_surface_z",
    "_add_box",
    "_get_tao_tex",
    "_normalize_xy",
    "_static_world_cache_file",
    "_static_world_cache_meta_file",
    "_build_static_world_cache_meta",
    "_write_static_world_cache_meta",
    "_read_static_world_cache_meta",
    "_invalidate_static_world_cache",
    "_build_static_world",
    "_save_static_world_cache_from_client",
    "_try_load_static_world_cache",
    "_find_clear_platform_position",
]
