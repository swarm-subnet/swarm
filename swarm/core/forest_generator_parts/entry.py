"""Public entrypoints for forest map generation."""

from ._shared import *
from .assets import _clamp_mode_id
from .ground import _ground_rgba_for_mode, _spawn_ground
from .hills import _spawn_hills
from .spawning import _spawn_forest_assets


def get_forest_subtype(seed: int) -> Tuple[int, int]:
    from swarm.constants import (
        FOREST_DIFFICULTY_DISTRIBUTION,
        FOREST_MODE_DISTRIBUTION,
    )

    mode_rng = random.Random(seed + 777777)
    modes = list(FOREST_MODE_DISTRIBUTION.keys())
    mode_weights = list(FOREST_MODE_DISTRIBUTION.values())
    mode_id = mode_rng.choices(modes, weights=mode_weights, k=1)[0]

    diff_rng = random.Random(seed + 888888)
    diffs = list(FOREST_DIFFICULTY_DISTRIBUTION.keys())
    diff_weights = list(FOREST_DIFFICULTY_DISTRIBUTION.values())
    difficulty_id = diff_rng.choices(diffs, weights=diff_weights, k=1)[0]

    return mode_id, difficulty_id


def build_forest(
    cli: int,
    seed: int,
    safe_zones: list,
    safe_zone_radius: float,
    hills_enabled: bool = False,
    forced_mode: Optional[int] = None,
    forced_difficulty: Optional[int] = None,
) -> None:
    """Build a Type 6 forest map on the given PyBullet client.

    Mode and difficulty are deterministic per seed via get_forest_subtype().
    Use forced_mode / forced_difficulty to override.
    """
    _ = safe_zones, safe_zone_radius
    sub_mode, sub_diff = get_forest_subtype(seed)
    mode_id = _clamp_mode_id(forced_mode) if forced_mode is not None else sub_mode
    difficulty_id = (
        max(1, min(3, int(forced_difficulty)))
        if forced_difficulty is not None
        else sub_diff
    )

    _reset_client_caches(cli)

    _spawn_ground(cli, mode_id)
    if hills_enabled:
        rgba = _ground_rgba_for_mode(mode_id)
        _spawn_hills(cli, rgba=rgba, apply_texture=(mode_id == 1))
    _spawn_forest_assets(cli, seed, mode_id=mode_id, difficulty_id=difficulty_id)


__all__ = [name for name in globals() if not name.startswith("__")]
