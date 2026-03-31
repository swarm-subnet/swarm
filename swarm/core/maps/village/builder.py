"""Builders for the Type 4 village benchmark map."""

from swarm.core.mountain_generator import build_mountains


def build_village_map(cli, seed, safe_zones, safe_zone_radius):
    return build_mountains(
        cli,
        seed,
        safe_zones,
        safe_zone_radius,
        forced_subtype=2,
    )

