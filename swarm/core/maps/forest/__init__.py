"""Type 6 forest map family."""

from swarm.core.forest_generator import build_forest


def build_forest_map(cli, seed, safe_zones, safe_zone_radius):
    return build_forest(
        cli,
        seed,
        safe_zones,
        safe_zone_radius,
        hills_enabled=True,
    )


__all__ = ["build_forest", "build_forest_map"]
