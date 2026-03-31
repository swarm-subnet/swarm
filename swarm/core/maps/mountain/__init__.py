"""Type 3 mountain map family."""

from swarm.core import mountain_generator as _mountain_generator


def build_mountain_map(cli, seed, safe_zones, safe_zone_radius):
    return _mountain_generator.build_mountains(
        cli,
        seed,
        safe_zones,
        safe_zone_radius,
        forced_subtype=1,
    )


__all__ = list(getattr(_mountain_generator, "__all__", ())) + ["build_mountain_map"]

for _name in getattr(_mountain_generator, "__all__", ()):
    globals()[_name] = getattr(_mountain_generator, _name)

if "_name" in globals():
    del _name
