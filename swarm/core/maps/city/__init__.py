"""Type 1 city map family."""

from swarm.core import city_generator as _city_generator

__all__ = list(getattr(_city_generator, "__all__", ()))

for _name in __all__:
    globals()[_name] = getattr(_city_generator, _name)

del _city_generator
if "_name" in globals():
    del _name

