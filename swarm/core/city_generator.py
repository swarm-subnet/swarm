"""OBJ mesh city generator for Type 1 challenge maps."""

from .city_generator_parts import *

__all__ = [name for name in globals() if not name.startswith("__")]
