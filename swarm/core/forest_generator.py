"""OBJ mesh forest generator for Type 6 challenge maps."""

from .forest_generator_parts import *

__all__ = [name for name in globals() if not name.startswith("__")]
