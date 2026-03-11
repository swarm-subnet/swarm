"""OBJ mesh mountain generator for Type 3 challenge maps."""

from .mountain_generator_parts import *

__all__ = [name for name in globals() if not name.startswith("__")]
