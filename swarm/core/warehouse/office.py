"""Embedded office room for the warehouse map."""

from .office_parts import *

__all__ = [name for name in globals() if not name.startswith("__")]
