# The MIT License (MIT)
# Copyright © 2026 Swarm

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Dot products, lengths and turns for the patrol, summed in a written order of plain float operations.

numpy hands `@`, `np.dot` and the length of a vector to OpenBLAS, which picks its kernel for the CPU it runs on; the
kernels add in different orders and some fuse a multiply into an add, so two validators can get different last bits
from the same numbers, and a flight, a picture or a decision built on them can part. Every operation here is one
IEEE rounding in the order written, the same on every machine.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    """a . b of two short vectors, summed left to right."""
    a, b = np.asarray(a, dtype=float).tolist(), np.asarray(b, dtype=float).tolist()
    total = a[0] * b[0]
    for x, y in zip(a[1:], b[1:]):
        total += x * y
    return total


def norm(a: Sequence[float]) -> float:
    """Length of a short vector."""
    return math.sqrt(dot(a, a))


def rotate(matrix: Sequence[float], v: Sequence[float]) -> np.ndarray:
    """v turned by a 3x3 matrix, given as pybullet's nine row-major numbers or as rows."""
    m = np.asarray(matrix, dtype=float).ravel().tolist()
    x, y, z = np.asarray(v, dtype=float).tolist()
    return np.array([m[0] * x + m[1] * y + m[2] * z, m[3] * x + m[4] * y + m[5] * z, m[6] * x + m[7] * y + m[8] * z])


def rows_dot(points: np.ndarray, v: Sequence[float]) -> np.ndarray:
    """Every row of points (..., k) dotted with v (k,), in elementwise numpy operations in a fixed order."""
    points, v = np.asarray(points, dtype=float), np.asarray(v, dtype=float).tolist()
    out = points[..., 0] * v[0]
    for i in range(1, len(v)):
        out = out + points[..., i] * v[i]
    return out
