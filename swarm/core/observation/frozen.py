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
"""Frames frozen by the code that made them, which the observation hand-over may trust to keep their bytes."""

from __future__ import annotations

import weakref

import numpy as np

# Keyed by id with weak values, since arrays cannot be hashed; a dead array drops out, so a reused id never matches.
_FROZEN: "weakref.WeakValueDictionary[int, np.ndarray]" = weakref.WeakValueDictionary()


def freeze(frame: np.ndarray) -> np.ndarray:
    """Make a frame its producer has just created read-only and record that its bytes never change.

    Only the producer may call this, while it holds the only reference: a writable view taken earlier could still
    change the bytes, which no flag can detect."""
    frame.flags.writeable = False
    _FROZEN[id(frame)] = frame
    return frame


def is_frozen(frame: np.ndarray) -> bool:
    """Whether this exact array was frozen by its producer."""
    return _FROZEN.get(id(frame)) is frame
