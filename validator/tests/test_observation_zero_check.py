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

"""The zero check that keeps blank tensors out of a model's observation message: it reads a payload's first bytes
first, and still calls a payload blank exactly when every one of its bytes is zero."""

from __future__ import annotations

import numpy as np
import pytest

from swarm.validator.docker.docker_evaluator_parts import submission


def _blank(arr: np.ndarray) -> bool:
    """The rule itself: every byte of the payload is zero."""
    return arr.tobytes() == bytes(arr.nbytes)


def _cases():
    """Payloads around the first bytes' edge: blank, a -0.0, data inside and past the first bytes, odd layouts."""
    frame = np.zeros((480, 640, 3), dtype=np.float32)
    yield "blank frame", frame
    for index in (0, 1023, 1024, 1025, frame.size - 1):
        for value in (1e-30, -0.0, 0.5):
            hit = frame.copy()
            hit.flat[index] = value
            yield f"{value} at {index}", hit
    yield "empty", np.zeros((0, 3), dtype=np.float32)
    yield "scalar", np.zeros((), dtype=np.float32)
    yield "short", np.array([0.0, -0.0], dtype=np.float32)
    for name, cell, value in (("strided blank", (0, 1), 1.0), ("strided -0.0", (3, 2), -0.0),
                              ("strided data", (5, 4), 2.0)):
        strided = np.zeros((64, 64), dtype=np.float32)
        strided[cell] = value
        yield name, strided[:, ::2]


@pytest.mark.parametrize("name,arr", list(_cases()), ids=[name for name, _ in _cases()])
def test_a_payload_is_blank_exactly_when_every_byte_is_zero(name, arr):
    """The check agrees with reading every byte, wherever the first non-zero byte lies."""
    assert submission._all_zero_bits(arr) == _blank(arr)
