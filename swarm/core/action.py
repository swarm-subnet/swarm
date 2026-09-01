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

"""Action shaping for code-agent submissions.

A miner's ``act()`` may return any array-like, so the validator owns the final
shape: non-finite values and wrong-sized arrays fall back to a zero action
rather than failing the seed, bounds come from the family's action space, and
the result is quantised so two validators stepping the same seed agree
bit-for-bit even when the miner's own inference differs in the last bits.
"""

from __future__ import annotations

import numpy as np

from swarm.constants import ACTION_QUANT_STEP


def canonicalize_action(
    raw,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    n_drones: int | None = None,
    act_dim: int,
) -> np.ndarray:
    """Return the miner action as a bounded, quantised float32 array.

    ``n_drones`` shapes the result as ``(n_drones, act_dim)`` for swarm families
    and ``(act_dim,)`` otherwise.
    """
    expected_size = act_dim * n_drones if n_drones else act_dim

    arr = np.asarray(raw)
    if arr.dtype.kind not in "fiub":
        # a non-numeric dtype is the miner's output, never an evaluation error
        arr = np.zeros(expected_size, dtype=np.float32)

    flat = np.nan_to_num(
        arr.astype(np.float32, copy=False).reshape(-1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if flat.size != expected_size:
        flat = np.zeros(expected_size, dtype=np.float32)

    shaped = flat.reshape(n_drones, act_dim) if n_drones else flat
    action = np.clip(shaped, lower, upper).astype(np.float32, copy=False)
    return (np.rint(action / ACTION_QUANT_STEP) * ACTION_QUANT_STEP).astype(np.float32)
