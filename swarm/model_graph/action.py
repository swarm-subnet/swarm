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

"""Canonical action adapter.

The runner owns action validation: shape and dtype are contract, non-finite
output is an artifact fault, bounds are clipped, continuous components are
quantized so two validators agree bit-for-bit, and the SAR RGB request is
binarized before it reaches the simulator.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .constants import (
    ACTION_QUANT_STEP,
    FAMILY_GRAPH_CONTRACTS,
    RGB_REQUEST_THRESHOLD,
    resolve_shape,
)
from .errors import ModelGraphError, ReasonCode


def canonicalize_action(
    raw: np.ndarray, family_id: str, num_drones: int | None = None
) -> np.ndarray:
    contract = FAMILY_GRAPH_CONTRACTS[family_id]
    expected = resolve_shape(contract["action_shape"], num_drones)

    if raw.dtype != np.float32:
        raise ModelGraphError(
            ReasonCode.OUTPUT_CONTRACT, f"action dtype {raw.dtype} is not float32"
        )
    if tuple(raw.shape) != expected:
        raise ModelGraphError(
            ReasonCode.OUTPUT_CONTRACT,
            f"action shape {tuple(raw.shape)} does not match {expected}",
        )
    if not np.isfinite(raw).all():
        raise ModelGraphError(ReasonCode.OUTPUT_NONFINITE, "action contains NaN or infinity")

    lower = np.asarray(contract["action_lower"], dtype=np.float32)
    upper = np.asarray(contract["action_upper"], dtype=np.float32)
    action = np.clip(raw, lower, upper).astype(np.float32, copy=True)
    action = (np.rint(action / ACTION_QUANT_STEP) * ACTION_QUANT_STEP).astype(np.float32)

    rgb_index = contract["rgb_request_index"]
    if rgb_index is not None:
        request = action[..., rgb_index]
        action[..., rgb_index] = np.where(request > RGB_REQUEST_THRESHOLD, 1.0, 0.0)

    return action


def action_digest(action: np.ndarray) -> str:
    """Stable digest of a canonical action; identical runs must produce identical digests."""
    return hashlib.sha256(np.ascontiguousarray(action, dtype=np.float32).tobytes()).hexdigest()
