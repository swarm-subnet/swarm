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

import json
import re
from pathlib import Path

import bittensor as bt
import numpy as np

from swarm.constants import DOCKER_PIP_WHITELIST


def _normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _validate_requirements(self, requirements_path: Path, uid: int) -> bool:
    """Gate a miner's requirements.txt against the approved package list.

    Only plain, whitelisted package specifiers are accepted: installer options,
    direct URL/path installs and PEP 508 direct references are all refused, so the
    dependency phase cannot fetch arbitrary code.
    """
    try:
        lines = requirements_path.read_text().splitlines()
    except Exception as e:
        bt.logging.warning(f"UID {uid}: Failed to read requirements.txt: {e}")
        return False

    rejected = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("-"):
            bt.logging.warning(f"UID {uid}: Installer option not allowed: {line}")
            return False

        if line.startswith(("git+", "http://", "https://", "file:", "./", "/")):
            bt.logging.warning(f"UID {uid}: Direct URL/path install not allowed: {line}")
            return False

        if " @ " in line:
            bt.logging.warning(f"UID {uid}: PEP 508 direct reference not allowed: {line}")
            return False

        line = line.split("#")[0].strip()
        if not line:
            continue

        line = line.split(";")[0].strip()
        if " " in line:
            bt.logging.warning(
                f"UID {uid}: Unexpected token after requirement: {raw_line.strip()!r}"
            )
            return False
        name = re.split(r"[>=<!~\[]", line)[0].strip()
        if not name:
            continue

        normalized = self._normalize_package_name(name)
        if normalized not in DOCKER_PIP_WHITELIST:
            rejected.append(normalized)

    if rejected:
        bt.logging.warning(
            f"UID {uid}: Requirements rejected — packages not whitelisted: {', '.join(rejected)}"
        )
        return False

    return True


def _all_zero_bits(arr):
    if arr.flags["C_CONTIGUOUS"]:
        return not arr.view(np.uint8).any()
    return not arr.any() and not np.signbit(arr).any()


def _serialize_observation_shm(agent_capnp, obs, shm_buf):
    """Like _serialize_observation, but tensor payloads go into the shared-memory
    buffer the container mounts read-only; the message carries only shapes plus a
    __shm__ manifest of (key, offset, nbytes). All-zero tensors stay compact.
    Raises BufferError when the observation does not fit."""
    items = list(obs.items()) if isinstance(obs, dict) else [("__value__", obs)]
    message = agent_capnp.Observation.new_message()
    entries = message.init("entries", len(items) + 1)
    manifest = []
    offset = 0
    for i, (key, value) in enumerate(items):
        arr = np.asarray(value, dtype=np.float32)
        entries[i].key = key
        entries[i].tensor.data = b""
        entries[i].tensor.shape = list(arr.shape)
        entries[i].tensor.dtype = str(arr.dtype)
        if arr.nbytes and not _all_zero_bits(arr):
            nbytes = arr.nbytes
            end = offset + nbytes
            if end > len(shm_buf):
                raise BufferError("observation exceeds shm buffer")
            dst = np.frombuffer(shm_buf, dtype=np.uint8, count=nbytes, offset=offset)
            src = arr if arr.flags["C_CONTIGUOUS"] else np.ascontiguousarray(arr)
            dst[:] = src.reshape(-1).view(np.uint8)
            manifest.append([key, offset, nbytes])
            offset = (end + 63) & ~63
    tail = entries[len(items)]
    tail.key = "__shm__"
    tail.tensor.data = json.dumps(manifest).encode()
    tail.tensor.shape = []
    tail.tensor.dtype = "manifest"
    return message

@staticmethod
def _serialize_observation(agent_capnp, obs):
    """Serialize a numpy observation dict into a Cap'n Proto Observation message.

    All-zero tensors (e.g. the on-demand RGB frame on steps where no drone
    requested one) are sent with empty data; the receiver rebuilds the zero
    array from shape+dtype, so the miner sees byte-identical observations.
    The zero check is byte-exact, so -0.0 payloads still ship in full.
    """
    message = agent_capnp.Observation.new_message()
    if isinstance(obs, dict):
        entries = message.init("entries", len(obs))
        for i, (key, value) in enumerate(obs.items()):
            arr = np.asarray(value, dtype=np.float32)
            entries[i].key = key
            entries[i].tensor.data = b"" if _all_zero_bits(arr) else arr.tobytes()
            entries[i].tensor.shape = list(arr.shape)
            entries[i].tensor.dtype = str(arr.dtype)
    else:
        arr = np.asarray(obs, dtype=np.float32)
        entry = message.init("entries", 1)[0]
        entry.key = "__value__"
        entry.tensor.data = b"" if _all_zero_bits(arr) else arr.tobytes()
        entry.tensor.shape = list(arr.shape)
        entry.tensor.dtype = str(arr.dtype)
    return message
