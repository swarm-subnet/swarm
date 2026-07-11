import json

import numpy as np


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
