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

"""Cap'n Proto RPC server that feeds each validator observation to the miner's flight controller."""

import asyncio
import json
import mmap
import os
import sys
import time
from pathlib import Path

from drone_agent import DroneFlightController

try:
    import capnp
    import numpy as np
except ImportError:
    print("ERROR: pycapnp not installed")
    sys.exit(1)

schema_file = Path(__file__).parent / "agent.capnp"
agent_capnp = capnp.load(str(schema_file))

_obs_shm = None
_obs_shm_path = os.environ.get("SWARM_OBS_SHM")
if _obs_shm_path and os.path.exists(_obs_shm_path):
    try:
        _obs_shm_file = open(_obs_shm_path, "rb")
        _obs_shm = mmap.mmap(_obs_shm_file.fileno(), 0, access=mmap.ACCESS_READ)
    except OSError:
        _obs_shm = None


def tensor_to_array(tensor):
    """Empty data means an all-zero tensor sent compactly; rebuild it locally."""
    shape = tuple(tensor.shape)
    dtype = np.dtype(tensor.dtype)
    if len(tensor.data) == 0:
        arr = np.zeros(shape, dtype=dtype)
        arr.flags.writeable = False
        return arr
    return np.frombuffer(tensor.data, dtype=dtype).reshape(shape)


def decode_observation(entries):
    """Rebuild the observation dict; tensors may arrive inline, as compact zeros,
    or via the read-only shared-memory file the validator writes each step."""
    manifest = {}
    tensor_entries = []
    for entry in entries:
        if entry.key == "__shm__":
            for key, offset, nbytes in json.loads(bytes(entry.tensor.data).decode()):
                manifest[key] = (int(offset), int(nbytes))
        else:
            tensor_entries.append(entry)

    obs = {}
    for entry in tensor_entries:
        key = entry.key
        if key in manifest:
            if _obs_shm is None:
                raise RuntimeError("observation shm referenced but not mounted")
            offset, nbytes = manifest[key]
            dtype = np.dtype(entry.tensor.dtype)
            arr = np.frombuffer(
                _obs_shm, dtype=dtype, count=nbytes // dtype.itemsize, offset=offset
            ).reshape(tuple(entry.tensor.shape)).copy()
            arr.flags.writeable = False
            obs[key] = arr
        else:
            obs[key] = tensor_to_array(entry.tensor)

    if len(obs) == 1 and "__value__" in obs:
        return obs["__value__"]
    return obs


class AgentServer(agent_capnp.Agent.Server):
    """Bootstrap capability the validator calls into, wrapping one controller instance."""

    def __init__(self, agent):
        """Hold the controller the act and reset calls are dispatched to."""
        self.agent = agent

    async def ping(self, message, **kwargs):
        """Answer pong so the caller can confirm the socket is live."""
        return "pong"

    async def act(self, obs, **kwargs):
        """Decode the observation, ask the controller for a move, and send it back as a float32 tensor."""
        obs_array = decode_observation(list(obs.entries))

        action = self.agent.act(obs_array)

        action_np = np.array(action, dtype=np.float32)
        response = agent_capnp.Tensor.new_message()
        response.data = action_np.tobytes()
        response.shape = list(action_np.shape)
        response.dtype = str(action_np.dtype)

        return response

    async def calibrate(self, obs, **kwargs):
        """Return a zero move plus the nanoseconds three 512x512 matrix products took, as a speed reading of the host."""
        _ = decode_observation(list(obs.entries))

        a = np.random.randn(512, 512).astype(np.float32)
        b = np.random.randn(512, 512).astype(np.float32)
        t0 = time.perf_counter_ns()
        for _ in range(3):
            np.dot(a, b)
        benchmark_ns = time.perf_counter_ns() - t0

        action_np = np.zeros(5, dtype=np.float32)
        response = agent_capnp.Tensor.new_message()
        response.data = action_np.tobytes()
        response.shape = list(action_np.shape)
        response.dtype = str(action_np.dtype)
        return response, benchmark_ns

    async def reset(self, **kwargs):
        """Clear the controller state so the next episode starts from scratch."""
        self.agent.reset()


async def serve(agent, port=8000):
    """Listen on all interfaces at port and stay up until the enclosing task is cancelled."""

    async def new_connection(stream):
        """Bind one accepted stream to an AgentServer and wait for the peer to drop."""
        server = capnp.TwoPartyServer(stream, bootstrap=AgentServer(agent))
        await server.on_disconnect()

    server = await capnp.AsyncIoStream.create_server(new_connection, "0.0.0.0", port)

    async with server:
        await server.serve_forever()


def start_server(agent, port=8000):
    """Block on the RPC loop, swallowing a Ctrl-C so shutdown is silent."""

    async def run_with_kj():
        """Open the capnp kj event loop, then serve inside it."""
        async with capnp.kj_loop():
            await serve(agent, port)

    try:
        asyncio.run(run_with_kj())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    try:
        sys.stderr.write("Initializing DroneFlightController...\n")
        sys.stderr.flush()
        agent = DroneFlightController()
        sys.stderr.write("Starting RPC server on port 8000...\n")
        sys.stderr.flush()
        start_server(agent, port=8000)
    except Exception as e:
        sys.stderr.write(f"Fatal error: {e}\n")
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
