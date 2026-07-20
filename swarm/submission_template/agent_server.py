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
    def __init__(self, agent):
        self.agent = agent

    async def ping(self, message, **kwargs):
        return "pong"

    async def act(self, obs, **kwargs):
        obs_array = decode_observation(list(obs.entries))

        action = self.agent.act(obs_array)

        action_np = np.array(action, dtype=np.float32)
        response = agent_capnp.Tensor.new_message()
        response.data = action_np.tobytes()
        response.shape = list(action_np.shape)
        response.dtype = str(action_np.dtype)

        return response

    async def calibrate(self, obs, **kwargs):
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
        self.agent.reset()


async def serve(agent, port=8000):
    async def new_connection(stream):
        server = capnp.TwoPartyServer(stream, bootstrap=AgentServer(agent))
        await server.on_disconnect()

    server = await capnp.AsyncIoStream.create_server(new_connection, "0.0.0.0", port)

    async with server:
        await server.serve_forever()


def start_server(agent, port=8000):
    async def run_with_kj():
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
