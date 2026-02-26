import asyncio
import sys
import os
import time
from pathlib import Path

try:
    import capnp
    import numpy as np
except ImportError:
    print("ERROR: pycapnp not installed")
    sys.exit(1)

schema_file = Path(__file__).parent / "agent.capnp"
agent_capnp = capnp.load(str(schema_file))

from drone_agent import DroneFlightController


class AgentServer(agent_capnp.Agent.Server):
    def __init__(self, agent):
        self.agent = agent
    
    async def ping(self, message, **kwargs):
        return "pong"
    
    async def act(self, obs, **kwargs):
        entries = list(obs.entries)
        
        if len(entries) == 1 and entries[0].key == "__value__":
            obs_array = np.frombuffer(
                entries[0].tensor.data, dtype=np.dtype(entries[0].tensor.dtype)
            ).reshape(tuple(entries[0].tensor.shape))
        else:
            obs_dict = {
                entry.key: np.frombuffer(
                    entry.tensor.data, dtype=np.dtype(entry.tensor.dtype)
                ).reshape(tuple(entry.tensor.shape))
                for entry in entries
            }
            obs_array = obs_dict
        
        action = self.agent.act(obs_array)
        
        action_np = np.array(action, dtype=np.float32)
        response = agent_capnp.Tensor.new_message()
        response.data = action_np.tobytes()
        response.shape = list(action_np.shape)
        response.dtype = str(action_np.dtype)
        
        return response

    async def calibrate(self, obs, **kwargs):
        entries = list(obs.entries)
        for entry in entries:
            _ = np.frombuffer(
                entry.tensor.data, dtype=np.dtype(entry.tensor.dtype)
            ).reshape(tuple(entry.tensor.shape))

        t0 = time.perf_counter_ns()
        a = np.random.randn(256, 256).astype(np.float32)
        b = np.random.randn(256, 256).astype(np.float32)
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
    
    server = await capnp.AsyncIoStream.create_server(
        new_connection, "0.0.0.0", port
    )
    
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

