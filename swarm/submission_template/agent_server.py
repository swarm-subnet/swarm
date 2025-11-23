import asyncio
import sys
import os
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
    def __init__(self, agent, auth_token=None):
        self.agent = agent
        self.auth_token = auth_token
        self.authenticated = auth_token is None
    
    async def ping(self, message, **kwargs):
        if self.auth_token and not self.authenticated:
            if message == self.auth_token:
                self.authenticated = True
                return "pong"
            else:
                raise ValueError("Authentication failed")
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
    
    async def reset(self, **kwargs):
        self.agent.reset()


async def serve(agent, port=9000, auth_token=None):
    async def new_connection(stream):
        server = capnp.TwoPartyServer(stream, bootstrap=AgentServer(agent, auth_token))
        await server.on_disconnect()
    
    server = await capnp.AsyncIoStream.create_server(
        new_connection, "0.0.0.0", port
    )
    
    async with server:
        await server.serve_forever()


def start_server(agent, port=9000, auth_token=None):
    async def run_with_kj():
        async with capnp.kj_loop():
            await serve(agent, port, auth_token)
    
    try:
        asyncio.run(run_with_kj())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    agent = DroneFlightController()
    start_server(agent, port=9000)

