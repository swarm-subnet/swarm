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

"""Trusted Cap'n Proto service for the model-graph runner."""

from __future__ import annotations

import argparse
import asyncio
import json
import mmap
import os
from pathlib import Path

import numpy as np

from .constants import FAMILY_GRAPH_CONTRACTS, family_has_swarm_axis, resolve_shape
from .errors import ModelGraphError
from .runner import GraphRunner
from .session_probe import probe_artifact_subprocess


class ObservationDecoder:
    """Decode inline, compact-zero, and read-only shared-memory tensors."""

    def __init__(self, shm_path: str | None = None) -> None:
        self._file = None
        self._shm = None
        if shm_path and os.path.isfile(shm_path):
            self._file = open(shm_path, "rb")
            self._shm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    def close(self) -> None:
        if self._shm is not None:
            self._shm.close()
        if self._file is not None:
            self._file.close()

    def decode(self, entries) -> dict[str, np.ndarray]:
        manifest: dict[str, tuple[int, int]] = {}
        tensors = []
        for entry in entries:
            if entry.key == "__shm__":
                decoded = json.loads(bytes(entry.tensor.data).decode("utf-8"))
                manifest = {str(k): (int(o), int(n)) for k, o, n in decoded}
            else:
                tensors.append(entry)

        observation: dict[str, np.ndarray] = {}
        for entry in tensors:
            shape = tuple(int(d) for d in entry.tensor.shape)
            if entry.tensor.dtype != "float32":
                raise ValueError(f"observation '{entry.key}' dtype must be float32")
            if entry.key in manifest:
                if self._shm is None:
                    raise ValueError("observation references unavailable shared memory")
                offset, nbytes = manifest[entry.key]
                expected = int(np.prod(shape, dtype=np.int64)) * 4
                if offset < 0 or nbytes != expected or offset + nbytes > len(self._shm):
                    raise ValueError(f"invalid shared-memory range for '{entry.key}'")
                value = np.frombuffer(
                    self._shm, dtype=np.float32, count=expected // 4, offset=offset
                ).reshape(shape).copy()
            else:
                data = bytes(entry.tensor.data)
                expected = int(np.prod(shape, dtype=np.int64)) * 4
                if not data:
                    value = np.zeros(shape, dtype=np.float32)
                elif len(data) == expected:
                    value = np.frombuffer(data, dtype=np.float32).reshape(shape).copy()
                else:
                    raise ValueError(f"invalid inline tensor size for '{entry.key}'")
            value.flags.writeable = False
            observation[str(entry.key)] = value
        return observation


def _zero_action(family_id: str, observation: dict[str, np.ndarray]) -> np.ndarray:
    num_drones = None
    if family_has_swarm_axis(family_id):
        num_drones = int(next(iter(observation.values())).shape[0])
    return np.zeros(
        resolve_shape(FAMILY_GRAPH_CONTRACTS[family_id]["action_shape"], num_drones),
        dtype=np.float32,
    )


def make_agent_server(agent_capnp, runner: GraphRunner, decoder: ObservationDecoder):
    """Create the pycapnp server class lazily so unit tests need no KJ loop."""

    class AgentServer(agent_capnp.Agent.Server):
        async def ping(self, message, **kwargs):
            return "pong"

        @staticmethod
        def _tensor(value: np.ndarray):
            response = agent_capnp.Tensor.new_message()
            response.data = np.ascontiguousarray(value, dtype=np.float32).tobytes()
            response.shape = list(value.shape)
            response.dtype = "float32"
            return response

        async def act(self, obs, **kwargs):
            observation = decoder.decode(list(obs.entries))
            return self._tensor(runner.act(observation))

        async def reset(self, **kwargs):
            runner.reset()

        async def calibrate(self, obs, **kwargs):
            observation = decoder.decode(list(obs.entries))
            return self._tensor(_zero_action(runner.manifest.family_id, observation)), 0

    return AgentServer()


async def serve(artifact: Path, schema_path: Path, port: int, shm_path: str | None) -> None:
    import capnp

    probe_artifact_subprocess(artifact)
    runner = GraphRunner.from_artifact(artifact)
    decoder = ObservationDecoder(shm_path)
    agent_capnp = capnp.load(str(schema_path))
    bootstrap = make_agent_server(agent_capnp, runner, decoder)

    async def new_connection(stream):
        server = capnp.TwoPartyServer(stream, bootstrap=bootstrap)
        await server.on_disconnect()

    try:
        server = await capnp.AsyncIoStream.create_server(new_connection, "0.0.0.0", port)
        async with server:
            await server.serve_forever()
    finally:
        decoder.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a model_graph.v1 artifact")
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    async def run() -> None:
        import capnp

        async with capnp.kj_loop():
            await serve(args.artifact, args.schema, args.port, os.getenv("SWARM_OBS_SHM"))

    try:
        asyncio.run(run())
    except ModelGraphError as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
