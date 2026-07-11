"""Image-owned bootstrap for the trusted graph runner."""

from __future__ import annotations

import os
import time
from pathlib import Path

from swarm.model_graph.server import main as run_server


def wait_for_start_gate(timeout_sec: float = 120.0) -> None:
    """Block until the validator opens the start gate.

    The gate file appears only after the validator has locked down this
    container's network, so the artifact is never parsed with the network open.
    """
    gate = os.environ.get("SWARM_START_GATE")
    if not gate:
        return
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if os.path.exists(gate):
            return
        time.sleep(0.05)
    raise SystemExit("start gate never opened; refusing to load the artifact")


def main() -> int:
    wait_for_start_gate()
    artifact = Path(os.environ.get("SWARM_MODEL_GRAPH_ARTIFACT", "/workspace/model_graph.zip"))
    schema = Path(__file__).with_name("agent.capnp")
    port = int(os.environ.get("SWARM_AGENT_PORT", "8000"))
    return run_server([
        "--artifact", str(artifact), "--schema", str(schema), "--port", str(port)
    ])


if __name__ == "__main__":
    raise SystemExit(main())
