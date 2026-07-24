"""Image-owned bootstrap for the trusted graph runner.

Serves a legacy model-graph champion over the same RPC the code-agent lane
uses. Nothing from the archive is imported or executed: the runner reads the
declared ONNX weights and evaluates a fixed tensor graph.
"""

from __future__ import annotations

import os
from pathlib import Path

from swarm.model_graph.server import main as run_server
from swarm.submission_template.main import wait_for_start_gate


def main() -> int:
    wait_for_start_gate()
    artifact = Path(
        os.environ.get("SWARM_MODEL_GRAPH_ARTIFACT", "/workspace/submission/model_graph.zip")
    )
    schema = Path(__file__).with_name("agent.capnp")
    port = int(os.environ.get("SWARM_AGENT_PORT", "8000"))
    return run_server(
        ["--artifact", str(artifact), "--schema", str(schema), "--port", str(port)]
    )


if __name__ == "__main__":
    raise SystemExit(main())
