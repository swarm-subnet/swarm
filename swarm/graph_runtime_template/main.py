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
