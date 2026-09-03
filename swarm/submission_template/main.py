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

from __future__ import annotations

import os
import time

try:
    from runtime_caps import install_runtime_thread_caps
except ModuleNotFoundError:
    from .runtime_caps import install_runtime_thread_caps


def wait_for_start_gate(timeout_sec: float = 120.0) -> None:
    """Block until the validator opens the start gate.

    The gate file appears only after the validator has locked down this
    container's network, so the agent is never imported with the network open.
    """
    gate = os.environ.get("SWARM_START_GATE")
    if not gate:
        return
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if os.path.exists(gate):
            return
        time.sleep(0.05)
    raise SystemExit("start gate never opened; refusing to load the agent")


def main() -> None:
    # nothing may import from the submission directory before the gate opens:
    # the validator only opens it once this container's network is locked down
    wait_for_start_gate()
    install_runtime_thread_caps()

    from agent_server import start_server
    from drone_agent import DroneFlightController

    agent = DroneFlightController()
    start_server(agent, port=int(os.environ.get("SWARM_AGENT_PORT", "8000")))


if __name__ == "__main__":
    main()
