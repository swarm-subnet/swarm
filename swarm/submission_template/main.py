from __future__ import annotations

import os
import time


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


def _apply_torch_thread_caps() -> None:
    num_threads = os.getenv("SWARM_TORCH_NUM_THREADS")
    interop_threads = os.getenv("SWARM_TORCH_INTEROP_THREADS")
    if num_threads is None and interop_threads is None:
        return

    try:
        import torch
    except Exception:
        return

    try:
        if num_threads is not None:
            torch.set_num_threads(max(1, int(num_threads)))
    except Exception:
        pass

    try:
        if interop_threads is not None and hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(max(1, int(interop_threads)))
    except Exception:
        pass


def main() -> None:
    # nothing may import from the submission directory before the gate opens:
    # the validator only opens it once this container's network is locked down
    wait_for_start_gate()
    _apply_torch_thread_caps()

    from agent_server import start_server
    from drone_agent import DroneFlightController

    agent = DroneFlightController()
    start_server(agent, port=int(os.environ.get("SWARM_AGENT_PORT", "8000")))


if __name__ == "__main__":
    main()
