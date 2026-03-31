from __future__ import annotations

import os


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
    _apply_torch_thread_caps()

    from agent_server import start_server
    from drone_agent import DroneFlightController

    agent = DroneFlightController()
    start_server(agent, port=8000)


if __name__ == "__main__":
    main()
