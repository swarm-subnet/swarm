"""
Minimal reference Pilot shipped with the starter‑kit.

• Pure‑Python (complies with validator sandbox).
• No external network calls.
• Keeps to ≤ 100 MB wheel size.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch


class _Net(torch.nn.Module):
    """Tiny MLP to keep the demo wheel small."""

    def __init__(self, in_dim: int = 12, out_dim: int = 4) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, out_dim),
            torch.nn.Tanh(),  # maps to (‑1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x) * 1_000.0  # scale to ±1 000 RPM


class Pilot:
    """
    Contract expected by validators (SDK v2.0.0).

    Required methods
    ----------------
    • `reset(task: MapTask) -> None`
    • `act(obs: np.ndarray, t: float) -> np.ndarray`
    """

    def __init__(self) -> None:
        weights = Path(__file__).with_name("weights.pt")
        self.net = _Net()
        self.net.load_state_dict(torch.load(weights, map_location="cpu"))
        self.net.eval()

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------
    def reset(self, task: Any) -> None:  # `MapTask` typed as Any to avoid heavy imports
        self._t0 = 0.0  # placeholder for episodic state

    def act(self, obs: np.ndarray, t: float) -> np.ndarray:  # noqa: D401
        """Return a 4‑element RPM vector clipped by the network’s Tanh."""
        with torch.no_grad():
            rpm = self.net(torch.from_numpy(obs).float().unsqueeze(0))  # (1,4)
        return rpm.squeeze(0).numpy()
