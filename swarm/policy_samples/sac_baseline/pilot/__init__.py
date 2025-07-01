"""
Baseline pilot packaged as a standard Python wheel.

* Loads a pre‑trained Soft‑Actor‑Critic (SAC) policy.
* Exposes the API required by Swarm validators (reset / act).

The network architecture and weights are frozen; feel free to
use this as a template for your own policies.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import nn


class _MLPPolicy(nn.Module):
    """Simple 2‑layer MLP → tanh‑squashed actions."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Tuple[int, int] = (256, 256),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, obs_dim) → (B, act_dim)
        # tanh keeps actions in (‑1, 1); scaling to RPM happens in caller
        return torch.tanh(self.net(x))


class Pilot:
    """
    Minimal wrapper around the trained policy.

    The class is imported dynamically by the validator; keep global side
    effects (e.g. GPU allocation) to a minimum.
    """

    # ----‑‑‑ Static config --------------------------------------------------
    _OBS_DIM: int = 24   # change if your env’s observation size differs
    _ACT_DIM: int = 4    # 4 propellers → 4 RPM commands
    _RPM_CLIP: float = 1.0  # network already outputs in (‑1, 1)

    # -----------------------------------------------------------------------

    def __init__(self) -> None:
        self.net = _MLPPolicy(self._OBS_DIM, self._ACT_DIM)
        weights_path = Path(__file__).with_name("weights.pt")
        self.net.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.net.eval()          # important: disable dropout / BN
        self._t0: float = 0.0     # episodic timer

    # -----------------------------------------------------------------------#
    #   Swarm interface                                                      #
    # -----------------------------------------------------------------------#

    def reset(self, task) -> None:
        """Called once at episode start with the full `MapTask` dataclass."""
        self._t0 = 0.0            # clear any state you keep between steps

    @torch.no_grad()
    def act(self, obs: np.ndarray, t: float) -> np.ndarray:
        """
        Parameters
        ----------
        obs : np.ndarray
            Observation vector shaped (`OBS_DIM`,).
        t : float
            Sim‑time (seconds) since episode start.

        Returns
        -------
        np.ndarray
            Array of propeller RPM targets shaped (`ACT_DIM`,).
        """
        if obs.shape[0] != self._OBS_DIM:
            raise ValueError(
                f"Expected obs dim {self._OBS_DIM}, got {obs.shape[0]}"
            )

        obs_t = torch.from_numpy(obs).float().unsqueeze(0)  # (1, obs_dim)
        act = self.net(obs_t)[0].cpu().numpy()              # (act_dim,)
        return np.clip(act * self._RPM_CLIP, -1.0, 1.0)
