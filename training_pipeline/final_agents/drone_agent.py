"""Placeholder deployable controller for the training-pipeline scaffold."""

from __future__ import annotations

import numpy as np


class DroneFlightController:
    """Non-functional placeholder controller.

    Replace this with a real export once the pipeline produces a trained policy.
    """

    def __init__(self) -> None:
        self._step_count = 0

    def act(self, observation: dict) -> np.ndarray:
        self._step_count += 1
        action = np.zeros(5, dtype=np.float32)
        action[0] = 1.0
        action[3] = 0.25
        return action

    def reset(self) -> None:
        self._step_count = 0
