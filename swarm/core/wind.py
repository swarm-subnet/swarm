# swarm/core/wind.py
"""Seeded wind for opted-in maps: a steady vector, Dryden-style turbulence and gust bumps,
all drawn from the map seed so the same seed always blows the same way."""
from __future__ import annotations

import math

import numpy as np

from swarm.constants import (
    WIND_GUST_DURATION_SEC,
    WIND_GUST_PEAK,
    WIND_MEAN_FRACTION,
    WIND_SEED_OFFSET,
    WIND_TURB_SIGMA_XY,
    WIND_TURB_SIGMA_Z,
    WIND_TURB_TAU_XY_SEC,
    WIND_TURB_TAU_Z_SEC,
)


def _unit_xy(rng: np.random.Generator) -> np.ndarray:
    """Uniformly random horizontal unit vector, drawn without trigonometry so every CPU agrees."""
    while True:
        v = rng.standard_normal(2)
        n = math.sqrt(float(v[0]) * float(v[0]) + float(v[1]) * float(v[1]))
        if n > 1e-6:
            return np.array([float(v[0]) / n, float(v[1]) / n, 0.0])


def _bump(s: float) -> float:
    """Smooth 0-1-0 bump over s in [0, 1], peaking at 1 in the middle."""
    if s <= 0.0 or s >= 1.0:
        return 0.0
    return 16.0 * s * s * (1.0 - s) * (1.0 - s)


class SeededWind:
    """Wind velocity for one episode as a function of time.

    steady:     horizontal vector, direction from the seed, speed a seeded fraction of the cap
    turbulence: first-order filtered noise per axis, intensity per the Dryden low-altitude
                model (0.2 x mean sideways, 0.1 x mean vertical), advanced once per call
    gusts:      a few seeded bumps along the steady direction peaking at WIND_GUST_PEAK x mean
    The total is clamped to max_mps so the map's cap is a hard limit.
    """

    def __init__(self, seed: int, *, max_mps: float, turbulence: float, gusts: int,
                 dt: float, horizon: float):
        """Draw the episode's steady wind, gust schedule and filter gains from the seed."""
        self._seed = int(seed) ^ WIND_SEED_OFFSET
        self.max_mps = float(max_mps)
        self._turbulence = float(turbulence)
        self._gusts = int(gusts)
        self._dt = float(dt)
        self._horizon = float(horizon)
        self.reset()

    def reset(self) -> None:
        """Re-draw everything from the seed so a reset episode sees the same wind again."""
        rng = np.random.default_rng(self._seed)
        lo, hi = WIND_MEAN_FRACTION
        self.mean_mps = self.max_mps * float(rng.uniform(lo, hi))
        self.direction = _unit_xy(rng)
        self.steady = self.direction * self.mean_mps
        d_lo, d_hi = WIND_GUST_DURATION_SEC
        self._gust_windows = []
        for _ in range(self._gusts):
            duration = float(rng.uniform(d_lo, d_hi))
            start = float(rng.uniform(0.0, max(self._horizon - duration, 0.0)))
            self._gust_windows.append((start, duration))
        self._gust_amp = (WIND_GUST_PEAK - 1.0) * self.mean_mps
        sigma = self._turbulence * self.mean_mps * np.array(
            [WIND_TURB_SIGMA_XY, WIND_TURB_SIGMA_XY, WIND_TURB_SIGMA_Z]
        )
        tau = np.array([WIND_TURB_TAU_XY_SEC, WIND_TURB_TAU_XY_SEC, WIND_TURB_TAU_Z_SEC])
        # Euler-discretised Ornstein-Uhlenbeck: keeps the stationary std at sigma within 1 %.
        self._decay = 1.0 - self._dt / tau
        self._gain = sigma * np.sqrt(2.0 * self._dt / tau)
        self._turb = np.zeros(3)
        self._rng = rng

    def velocity(self, t: float) -> np.ndarray:
        """Wind vector (m/s, world frame) at time t; advances the turbulence by one step."""
        self._turb = self._decay * self._turb + self._gain * self._rng.standard_normal(3)
        gust = 0.0
        for start, duration in self._gust_windows:
            gust += _bump((float(t) - start) / duration)
        wind = self.steady + self._turb + self.direction * (self._gust_amp * gust)
        speed = math.sqrt(float(wind[0]) ** 2 + float(wind[1]) ** 2 + float(wind[2]) ** 2)
        if speed > self.max_mps:
            wind = wind * (self.max_mps / speed)
        return wind
