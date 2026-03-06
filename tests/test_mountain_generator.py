from __future__ import annotations

import math
import random

from swarm.core import mountain_generator as mg


def test_get_global_scale_is_deterministic_and_bounded():
    s1 = mg.get_global_scale(123)
    s2 = mg.get_global_scale(123)
    assert s1 == s2
    assert mg.TYPE_3_SCALE_MIN <= s1 <= mg.TYPE_3_SCALE_MAX


def test_make_noise_params_returns_expected_shape():
    params = mg._make_noise_params(seed=7, gs=0.7)
    assert len(params) == mg.TERRAIN_N_OCTAVES
    assert all({"amp", "fx", "fy", "px", "py"} <= set(p.keys()) for p in params)


def test_get_terrain_z_is_deterministic():
    gs = mg.get_global_scale(88)
    z1 = mg.get_terrain_z(10.0, -4.0, seed=88, gs=gs)
    z2 = mg.get_terrain_z(10.0, -4.0, seed=88, gs=gs)
    assert math.isclose(z1, z2, rel_tol=1e-12, abs_tol=1e-12)


def test_too_close_detects_overlap_threshold():
    placed = [mg._Placed(x=0.0, y=0.0, radius=2.0)]
    assert mg._too_close(0.5, 0.0, radius=2.0, placed=placed, max_overlap=0.60) is True
    assert mg._too_close(10.0, 10.0, radius=1.0, placed=placed, max_overlap=0.60) is False


def test_sample_point_square_within_half_range():
    rng = random.Random(1)
    x, y = mg._sample_point_square(rng, half=5.0)
    assert -5.0 <= x <= 5.0
    assert -5.0 <= y <= 5.0
