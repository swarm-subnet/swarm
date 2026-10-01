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

"""Fixed-order arithmetic: the patrol's dot products, lengths and turns, and the geometry built on them, give the same
bits whichever BLAS kernel the CPU would pick."""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
import sys
import types

import numpy as np

from swarm.challenge_families.solar_patrol import airframe, camera, decoys, dock, drone_state, laser, theft, theft_moves, theft_site, zoom
from swarm.challenge_families.solar_patrol.contract import Box
from swarm.challenge_families.solar_patrol.fixed_order import dot, norm, rotate, rows_dot
from swarm.core.maps.solar import builder


def test_the_sums_run_left_to_right_in_plain_floats():
    """Each helper gives the left-to-right sum of single roundings it promises."""
    a, b = [0.1, 0.2, 0.3], [0.7, -1.3, 2.9]
    assert dot(a, b) == (0.1 * 0.7 + 0.2 * -1.3) + 0.3 * 2.9
    assert norm(a) == math.sqrt((0.1 * 0.1 + 0.2 * 0.2) + 0.3 * 0.3)
    m = [0.5, -0.25, 1.5, 2.0, 0.125, -1.0, 3.0, 0.75, -0.5]
    assert rotate(m, a).tolist() == [dot(m[0:3], a), dot(m[3:6], a), dot(m[6:9], a)]
    assert rows_dot(np.array([a, b]), a).tolist() == [dot(a, a), dot(b, a)]


def _geometry_sha() -> str:
    """Hash of the patrol's geometry over random inputs: camera pose, laser and zoom rays, thief and dog visibility,
    walking tracks, site-map footprints, outlines and tables, the dock's slope and the park's gaps."""
    rng = np.random.default_rng(5)
    digest = hashlib.sha256()
    for _ in range(200):
        q = rng.normal(size=4)
        q /= np.sqrt((q * q).sum())
        env = types.SimpleNamespace(quat=[q], pos=[rng.normal(0.0, 50.0, 3)])
        eye, forward, up = airframe.camera_pose(env, float(rng.uniform(-90.0, 30.0)))
        beam = laser._Beam(None, eye, forward)
        view = camera.View(feed="colour", eye=tuple(eye.tolist()), forward=tuple(forward.tolist()), up=tuple(up.tolist()),
                           width=640, height=480, vertical_fov_deg=50.0, sees=True, step=0)
        point = eye + rng.normal(0.0, 20.0, 3)
        route = theft_moves.Route(rng.normal(0.0, 10.0, (5, 2)))
        lane = theft_moves.Lane(rng.normal(0.0, 10.0, 2), rng.normal(0.0, 10.0, 2))
        pos = rng.normal(0.0, 10.0, 2)
        place = {"quaternion": q.tolist(), "scale": [1.0, 1.0, 1.0], "position": rng.normal(0.0, 50.0, 3).tolist()}
        outline = theft_site._outline({"bounds_min": [-1.0, -0.5, 0.0], "bounds_max": [1.5, 0.5, 2.0]}, place)
        piece = {"low": np.array([-1.0, -0.5]), "high": np.array([1.5, 0.5])}
        corners = [builder._corners(piece, rng.normal(0.0, 3.0, 2), float(rng.uniform(-3.0, 3.0)), 1.0) for _ in range(2)]
        ground = np.column_stack([rng.uniform(-2.0, 2.0, (60, 2)), np.zeros(60)])
        ground[:, 2] = rows_dot(ground[:, :2], [0.05, -0.02]) + rng.normal(0.0, 0.01, 60)
        values = [*eye, *forward, *up, *beam.up, *beam.right, *zoom.box_ray(view, Box(0.3, 0.7, 0.1, 0.1)),
                  theft._in_view(view, point), decoys._pixels(view, point, 1.0), route.heading(pos), route.left(pos),
                  lane.heading(pos), lane.left(pos), *drone_state._footprint(np.array([-1.0, -0.5, 0.0]),
                                                                             np.array([1.5, 0.5, 2.0]), place),
                  *outline.ravel(), builder._gap(*corners), dock._slope_deg(ground, np.zeros(2))]
        digest.update(np.asarray(values, dtype=np.float64).tobytes() + repr(theft_site._table(outline, 0)).encode())
    return digest.hexdigest()


def test_the_geometry_is_the_same_bits_on_every_blas_kernel():
    """The patrol's geometry over random inputs hashes the same under the oldest and a fused multiply-add BLAS
    kernel, so validators on different CPU families see, track and judge the same."""
    script = "from validator.tests.test_solar_fixed_order import _geometry_sha; print(_geometry_sha())"
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    hashes = {subprocess.run([sys.executable, "-c", script], cwd=root, env=dict(os.environ, OPENBLAS_CORETYPE=core),
                             capture_output=True, text=True, check=True).stdout.split()[-1]
              for core in ("Prescott", "Haswell")}
    assert hashes == {_geometry_sha()}
