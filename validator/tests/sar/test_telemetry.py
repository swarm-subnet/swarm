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

import contextlib
import io

import pytest

from swarm.protocol import MapTask


def _task():
    return MapTask(
        map_seed=314,
        start=(0.0, 0.0, 1.5),
        goal=(8.0, 8.0, 1.5),
        sim_dt=1 / 30,
        horizon=60.0,
        challenge_type=2,
        family_id="cf_search_and_rescue",
        version="5.0.0",
    )


def _build():
    from swarm.core.moving_drone import MovingDroneAviary

    with contextlib.redirect_stdout(io.StringIO()):
        env = MovingDroneAviary(
            _task(), ctrl_freq=30, pyb_freq=30, sar_mode=True,
        )
        env.reset(seed=_task().map_seed)
    return env


@pytest.mark.timeout(180)
def test_info_carries_sar_telemetry_fields():
    env = _build()
    try:
        info = env._computeInfo()
        for key in (
            "failure_reason",
            "sar_min_horizontal_distance",
            "sar_min_sphere_distance",
            "sar_max_dwell",
            "sar_spawn_attempts",
            "t_to_confirm",
            "schema_version",
            "task_version",
            "success",
        ):
            assert key in info, f"{key} missing"
        assert info["failure_reason"] == "NONE"
        assert info["schema_version"] == "5.0.0"
        assert info["task_version"] == "5.0.0"
    finally:
        try:
            env.close()
        except Exception:
            pass
