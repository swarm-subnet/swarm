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
from swarm.core.env_builder.sar_types import BodyCategory, SARWorld


def _task(challenge_type=2):
    return MapTask(
        map_seed=2024,
        start=(0.0, 0.0, 1.5),
        goal=(8.0, 8.0, 1.5),
        sim_dt=1 / 30,
        horizon=60.0,
        challenge_type=challenge_type,
        family_id="cf_search_and_rescue",
        version="5.0.0",
    )


def _build_aviary(sar_mode: bool):
    from swarm.core.moving_drone import MovingDroneAviary

    with contextlib.redirect_stdout(io.StringIO()):
        env = MovingDroneAviary(
            _task(2),
            ctrl_freq=30,
            pyb_freq=30,
            sar_mode=sar_mode,
        )
        env.reset(seed=_task(2).map_seed)
    return env


@pytest.mark.timeout(180)
def test_sar_mode_loads_sar_world():
    env = _build_aviary(sar_mode=True)
    try:
        assert env.sar_mode is True
        assert isinstance(env.sar_world, SARWorld)
        assert len(env.sar_world.victim_uids) >= 1
        assert env.sar_world.search_centre is not None
    finally:
        try:
            env.close()
        except Exception:
            pass


@pytest.mark.timeout(180)
def test_sar_mode_writes_search_centre_into_env_and_task():
    env = _build_aviary(sar_mode=True)
    try:
        sc = env.sar_world.search_centre
        assert sc is not None
        assert env._search_area_center[0] == pytest.approx(sc[0])
        assert env._search_area_center[1] == pytest.approx(sc[1])
        assert env.task.search_centre[0] == pytest.approx(sc[0])
        assert env.task.search_centre[1] == pytest.approx(sc[1])
    finally:
        try:
            env.close()
        except Exception:
            pass


@pytest.mark.timeout(180)
def test_sar_mode_suppresses_open_scenery():
    env = _build_aviary(sar_mode=True)
    try:
        body_tags = env.sar_world.body_tags
        victim_uids = set(env.sar_world.victim_uids)
        spurious_victims = [
            u for u, t in body_tags.items()
            if t == BodyCategory.VICTIM.value and u not in victim_uids
        ]
        assert not spurious_victims
    finally:
        try:
            env.close()
        except Exception:
            pass


@pytest.mark.timeout(180)
def test_sar_mode_is_always_on_post_cutover():
    """Even when the caller passes sar_mode=False the env normalises to True
    after D.4 cutover; the legacy landing path no longer exists."""
    env = _build_aviary(sar_mode=False)
    try:
        assert env.sar_mode is True
    finally:
        try:
            env.close()
        except Exception:
            pass
