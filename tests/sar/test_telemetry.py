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
