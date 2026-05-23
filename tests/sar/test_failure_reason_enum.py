from __future__ import annotations

from swarm.protocol import FailureReason


def test_eight_members_exist():
    expected = {
        "NONE": "NONE",
        "OBSTACLE_COLLISION": "OBSTACLE_COLLISION",
        "NO_TOUCH_SPHERE": "NO_TOUCH_SPHERE",
        "TILT": "TILT",
        "TIMEOUT": "TIMEOUT",
        "INFEASIBLE": "INFEASIBLE",
        "SPAWN_FAILURE": "SPAWN_FAILURE",
        "EVAL_ERROR": "EVAL_ERROR",
    }
    members = {m.name: m.value for m in FailureReason}
    assert members == expected
    for name, value in expected.items():
        assert FailureReason[name].value == value
        assert FailureReason(value).name == name
