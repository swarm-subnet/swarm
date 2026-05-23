"""D.4 guardrail: moving_platform field, constants, and methods are deleted.

The only allowed mention of the literal string ``moving_platform`` in
production code is the back-compat shim in ``MapTask.unpack`` that drops the
field from legacy payloads.
"""
from __future__ import annotations

import pathlib
import re

import msgpack

from swarm.protocol import MapTask

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROD_DIRS = ("swarm", "scripts")

ALLOWED_PATHS = {
    ROOT / "swarm" / "protocol.py",
}

PATTERN = re.compile(r"moving_platform|MOVING_PLATFORM|MovingPlatform")


def _iter_prod_files():
    for d in PROD_DIRS:
        for path in (ROOT / d).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            yield path


def test_no_moving_platform_in_production_code():
    offenders = []
    for path in _iter_prod_files():
        if path in ALLOWED_PATHS:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for ln, line in enumerate(text.splitlines(), 1):
            if PATTERN.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{ln}: {line.strip()}")
    assert not offenders, "moving_platform leaked into production:\n" + "\n".join(offenders)


def test_maptask_has_no_moving_platform_attribute():
    task = MapTask(
        map_seed=1,
        start=(0.0, 0.0, 1.5),
        goal=(8.0, 8.0, 1.5),
        sim_dt=1 / 30,
        horizon=60.0,
        challenge_type=2,
        version="5.0.0",
    )
    assert not hasattr(task, "moving_platform"), \
        "MapTask.moving_platform field should be deleted in D.4"


def test_legacy_payload_drops_moving_platform_silently():
    legacy_payload = {
        "map_seed": 1,
        "start": [0.0, 0.0, 1.5],
        "goal": [8.0, 8.0, 1.5],
        "sim_dt": 1 / 30,
        "horizon": 60.0,
        "challenge_type": 2,
        "version": "5.0.0",
        "moving_platform": True,
    }
    blob = msgpack.packb(legacy_payload, use_bin_type=True)
    task = MapTask.unpack(blob)
    assert not hasattr(task, "moving_platform")
    assert task.challenge_type == 2


def test_constants_module_has_no_moving_platform_constants():
    from swarm import constants

    for name in (
        "MOVING_PLATFORM_RADIUS",
        "MOVING_PLATFORM_SPEED",
        "MOVING_PLATFORM_PERIOD",
        "PLATFORM_ORBIT_RADIUS",
    ):
        assert not hasattr(constants, name), f"constants.{name} should be gone"
