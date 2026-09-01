from __future__ import annotations

from swarm.protocol import SCHEMA_VERSION
from swarm.validator.task_gen import random_task


def test_task_gen_emits_v5():
    task = random_task(sim_dt=1 / 30, seed=42)
    assert task.version == SCHEMA_VERSION
    assert task.version.startswith("5.")
