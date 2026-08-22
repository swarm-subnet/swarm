"""A pre-evaluated seed must be reported under the epoch that leased it.

The backend assigns a champion the NEXT epoch's seeds. If the validator uploads
those scores under today's epoch instead, no lease matches, the pool never
completes, and the reaper reopens it from seed zero — the evaluation appears to
run, then reset, forever.
"""
from __future__ import annotations

import inspect

from swarm.validator.utils_parts import evaluation


def test_the_upload_epoch_comes_from_the_task_not_the_clock():
    src = inspect.getsource(evaluation._run_full_benchmark)
    assert "epoch_number if epoch_number is not None" in src, (
        "_run_full_benchmark must honour the assigned task epoch"
    )
    assert "epoch = self.seed_manager.epoch_number" not in src, (
        "the task epoch must not be overwritten with the validator's current epoch"
    )


def test_todays_epoch_is_still_used_when_the_task_names_none():
    """The ordinary benchmark path passes no epoch and must be unchanged."""
    src = inspect.getsource(evaluation._run_full_benchmark)
    assert "else self.seed_manager.epoch_number" in src
