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

"""How the local RL test script builds the tasks it evaluates a model on."""
from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

import pytest

from swarm.domain_model import DEFAULT_CLI_FAMILY_ID

TEST_RL = Path(__file__).resolve().parents[1] / "src" / "RL" / "test_RL.py"


class _StopAfterTasks(Exception):
    """Raised by the stand-in builder so the script stops before it starts Docker."""


def test_default_family_tasks_come_from_the_family_runtime(monkeypatch):
    """With no --family_id, every seed is built by the default family's own task builder."""
    spec = importlib.util.spec_from_file_location("miner_test_rl", TEST_RL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    def build(**kwargs):
        """Record the call and stop the run before any task is evaluated."""
        calls.append(kwargs)
        raise _StopAfterTasks

    monkeypatch.setattr(module, "build_random_task", build)
    with pytest.raises(_StopAfterTasks):
        asyncio.run(
            module._evaluate_seeds(
                model_path=Path("model.zip"),
                uid=0,
                family_id=DEFAULT_CLI_FAMILY_ID,
                seeds=[7],
                workers=1,
                show_actions=False,
                action_log_every=1,
                verbose_eval=False,
                progress_log_every=1,
                timeout_multiplier=1.0,
                extend_timeout_on_progress=False,
                timeout_extend_sec=0.0,
                timeout_progress_stale_sec=0.0,
                timeout_progress_min_sim_advance=0.0,
                max_seed_walltime_sec=0.0,
            )
        )
    assert calls == [{"sim_dt": module.SIM_DT, "seed": 7, "family_id": "cf_solar_patrol"}]
