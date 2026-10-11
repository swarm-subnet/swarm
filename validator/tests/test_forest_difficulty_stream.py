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

"""The forest difficulty, which a drone can see, must not reveal the hidden search radius."""

from __future__ import annotations

from collections import defaultdict

from swarm.constants import SIM_DT
from swarm.core.forest_generator_parts.entry import get_forest_subtype
from swarm.validator.task_gen import task_for_seed_and_type


def test_every_forest_difficulty_sees_both_small_and_large_search_radii():
    """Each difficulty level draws search radii across the range, so tree density says nothing about the clue."""
    radii = defaultdict(list)
    for seed in range(1000):
        task = task_for_seed_and_type(sim_dt=SIM_DT, seed=seed, challenge_type=6, family_id="cf_autopilot")
        radii[get_forest_subtype(seed)[1]].append(task.search_radius)

    assert set(radii) == {1, 2, 3}
    for difficulty, values in radii.items():
        assert min(values) < 10.0 and max(values) > 14.0, (
            f"difficulty {difficulty} only draws radii {min(values):.1f}-{max(values):.1f} m"
        )
