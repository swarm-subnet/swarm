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

import random

from swarm.challenge_families import get_challenge_family
from swarm.constants import SWARM_COUNT_SEED_OFFSET, SWARM_MAX_DRONES, SWARM_MIN_DRONES
from swarm.validator.task_gen import screening_task


def _strip_n_drones(slots: tuple[dict, ...]) -> list[dict]:
    stripped = []
    for slot in slots:
        copy = dict(slot)
        copy.pop("n_drones", None)
        stripped.append(copy)
    return stripped


def test_swarm_templates_carry_deterministic_drone_counts():
    span = SWARM_MAX_DRONES - SWARM_MIN_DRONES + 1
    for family_id in ("cf_swarm_autopilot", "cf_swarm_sar"):
        family = get_challenge_family(family_id)
        for template in (family.screening_template(), family.benchmark_template()):
            assert template
            for i, slot in enumerate(template):
                assert SWARM_MIN_DRONES <= slot["n_drones"] <= SWARM_MAX_DRONES
                assert slot["n_drones"] == SWARM_MIN_DRONES + (i % span)


def test_swarm_templates_match_parent_templates_without_drone_counts():
    swarm_autopilot = get_challenge_family("cf_swarm_autopilot")
    autopilot = get_challenge_family("cf_autopilot")
    swarm_sar = get_challenge_family("cf_swarm_sar")
    sar = get_challenge_family("cf_search_and_rescue")

    def no_warehouse(template):
        return [slot for slot in template if slot["challenge_type"] != 5]

    assert _strip_n_drones(swarm_autopilot.screening_template()) == no_warehouse(autopilot.screening_template())
    assert _strip_n_drones(swarm_autopilot.benchmark_template()) == no_warehouse(autopilot.benchmark_template())
    assert _strip_n_drones(swarm_sar.screening_template()) == no_warehouse(sar.screening_template())
    assert all(slot["challenge_type"] != 5 for slot in swarm_sar.benchmark_template())


def test_screening_task_accepts_template_drone_count_and_keeps_random_fallback():
    explicit = screening_task(
        sim_dt=0.02,
        seed=4242,
        challenge_type=2,
        distance_range=(14, 20),
        family_id="cf_swarm_sar",
        n_drones=5,
    )
    assert explicit.num_drones == 5
    assert len(explicit.starts) == 5
    assert len(explicit.goals) == 5

    fallback = screening_task(
        sim_dt=0.02,
        seed=4242,
        challenge_type=2,
        distance_range=(14, 20),
        family_id="cf_swarm_sar",
    )
    expected = random.Random((4242 + SWARM_COUNT_SEED_OFFSET) & 0xFFFFFFFF).randint(
        SWARM_MIN_DRONES,
        SWARM_MAX_DRONES,
    )
    assert fallback.num_drones == expected
