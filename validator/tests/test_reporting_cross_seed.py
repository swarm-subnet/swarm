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

"""The local benchmark shows each seed's missed threats and false alarms, and the score the cross-seed rule leaves."""
from collections import deque

import pytest

from swarm.benchmark.engine_parts.reporting import _print_results, _zero_best_seeds
from swarm.domain_model import get_family_seeds_zeroed_per_mistake
from swarm.protocol import ValidationResult


def _run(rows, group, challenge_type, family_id):
    """Print a finished run of (seed, score, metrics) rows in one group and return its JSON record."""
    task_meta = [{"group": group, "seed": seed, "challenge_type": challenge_type} for seed, _, _ in rows]
    results = [ValidationResult(0, True, 200.0, score, metrics=metrics) for _, score, metrics in rows]
    keys = {(seed, challenge_type) for seed, _, _ in rows}
    return _print_results(
        task_meta,
        results,
        [],
        {key: deque([60.0]) for key in keys},
        {key: deque(["seed_done"]) for key in keys},
        {key: 60.0 for key in keys},
        [],
        elapsed=60.0,
        eval_start=0.0,
        num_workers=1,
        family_id=family_id,
    )


def _patrol(seed, score, missed, false_alarms):
    """One Swarm Sentinel seed row with its two mistake counts."""
    return seed, score, {"missed_threats": missed, "false_alarms": false_alarms}


def test_registry_carries_the_rule_for_swarm_sentinel_only():
    """Swarm Sentinel zeroes 10 seeds per mistake and a family without the rule reads 0."""
    assert get_family_seeds_zeroed_per_mistake("cf_solar_patrol") == 10
    assert get_family_seeds_zeroed_per_mistake("cf_search_and_rescue") == 0


def test_zeroing_takes_the_best_seeds_first_and_ties_go_to_the_earliest():
    """The best seeds go to 0 first, an equal score is taken from the earlier seed, and the count never passes the run."""
    assert _zero_best_seeds([0.5, 0.9, 0.9, 0.1], 1, 2) == [0.5, 0.0, 0.0, 0.1]
    assert _zero_best_seeds([0.9, 0.9, 0.5], 1, 1) == [0.0, 0.9, 0.5]
    assert _zero_best_seeds([0.9, 0.8], 3, 10) == [0.0, 0.0]
    assert _zero_best_seeds([0.9, 0.8], 0, 10) == [0.9, 0.8]


def test_sentinel_run_prints_each_seeds_mistakes_and_the_score_after_the_rule(capsys):
    """Eleven clean seeds and one missed thief: the table shows the miss and ten of the best seeds go to 0."""
    rows = [_patrol(1000 + i, 0.95, 0, 0) for i in range(11)] + [_patrol(2000, 0.28, 1, 0)]
    record = _run(rows, "type8_solar", 8, "cf_solar_patrol")
    out = capsys.readouterr().out

    assert "Missed" in out and "FalseAlarms" in out
    seed_line = next(line for line in out.splitlines() if " 2000 " in line)
    assert seed_line.split()[-2:] == ["1", "0"]
    assert "Seeds set to 0:            10/12" in out

    penalty = record["cross_seed_penalty"]
    assert penalty["missed_threats"] == 1 and penalty["false_alarms"] == 0
    assert penalty["mean_score"] == pytest.approx((0.95 * 11 + 0.28) / 12)
    assert penalty["penalised_score"] == pytest.approx((0.95 + 0.28) / 12)
    assert f"Score after the rule:      {penalty['penalised_score']:.4f}" in out
    assert record["group_results"]["type8_solar"][-1]["missed_threats"] == 1


def test_false_alarms_count_like_misses_and_the_score_floors_at_zero(capsys):
    """Two false alarms over three seeds set every seed to 0, never below."""
    rows = [_patrol(1, 0.9, 0, 2), _patrol(2, 0.8, 0, 0), _patrol(3, 0.7, 0, 0)]
    record = _run(rows, "type8_solar", 8, "cf_solar_patrol")
    assert record["cross_seed_penalty"]["seeds_zeroed"] == 3
    assert record["cross_seed_penalty"]["penalised_score"] == 0.0


def test_family_without_mistake_counts_prints_the_table_unchanged(capsys):
    """A Search and Rescue run gains no columns and no rule block."""
    record = _run([(7, 0.6, {}), (8, 0.4, {})], "type1_city", 1, "cf_search_and_rescue")
    out = capsys.readouterr().out
    assert "Missed" not in out and "Cross-seed rule" not in out
    assert record["cross_seed_penalty"] is None


def test_a_seed_without_counts_shows_a_dash_and_adds_no_mistake(capsys):
    """A seed whose flight failed before scoring carries no counts: a dash in the table and nothing added to the rule."""
    rows = [_patrol(1, 0.9, 0, 0), (2, 0.0, {})]
    record = _run(rows, "type8_solar", 8, "cf_solar_patrol")
    out = capsys.readouterr().out
    seed_line = next(line for line in out.splitlines() if line.split()[:2] == ["type8_solar", "2"])
    assert seed_line.split()[-2:] == ["-", "-"]
    assert record["cross_seed_penalty"]["seeds_zeroed"] == 0
