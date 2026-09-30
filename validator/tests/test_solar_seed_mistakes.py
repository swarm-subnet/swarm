"""Cross-seed penalty counts (task 25): every Solar Patrol seed hands the backend its missed threats and false
alarms, including a seed the model stalled out of, while every other family's results stay as they were."""

from __future__ import annotations

from dataclasses import asdict

from swarm.challenge_families import get_challenge_family, list_registered_challenge_families
from swarm.challenge_families.solar_patrol.contract import FAMILY_ID, Outcome
from swarm.domain_model import get_challenge_family_definition
from swarm.validator.utils_parts.evaluation import _seed_mistakes


def test_a_stalled_patrol_keeps_the_counts_it_reached():
    """A patrol cut off by slow-act strikes still reports the threat it left unreported and its false alarms."""
    outcome = Outcome(threats=2, valid_reports=1, missed_threats=1, false_alarms=3, reports_made=4)
    metrics = get_challenge_family(FAMILY_ID).stalled_rollout_metrics(None, {"solar_outcome": asdict(outcome)})
    assert metrics == asdict(outcome)
    assert _seed_mistakes(metrics) == {"missed_threats": 1, "false_alarms": 3}


def test_a_patrol_stalled_before_its_first_step_counts_nothing():
    """With no step taken there is no outcome yet, so the seed carries zero of each."""
    metrics = get_challenge_family(FAMILY_ID).stalled_rollout_metrics(None, {})
    assert _seed_mistakes(metrics) == {"missed_threats": 0, "false_alarms": 0}


def test_every_other_family_keeps_nothing_from_a_stalled_seed():
    """Only Solar Patrol answers for a stalled seed, so the other families' zero results are unchanged."""
    for family_id in list_registered_challenge_families():
        if family_id != FAMILY_ID:
            assert get_challenge_family(family_id).stalled_rollout_metrics(None, {"collision": True}) == {}


def test_the_registry_zeroes_ten_seeds_per_mistake_for_solar_patrol_only():
    """Solar Patrol's registry entry carries the ×10 setting the backend reads; no other family has one."""
    for family_id in list_registered_challenge_families():
        expected = 10 if family_id == FAMILY_ID else None
        assert get_challenge_family_definition(family_id).get("seeds_zeroed_per_mistake") == expected
