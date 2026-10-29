"""Cross-seed penalty counts (tasks 25 and 56): every Solar Patrol seed hands the backend its missed threats and false
alarms, a seed the model abandoned counting its whole crew as missed, while every other family's results stay as they
were."""

from __future__ import annotations

from dataclasses import asdict

from swarm.challenge_families import get_challenge_family, list_registered_challenge_families
from swarm.challenge_families.solar_patrol.contract import FAMILY_ID, Outcome
from swarm.domain_model import get_challenge_family_definition
from swarm.validator.utils_parts.evaluation import _seed_mistakes


def test_an_abandoned_patrol_misses_every_intruder_even_the_reported_ones():
    """A patrol the model abandons answers for its whole crew, the man it reported included, and keeps its false
    alarms; everything else it reached stands."""
    outcome = Outcome(threats=2, valid_reports=1, missed_threats=1, false_alarms=3, reports_made=4)
    metrics = get_challenge_family(FAMILY_ID).stalled_rollout_metrics(None, {"solar_outcome": asdict(outcome)})
    assert metrics == dict(asdict(outcome), missed_threats=2)
    assert _seed_mistakes(metrics) == {"missed_threats": 2, "false_alarms": 3}


def test_a_patrol_abandoned_before_its_first_step_misses_the_whole_crew():
    """At reset the crew is counted but no step has refreshed the misses yet; abandoning then misses every man."""
    outcome = Outcome(threats=3)
    metrics = get_challenge_family(FAMILY_ID).stalled_rollout_metrics(None, {"solar_outcome": asdict(outcome)})
    assert _seed_mistakes(metrics) == {"missed_threats": 3, "false_alarms": 0}


def test_an_abandoned_seed_without_intruders_misses_nothing():
    """A seed with no theft has nobody to miss, whenever the model leaves it."""
    family = get_challenge_family(FAMILY_ID)
    assert _seed_mistakes(family.stalled_rollout_metrics(None, {})) == {"missed_threats": 0, "false_alarms": 0}
    reached = {"solar_outcome": asdict(Outcome(false_alarms=1, reports_made=1))}
    assert _seed_mistakes(family.stalled_rollout_metrics(None, reached)) == {"missed_threats": 0, "false_alarms": 1}


def test_every_other_family_keeps_nothing_from_a_stalled_seed():
    """Only Solar Patrol answers for a stalled seed, so the other families' zero results are unchanged."""
    for family_id in list_registered_challenge_families():
        if family_id != FAMILY_ID:
            assert get_challenge_family(family_id).stalled_rollout_metrics(None, {"collision": True}) == {}


def test_the_registry_zeroes_five_seeds_per_mistake_for_solar_patrol_only():
    """Solar Patrol's registry entry carries the ×5 setting the backend reads; no other family has one."""
    for family_id in list_registered_challenge_families():
        expected = 5 if family_id == FAMILY_ID else None
        assert get_challenge_family_definition(family_id).get("seeds_zeroed_per_mistake") == expected
