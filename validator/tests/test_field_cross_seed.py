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

"""The Test Field's cross-seed rule: Sentinel's ×5 rule over 1,000 seeds, with tripping the edge rule counted as a
mistake beside every missed person and false alarm, while Sentinel's own count stays missed threats plus false
alarms."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from swarm.benchmark.engine_parts.reporting import _cross_seed_penalty, _print_results
from swarm.challenge_families import (
    family_seed_mistake_fields,
    get_challenge_family,
    list_registered_challenge_families,
)
from swarm.challenge_families.solar_patrol.contract import FAMILY_ID as SENTINEL_ID
from swarm.challenge_families.solar_patrol.contract import Outcome
from swarm.challenge_families.test_field import flight_limit
from swarm.challenge_families.test_field.contract import FAMILY_ID, FieldOutcome
from swarm.challenge_families.test_field.episode import FieldEpisode
from swarm.domain_model import (
    get_challenge_family_definition,
    get_family_benchmark_seed_count,
    get_family_seed_mistake_fields,
    get_family_seeds_zeroed_per_mistake,
)
from swarm.protocol import ValidationResult
from swarm.validator import utils as validator_utils
from swarm.validator.utils_parts import evaluation as validator_evaluation
from swarm.validator.utils_parts.evaluation import _seed_mistakes
from swarm.validator.utils_parts.heartbeat import HeartbeatManager

_OUTLINE = np.array([[-30.0, -30.0], [30.0, -30.0], [30.0, 30.0], [-30.0, 30.0]])


def _flying_episode():
    """A flying episode on a square stand-in outline with its edge rule reset."""
    ep = FieldEpisode(seed=3)
    ep.fence, ep.dock_position, ep.phase = _OUTLINE.copy(), np.array([5.0, -5.0, 2.0]), "flying"
    flight_limit.reset(None, ep)
    return ep


def _at(x):
    """An environment whose drone stands at x on the outline's east-west axis."""
    return SimpleNamespace(pos=np.array([[x, 0.0, 20.0]]))


def test_the_registry_runs_the_field_on_1000_seeds_under_sentinels_rule():
    """The Test Field carries the same two registry fields as Sentinel: 1,000 seeds an epoch, 5 zeroed per mistake."""
    assert get_family_benchmark_seed_count(FAMILY_ID) == 1000
    assert get_family_seeds_zeroed_per_mistake(FAMILY_ID) == get_family_seeds_zeroed_per_mistake(SENTINEL_ID) == 5


def test_tripping_the_edge_counts_one_edge_trip():
    """The patrol the edge rule ends carries edge_trips 1; one that stays clear carries 0."""
    tripped, clear = _flying_episode(), _flying_episode()
    flight_limit.update(_at(29.5), tripped)
    flight_limit.update(_at(20.0), clear)
    assert tripped.outcome.end_reason == "flight_limit" and tripped.outcome.edge_trips == 1
    assert clear.outcome.end_reason == "" and clear.outcome.edge_trips == 0


def test_a_patrol_another_end_closed_first_is_no_edge_trip():
    """A collision that ended the patrol first stays a zero for that seed only, even if the drone then nears the edge."""
    ep = _flying_episode()
    ep.end("collision")
    flight_limit.update(_at(29.5), ep)
    assert ep.outcome.end_reason == "collision" and ep.outcome.edge_trips == 0


def test_the_fields_mistake_counts_come_from_its_registry_entry():
    """The Test Field's registry entry names edge_trips beside the default pair, and its runtime reads that list."""
    listed = get_challenge_family_definition(FAMILY_ID)["seed_mistake_fields"]
    assert listed == ["missed_threats", "false_alarms", "edge_trips"]
    assert get_challenge_family(FAMILY_ID).seed_mistake_fields() == tuple(listed)
    assert family_seed_mistake_fields(FAMILY_ID) == tuple(listed)


def test_a_family_without_the_key_counts_missed_threats_and_false_alarms():
    """Sentinel and every other family but Swarm Test Field v2 carry no list and keep the default pair, as does an
    unknown id."""
    for family_id in list_registered_challenge_families():
        if family_id not in (FAMILY_ID, "cf_test_field_v2"):
            assert "seed_mistake_fields" not in get_challenge_family_definition(family_id)
            assert family_seed_mistake_fields(family_id) == ("missed_threats", "false_alarms")
    assert family_seed_mistake_fields(None) == family_seed_mistake_fields("cf_unknown") == ("missed_threats", "false_alarms")


@pytest.mark.parametrize("value", [[], "edge_trips", ["missed_threats", "missed_threats"], ["missed_threats", ""], [3]])
def test_a_malformed_list_in_the_registry_is_refused(value):
    """The list must be distinct, non-empty names; anything else stops the reader rather than counting wrongly."""
    registry = {"challenge_families": {"cf_fixture": {"family_id": "cf_fixture", "seed_mistake_fields": value}}}
    with pytest.raises(ValueError, match="invalid_family_seed_mistake_fields"):
        get_family_seed_mistake_fields("cf_fixture", registry=registry)


def test_the_field_uploads_its_edge_trip_and_sentinel_its_two_counts_only():
    """A Test Field seed sends edge_trips beside its misses and false alarms; Sentinel's metrics send the pair alone,
    whatever else they carry."""
    field = get_challenge_family(FAMILY_ID).evaluate_rollout(
        task=None, success=False, t=40.0, horizon=180.0, min_clearance=None, collision=False,
        failure_reason="FLIGHT_LIMIT", info={"field_outcome": asdict(FieldOutcome(
            threats=2, valid_reports=1, missed_threats=1, false_alarms=1, end_reason="flight_limit", edge_trips=1))})
    assert field.metrics["edge_trips"] == 1
    assert _seed_mistakes(field.metrics, FAMILY_ID) == {"missed_threats": 1, "false_alarms": 1, "edge_trips": 1}
    sentinel = dict(asdict(Outcome(threats=2, missed_threats=1, false_alarms=1)), edge_trips=1)
    assert _seed_mistakes(sentinel, SENTINEL_ID) == {"missed_threats": 1, "false_alarms": 1}


def test_an_abandoned_field_patrol_misses_everyone_and_keeps_its_edge_trip():
    """A patrol the model abandons counts every person as missed and keeps whether the edge rule had ended it."""
    reached = {"field_outcome": asdict(FieldOutcome(threats=3, valid_reports=2, missed_threats=1, edge_trips=1))}
    metrics = get_challenge_family(FAMILY_ID).stalled_rollout_metrics(None, reached)
    assert _seed_mistakes(metrics, FAMILY_ID) == {"missed_threats": 3, "false_alarms": 0, "edge_trips": 1}
    assert _seed_mistakes(get_challenge_family(FAMILY_ID).stalled_rollout_metrics(None, {}), FAMILY_ID) == {
        "missed_threats": 0, "false_alarms": 0, "edge_trips": 0}


def test_an_edge_trip_zeroes_five_seeds_like_a_miss():
    """One edge trip over twelve clean seeds zeroes five of the best; the same rows under Sentinel's fields zero none."""
    rows = [{"score": 0.9, "missed_threats": 0, "false_alarms": 0, "edge_trips": 0} for _ in range(11)]
    rows.append({"score": 0.0, "missed_threats": 0, "false_alarms": 0, "edge_trips": 1})
    field = _cross_seed_penalty(rows, 5, family_seed_mistake_fields(FAMILY_ID))
    assert field["edge_trips"] == 1 and field["seeds_zeroed"] == 5
    assert field["penalised_score"] == pytest.approx(0.9 * 6 / 12)
    sentinel = _cross_seed_penalty(rows, 5, family_seed_mistake_fields(SENTINEL_ID))
    assert "edge_trips" not in sentinel and sentinel["seeds_zeroed"] == 0


def test_the_benchmark_summary_shows_the_fields_edge_trips(capsys):
    """A Test Field run gains an EdgeTrips column and line, and its trip zeroes five of the best seeds."""
    counts = [(0, 0, 0)] * 11 + [(0, 0, 1)]
    task_meta = [{"group": "type9_testsite", "seed": 100 + i, "challenge_type": 9} for i in range(12)]
    results = [ValidationResult(0, True, 120.0, 0.0 if trip else 0.9,
                                metrics={"missed_threats": m, "false_alarms": f, "edge_trips": trip})
               for m, f, trip in counts]
    keys = {(meta["seed"], 9) for meta in task_meta}
    record = _print_results(task_meta, results, [], {key: deque([60.0]) for key in keys},
                            {key: deque(["seed_done"]) for key in keys}, {key: 60.0 for key in keys}, [],
                            elapsed=60.0, eval_start=0.0, num_workers=1, family_id=FAMILY_ID)
    out = capsys.readouterr().out
    assert "Missed FalseAlarms EdgeTrips" in out
    assert "every missed threat, false alarm or edge trip sets 5 of the best seeds to 0" in out
    assert "Edge trips:                1" in out and "Seeds set to 0:            5/12" in out
    assert record["cross_seed_penalty"]["edge_trips"] == 1


def test_the_streaming_upload_carries_the_fields_edge_trips(monkeypatch, tmp_path):
    """Rows rebuilt from a Test Field run's details carry each seed's edge trip for the backend."""
    uploads: list[dict] = []
    model = tmp_path / "model.zip"
    model.write_bytes(b"field cross-seed fixture")

    async def _capture_upload(**kwargs):
        """Collect every uploaded row and acknowledge the batch."""
        uploads.extend(kwargs["scores"])
        return {"recorded": True}

    async def _acknowledge(**_kwargs):
        """Acknowledge a heartbeat."""
        return {"ok": True}

    async def _evaluate(_self, _uid, _model_path, seeds, *args, **kwargs):
        """Score every seed without the live callback, seed 1 tripping the edge rule."""
        details = [{"score": 0.0 if i == 1 else 0.8, "map_type": "testsite", "failure_reason": "NONE",
                    "metrics": {"missed_threats": 0, "false_alarms": 0, "edge_trips": int(i == 1)}}
                   for i, _seed in enumerate(seeds)]
        return [d["score"] for d in details], {"testsite": [d["score"] for d in details]}, details

    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _evaluate)
    validator = SimpleNamespace(
        backend_api=SimpleNamespace(post_heartbeat=_acknowledge, post_seed_scores_batch=_capture_upload),
        docker_evaluator=SimpleNamespace(_get_image_hash_label=lambda: "image", _calculate_docker_hash=lambda: "image"),
    )

    async def _run():
        """Stream three Test Field seeds in one chunk."""
        hb = HeartbeatManager(validator.backend_api, asyncio.get_event_loop())
        try:
            return await validator_evaluation._run_streaming_phase(
                validator, uid=7, model_path=model, seeds=list(range(3)), phase_description="benchmark",
                family_id=FAMILY_ID, seed_offset=0, epoch_number=1, hb=hb, chunk_size=3)
        finally:
            hb.finish()

    asyncio.run(_run())
    assert [row["edge_trips"] for row in uploads] == [0, 1, 0]
    assert all(row["missed_threats"] == 0 and row["false_alarms"] == 0 for row in uploads)
