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

"""The Solar Patrol skeleton: its contract, and whole patrols flown end to end on a flat stand-in park.

The real park's assets are too large for this repository, so the patrols here stand the family on flat ground
with a square fence; everything above the world build is the family's own code.
"""
from __future__ import annotations

import contextlib
import io
import os
import zipfile

import numpy as np
import pybullet as p
import pytest
import swarm_worlds

from swarm.challenge_families import (
    build_benchmark_tasks,
    evaluate_rollout,
    get_challenge_family,
    list_registered_challenge_families,
)
from swarm.challenge_families.solar_patrol import airframe, drone_state, park
from swarm.challenge_families.solar_patrol.contract import (
    ACTION_DIM,
    ACTION_FIELDS,
    ACTION_HIGH,
    ACTION_INDEX,
    ACTION_LOW,
    DECISION_STEPS,
    END_REASONS,
    FAMILY_ID,
    HORIZON_S,
    OBSERVATION_SHAPES,
    RESULT_METRICS,
    SCORE_TERMS,
    SITE_MAP_FIELDS,
    SITE_MAP_SLICES,
    STATE_FIELDS,
    STATE_SLICES,
    TIME_BUDGET_S,
    decode_action,
)
from swarm.constants import SIM_DT
from swarm.domain_model import CHALLENGE_TYPE_TO_ENVIRONMENT_TYPE, get_challenge_family_definition
from swarm.policy_interface import (
    POLICY_CONTRACT_FILENAME,
    build_artifact_policy_contract,
    build_smoke_test_observation,
    render_artifact_policy_contract,
    smoke_test_policy_package,
)
from swarm.utils.env_factory import make_env_with_initial_obs

# A patrol flies the M4TD, which ships in swarm-worlds; an installed release without it cannot fly one.
_M4TD_SHIPPED = os.path.isfile(os.path.join(swarm_worlds.robots_dir(), airframe.URDF))
_FENCE = np.array([[0.0, 40.0], [120.0, 40.0], [120.0, 160.0], [0.0, 160.0]])
_TABLES = np.array([[60.0, 120.0, 20.0, 4.0, 89.0], [40.0, 70.0, 10.0, 4.0, 88.0]])
_BUILDINGS = np.array([[20.0, 50.0, 3.8, 3.7, 0.0]])


class _StillMovers:
    """Movers for a park that has none."""

    body_uids = frozenset()

    def advance(self, step=None):
        """Nothing moves."""


@pytest.fixture
def flat_park(monkeypatch):
    """Build the patrol's world as flat ground with a square fence around the stand-in dock spot."""
    def build(seed=0, cli=0, asset_dir=None, groups=None):
        """A flat slab standing in for the park's terrain."""
        shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[200.0, 200.0, 0.5], physicsClientId=cli)
        ground = p.createMultiBody(0, shape, -1, [60.0, 100.0, -0.5], physicsClientId=cli)
        return {"bodies": {"terrain": [ground]}, "movers": [], "asset_dir": "flat"}

    monkeypatch.setattr(park, "build_solar_map", build)
    monkeypatch.setattr(park, "build_solar_movers", lambda world, seed=0, cli=0: _StillMovers())
    monkeypatch.setattr(park, "fence_line", lambda asset_dir: _FENCE)
    monkeypatch.setattr(drone_state, "survey", lambda asset_dir: (_TABLES, _BUILDINGS))


def _action(**values):
    """An action vector at rest, with the named fields set."""
    a = np.zeros(ACTION_DIM, dtype=np.float32)
    for name, value in values.items():
        a[ACTION_INDEX[name]] = value
    return a


def _patrol(seed, pilot, max_decisions=None):
    """Fly one patrol with a pilot that maps (decision index, observation) to an action; returns the log."""
    if not _M4TD_SHIPPED:
        pytest.skip(f"the installed swarm-worlds has no {airframe.URDF} yet")
    task = build_benchmark_tasks(sim_dt=SIM_DT, seeds=[seed], family_id=FAMILY_ID)[0]
    with contextlib.redirect_stdout(io.StringIO()):
        env, obs = make_env_with_initial_obs(task)
    log = {"observations": [obs], "info": {}, "decisions": 0}
    try:
        for i in range(max_decisions or 10 ** 6):
            obs, _r, terminated, truncated, info = env.step(pilot(i, obs)[None, :])
            log["observations"].append(obs)
            log["decisions"] = i + 1
            log["info"] = info
            if terminated or truncated:
                break
        log["time_s"] = env._time_alive
        log["episode"] = env._solar
    finally:
        env.close()
    return log


def test_family_is_registered_incubating_on_its_own_map_type():
    """The family is registered, incubating with no emissions, on the solar map type 8."""
    assert FAMILY_ID in list_registered_challenge_families()
    definition = get_challenge_family_definition(FAMILY_ID)
    assert definition["family_state"] == "incubating"
    assert definition["emission_allocation"] == 0.0
    assert definition["environment_types"] == ["solar"]
    assert CHALLENGE_TYPE_TO_ENVIRONMENT_TYPE[8] == "solar"


def test_every_other_family_still_steps_once_per_decision():
    """Only Solar Patrol holds a decision over several control steps, so no other family's flight changes."""
    for family_id in list_registered_challenge_families():
        expected = DECISION_STEPS if family_id == FAMILY_ID else 1
        assert get_challenge_family(family_id).decision_steps == expected


def test_decided_timing():
    """Ten decisions a second over the shared 50 Hz physics, inside a 390 s patrol budget."""
    assert DECISION_STEPS * SIM_DT == pytest.approx(0.1)
    assert TIME_BUDGET_S == {"take_off": 40.0, "sweep": 248.0, "zoom_stops": 62.0, "landing": 40.0}
    assert HORIZON_S == 390.0


def test_published_contract_matches_the_contract_module():
    """The registry's contract carries exactly the shapes, fields and bounds the contract module declares."""
    art = build_artifact_policy_contract(FAMILY_ID, "submission_zip.v1")
    fields = art["observation_space"]["fields"]
    assert set(fields) == set(OBSERVATION_SHAPES)
    for key, shape in OBSERVATION_SHAPES.items():
        assert tuple(fields[key]["shape"]) == tuple(shape)
    assert fields["state"]["semantic_channels"] == [name for name, _ in STATE_FIELDS]
    assert fields["site_map"]["semantic_channels"] == [name for name, _ in SITE_MAP_FIELDS]
    action = art["action_space"]
    assert action["component_names"] == [f.name for f in ACTION_FIELDS]
    assert tuple(action["lower_bound"]) == ACTION_LOW
    assert tuple(action["upper_bound"]) == ACTION_HIGH
    smoke = build_smoke_test_observation(FAMILY_ID, "submission_zip.v1")
    assert {key: value.shape for key, value in smoke.items()} == OBSERVATION_SHAPES


def test_buttons_count_once_while_held():
    """A button pressed and held fires on the first step only, and fires again after a release."""
    press = _action(zoom=1.0, report=1.0, take_off=1.0, return_home=1.0, cancel_return=1.0)
    first = decode_action(press, None)
    held = decode_action(press, press)
    again = decode_action(press, _action())
    assert first.zoom and first.report and first.take_off and first.return_home and first.cancel_return
    assert not (held.zoom or held.report or held.take_off or held.return_home or held.cancel_return)
    assert again.take_off


def test_choices_decode_from_their_ranges():
    """Lens, class, image and night mode are read from the value's side of the threshold or its third."""
    command = decode_action(_action(zoom=1.0, zoom_lens=0.9, report=1.0, report_class=0.9, report_image=0.1,
                                    night_mode=0.5, thermal=0.7, night_vision=0.2), None)
    assert command.zoom.lens == 7
    assert command.report.kind == "vehicle"
    assert command.report.image == "feed"
    assert command.night_mode == "on"
    assert command.thermal and not command.night_vision
    assert decode_action(_action(night_mode=1.0), None).night_mode == "auto"


def test_a_random_action_controller_passes_the_package_smoke_test(tmp_path):
    """A submission sending random actions in the contract's bounds clears the one-step package check."""
    zip_path = tmp_path / "submission.zip"
    source = "\n".join([
        "import numpy as np",
        "",
        "class DroneFlightController:",
        "    def reset(self):",
        "        self.rng = np.random.default_rng(0)",
        "",
        "    def act(self, observation):",
        f"        return self.rng.uniform({list(ACTION_LOW)}, {list(ACTION_HIGH)}).astype(np.float32)",
        "",
    ])
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("drone_agent.py", source)
        zf.writestr(POLICY_CONTRACT_FILENAME, render_artifact_policy_contract(FAMILY_ID, "submission_zip.v1"))
    assert smoke_test_policy_package(zip_path) == (True, "ok")


@pytest.mark.timeout(300)
def test_random_actions_fly_a_whole_patrol(flat_park):
    """Random actions run a patrol to one of its end reasons, every observation in the contract's shapes."""
    rng = np.random.default_rng(7)
    log = _patrol(11, lambda i, obs: rng.uniform(ACTION_LOW, ACTION_HIGH).astype(np.float32))
    assert log["episode"].outcome.end_reason in END_REASONS
    assert log["time_s"] <= HORIZON_S + 1e-6
    for obs in log["observations"]:
        assert {key: value.shape for key, value in obs.items()} == OBSERVATION_SHAPES
        assert all(value.dtype == np.float32 for value in obs.values())
    assert log["observations"][0]["site_map"][SITE_MAP_SLICES["fence_count"]][0] == len(_FENCE)
    assert all(not obs["site_map"].any() for obs in log["observations"][1:])


@pytest.mark.timeout(300)
def test_a_patrol_that_never_takes_off_runs_out_of_time(flat_park):
    """A drone left in its dock ends the patrol on the clock at 390 s, one decision every 0.1 s."""
    log = _patrol(12, lambda i, obs: _action())
    episode = log["episode"]
    assert episode.outcome.end_reason == "timeout"
    assert not episode.outcome.took_off
    assert log["decisions"] == pytest.approx(HORIZON_S / 0.1, abs=1)
    assert log["info"]["failure_reason"] == "TIMEOUT"


def _home_and_back(outbound_decisions):
    """A pilot that takes off, flies forward for a while, then asks the dock to bring it home."""
    def pilot(i, obs):
        """Take off at once, fly out once the dock hands over, then press return home."""
        phase = int(obs["state"][STATE_SLICES["flight_phase"]][0])
        if i == 0:
            return _action(take_off=1.0)
        if phase == 2 and pilot.flown < outbound_decisions:
            pilot.flown += 1
            return _action(move_forward=0.6)
        if phase == 2:
            return _action(return_home=1.0)
        return _action()
    pilot.flown = 0
    return pilot


@pytest.mark.timeout(300)
def test_take_off_fly_return_and_land(flat_park):
    """Take-off, a leg out, return home and landing close the patrol as a success in the dock."""
    log = _patrol(13, _home_and_back(outbound_decisions=40))
    outcome = log["episode"].outcome
    assert outcome.end_reason == "landed"
    assert outcome.took_off and outcome.returned_by_model and outcome.landed_in_dock
    assert outcome.max_height_m == pytest.approx(20.0, abs=1.5)
    assert log["info"]["success"] is True
    assert log["time_s"] < HORIZON_S


@pytest.mark.timeout(300)
def test_the_state_counts_from_the_dock_through_a_whole_patrol(flat_park):
    """In the dock the drone reads 0, 0, 0 with 390 s and 95 % left; flying forward from its east-facing dock it
    heads 90 and moves east at the speed it reports, 20 m above take-off; the clock loses 0.1 s a decision up to
    the landing, the battery only ever drains, and the site map places the survey from the dock."""
    log = _patrol(16, _home_and_back(outbound_decisions=40))
    ep = log["episode"]
    states = np.array([obs["state"] for obs in log["observations"]])
    position = states[:, STATE_SLICES["position_m"]]
    velocity = states[:, STATE_SLICES["velocity_mps"]]
    battery = states[:, STATE_SLICES["battery_pct"]][:, 0]
    assert position[0] == pytest.approx([0.0, 0.0, 0.0], abs=0.05)
    assert states[0, STATE_SLICES["heading_deg"]][0] == pytest.approx(90.0, abs=0.5)
    time_left = states[:, STATE_SLICES["time_left_s"]][:, 0]
    assert time_left[:-1] == pytest.approx(HORIZON_S - 0.1 * np.arange(len(states) - 1), abs=1e-3)
    # The landing closes the patrol on the control step it happens, part way through the last decision.
    assert time_left[-2] - 0.1 - 1e-3 <= time_left[-1] < time_left[-2]
    assert states[:, STATE_SLICES["height_above_takeoff_m"]][:, 0] == pytest.approx(position[:, 2])
    assert position[:, 2].max() == pytest.approx(20.0, abs=1.5)
    assert position[:, 0].max() > 5.0 and np.abs(position[:, 1]).max() < 1.0
    assert velocity[:, 0].max() == pytest.approx(3.0, abs=0.5)
    assert np.median(np.abs(np.diff(position, axis=0) / 0.1 - velocity[1:])) < 0.2
    assert battery[0] == 95.0 and battery[-1] < 95.0 and np.all(np.diff(battery) <= 0.0)
    assert 95.0 - ep.drone_state["battery_pct"] <= 100.0 * log["time_s"] / (47 * 60)
    site = log["observations"][0]["site_map"]
    shift = np.array([ep.dock_position[0], ep.dock_position[1], 0.0, 0.0, 0.0])
    assert site[SITE_MAP_SLICES["table_count"]][0] == len(_TABLES)
    assert site[SITE_MAP_SLICES["tables"]].reshape(-1, 5)[:len(_TABLES)] == pytest.approx(_TABLES - shift, abs=1e-4)
    assert site[SITE_MAP_SLICES["building_count"]][0] == len(_BUILDINGS)
    assert site[SITE_MAP_SLICES["buildings"]].reshape(-1, 5)[:1] == pytest.approx(_BUILDINGS - shift, abs=1e-4)


@pytest.mark.timeout(300)
def test_the_stop_line_ends_the_patrol(flat_park):
    """A drone flown straight at the fence ends the patrol at the flight limit's stop line."""
    def pilot(i, obs):
        """Take off, then fly one way until the patrol ends."""
        return _action(take_off=1.0) if i == 0 else _action(move_forward=1.0)
    log = _patrol(14, pilot)
    assert log["episode"].outcome.end_reason == "flight_limit"
    assert log["info"]["success"] is False


@pytest.mark.timeout(300)
def test_flying_into_the_ground_is_a_collision(flat_park):
    """Descending into the ground after take-off ends the patrol as a collision."""
    def pilot(i, obs):
        """Take off, then descend at full rate once the dock hands over."""
        phase = int(obs["state"][STATE_SLICES["flight_phase"]][0])
        if i == 0:
            return _action(take_off=1.0)
        return _action(move_forward=0.5, move_up=-1.0) if phase == 2 else _action()
    log = _patrol(15, pilot)
    assert log["episode"].outcome.end_reason == "collision"
    assert log["info"]["failure_reason"] == "OBSTACLE_COLLISION"


@pytest.mark.timeout(300)
def test_reports_do_not_end_the_patrol_and_reach_the_result(flat_park):
    """Reports and zooms are counted into the outcome, the patrol goes on, and the seed result carries them."""
    def pilot(i, obs):
        """Take off, then press report and zoom on every other decision."""
        if i == 0:
            return _action(take_off=1.0)
        return _action(report=float(i % 2), zoom=float(i % 2))
    log = _patrol(16, pilot, max_decisions=200)
    outcome = log["episode"].outcome
    assert outcome.end_reason == ""
    assert outcome.reports_made == 100 and outcome.false_alarms == 100
    assert outcome.zooms_used == 80
    evaluation = evaluate_rollout(task=build_benchmark_tasks(sim_dt=SIM_DT, seeds=[16], family_id=FAMILY_ID)[0],
                                  success=False, t=log["time_s"], horizon=HORIZON_S, min_clearance=None,
                                  collision=False, failure_reason="NONE", info=log["info"])
    assert set(RESULT_METRICS) <= set(evaluation.metrics)
    assert evaluation.metrics["false_alarms"] == 100
    assert set(evaluation.normalized_metrics) == set(SCORE_TERMS)
