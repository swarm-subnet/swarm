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

"""Test Field v2 family: its contract and registry entry, and whole patrols flown on a stand-in area.

The real areas are not in the released swarm-worlds, so the patrols here cut their fields from the stand-in area
package (field_v2_stand_in), built by the real loader; everything above the world build is the family's own code.
"""
from __future__ import annotations

import contextlib
import io
import os

import numpy as np
import pytest
import swarm_description
import swarm_worlds

from swarm.challenge_families import build_benchmark_tasks, get_challenge_family, list_registered_challenge_families
from swarm.challenge_families.solar_patrol import airframe, camera, intruders
from swarm.challenge_families.solar_patrol import contract as sentinel
from swarm.challenge_families.solar_patrol import install as sentinel_install
from swarm.challenge_families.test_field_v2 import install, site
from swarm.challenge_families.test_field_v2 import task as field_task
from swarm.challenge_families.test_field_v2.contract import (
    ACTION_DIM,
    ACTION_FIELDS,
    ACTION_HIGH,
    ACTION_INDEX,
    ACTION_LOW,
    CHALLENGE_TYPE,
    FAMILY_ID,
    HORIZON_S,
    REPORT_REACH_M,
    TIME_BUDGET_S,
    FieldReport,
    decode_action,
)
from swarm.constants import SIM_DT
from swarm.domain_model import (
    CHALLENGE_TYPE_TO_ENVIRONMENT_TYPE,
    get_challenge_family_definition,
    get_family_benchmark_seed_count,
    get_family_seeds_zeroed_per_mistake,
)
from swarm.policy_interface import build_artifact_policy_contract, build_smoke_test_observation
from swarm.utils.env_factory import make_env_with_initial_obs
from validator.tests import field_v2_stand_in

# A patrol flies the M4TD, which ships in swarm-description, and puts Sentinel's bodies on the field, from swarm-worlds.
_SHIPPED = os.path.isfile(swarm_description.get_urdf(airframe.DRONE)) and os.path.isfile(
    os.path.join(swarm_worlds.maps_dir(), intruders.ASSET_DIR, "intruders.json"))


@pytest.fixture
def stand_in_field(tmp_path, monkeypatch):
    """Cut every field from the stand-in area, and dress the people in their everyday colours."""
    monkeypatch.setattr(install, "area_problems", lambda: [])
    field_v2_stand_in.use_area(tmp_path, monkeypatch)
    if _SHIPPED:
        field_v2_stand_in.use_people(tmp_path / "people", monkeypatch)


@pytest.fixture
def blank_camera(monkeypatch):
    """Hand back a black colour frame instead of drawing one: it is most of a flown step and no test here reads it."""
    monkeypatch.setattr(camera, "colour_frame",
                        lambda env, shot, *_night, **_options: (np.zeros((shot.height, shot.width, 3), dtype=np.float32),
                                                                np.full((shot.height, shot.width), -1, dtype=np.int32)))


def _action(**values):
    """An action vector at rest, with the named fields set."""
    a = np.zeros(ACTION_DIM, dtype=np.float32)
    for name, value in values.items():
        a[ACTION_INDEX[name]] = value
    return a


def _patrol(seed, pilot):
    """Fly one patrol on the stand-in with a pilot that maps (decision, observation, episode) to an action."""
    if not _SHIPPED:
        pytest.skip("the installed swarm-description has no M4TD or swarm-worlds no intruder bodies")
    task = build_benchmark_tasks(sim_dt=SIM_DT, seeds=[seed], family_id=FAMILY_ID)[0]
    with contextlib.redirect_stdout(io.StringIO()):
        env, obs = make_env_with_initial_obs(task)
    log = {"first": obs, "info": {}, "peak_above_dock_m": 0.0}
    try:
        for i in range(10 ** 6):
            obs, _r, terminated, truncated, info = env.step(pilot(i, obs, env._solar)[None, :])
            log["info"] = info
            log["peak_above_dock_m"] = max(log["peak_above_dock_m"], float(env.pos[0][2] - env._solar.dock_position[2]))
            if terminated or truncated:
                break
        log["episode"], log["time_s"] = env._solar, env._time_alive
        log["evaluation"] = get_challenge_family(FAMILY_ID).evaluate_rollout(
            task=task, success=bool(info.get("success")), t=env._time_alive, horizon=task.horizon, min_clearance=None,
            collision=False, failure_reason=str(info.get("failure_reason")), info=info)
    finally:
        env.close()
    return log


def test_family_is_registered_open_at_ten_percent_on_its_own_map_type():
    """The family is a task of its own beside the first Swarm Test Field: registered, active and paid 10 % of
    emissions, starting at version 1.0.0 with nothing to restart, with 1,000 seeds an epoch on the testfield map type
    10 and Sentinel's ×5 rule across seeds."""
    assert FAMILY_ID == "cf_test_field_v2" and FAMILY_ID in list_registered_challenge_families()
    definition = get_challenge_family_definition(FAMILY_ID)
    assert definition["family_state"] == "active" and definition["emissions_state"] == "active"
    assert definition["emission_allocation"] == 0.1 and definition["benchmark_version"] == "1.0.0"
    assert "benchmark_version_reset" not in definition
    assert definition["display"] == {"label": "Swarm Test Field v2", "short_label": "Test Field v2",
                                     "slug": "swarm-test-field-v2"}
    assert get_family_benchmark_seed_count(FAMILY_ID) == 1000
    assert get_family_seeds_zeroed_per_mistake(FAMILY_ID) == 5
    assert definition["environment_types"] == ["testfield"] and CHALLENGE_TYPE == 10
    assert CHALLENGE_TYPE_TO_ENVIRONMENT_TYPE[10] == "testfield"
    assert get_challenge_family(FAMILY_ID) is not get_challenge_family("cf_test_field")


def test_the_runtime_is_sentinels_drone_by_day_and_two_nights_in_five():
    """The family decides ten times a second over the shared physics, under a seeded sun or two nights in five on the
    field's own night stream, in the light the areas were matched under."""
    family = get_challenge_family(FAMILY_ID)
    assert family.decision_steps == sentinel.DECISION_STEPS
    assert family.seeded_sun and family.night_share == 0.4 and family.area_light
    assert family.night_stream == site.NIGHT_STREAM
    assert family.prepares_seeds


def test_the_runtime_profile_keeps_a_worker_and_gives_a_patrol_room():
    """A patrol runs in the navigation class under its own profile name, a worker is kept past the field's measured
    2.95 GB, and a seed is given five times the slowest measured patrol."""
    profile = get_challenge_family(FAMILY_ID).runtime_profile(None)
    assert profile.profile_name == "test_field_v2" and profile.resource_class == "navigation"
    assert profile.worker_recycle_rss_mb > 2949.0
    assert profile.global_eval_per_seed_sec >= 5 * 228.0
    assert profile.docker_env["SWARM_CHALLENGE_FAMILY_ID"] == FAMILY_ID


def test_decided_timing():
    """A 200 s patrol for every field, split into take-off, people, sweep and landing, and the task carries it."""
    assert TIME_BUDGET_S == {"take_off": 15.0, "people": 40.0, "sweep": 105.0, "landing": 40.0}
    assert HORIZON_S == 200.0
    assert field_task.test_field_task(seed=5, sim_dt=SIM_DT).horizon == HORIZON_S


def test_published_contract_matches_the_contract_module():
    """The registry's contract keeps Sentinel's observation and first 24 actions and appends the two position values."""
    art = build_artifact_policy_contract(FAMILY_ID, "submission_zip.v1")
    assert {key: tuple(field["shape"]) for key, field in art["observation_space"]["fields"].items()} == sentinel.OBSERVATION_SHAPES
    action = art["action_space"]
    assert action["component_names"] == [f.name for f in ACTION_FIELDS]
    assert action["component_names"][:24] == [f.name for f in sentinel.ACTION_FIELDS]
    assert action["component_names"][24:] == ["report_east", "report_north"]
    assert tuple(action["lower_bound"]) == ACTION_LOW and tuple(action["upper_bound"]) == ACTION_HIGH
    assert ACTION_DIM == 26
    smoke = build_smoke_test_observation(FAMILY_ID, "submission_zip.v1")
    assert {key: value.shape for key, value in smoke.items()} == sentinel.OBSERVATION_SHAPES


def test_a_report_carries_where_the_person_stands_and_the_rest_decodes_as_sentinels():
    """The first 24 values decode exactly as Sentinel's, buttons counting once, and a report carries its position in
    metres from the dock."""
    a = _action(report=1.0, report_cx=0.4, report_cy=0.6, report_w=0.1, report_h=0.2, report_east=0.25,
                report_north=-0.5, take_off=1.0, gimbal_tilt=-1.0)
    command = decode_action(a, None)
    expected = sentinel.decode_action(a[:24], None)
    assert isinstance(command.report, FieldReport)
    assert command.report.box == expected.report.box and command.report.kind == "person"
    assert (command.report.east_m, command.report.north_m) == pytest.approx((0.25 * REPORT_REACH_M, -0.5 * REPORT_REACH_M))
    assert command.take_off == expected.take_off and command.gimbal_tilt == expected.gimbal_tilt
    assert decode_action(a, a).report is None
    assert decode_action(_action(), None).report is None


@pytest.mark.timeout(600)
def test_take_off_fly_return_and_land(stand_in_field, blank_camera):
    """Take-off, a leg out, return home and landing close the patrol as a success in the dock, well inside the edge."""
    def pilot(i, obs, ep):
        """Take off, fly a little north once the dock hands over, then press return home."""
        if i == 0:
            return _action(take_off=1.0)
        if ep.phase == "flying" and pilot.flown < 30:
            pilot.flown += 1
            return _action(move_forward=0.4)
        return _action(return_home=1.0) if ep.phase == "flying" else _action()
    pilot.flown = 0
    log = _patrol(13, pilot)
    outcome = log["episode"].outcome
    assert outcome.end_reason == "landed" and outcome.landed_in_dock and outcome.returned_by_model
    assert outcome.min_edge_m is not None and outcome.min_edge_m > 1.0
    assert log["info"]["success"] is True and log["evaluation"].score > 0.0
    assert log["first"]["site_map"][0] == len(log["episode"].fence)


@pytest.mark.timeout(600)
def test_flying_past_the_edge_ends_the_patrol_and_scores_nothing(stand_in_field, blank_camera):
    """A drone that flies on towards the edge ends the patrol on the flight limit within 1 m of it, and the seed is 0."""
    def pilot(i, obs, ep):
        """Take off, then fly straight on."""
        return _action(take_off=1.0) if i == 0 else _action(move_forward=1.0) if ep.phase == "flying" else _action()
    log = _patrol(14, pilot)
    outcome = log["episode"].outcome
    assert outcome.end_reason == "flight_limit" and outcome.min_edge_m <= 1.0
    assert log["info"]["failure_reason"] == "FLIGHT_LIMIT"
    assert log["evaluation"].score == 0.0


@pytest.mark.timeout(600)
def test_the_drone_never_climbs_past_ten_metres_above_the_dock(stand_in_field, blank_camera):
    """The dock's take-off hands the drone over about 9.5 m above the dock, and a climb stick held to the full stops
    at the 10 m ceiling, past it only by the flight controller's lag, measured at 0.15 m."""
    heights = []

    def pilot(i, obs, ep):
        """Take off, climb at full stick for 15 s once flying, then press return home."""
        if i == 0:
            return _action(take_off=1.0)
        if ep.phase == "flying" and pilot.flying == 0:
            pilot.handed = float(obs["state"][7])
        if ep.phase == "flying" and pilot.flying < 150:
            pilot.flying += 1
            heights.append(float(obs["state"][7]))
            return _action(move_up=1.0)
        return _action(return_home=1.0) if ep.phase == "flying" else _action()
    pilot.flying, pilot.handed = 0, None
    log = _patrol(15, pilot)
    assert log["episode"].outcome.end_reason == "landed"
    assert 8.9 <= pilot.handed <= 9.7, pilot.handed
    assert 9.9 < log["peak_above_dock_m"] <= 10.2, log["peak_above_dock_m"]


def test_an_abandoned_patrol_counts_everyone_as_missed():
    """A patrol the model abandoned misses every person on the field, and an outcome-less one earns nothing."""
    family = get_challenge_family(FAMILY_ID)
    outcome = family.stalled_rollout_metrics(None, {"field_outcome": {"threats": 4, "valid_reports": 2, "false_alarms": 1}})
    assert outcome["missed_threats"] == 4 and outcome["false_alarms"] == 1
    empty = family.evaluate_rollout(task=None, success=False, t=0.0, horizon=HORIZON_S, min_clearance=None,
                                    collision=False, failure_reason="TIMEOUT", info={})
    assert empty.score == 0.0 and empty.metrics["environment_type"] == "testfield"


def test_the_field_is_refused_without_the_aircraft_in_swarm_description(monkeypatch):
    """An install whose swarm-description lacks an M4TD or dock file cannot fly the field, and the file is named."""
    gone = os.path.join(swarm_description.drone_dir(airframe.DRONE), "missing.urdf")
    monkeypatch.setattr(install, "area_problems", lambda: [])
    monkeypatch.setattr(install, "asset_files", lambda: [])
    monkeypatch.setattr(sentinel_install, "description_files", lambda: [gone])
    monkeypatch.setattr(sentinel_install, "ENGINE_FLAGS", ())
    monkeypatch.setattr(install, "ENGINE_FLAGS", ())
    monkeypatch.setattr(sentinel_install, "cpu_features", lambda: set(sentinel_install.CPU_FEATURES))
    assert install.problems() == [
        f"swarm-description {sentinel_install._installed('swarm-description')} is missing {airframe.DRONE}/missing.urdf"]


def test_the_field_is_refused_on_an_engine_that_cannot_be_told_how_the_ground_holds_heat(monkeypatch):
    """Beyond Sentinel's flags the task needs the engine's setting for a passive surface's thermal read, or two
    validators on different engines would draw different thermal ground: an engine without it is named and refused,
    one with it is not."""
    monkeypatch.setattr(install, "area_problems", lambda: [])
    monkeypatch.setattr(install, "asset_files", lambda: [])
    monkeypatch.setattr(sentinel_install, "description_files", lambda: [])
    monkeypatch.setattr(sentinel_install, "ENGINE_FLAGS", ())
    monkeypatch.setattr(sentinel_install, "cpu_features", lambda: set(sentinel_install.CPU_FEATURES))
    assert install.ENGINE_FLAGS == ("THERMAL_FIELD_PASSIVE",)
    monkeypatch.delattr(install.p, "THERMAL_FIELD_PASSIVE", raising=False)
    assert install.problems() == [f"swarm-bullet3 {sentinel_install._installed('swarm-bullet3')} has no THERMAL_FIELD_PASSIVE"]
    monkeypatch.setattr(install.p, "THERMAL_FIELD_PASSIVE", 8, raising=False)
    assert install.problems() == []
