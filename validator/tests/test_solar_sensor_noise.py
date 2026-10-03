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

"""Sensor noise and delay: the RTK drift, the link's delays and the live stream's look, each against DJI's figures."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from swarm.challenge_families.solar_patrol import camera, sensor_noise, zoom
from swarm.challenge_families.solar_patrol.contract import (
    DECISION_STEPS,
    HORIZON_S,
    STATE_SLICES,
    decode_action,
    new_state,
)
from swarm.challenge_families.solar_patrol.episode import SolarEpisode
from swarm.challenge_families.solar_patrol.family import SolarPatrolChallengeFamily
from swarm.constants import SIM_DT
from validator.tests.test_solar_patrol_family import (
    _action,
    _patrol,
    blank_camera,  # noqa: F401
)
from validator.tests.test_solar_patrol_family import flat_park as _flat_park  # noqa: F401

_STEPS = int(round(HORIZON_S / SIM_DT)) + 1
# The stream look of _frame(), pinned so a validator on any machine that drew other bytes fails here.
_STREAM_SHA256 = "d02fe9460af74d5061cb6f586a608c41787672f86ecd1cde7121d627bffc1ed1"


def _episode(seed=0):
    """A patrol with only the sensor noise part reset."""
    ep = SolarEpisode(seed=seed)
    sensor_noise.reset(SimpleNamespace(), ep)
    return ep


def _frame(seed=3):
    """A colour frame with smooth shading, hard edges and fine grain, like a daylight render of the park."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:480, 0:640].astype(np.float32)
    frame = np.stack([xx / 640.0, yy / 480.0, 0.5 + 0.4 * np.sign(np.sin(xx / 9.0) * np.cos(yy / 13.0))], axis=-1)
    frame += rng.normal(0.0, 0.02, frame.shape).astype(np.float32)
    return np.clip(frame, 0.0, 1.0).astype(np.float32)


def _view(step=0, eye=(0.0, 0.0, 20.0)):
    """A colour frame's view, looking north and level."""
    return camera.View(feed="colour", eye=eye, forward=(0.0, 1.0, 0.0), up=(0.0, 0.0, 1.0), width=640, height=480,
                       vertical_fov_deg=55.1, sees=True, step=step)


def test_rtk_drift_stays_inside_dji_figures_and_spans_them():
    """Over 40 whole patrols the reading is never further off than 0.1 m across or up, and reaches near it."""
    drift = np.concatenate([sensor_noise.rtk_drift(seed, _STEPS) for seed in range(40)])
    across = np.hypot(drift[:, 0], drift[:, 1])
    assert across.max() <= sensor_noise.RTK_HORIZONTAL_M + 1e-12
    assert np.abs(drift[:, 2]).max() <= sensor_noise.RTK_VERTICAL_M
    assert 0.08 <= np.quantile(across, 0.95) <= 0.1
    assert 0.08 <= np.quantile(np.abs(drift[:, 2]), 0.95) <= 0.1


def test_rtk_drift_wanders_slowly():
    """The error moves under 1 cm a control step and a second later is still nearly the same error."""
    drift = sensor_noise.rtk_drift(5, _STEPS)
    assert np.abs(np.diff(drift, axis=0)).max() < 0.01
    lag = int(round(1.0 / SIM_DT))
    for axis in range(3):
        assert np.corrcoef(drift[:-lag, axis], drift[lag:, axis])[0, 1] > 0.9


def test_every_error_repeats_for_its_seed_and_changes_with_it():
    """The same seed draws the same drift and delays; another seed draws others."""
    a, b, c = _episode(9), _episode(9), _episode(10)
    assert np.array_equal(a.sensor_noise["drift"], b.sensor_noise["drift"])
    assert np.array_equal(a.sensor_noise["data_steps"], b.sensor_noise["data_steps"])
    assert not np.array_equal(a.sensor_noise["drift"], c.sensor_noise["drift"])
    assert not np.array_equal(a.sensor_noise["data_steps"], c.sensor_noise["data_steps"])


def test_what_the_model_sees_is_60_or_80_ms_old():
    """Every decision's data is 60 or 80 ms old, the two about equally often, as DJI's delay report gives."""
    delays = np.concatenate([_episode(seed).sensor_noise["data_steps"] for seed in range(20)]) * SIM_DT * 1000.0
    assert set(np.round(delays).astype(int)) == {60, 80}
    assert 0.45 < np.mean(np.isclose(delays, 60.0)) < 0.55


def test_one_snapshot_is_taken_before_each_decision_at_its_delay():
    """Between two decisions exactly one control step is snapshotted, the delay's number of steps before the next."""
    ep = _episode(4)
    for decision in range(1, 200):
        s = decision * DECISION_STEPS
        taken = []
        for step in range(s - DECISION_STEPS + 1, s + 1):
            ep.step = step
            if sensor_noise.snapshot_due(ep):
                taken.append(step)
        assert taken == [s - sensor_noise.data_delay_steps(ep, s)]


def test_a_command_acts_one_control_step_after_it_is_sent():
    """The aircraft holds still on the first step, then always acts on the command sent one step before."""
    ep = _episode()
    sent = [decode_action(_action(move_forward=v, take_off=1.0 if v == 0.25 else 0.0), None)
            for v in (0.25, 0.5, 0.75, 1.0)]
    acted = [sensor_noise.delay(None, ep, command) for command in sent]
    still = acted[0]
    assert (still.move_forward, still.move_up, still.turn, still.gimbal_tilt) == (0.0, 0.0, 0.0, 0.0)
    assert not (still.thermal or still.take_off or still.zoom or still.report) and still.night_mode == "off"
    assert acted[1:] == sent[:-1] and acted[1].take_off


def test_the_stream_look_is_the_same_bytes_on_every_machine():
    """The stream look of a fixed frame hashes to the pinned value, so no validator's CPU can draw other bytes."""
    streamed = sensor_noise.stream_look(_frame())
    assert streamed.dtype == np.float32 and streamed.shape == (480, 640, 3)
    assert hashlib.sha256(streamed.tobytes()).hexdigest() == _STREAM_SHA256


@pytest.mark.skipif(sensor_noise._ENGINE_LOOK is None, reason="the engine has no compiled stream look")
def test_the_engine_stream_look_is_the_numpy_one_to_the_byte():
    """The engine's compiled look gives the numpy look's bytes on a render-like frame, noise past 0 and 1, every
    rounding tie, other sizes and the whole quality scale."""
    rng = np.random.default_rng(16)
    ties = ((np.arange(480 * 640 * 3) % 256 + 0.5) / 255.0).astype(np.float32).reshape(480, 640, 3)
    frames = [_frame(), ties, rng.uniform(-0.1, 1.1, (480, 640, 3)).astype(np.float32),
              rng.uniform(0.0, 1.0, (32, 48, 3)).astype(np.float32)]
    for frame in frames:
        for quality in (1, 10, 49, 50, sensor_noise.VIDEO_QUALITY, 95, 100):
            engine = sensor_noise.stream_look(frame, quality)
            assert engine.tobytes() == sensor_noise._numpy_look(frame, quality).tobytes()


def test_the_stream_look_softens_detail_and_keeps_flat_colour():
    """A detailed frame loses fine grain like a compressed stream; a flat colour moves at most two levels, and a blank
    frame is passed through untouched."""
    frame = _frame()
    streamed = sensor_noise.stream_look(frame)
    psnr = 10.0 * np.log10(1.0 / np.mean((streamed - frame) ** 2))
    assert 20.0 < psnr < 40.0
    for colour in ((96, 96, 96), (40, 120, 200), (230, 60, 30)):
        flat = np.broadcast_to(np.asarray(colour, dtype=np.float32) / np.float32(255.0), (480, 640, 3))
        assert np.abs(sensor_noise.stream_look(flat) - flat).max() <= 2.0 / 255.0 + 1e-6
    ep = _episode()
    blank = np.zeros((480, 640, 3), dtype=np.float32)
    assert sensor_noise._streamed(ep.sensor_noise, "rgb", blank) is blank


def test_the_position_error_moves_position_and_height_together():
    """The drift at the snapshot's step is added to the position, and its up part to the height above take-off."""
    ep = _episode(2)
    state = new_state()
    state[STATE_SLICES["position_m"]] = (10.0, 20.0, 5.0)
    state[STATE_SLICES["height_above_takeoff_m"]] = 5.0
    image = np.zeros((480, 640, 3), dtype=np.float32)
    seen = sensor_noise.observe(None, ep, {"step": 123, "state": state, "rgb": image, "thermal": image[..., :1],
                                           "zoom": image, "feed_view": _view(), "zoom_view": None})
    error = ep.sensor_noise["drift"][123]
    assert seen["state"][STATE_SLICES["position_m"]] == pytest.approx(np.array([10.0, 20.0, 5.0]) + error)
    assert seen["state"][STATE_SLICES["height_above_takeoff_m"]][0] == pytest.approx(5.0 + error[2])


def test_a_zoom_box_is_read_on_the_frame_the_model_was_shown():
    """When a newer frame has been taken but not yet delivered, the zoom still aims from the frame the model saw."""
    ep = _episode()
    camera.reset(None, ep)
    zoom.reset(None, ep)
    shown, newer = _view(step=25), _view(step=50, eye=(3.0, 0.0, 20.0))
    ep.sensor_noise["shown"]["feed"] = shown
    ep.camera["view"] = newer
    zoom.request(None, ep, decode_action(_action(zoom=1.0, zoom_cx=0.3, zoom_cy=0.6), None))
    assert ep.zoom["seen"] == shown


def _flown(seed, pilot, max_decisions):
    """Fly a patrol while logging, every control step, the true position and the command the aircraft acted on."""
    log = {"position": [], "forward": []}
    step_once = SolarPatrolChallengeFamily.post_step_update

    def post_step_update(self, env):
        """The family's own bookkeeping, then a record of this step."""
        step_once(self, env)
        log["position"].append(np.asarray(env.pos[0], dtype=float) - env._solar.dock_position)
        log["forward"].append(env._solar.command.move_forward)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(SolarPatrolChallengeFamily, "post_step_update", post_step_update)
        flight = _patrol(seed, pilot, max_decisions=max_decisions)
    return flight, log


@pytest.mark.timeout(300)
@pytest.mark.usefixtures("_flat_park", "blank_camera")
def test_a_flown_patrol_is_seen_60_or_80_ms_late_and_within_the_rtk_error():
    """In a real flight each observed position is the true one of 3 or 4 control steps before, off by the drift."""
    def pilot(i, obs):
        """Take off, then fly a slow circle."""
        return _action(take_off=1.0) if i == 0 else _action(move_forward=0.6, turn=0.3)

    flight, log = _flown(21, pilot, max_decisions=400)
    ep, truth = flight["episode"], np.array(log["position"])
    lags = []
    for k, obs in enumerate(flight["observations"][1:], start=1):
        s = k * DECISION_STEPS
        # Only a moving aircraft shows its lag; one standing still reads the same at every delay.
        if np.linalg.norm(truth[s - 1] - truth[s - DECISION_STEPS - 1]) < 0.05:
            continue
        seen = obs["state"][STATE_SLICES["position_m"]]
        misses = [np.abs(seen - truth[s - lag - 1] - ep.sensor_noise["drift"][s - lag]).max() for lag in range(5)]
        lag = int(np.argmin(misses))
        assert misses[lag] < 1e-3 and lag == sensor_noise.data_delay_steps(ep, s)
        assert np.abs(seen - truth[s - lag - 1]).max() <= 0.1 + 1e-3
        lags.append(lag)
    assert set(lags) == {3, 4}


@pytest.mark.timeout(300)
@pytest.mark.usefixtures("_flat_park", "blank_camera")
def test_a_flown_command_reaches_the_aircraft_one_step_later():
    """A stick moved at a decision is acted on from that decision's second control step, 20 ms after it was sent."""
    def pilot(i, obs):
        """Take off, hover, then push forward at decision 300."""
        return _action(take_off=1.0) if i == 0 else _action(move_forward=1.0 if i >= 300 else 0.0)

    _flight, log = _flown(22, pilot, max_decisions=305)
    forward = np.array(log["forward"])
    first = int(np.argmax(forward > 0.0))
    assert first == 300 * DECISION_STEPS + 1
    assert np.all(forward[:first] == 0.0) and np.all(forward[first:] == 1.0)
