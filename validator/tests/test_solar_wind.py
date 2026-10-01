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

"""Wind strength per seed (task 18): the shares by day and night, the caps, the gusts, the two readings the dock
sends and when, and a patrol flown in still, light and strong air."""

from __future__ import annotations

import contextlib
import dataclasses
import io
import math
import os

import numpy as np
import pybullet as p
import pytest
import swarm_worlds

from swarm.challenge_families import build_benchmark_tasks
from swarm.challenge_families.solar_patrol import airframe, drone_state, park, wind
from swarm.challenge_families.solar_patrol.contract import (
    ACTION_DIM,
    ACTION_INDEX,
    FAMILY_ID,
    FLIGHT_PHASES,
    HORIZON_S,
    STATE_SLICES,
)
from swarm.challenge_families.solar_patrol.episode import SolarEpisode
from swarm.constants import SIM_DT, WIND_MEAN_FRACTION
from swarm.core.daylight import seeded_sun
from swarm.core.wind import SeededWind
from swarm.utils.env_factory import make_env_with_initial_obs
from validator.tests.test_solar_patrol_family import blank_camera  # noqa: F401

pytestmark = pytest.mark.usefixtures("blank_camera")
ROBOTS = swarm_worlds.robots_dir()
needs_m4td = pytest.mark.skipif(not os.path.isfile(os.path.join(ROBOTS, airframe.URDF)),
                                reason=f"the installed swarm-worlds has no {airframe.URDF} yet")


def _seeded_wind(seed: int) -> SeededWind:
    """The physical wind the environment builds for this seed's task."""
    fields = wind.for_seed(seed)
    return SeededWind(seed, max_mps=fields["wind_max_mps"], turbulence=fields["wind_turbulence"],
                      gusts=fields["wind_gusts"], dt=SIM_DT, horizon=HORIZON_S)


def _first(kind: str) -> int:
    """The first seed of the given strength."""
    return next(seed for seed in range(1000) if wind.strength(seed) == kind)


@pytest.fixture
def nights(monkeypatch):
    """Every seed flown at night, lit by the seeded moon."""
    monkeypatch.setattr(park, "SEEDED_SUN", True)
    monkeypatch.setattr(park, "NIGHT_SHARE", 1.0)


def test_the_same_seed_blows_the_same_wind():
    """A seed's strength, cap and gusts never change, and its wind replays to the byte."""
    assert all(wind.for_seed(seed) == wind.for_seed(seed) for seed in range(200))
    seed = _first("strong")
    first, second = _seeded_wind(seed), _seeded_wind(seed)
    a = np.array([first.velocity(k * SIM_DT) for k in range(2000)])
    b = np.array([second.velocity(k * SIM_DT) for k in range(2000)])
    assert a.tobytes() == b.tobytes()
    assert first.current.tobytes() == a[-1].tobytes()


@pytest.fixture
def days(monkeypatch):
    """Every seed flown by day, under the seeded sun."""
    monkeypatch.setattr(park, "SEEDED_SUN", True)
    monkeypatch.setattr(park, "NIGHT_SHARE", 0.0)


def test_day_seeds_take_the_sites_day_shares(days):
    """Over 20,000 day seeds, still, light and strong come in the shares measured at the site by day."""
    kinds = [wind.strength(seed) for seed in range(20000)]
    shares = [kinds.count(kind) / len(kinds) for kind in wind.STRENGTHS]
    assert shares == pytest.approx(wind.DAY_SHARES, abs=0.01)


def test_night_seeds_take_the_sites_night_shares(nights):
    """Over 20,000 night seeds, still, light and strong come in the site's calmer night shares."""
    kinds = [wind.strength(seed) for seed in range(20000)]
    shares = [kinds.count(kind) / len(kinds) for kind in wind.STRENGTHS]
    assert shares == pytest.approx(wind.NIGHT_SHARES, abs=0.01)


def test_the_parks_own_mix_gives_each_seed_its_own_air():
    """With the park's own day and night mix, the day seeds take the day shares and the night seeds the night ones."""
    kinds = {False: [], True: []}
    for seed in range(20000):
        kinds[wind.is_night(seed)].append(wind.strength(seed))
    for night, shares in ((False, wind.DAY_SHARES), (True, wind.NIGHT_SHARES)):
        drawn = kinds[night]
        assert len(drawn) > 5000
        assert [drawn.count(kind) / len(drawn) for kind in wind.STRENGTHS] == pytest.approx(shares, abs=0.015)


def test_night_is_the_night_the_environment_lights(monkeypatch):
    """With half the seeds at night, a seed's wind is a night wind exactly when the seeded sun gives it the moon."""
    monkeypatch.setattr(park, "SEEDED_SUN", True)
    monkeypatch.setattr(park, "NIGHT_SHARE", 0.5)
    nights = [wind.is_night(seed) for seed in range(400)]
    assert nights == [seeded_sun(seed, 0.5).night for seed in range(400)]
    assert 0.4 < sum(nights) / len(nights) < 0.6


def test_each_strength_gets_its_cap():
    """Still seeds switch the wind off; light and strong ones get a cap in their range, full turbulence and one or
    two gusts."""
    for seed in range(3000):
        kind, fields = wind.strength(seed), wind.for_seed(seed)
        if kind == "none":
            assert fields == wind.STILL_AIR
            continue
        low, high = wind.CAP_MPS[kind]
        assert low <= fields["wind_max_mps"] <= high
        assert fields["wind_turbulence"] == wind.TURBULENCE
        assert wind.GUSTS[0] <= fields["wind_gusts"] <= wind.GUSTS[1]


def test_no_gust_ever_passes_12_mps():
    """Across whole patrols of the strong seeds, the steady wind is 4 to 8 m/s and nothing, gusts and turbulence
    included, ever passes the seed's cap or 12 m/s."""
    strong = [seed for seed in range(200) if wind.strength(seed) == "strong"][:12]
    low, high = wind.CAP_MPS["strong"]
    for seed in strong:
        blow = _seeded_wind(seed)
        top = max(float(np.linalg.norm(blow.velocity(k * SIM_DT))) for k in range(int(HORIZON_S / SIM_DT)))
        assert top <= blow.max_mps + 1e-9 <= 12.0 + 1e-9
        assert low * WIND_MEAN_FRACTION[0] - 1e-9 <= blow.mean_mps <= high * WIND_MEAN_FRACTION[1] + 1e-9


def test_gusts_peak_like_the_sites():
    """The strongest 3 s of a patrol blows about 1.5 times its mean, as the station near the site measures."""
    windy = [seed for seed in range(40) if wind.strength(seed) != "none"][:8]
    window = int(round(3.0 / SIM_DT))
    factors = []
    for seed in windy:
        blow = _seeded_wind(seed)
        speed = np.array([math.hypot(*blow.velocity(k * SIM_DT)[:2]) for k in range(int(HORIZON_S / SIM_DT))])
        factors.append(np.convolve(speed, np.ones(window) / window, mode="valid").max() / speed.mean())
    assert 1.35 < float(np.median(factors)) < 1.65


class _Blowing:
    """The environment's seeded wind held at one vector."""

    def __init__(self, east: float, north: float):
        """Blowing towards (east, north), steady."""
        self.current = np.array([east, north, 0.0])
        self.steady = self.current.copy()


class _Env:
    """The environment fields the wind part reads: the aircraft's position and the seeded wind."""

    def __init__(self, blowing: _Blowing | None, height_m: float):
        """An aircraft this high above the pad at the origin."""
        self._wind = blowing
        self.pos = [np.array([0.0, 0.0, height_m])]


def _patrol(seed: int, env: _Env, phase: str, seconds: float) -> tuple[SolarEpisode, list[tuple]]:
    """Run the wind part for a while in one flight phase; every control step's (estimate, sector, gauge)."""
    ep = SolarEpisode(seed=seed)
    ep.phase = phase
    wind.reset(env, ep)
    readings = []
    for step in range(1, int(round(seconds / SIM_DT)) + 1):
        ep.step = step
        wind.update(env, ep)
        readings.append((ep.wind["estimate_mps"], ep.wind["estimate_sector"], ep.wind["dock_mps"]))
    return ep, readings


def test_on_the_pad_the_estimate_is_zero_and_the_gauge_reads():
    """Docked, the aircraft estimates nothing, as DJI reports 0 on the ground; the dock's gauge still reads the wind
    near the ground from the first step."""
    ep, readings = _patrol(3, _Env(_Blowing(6.0, 0.0), 0.0), "docked", 30.0)
    assert {(r[0], r[1]) for r in readings} == {(0.0, 0)}
    assert readings[0][2] > 0.0
    assert readings[-1][2] == pytest.approx(ep.wind["gauge_share"] * 6.0, abs=0.05)


def test_both_readings_are_pushed_every_2_s():
    """Each reading changes only on its own 2 s push, whatever the wind does in between."""
    ep = SolarEpisode(seed=5)
    ep.phase = "flying"
    blowing = _Blowing(0.0, 0.0)
    env = _Env(blowing, 20.0)
    wind.reset(env, ep)
    changed = {"estimate_mps": [], "dock_mps": []}
    for step in range(1, 3001):
        blowing.current = np.array([2.0 + math.sin(step * 0.05), 1.0, 0.0])
        before = dict(ep.wind)
        ep.step = step
        wind.update(env, ep)
        for key in changed:
            if ep.wind[key] != before[key]:
                changed[key].append(step)
    for key, offset in (("estimate_mps", "estimate_offset"), ("dock_mps", "gauge_offset")):
        assert len(changed[key]) > 10
        assert all((step + ep.wind[offset]) % wind.PUSH_STEPS == 0 for step in changed[key])
    assert wind.PUSH_STEPS * SIM_DT == pytest.approx(2.0)


def test_in_the_air_the_estimate_is_smoothed_and_reads_low():
    """Airborne in a steady 5 m/s east wind, the estimate climbs in over DJI's smoothing, settles at the seed's low
    reading in 0.1 m/s steps, and points within its fixed error of where the wind comes from."""
    for seed in range(20):
        ep, readings = _patrol(seed, _Env(_Blowing(-5.0, 0.0), 20.0), "flying", 90.0)
        pushed = int(round(wind.ESTIMATE_TAU_S / SIM_DT)) + wind.PUSH_STEPS
        assert readings[pushed][0] < readings[-1][0]
        assert readings[-1][0] == pytest.approx(round(5.0 * ep.wind["scale"], 1), abs=0.1)
        assert readings[-1][0] == round(readings[-1][0], 1)
        assert readings[-1][1] in (1, 2, 3)
        assert wind.ESTIMATE_SCALE[0] <= ep.wind["scale"] <= wind.ESTIMATE_SCALE[1]


def test_still_air_reads_zero_in_the_air():
    """With no wind the estimate and the gauge both read 0, and the sector says north as DJI's windless does."""
    _ep, readings = _patrol(42, _Env(None, 20.0), "flying", 20.0)
    assert set(readings) == {(0.0, 0, 0.0)}


@pytest.mark.parametrize("east, north, sector", [
    (0.0, -1.0, 0), (-1.0, -1.0, 1), (-1.0, 0.0, 2), (-1.0, 1.0, 3),
    (0.0, 1.0, 4), (1.0, 1.0, 5), (1.0, 0.0, 6), (1.0, -1.0, 7),
    (-0.4, -1.0, 0), (-1.0, -0.2, 2),
])
def test_the_sector_is_where_the_wind_comes_from(east, north, sector):
    """A wind blowing south comes from the north, sector 0; the sectors run clockwise, to northwest at 7."""
    assert wind._sector(np.array([east, north])) == sector


def test_speeds_are_in_tenths():
    """Readings come in DJI's 0.1 m/s steps, rounded to the nearest."""
    assert [wind._tenths(v) for v in (0.0, 0.04, 0.05, 3.449, 3.46, 11.96)] == [0.0, 0.0, 0.1, 3.4, 3.5, 12.0]


@pytest.mark.parametrize("night", [False, True])
def test_the_gauge_reads_the_slower_air_near_the_ground(monkeypatch, night):
    """The gauge's share of the 20 m wind is the site's by day, and lower by night when the ground air is still."""
    monkeypatch.setattr(park, "SEEDED_SUN", night)
    monkeypatch.setattr(park, "NIGHT_SHARE", 1.0)
    low, high = wind.NIGHT_GAUGE_SHARE if night else wind.DAY_GAUGE_SHARE
    for seed in range(100):
        ep = SolarEpisode(seed=seed)
        wind.reset(_Env(_Blowing(4.0, 3.0), 0.0), ep)
        assert low <= ep.wind["gauge_share"] <= high
        assert ep.wind["dock_mps"] == wind._tenths(ep.wind["gauge_share"] * 5.0)


def _action(**values) -> np.ndarray:
    """An action vector at rest, with the named fields set."""
    a = np.zeros(ACTION_DIM, dtype=np.float32)
    for name, value in values.items():
        a[ACTION_INDEX[name]] = value
    return a


@pytest.fixture
def flat_ground(monkeypatch):
    """The patrol's world as a flat slab, no movers, and a wide square fence."""
    class Still:
        """Movers for a park that has none."""

        body_uids = frozenset()

        def advance(self, step=None):
            """Nothing moves."""

    def build(seed=0, cli=0, asset_dir=None, groups=None):
        """A flat collision slab standing in for the park's terrain."""
        slab = p.createCollisionShape(p.GEOM_BOX, halfExtents=[400.0, 400.0, 0.5], physicsClientId=cli)
        floor = p.createMultiBody(0, slab, -1, [60.0, 100.0, -0.5], physicsClientId=cli)
        return {"bodies": {"terrain": [floor]}, "movers": [], "asset_dir": "flat"}

    monkeypatch.setattr(park, "build_solar_map", build)
    monkeypatch.setattr(park, "build_solar_movers", lambda world, seed=0, cli=0: Still())
    monkeypatch.setattr(park, "fence_line", lambda asset_dir: np.array([[-300.0, -300.0], [400.0, -300.0],
                                                                        [400.0, 500.0], [-300.0, 500.0]]))
    monkeypatch.setattr(drone_state, "survey", lambda asset_dir: (np.zeros((0, 5)), np.zeros((0, 5))))


def _fly(task, seconds_forward: float, hover_s: float) -> tuple[list, object]:
    """Take off, hover, then fly forward at full stick; the state after every decision, and the environment."""
    with contextlib.redirect_stdout(io.StringIO()):
        env, obs = make_env_with_initial_obs(task)
    states, flying = [obs["state"].copy()], None
    for i in range(int(HORIZON_S * 10)):
        phase = FLIGHT_PHASES[int(obs["state"][STATE_SLICES["flight_phase"]][0])]
        if phase == "flying" and flying is None:
            flying = i
        since = None if flying is None else (i - flying) * 0.1
        forward = 1.0 if since is not None and hover_s <= since < hover_s + seconds_forward else 0.0
        obs, _r, terminated, truncated, _info = env.step(_action(take_off=1.0 if i == 0 else 0.0,
                                                                 move_forward=forward)[None, :])
        assert not (terminated or truncated)
        states.append(obs["state"].copy())
        if since is not None and since >= hover_s + seconds_forward:
            break
    return states, env


@needs_m4td
@pytest.mark.timeout(600)
def test_still_air_flies_by_the_windy_drag(flat_ground):
    """A still seed and the same seed in a wind too weak to feel fly forward alike: rotor drag acts in still air
    too, so strength is the only thing a seed's wind changes."""
    task = build_benchmark_tasks(sim_dt=SIM_DT, seeds=[_first("none")], family_id=FAMILY_ID)[0]
    speeds = []
    for flown in (task, dataclasses.replace(task, wind_max_mps=0.001, wind_turbulence=0.0, wind_gusts=0)):
        _states, env = _fly(flown, 10.0, 5.0)
        try:
            speeds.append(float(np.hypot(*env.vel[0][:2])))
        finally:
            env.close()
    assert speeds[0] == pytest.approx(speeds[1], rel=0.01)


@needs_m4td
@pytest.mark.timeout(600)
def test_a_strong_wind_does_not_blow_the_hover_away(flat_ground):
    """With the sticks centred in a strong seed's wind, the aircraft holds its place within a metre for 30 s, as the
    real one holds position up to its 12 m/s rating."""
    task = build_benchmark_tasks(sim_dt=SIM_DT, seeds=[_first("strong")], family_id=FAMILY_ID)[0]
    states, env = _fly(task, 0.0, 45.0)
    env.close()
    held = np.array([s[STATE_SLICES["position_m"]][:2] for s in states[-300:]])
    assert float(np.linalg.norm(held - held.mean(axis=0), axis=1).max()) < 1.0


@needs_m4td
@pytest.mark.timeout(900)
def test_a_patrol_reads_still_light_and_strong_air(flat_ground):
    """Three seeds, one of each strength, hovering at 20 m: the estimate is 0 before take-off and grows with the
    strength once airborne, and the gauge reads nothing in still air and more in strong air than in light."""
    readings = {}
    for kind in wind.STRENGTHS:
        task = build_benchmark_tasks(sim_dt=SIM_DT, seeds=[_first(kind)], family_id=FAMILY_ID)[0]
        states, env = _fly(task, 0.0, 45.0)
        env.close()
        on_pad = [s for s in states if s[STATE_SLICES["height_above_takeoff_m"]][0] < 0.3]
        assert len(on_pad) > 10 and all(s[STATE_SLICES["wind_estimate_mps"]][0] == 0.0 for s in on_pad)
        late = states[-100:]
        readings[kind] = (np.mean([s[STATE_SLICES["wind_estimate_mps"]][0] for s in late]),
                          np.mean([s[STATE_SLICES["dock_wind_mps"]][0] for s in late]))
    assert readings["none"] == (0.0, 0.0)
    assert 0.0 < readings["light"][0] < readings["strong"][0]
    assert 0.0 < readings["light"][1] < readings["strong"][1]
