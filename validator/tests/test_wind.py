"""Seeded wind: determinism, the per-map cap, the off-by-default path and the push on a hovering drone."""
from __future__ import annotations

import dataclasses

import msgpack
import numpy as np

from swarm.constants import SIM_DT, WIND_MEAN_FRACTION
from swarm.core import moving_drone as moving_drone_mod
from swarm.core.wind import SeededWind
from swarm.protocol import MapTask
from swarm.utils.env_factory import make_env
from swarm.validator import task_gen

_HORIZON = 60.0


def _wind(seed: int, *, turbulence: float = 1.0, gusts: int = 2, horizon: float = _HORIZON) -> SeededWind:
    """A 4 m/s wind model for the given seed."""
    return SeededWind(seed, max_mps=4.0, turbulence=turbulence, gusts=gusts, dt=SIM_DT, horizon=horizon)


def _series(wind: SeededWind, steps: int) -> np.ndarray:
    """The wind vector at every control step from t = 0."""
    return np.array([wind.velocity(k * SIM_DT) for k in range(steps)])


def _windy(task: MapTask) -> MapTask:
    """The same task with a steady 4 m/s wind switched on."""
    return dataclasses.replace(task, wind_max_mps=4.0, wind_turbulence=0.0, wind_gusts=0)


def _open_task(seed: int) -> MapTask:
    """A still-air autopilot task on the open map."""
    return task_gen.task_for_seed_and_type(
        sim_dt=SIM_DT, seed=seed, challenge_type=2, moving_platform=False,
    )


def test_same_seed_gives_the_same_wind() -> None:
    """Two models from one seed produce identical vectors; a different seed does not."""
    a = _series(_wind(7), 3000)
    b = _series(_wind(7), 3000)
    c = _series(_wind(8), 3000)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_reset_replays_the_same_wind() -> None:
    """Resetting the model replays the exact same series."""
    wind = _wind(7)
    first = _series(wind, 3000)
    wind.reset()
    assert np.array_equal(first, _series(wind, 3000))


def test_total_wind_never_exceeds_the_map_cap() -> None:
    """The map's max_mps is a hard cap on steady plus turbulence plus gusts."""
    for seed in (1, 2, 3, 4, 5):
        speeds = np.linalg.norm(_series(_wind(seed), 3000), axis=1)
        assert speeds.max() <= 4.0 + 1e-9


def test_steady_wind_is_constant_horizontal_and_inside_the_mean_band() -> None:
    """With turbulence and gusts off the wind is one horizontal vector in the mean band."""
    series = _series(_wind(3, turbulence=0.0, gusts=0), 500)
    assert np.all(series == series[0])
    assert series[0, 2] == 0.0
    lo, hi = WIND_MEAN_FRACTION
    assert 4.0 * lo <= np.linalg.norm(series[0]) <= 4.0 * hi


def test_gusts_raise_the_speed_above_the_steady_wind() -> None:
    """Gust bumps lift the speed well above the steady value and fall back to it."""
    wind = _wind(3, turbulence=0.0, gusts=2)
    speeds = np.linalg.norm(_series(wind, 3000), axis=1)
    assert speeds.max() > 1.3 * wind.mean_mps
    assert speeds.min() == np.linalg.norm(wind.steady)


def test_turbulence_has_the_dryden_intensity() -> None:
    """Turbulence std is about 0.2 x mean sideways and 0.1 x mean vertically."""
    wind = _wind(5, turbulence=1.0, gusts=0, horizon=600.0)
    turb = _series(wind, 30000) - wind.steady
    std = turb.std(axis=0)
    assert abs(std[0] - 0.2 * wind.mean_mps) < 0.06 * wind.mean_mps
    assert abs(std[1] - 0.2 * wind.mean_mps) < 0.06 * wind.mean_mps
    assert abs(std[2] - 0.1 * wind.mean_mps) < 0.03 * wind.mean_mps


def test_task_without_wind_applies_no_force_to_the_base(monkeypatch) -> None:
    """A map that did not opt in has no wind model and never pushes the drone body."""
    env = make_env(_open_task(11), gui=False)
    try:
        assert env._wind is None
        base_calls = []
        orig = moving_drone_mod.p.applyExternalForce

        def spy(uid, link, *args, **kwargs):
            """Record every force aimed at the drone base, then apply it as normal."""
            if int(link) == -1:
                base_calls.append(args or kwargs)
            return orig(uid, link, *args, **kwargs)

        monkeypatch.setattr(moving_drone_mod.p, "applyExternalForce", spy)
        env.step(np.zeros((1, env.action_space.shape[-1]), dtype=np.float32))
        assert base_calls == []
    finally:
        env.close()


def test_wind_pushes_a_hovering_drone_downwind() -> None:
    """Hovering for 5 s under a 4 m/s wind drifts the drone along the wind direction."""
    still = make_env(_open_task(11), gui=False)
    windy = make_env(_windy(_open_task(11)), gui=False)
    try:
        assert windy._wind is not None
        hover = np.zeros((1, still.action_space.shape[-1]), dtype=np.float32)
        for _ in range(250):
            still.step(hover)
            windy.step(hover)
        drift = windy.pos[0, :2] - still.pos[0, :2]
        along_wind = float(np.dot(drift, windy._wind.direction[:2]))
        assert along_wind > 0.15, f"drift along the wind was {along_wind:.3f} m"
    finally:
        still.close()
        windy.close()


def test_same_seed_gives_the_same_flight_under_wind() -> None:
    """Flying the same actions twice on a windy task, with a reset in between, traces bit-identical positions."""
    task = dataclasses.replace(_open_task(11), wind_max_mps=4.0, wind_turbulence=1.0, wind_gusts=2)
    env = make_env(task, gui=False)
    try:
        n_act = int(env.action_space.shape[-1])

        def fly() -> np.ndarray:
            """Forty scripted steps; returns every position visited."""
            trace = []
            for k in range(40):
                env.step(0.5 * np.sin(np.arange(n_act, dtype=np.float32) + 0.1 * k)[None, :])
                trace.append(env.pos.copy())
            return np.array(trace)

        first = fly()
        env.reset(seed=task.map_seed)
        assert np.array_equal(first, fly())
    finally:
        env.close()


def test_task_gen_reads_the_per_map_table(monkeypatch) -> None:
    """Only the (family, map) pairs listed in the table get wind fields; everything else stays at zero."""
    monkeypatch.setattr(
        task_gen, "WIND_BY_MAP",
        {("cf_autopilot", 2): {"max_mps": 4.0, "turbulence": 1.0, "gusts": 2}},
    )
    windy = task_gen.task_for_seed_and_type(sim_dt=SIM_DT, seed=1, challenge_type=2)
    assert (windy.wind_max_mps, windy.wind_turbulence, windy.wind_gusts) == (4.0, 1.0, 2)
    other_map = task_gen.task_for_seed_and_type(sim_dt=SIM_DT, seed=1, challenge_type=1)
    other_family = task_gen.task_for_seed_and_type(
        sim_dt=SIM_DT, seed=1, challenge_type=2, family_id="cf_search_and_rescue",
    )
    for task in (other_map, other_family):
        assert (task.wind_max_mps, task.wind_turbulence, task.wind_gusts) == (0.0, 0.0, 0)


def test_default_table_gives_no_wind_anywhere() -> None:
    """The shipped table is empty, so every family on every map keeps still air."""
    for challenge_type in (1, 2, 3, 4, 5, 6):
        task = task_gen.task_for_seed_and_type(sim_dt=SIM_DT, seed=3, challenge_type=challenge_type)
        assert task.wind_max_mps == 0.0


def test_maptask_wind_fields_survive_pack_and_old_blobs() -> None:
    """Wind fields round-trip through pack/unpack and default to zero for blobs written before them."""
    task = _windy(_open_task(11))
    again = MapTask.unpack(task.pack())
    assert (again.wind_max_mps, again.wind_turbulence, again.wind_gusts) == (4.0, 0.0, 0)
    old = {k: v for k, v in dataclasses.asdict(task).items() if not k.startswith("wind_")}
    legacy = MapTask.unpack(msgpack.packb(old, use_bin_type=True))
    assert (legacy.wind_max_mps, legacy.wind_turbulence, legacy.wind_gusts) == (0.0, 0.0, 0)
