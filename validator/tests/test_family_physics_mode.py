"""Tests for the per-family physics mode: plain rotor thrust today, the URDF's drag, ground
effect and downwash for families that opt in."""

from __future__ import annotations

import dataclasses

import numpy as np
import pybullet as p
from gym_pybullet_drones.utils.enums import Physics

import swarm.challenge_families  # noqa: F401  registers every family runtime class
from swarm.challenge_families.autopilot import AutopilotChallengeFamily
from swarm.challenge_families.base import ChallengeFamilyRuntime
from swarm.constants import SIM_DT
from swarm.core.moving_drone import MovingDroneAviary
from swarm.protocol import MapTask
from swarm.utils.env_factory import make_env
from swarm.validator import task_gen

AERO = Physics.PYB_GND_DRAG_DW.value


def _every_family_class(cls=ChallengeFamilyRuntime):
    """Yield every runtime family class registered under the base class."""
    for sub in cls.__subclasses__():
        yield sub
        yield from _every_family_class(sub)


def _open_task(seed: int) -> MapTask:
    """A still-air autopilot task on the open map."""
    return task_gen.task_for_seed_and_type(
        sim_dt=SIM_DT, seed=seed, challenge_type=2, moving_platform=False,
    )


def _fly_forward(env: MovingDroneAviary, steps: int) -> None:
    """Command full speed along +x for the given number of control steps."""
    action = np.zeros((1, env.action_space.shape[-1]), dtype=np.float32)
    action[0, 0] = 1.0
    action[0, 3] = 1.0
    for _ in range(steps):
        env.step(action)


def test_every_family_stays_on_plain_physics():
    """No family opts into the aerodynamic terms on its own; the switch is explicit per family."""
    assert ChallengeFamilyRuntime.physics_mode == Physics.PYB.value
    for family in _every_family_class():
        assert family.physics_mode == Physics.PYB.value, family.__name__


def test_env_takes_the_physics_mode_from_the_family(monkeypatch):
    """The env resolves its physics mode from the family unless the caller passes one."""
    task = _open_task(11)
    plain = make_env(task, gui=False)
    plain.close()
    assert plain.PHYSICS == Physics.PYB

    monkeypatch.setattr(AutopilotChallengeFamily, "physics_mode", AERO)
    aero = make_env(task, gui=False)
    aero.close()
    assert aero.PHYSICS == Physics.PYB_GND_DRAG_DW

    explicit = MovingDroneAviary(task, physics=Physics.PYB, ctrl_freq=30, pyb_freq=30)
    explicit.close()
    assert explicit.PHYSICS == Physics.PYB


def test_air_drag_shortens_a_full_speed_flight(monkeypatch):
    """Three seconds at full speed cover less ground with the air terms on than off."""
    task = _open_task(11)
    plain = make_env(task, gui=False)
    monkeypatch.setattr(AutopilotChallengeFamily, "physics_mode", AERO)
    aero = make_env(task, gui=False)
    try:
        steps = int(round(3.0 / SIM_DT))
        _fly_forward(plain, steps)
        _fly_forward(aero, steps)
        plain_dx = float(plain.pos[0, 0] - task.start[0])
        aero_dx = float(aero.pos[0, 0] - task.start[0])
        assert plain_dx > 3.0
        assert aero_dx < plain_dx - 0.05, f"plain {plain_dx:.3f} m, aero {aero_dx:.3f} m"
    finally:
        plain.close()
        aero.close()


def test_ground_effect_height_comes_from_the_downward_ray(monkeypatch):
    """The cushion follows the measured height above the surface, not the world z."""
    task = _open_task(11)
    monkeypatch.setattr(AutopilotChallengeFamily, "physics_mode", AERO)
    env = make_env(task, gui=False)
    try:
        hover = np.full(4, float(env.HOVER_RPM))

        def lift_for(height: float) -> float:
            """Vertical speed after one physics step with only the cushion at this height."""
            p.resetBaseVelocity(env.DRONE_IDS[0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], physicsClientId=env.CLIENT)
            monkeypatch.setattr(env, "_get_altitude_distance", lambda nth_drone=0: height)
            env._groundEffect(hover, 0)
            p.stepSimulation(physicsClientId=env.CLIENT)
            return float(p.getBaseVelocity(env.DRONE_IDS[0], physicsClientId=env.CLIENT)[0][2])

        near, far = lift_for(0.05), lift_for(5.0)
        assert near > far + 0.01, f"near {near:.4f} m/s, far {far:.4f} m/s"
    finally:
        env.close()


def test_wind_replaces_the_still_air_drag_term(monkeypatch):
    """With wind on, the wind force carries the rotor drag and the still-air drag is skipped."""
    task = _open_task(11)
    monkeypatch.setattr(AutopilotChallengeFamily, "physics_mode", AERO)
    still = make_env(task, gui=False)
    windy = make_env(dataclasses.replace(task, wind_max_mps=4.0), gui=False)
    try:
        calls = {"still": 0, "windy": 0}
        monkeypatch.setattr(still, "_drag", lambda rpm, i: calls.__setitem__("still", calls["still"] + 1))
        monkeypatch.setattr(windy, "_drag", lambda rpm, i: calls.__setitem__("windy", calls["windy"] + 1))
        hover = np.zeros((1, still.action_space.shape[-1]), dtype=np.float32)
        still.step(hover)
        windy.step(hover)
        assert calls == {"still": 1, "windy": 0}
    finally:
        still.close()
        windy.close()
