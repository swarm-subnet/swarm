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

"""Swarm Sentinel on an incomplete install (task 52): every missing asset, engine flag and CPU feature is named; a
patrol and a seed check are refused before anything is built or flown, and leave no verdict behind; once the install
is repaired the seed is judged and passes; the validator turns the family's tasks away, says why without flooding
its log, and waits instead of asking for the same task again.

Each test builds its own install: a swarm-worlds tree in a folder of its own, the engine flags set or removed on
the pybullet module, and a CPU feature list in a file, so the outcome never depends on the machine the suite runs on.
The one exception flies the real reference pilot on the machine's own install, and runs only where that install is
incomplete, as the CI image's is.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pybullet as p
import pytest
import swarm_worlds

from swarm.challenge_families import require_runtime_family
from swarm.challenge_families.base import IncompleteInstall
from swarm.challenge_families.solar_patrol import decoys, install, reference_pilot, seed_checks
from swarm.challenge_families.solar_patrol.contract import FAMILY_ID
from swarm.challenge_families.solar_patrol.seed_checks import Verdict
from swarm.challenge_families.solar_patrol.task import solar_patrol_task
from swarm.constants import INSTALL_REFUSAL_IDLE_SEC, INSTALL_REFUSAL_LOG_SEC, SIM_DT
from swarm.core.maps.solar import builder
from swarm.utils.env_factory import make_env_with_initial_obs
from swarm.validator import forward as forward_mod
from swarm.validator.utils_parts import run_task as run_task_module

_APPROVE = seed_checks.approve
_PREPARE = seed_checks.prepare
_PACKAGE = Path(__file__).resolve().parents[2] / "swarm"
# Flags the patrol reads that change no pixel: the stock renderer every engine has, and the switch that only skips
# converting a depth image the colour and thermal frames never read.
_PIXEL_NEUTRAL_FLAGS = {"ER_TINY_RENDERER", "ER_SWARM_NO_DEPTH"}
_FLAG_NAME = re.compile(r"""(?:p\.|["'])((?:ER|VISUAL_SHAPE)_[A-Z0-9_]+)""")
_PASSING_PILOT = '''"""A reference pilot that flies nothing and passes every seed it reads."""
import sys

from swarm.challenge_families.solar_patrol.seed_checks import Verdict, keep

for line in sys.stdin:
    keep(Verdict(seed=int(line), passed=True))
'''


@pytest.fixture
def complete(monkeypatch, tmp_path):
    """A complete install: a swarm-worlds tree holding every file Sentinel loads, an engine with every flag and a CPU
    with AVX2 and FMA. Returns the tree's root."""
    root = tmp_path / "swarm_worlds"
    solar = root / "maps" / "custom" / "solar"
    monkeypatch.setattr(swarm_worlds, "__file__", str(root / "__init__.py"))
    monkeypatch.setattr(swarm_worlds, "maps_dir", lambda: str(root / "maps"))
    monkeypatch.setattr(swarm_worlds, "robots_dir", lambda: str(root / "robots"))
    monkeypatch.setattr(builder, "SOLAR_ASSET_DIR", str(solar))
    monkeypatch.setattr(decoys, "DOGS_DIR", str(solar / "dogs"))
    solar.mkdir(parents=True)
    (solar / "manifest.json").write_text(json.dumps({
        "items": {"pickup_body": {"folder": "movers"}, "goat": {"folder": "goats"}},
        "placements": [{"item": "pickup_body", "mover": "pickup"}, {"item": "goat", "mover": "goat"}],
    }))
    for path in install.asset_files():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).touch()
    for bit, name in enumerate(install.ENGINE_FLAGS):
        monkeypatch.setattr(p, name, 1 << bit, raising=False)
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor\t: 0\nflags\t\t: fpu sse4_2 avx avx2 fma bmi2\n")
    monkeypatch.setattr(install, "CPUINFO", str(cpuinfo))
    return root


@pytest.fixture
def no_thieves(complete):
    """A complete install but for a swarm-worlds from before the thieves: no intruder catalogue."""
    (complete / "maps" / "custom" / "solar" / "intruders" / "intruders.json").unlink()
    return complete


@pytest.fixture
def kept(monkeypatch, tmp_path):
    """The real seed checks, which the suite otherwise stubs out, keeping their verdicts in a folder of the test's own."""
    monkeypatch.setattr(seed_checks, "approve", _APPROVE)
    monkeypatch.setattr(seed_checks, "prepare", _PREPARE)
    monkeypatch.setenv(seed_checks.CACHE_ENV, str(tmp_path / "verdicts"))
    return tmp_path / "verdicts"


def _named(root: Path, path: str) -> str:
    """A swarm-worlds file as the refusal names it, relative to the package."""
    return os.path.relpath(path, root)


def test_a_complete_install_has_nothing_missing(complete):
    """An install with every file, flag and CPU feature reports nothing and is not refused."""
    assert install.problems() == []
    install.require()
    assert require_runtime_family(FAMILY_ID).install_problems() == []


def test_every_file_a_patrol_needs_is_checked_and_named(complete):
    """Each swarm-worlds file a patrol loads, the road file found through the park's own manifest among them, is
    refused by its own path when it is missing, and the install is whole again once it is back."""
    files = install.asset_files()
    names = {_named(complete, path) for path in files}
    assert {"maps/custom/solar/intruders/intruders.json", "maps/custom/solar/motions/motions.json",
            "maps/custom/solar/dogs/dogs.json", "maps/custom/solar/movers/public_road.json",
            "maps/custom/solar/manifest.json", "robots/m4td.urdf", "robots/dock3.urdf"} <= names
    for path in files:
        if path.endswith("manifest.json"):
            continue  # without the manifest the road cannot be looked up; the manifest's own case is the next test
        held = Path(path).read_bytes()
        Path(path).unlink()
        assert install.problems() == [f"swarm-worlds {install._installed('swarm-worlds')} is missing "
                                      f"{_named(complete, path)}"]
        Path(path).write_bytes(held)
    assert install.problems() == []


def test_a_swarm_worlds_without_the_park_is_refused(complete):
    """A swarm-worlds with no solar map at all, as the one the shipped pin installs, is refused for the manifest."""
    (complete / "maps" / "custom" / "solar" / "manifest.json").unlink()
    [line] = install.problems()
    assert "maps/custom/solar/manifest.json" in line


@pytest.mark.parametrize("flag", ["ER_SWARM_RAYCAST", "ER_SWARM_THERMAL", "ER_SWARM_LOW_LIGHT",
                                  "ER_SWARM_NEAR_INFRARED", "ER_SWARM_DAYLIGHT", "VISUAL_SHAPE_GLASS_BACKED"])
def test_an_engine_without_a_flag_is_refused_by_name(complete, monkeypatch, flag):
    """An engine built without one of the flags the pictures rest on is refused, the flag named."""
    monkeypatch.delattr(p, flag)
    assert install.problems() == [f"swarm-bullet3 {install._installed('swarm-bullet3')} has no {flag}"]


def test_every_engine_flag_the_patrol_reads_is_checked():
    """Every engine flag Sentinel's code reads is one the install check requires, or changes no pixel; a flag added
    to the patrol later without being added here fails this test."""
    read = set()
    for folder in (_PACKAGE / "challenge_families" / "solar_patrol", _PACKAGE / "core" / "maps" / "solar"):
        for source in folder.glob("*.py"):
            if source.name != "install.py":
                read |= set(_FLAG_NAME.findall(source.read_text(encoding="utf-8")))
    assert read and read - _PIXEL_NEUTRAL_FLAGS <= set(install.ENGINE_FLAGS)


@pytest.mark.parametrize(("flags", "lacking"), [
    ("fpu sse4_2 avx fma bmi2", "avx2"),
    ("fpu sse4_2 avx avx2 bmi2", "fma"),
    ("fpu sse4_2 avx", "avx2 or fma"),
])
def test_a_cpu_without_avx2_or_fma_is_refused(complete, flags, lacking):
    """A CPU the ray caster cannot run on is refused, naming what it lacks."""
    Path(install.CPUINFO).write_text(f"processor\t: 0\nflags\t\t: {flags}\n")
    assert install.problems() == [f"the CPU has no {lacking}, which the ray caster needs"]


def test_a_cpu_whose_features_cannot_be_read_is_refused(complete, monkeypatch, tmp_path):
    """A CPU whose features cannot be read counts as lacking them: the patrol cannot be shown to draw its scene."""
    monkeypatch.setattr(install, "CPUINFO", str(tmp_path / "absent"))
    [line] = install.problems()
    assert line.startswith("the CPU's features cannot be read")


def test_every_missing_piece_is_named_at_once_by_an_error_that_is_not_a_seeds(no_thieves, monkeypatch):
    """The refusal names the files, the flags and the CPU together, and is not a RuntimeError: the seed checks keep
    a RuntimeError as the seed's own failure, and this one is the machine's."""
    monkeypatch.delattr(p, "ER_SWARM_THERMAL")
    Path(install.CPUINFO).write_text("flags\t\t: fpu avx fma\n")
    with pytest.raises(IncompleteInstall) as refused:
        install.require()
    message = str(refused.value)
    assert message.startswith("Swarm Sentinel cannot run on this install: ")
    assert all(part in message for part in ("intruders/intruders.json", "ER_SWARM_THERMAL", "no avx2"))
    assert not isinstance(refused.value, RuntimeError)


def test_a_patrol_environment_is_refused_before_anything_is_built(no_thieves):
    """Building a Sentinel environment on an incomplete install stops at the family's first hook, naming why."""
    with pytest.raises(IncompleteInstall, match="intruders/intruders.json"):
        make_env_with_initial_obs(solar_patrol_task(seed=11, sim_dt=SIM_DT))
    env = SimpleNamespace(task=SimpleNamespace(map_seed=11))
    with pytest.raises(IncompleteInstall):
        require_runtime_family(FAMILY_ID).initialise_env_state(env)
    assert not hasattr(env, "_solar")


def test_a_seed_check_on_an_incomplete_install_flies_nothing_and_keeps_no_verdict(no_thieves, kept, monkeypatch):
    """Judging a seed on an incomplete install starts no reference flight and keeps no verdict, so nothing the
    machine got wrong is ever held against the seed."""
    def no_flights(*args, **kwargs):
        """Fail the test if a reference flight is started."""
        raise AssertionError("no flight may start on an incomplete install")

    monkeypatch.setattr(seed_checks, "Flights", no_flights)
    with pytest.raises(IncompleteInstall):
        seed_checks.prepare([11], workers=1)
    with pytest.raises(IncompleteInstall):
        seed_checks.approve(solar_patrol_task(seed=11, sim_dt=SIM_DT))
    assert not kept.exists() or not any(kept.rglob("*.json"))


def test_after_the_install_is_repaired_the_seed_is_checked_again_and_passes(no_thieves, kept, monkeypatch, tmp_path):
    """The same seed refused on the broken install is judged by a real flight process once the missing file is back,
    and flies as itself."""
    (tmp_path / "passing_pilot.py").write_text(_PASSING_PILOT)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.environ.get("PYTHONPATH", "")]))
    monkeypatch.setattr(seed_checks, "PILOT_MODULE", "passing_pilot")
    task = solar_patrol_task(seed=11, sim_dt=SIM_DT)
    with pytest.raises(IncompleteInstall):
        seed_checks.approve(task)
    assert seed_checks.cached(11) is None
    (no_thieves / "maps" / "custom" / "solar" / "intruders" / "intruders.json").touch()
    assert seed_checks.approve(task) is task
    assert seed_checks.cached(11) == Verdict(seed=11, passed=True)


def test_the_pilot_lets_a_refused_install_through_and_keeps_nothing(kept, monkeypatch):
    """The reference pilot keeps a seed's own failure as its verdict, but raises a refused install and keeps
    nothing, so its process ends and the preparation fails instead."""
    def refused(seed):
        """A flight whose environment was refused for the machine's install."""
        raise IncompleteInstall("Swarm Sentinel cannot run on this install: swarm-bullet3 has no ER_SWARM_THERMAL")

    monkeypatch.setattr(reference_pilot, "check", refused)
    with pytest.raises(IncompleteInstall):
        reference_pilot.main(["11"])
    assert seed_checks.cached(11) is None


def test_verdicts_kept_before_this_check_existed_are_not_read(kept):
    """A failed verdict an incomplete install kept under the previous check version is never read again."""
    old = seed_checks.cache_dir().parent / f"{seed_checks.cache_dir().name.rsplit('-c', 1)[0]}-c2"
    old.mkdir(parents=True)
    (old / "11.json").write_text(json.dumps({"seed": 11, "passed": False,
                                             "reason": "error:the solar camera needs a swarm-bullet3 wheel"}))
    assert seed_checks.CHECK_VERSION > 2 and seed_checks.cached(11) is None


@pytest.mark.skipif(not install.problems(), reason="this machine's install can fly Swarm Sentinel")
def test_the_real_reference_pilot_on_this_incomplete_install_is_refused(tmp_path):
    """The real reference pilot, on this machine's own incomplete install, ends with the reason and keeps nothing."""
    run = subprocess.run([sys.executable, "-m", seed_checks.PILOT_MODULE, "11"], capture_output=True, text=True,
                         env={**os.environ, seed_checks.CACHE_ENV: str(tmp_path)}, timeout=300)
    assert run.returncode != 0
    assert "IncompleteInstall: Swarm Sentinel cannot run on this install" in run.stderr
    assert not any(tmp_path.rglob("*.json"))


def _validator():
    """The validator stand-in run_task and the seed preparation read."""
    manager = SimpleNamespace(epoch_number=5, get_all_seeds=lambda family_id=None, epoch=None: [11, 12])
    return SimpleNamespace(seed_manager=manager)


@pytest.mark.asyncio
async def test_a_sentinel_task_is_refused_before_its_model_is_fetched(no_thieves, monkeypatch):
    """A Sentinel task on an incomplete install is refused with the reason before the model is fetched."""
    async def fetch(_self, models):
        """Fail the test if the model is fetched."""
        raise AssertionError("a refused task must fetch nothing")

    monkeypatch.setattr(run_task_module, "_ensure_models_from_backend", fetch)
    with pytest.raises(IncompleteInstall, match="intruders/intruders.json"):
        await run_task_module.run_task(
            _validator(), {"task_id": 1, "uid": 42, "phase": "SCREENING", "family_id": FAMILY_ID,
                           "model_hash": "0" * 64},
            cancel_flag=asyncio.Event(), wake_flag=asyncio.Event())


def test_the_reason_is_logged_when_found_and_then_at_most_every_interval(no_thieves, monkeypatch):
    """The refusal says why the first time, stays quiet while the same refusal repeats, and says it again once the
    interval has passed; a family the install can run is never refused."""
    logged, now = [], [1000.0]
    monkeypatch.setattr(run_task_module.bt.logging, "error", logged.append)
    monkeypatch.setattr(run_task_module.time, "monotonic", lambda: now[0])
    validator = _validator()
    for _ in range(3):
        assert "intruders/intruders.json" in run_task_module.install_refusal(validator, FAMILY_ID)
    assert len(logged) == 1 and "Refusing cf_solar_patrol tasks" in logged[0] and "intruders.json" in logged[0]
    now[0] += INSTALL_REFUSAL_LOG_SEC
    run_task_module.install_refusal(validator, FAMILY_ID)
    assert len(logged) == 2
    assert run_task_module.install_refusal(validator, "cf_autopilot") == ""


def test_seeds_are_not_judged_on_an_incomplete_install(no_thieves, monkeypatch):
    """The background preparation is not started for a family the install cannot run, so no flight is spent and no
    verdict kept on the broken install."""
    monkeypatch.setattr(run_task_module.bt.logging, "error", lambda message: None)
    monkeypatch.setattr(require_runtime_family(FAMILY_ID), "seeds_prepared",
                        lambda seeds: pytest.fail("a refused family's seeds must not be looked at"))
    validator = _validator()
    run_task_module.start_seed_preparations(validator, 5)
    assert not validator.__dict__.get("_seed_preparations")


@pytest.mark.asyncio
async def test_a_refused_task_makes_the_loop_wait_instead_of_asking_again(monkeypatch):
    """When a task is refused for the install, the loop pauses for the refusal's interval before asking the backend
    again, skips the cleanup of a task that never ran, and raises nothing."""
    idled, cleaned = [], []

    async def skipped(*args, **kwargs):
        """Stand in for a step of the loop this test does not exercise."""

    async def may_score(_self):
        """The host is calibrated and its image verified."""
        return True

    async def sync():
        """A sync that changes nothing."""
        return {}

    async def next_task():
        """The backend hands out a Sentinel task."""
        return {"task_id": 1, "uid": 42, "phase": "SCREENING", "family_id": FAMILY_ID}

    async def refuse(_self, leased, **kwargs):
        """The install cannot run the task."""
        raise IncompleteInstall("Swarm Sentinel cannot run on this install: the CPU has no avx2")

    async def idle(_self, timeout):
        """Record how long the loop waits."""
        idled.append(timeout)

    async def cleanup(_self, **kwargs):
        """Record a cleanup."""
        cleaned.append(kwargs)

    monkeypatch.setattr(forward_mod, "_ensure_components", skipped)
    monkeypatch.setattr(forward_mod, "_publish_pending_epoch_seeds", skipped)
    monkeypatch.setattr(forward_mod, "_cleanup_off_loop", cleanup)
    monkeypatch.setattr(forward_mod, "_host_may_score", may_score)
    monkeypatch.setattr(forward_mod, "_idle_until_wake", idle)
    monkeypatch.setattr(forward_mod, "start_seed_preparations", lambda _self, epoch: None)
    monkeypatch.setattr(forward_mod, "run_task", refuse)
    validator = _validator()
    validator.seed_manager.seeds_ready = lambda: True
    validator.backend_api = SimpleNamespace(sync=sync, next_task=next_task)
    validator.metagraph = None
    validator._sse_fatal = None
    validator._cancel_flag, validator._wake_flag = asyncio.Event(), asyncio.Event()
    await forward_mod._forward_iteration(validator)
    assert idled == [INSTALL_REFUSAL_IDLE_SEC] and cleaned == []
