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

"""Seed checks: no seed is scored that a pilot could not finish.

Every seed is flown once by the reference pilot (reference_pilot.py), a pilot that knows where everyone is, before it
is scored. The seed passes when the pilot reported every person on the field with a valid report, made no false
alarm, never tripped the edge rule or hit anything, and landed back in the dock before the patrol's time ran out. A
person no legal viewpoint shows, under a tree crown or behind a wall, fails the seed. A seed that fails is replaced by
the next candidate drawn from it, the same on every machine, so the epoch's shared list stays one list.

The admission and replacement mechanics are Sentinel's (solar's seed_checks module): verdicts are worked out once per
benchmark version, kept on disk beside Sentinel's under the family's own folder, and flown in processes of their own.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

from swarm.domain_model import get_family_benchmark_version
from swarm.protocol import MapTask

from ..solar_patrol import seed_checks as sentinel_checks
from ..solar_patrol.seed_checks import CACHE_DEFAULT, CACHE_ENV, SeedCheckError, Verdict
from . import install
from .contract import FAMILY_ID
from .task import test_field_task

# 3 drops the verdicts flown on the site before its rebuild.
CHECK_VERSION = 3
CHECK_TRIES = sentinel_checks.CHECK_TRIES
REPLACEMENT_TAG = "test-field-seed-check"
PILOT_MODULE = "swarm.challenge_families.test_field.reference_pilot"

BACKGROUND_NICENESS = sentinel_checks.BACKGROUND_NICENESS
workers = sentinel_checks.workers


@dataclass
class FieldVerdict(Verdict):
    """Sentinel's verdict, where threats counts the people on the field and seen the ones validly reported, plus the
    pilot's false alarms, the closest it came to the field's edge and to a crown, and its peak above the dock."""

    false_alarms: int = 0
    min_edge_m: Optional[float] = None
    min_crown_m: Optional[float] = None
    peak_above_dock_m: Optional[float] = None


def candidates(seed: int) -> Iterator[int]:
    """The seed itself, then the replacements drawn from it in order: 32-bit seeds for a list seed, and as wide as
    the seed for a world seed, so a replaced world is no easier to find than the one it stands in for."""
    yield int(seed)
    width = 4 if int(seed) < 2**32 else 16
    for attempt in range(1, CHECK_TRIES):
        digest = hashlib.sha256(f"{REPLACEMENT_TAG}|{int(seed)}|{attempt}".encode("utf-8")).digest()
        yield int.from_bytes(digest[:width], "big")


def cache_dir() -> Path:
    """Where this version's verdicts are kept."""
    base = Path(os.environ.get(CACHE_ENV) or CACHE_DEFAULT)
    return base / f"{FAMILY_ID}-{get_family_benchmark_version(FAMILY_ID)}-c{CHECK_VERSION}"


def cached(seed: int) -> Optional[FieldVerdict]:
    """The kept verdict of a seed, or None when it has not been checked under this version."""
    try:
        return FieldVerdict(**json.loads((cache_dir() / f"{int(seed)}.json").read_text()))
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        return None


def keep(result: FieldVerdict) -> None:
    """Keep a verdict, written whole or not at all."""
    directory = cache_dir()
    directory.mkdir(parents=True, exist_ok=True)
    tmp = directory / f".{result.seed}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(asdict(result)))
    tmp.replace(directory / f"{result.seed}.json")


def replacement(seed: int) -> Optional[int]:
    """The seed a list's seed is flown as, from the kept verdicts alone, or None while one it needs is missing."""
    for candidate in candidates(seed):
        found = cached(candidate)
        if found is None:
            return None
        if found.passed:
            return candidate
    raise SeedCheckError(f"no candidate of seed {seed} passed the seed checks")


def prepare(seeds: Iterable[int], workers: int, *, niceness: int = 0) -> Dict[int, int]:
    """Judge a list of seeds on parallel flights at the given CPU niceness, following each seed's replacements only as
    far as it needs, and return what every list seed is flown as. An install that cannot fly the field flies nothing
    and keeps no verdict: what fails there is the machine, not the seed."""
    install.require()
    chains = {int(seed): candidates(seed) for seed in seeds}
    heads = {seed: next(chain) for seed, chain in chains.items()}
    flown: Dict[int, int] = {}
    with sentinel_checks.Flights(workers, niceness, pilot=PILOT_MODULE, verdict=cached) as flights:
        while heads:
            for seed, head in list(heads.items()):
                found = cached(head)
                if found is None:
                    continue
                if found.passed:
                    flown[seed] = heads.pop(seed)
                    continue
                nxt = next(chains[seed], None)
                if nxt is None:
                    raise SeedCheckError(f"no candidate of seed {seed} passed the seed checks")
                heads[seed] = nxt
            flights.fly([head for head in heads.values() if cached(head) is None])
    return flown


def approve(task: MapTask, *, judge: bool = True) -> MapTask:
    """The task to fly for this seed: the task itself when it passes, its replacement when it does not. A seed not
    judged yet is flown by the reference pilot first, or with judge off comes back as it is."""
    seed = int(task.map_seed)
    flown = replacement(seed)
    if flown is None and not judge:
        return task
    if flown is None:
        flown = prepare([seed], workers=1)[seed]
    return task if flown == seed else test_field_task(seed=flown, sim_dt=task.sim_dt)
