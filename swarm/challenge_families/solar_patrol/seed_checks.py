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

"""Seed checks (task 24): no seed is scored that a pilot could not finish.

Every seed is flown once by the reference pilot (reference_pilot.py) before it is scored. The seed passes when every
thief who stepped inside the fence was seen well enough at least once (check 1) and the pilot landed back in the
dock before the patrol's time ran out (check 2). A seed that fails is replaced by the next candidate drawn from it,
the same on every machine, so the epoch's shared list stays one list.

A verdict costs a whole flight, so each is worked out once per benchmark version and kept on disk. The flights run in
processes of their own: the environment they fly imports the family registry, which imports this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from swarm.constants import BENCHMARK_VERSION
from swarm.protocol import MapTask

from .task import solar_patrol_task

CHECK_VERSION = 1                      # part of the kept verdicts' folder: a change to the checks is a new version
CHECK_TRIES = 16                       # candidates tried per seed; at any real drop rate the last is never reached
REPLACEMENT_TAG = "solar-seed-check"
CACHE_ENV = "SWARM_SEED_CHECK_DIR"
CACHE_DEFAULT = Path.home() / ".cache" / "swarm" / "seed_checks"
WORKERS_ENV = "SWARM_SEED_CHECK_WORKERS"
CPU_SHARE = 4                          # by default the flights take a quarter of the machine's cores
PILOT_MODULE = "swarm.challenge_families.solar_patrol.reference_pilot"
# One BLAS and OpenMP thread in every reference flight, as in the validator image, whatever the host sets.
THREAD_CAPS = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


@dataclass
class Verdict:
    """What one seed's check found: whether it passed, why not, and the numbers behind it."""

    seed: int
    passed: bool
    reason: str = ""                   # empty when passed, else what failed first
    end_reason: str = ""
    landed_s: Optional[float] = None   # patrol time at the landing
    threats: int = 0                   # thieves who stepped inside the fence
    seen: int = 0                      # of them, seen well enough at least once
    best_px: List[float] = field(default_factory=list)  # each thief's best pixels across, seen or not
    night: bool = False
    wind: str = ""
    route_m: float = 0.0
    cpu_s: float = 0.0


class SeedCheckError(RuntimeError):
    """A seed could not be judged, or no candidate drawn from it passed."""


def candidates(seed: int) -> Iterator[int]:
    """The seed itself, then the replacements drawn from it in order, each a 32-bit seed."""
    yield int(seed)
    for attempt in range(1, CHECK_TRIES):
        digest = hashlib.sha256(f"{REPLACEMENT_TAG}|{int(seed)}|{attempt}".encode("utf-8")).digest()
        yield int.from_bytes(digest[:4], "big")


def workers() -> int:
    """How many reference flights run side by side: the machine's setting, else a share of its cores."""
    return max(1, int(os.environ.get(WORKERS_ENV) or (os.cpu_count() or 1) // CPU_SHARE))


def cache_dir() -> Path:
    """Where this version's verdicts are kept."""
    return Path(os.environ.get(CACHE_ENV) or CACHE_DEFAULT) / f"{BENCHMARK_VERSION}-c{CHECK_VERSION}"


def cached(seed: int) -> Optional[Verdict]:
    """The kept verdict of a seed, or None when it has not been checked under this version."""
    try:
        return Verdict(**json.loads((cache_dir() / f"{int(seed)}.json").read_text()))
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        return None


def keep(result: Verdict) -> None:
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


def prepare(seeds: Iterable[int], workers: int) -> Dict[int, int]:
    """Judge a list of seeds on parallel flights, following each seed's replacements only as far as it needs, and
    return what every list seed is flown as."""
    chains = {int(seed): candidates(seed) for seed in seeds}
    heads = {seed: next(chain) for seed, chain in chains.items()}
    flown: Dict[int, int] = {}
    while heads:
        fly([head for head in heads.values() if cached(head) is None], workers)
        following = {}
        for seed, head in heads.items():
            found = cached(head)
            if found is None:
                raise SeedCheckError(f"seed {head} was flown but left no verdict")
            if found.passed:
                flown[seed] = head
                continue
            nxt = next(chains[seed], None)
            if nxt is None:
                raise SeedCheckError(f"no candidate of seed {seed} passed the seed checks")
            following[seed] = nxt
        heads = following
    return flown


def fly(seeds: List[int], workers: int) -> None:
    """Fly the reference pilot over the seeds, split across parallel processes, each keeping its verdicts."""
    shares = [seeds[i::max(1, int(workers))] for i in range(max(1, int(workers)))]
    runs = [subprocess.Popen([sys.executable, "-m", PILOT_MODULE, *map(str, share)], stdout=subprocess.DEVNULL,
                             stderr=subprocess.PIPE, env={**os.environ, **THREAD_CAPS}) for share in shares if share]
    failures = []
    for run in runs:
        _out, err = run.communicate()
        if run.returncode != 0:
            failures.append(err.decode("utf-8", "replace")[-2000:])
    if failures:
        raise SeedCheckError("reference flights failed:\n" + "\n".join(failures))


def approve(task: MapTask) -> MapTask:
    """The task to fly for this seed: the task itself when it passes, its replacement when it does not. A seed not
    judged yet is flown by the reference pilot first."""
    seed = int(task.map_seed)
    flown = replacement(seed)
    if flown is None:
        flown = prepare([seed], workers=1)[seed]
    return task if flown == seed else solar_patrol_task(seed=flown, sim_dt=task.sim_dt)
