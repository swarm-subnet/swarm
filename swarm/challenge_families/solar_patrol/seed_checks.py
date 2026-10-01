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

import contextlib
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Dict, Iterable, Iterator, List, Optional

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
POLL_S = 0.5                           # how often a busy flight's kept verdict is looked for


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
    return what every list seed is flown as. A failed candidate's replacement goes to the next free flight at once,
    so no flight waits for the others to finish a round."""
    chains = {int(seed): candidates(seed) for seed in seeds}
    heads = {seed: next(chain) for seed, chain in chains.items()}
    flown: Dict[int, int] = {}
    with Flights(workers) as flights:
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


class Flights:
    """Reference flights in processes of their own, at most workers at a time. Each process judges the seeds it is
    handed one after another and keeps their verdicts, so only its first world is built cold."""

    def __init__(self, workers: int):
        """No process yet: one starts when a seed finds every running one busy."""
        self.workers = max(1, int(workers))
        self.runs: Dict[subprocess.Popen, Optional[int]] = {}   # each process and the seed it is judging
        self.logs: Dict[subprocess.Popen, IO[bytes]] = {}

    def __enter__(self) -> "Flights":
        """The flights, for the length of a preparation."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Let every process end once it has no more seeds, or stop them all when the preparation failed."""
        for run in self.runs:
            run.stdin.close()
            if exc[0] is not None:
                run.kill()
            run.wait()
            self.logs[run].close()

    def fly(self, seeds: List[int]) -> None:
        """Hand the seeds not being judged yet to free processes, then wait until a seed being judged has its
        verdict. A process that ends while it still holds a seed fails the preparation."""
        for run, seed in self.runs.items():
            if seed is not None and cached(seed) is not None:
                self.runs[run] = None
        flying = set(self.runs.values())
        for seed in dict.fromkeys(seeds):
            if seed in flying:
                continue
            free = next((run for run, held in self.runs.items() if held is None), None)
            if free is None and len(self.runs) < self.workers:
                log = tempfile.TemporaryFile()
                free = subprocess.Popen([sys.executable, "-m", PILOT_MODULE], stdin=subprocess.PIPE, bufsize=0,
                                        stdout=subprocess.DEVNULL, stderr=log)
                self.logs[free] = log
            if free is None:
                break
            with contextlib.suppress(BrokenPipeError):  # a process that has ended is caught below, holding the seed
                free.stdin.write(f"{seed}\n".encode())
            self.runs[free] = seed
        busy = {run: seed for run, seed in self.runs.items() if seed is not None}
        while busy and all(cached(seed) is None for seed in busy.values()):
            for run, seed in busy.items():
                if run.poll() is not None and cached(seed) is None:
                    self.logs[run].seek(0)
                    raise SeedCheckError(f"seed {seed} was flown but left no verdict:\n"
                                         + self.logs[run].read().decode("utf-8", "replace")[-2000:])
            time.sleep(POLL_S)


def approve(task: MapTask) -> MapTask:
    """The task to fly for this seed: the task itself when it passes, its replacement when it does not. A seed not
    judged yet is flown by the reference pilot first."""
    seed = int(task.map_seed)
    flown = replacement(seed)
    if flown is None:
        flown = prepare([seed], workers=1)[seed]
    return task if flown == seed else solar_patrol_task(seed=flown, sim_dt=task.sim_dt)
