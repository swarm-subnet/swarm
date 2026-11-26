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

"""Swarm Test Field v2 reference models: cheap strategies and a perfect one, each flown as a whole patrol.

A strategy that is cheap to build must not score well, and a perfect one must score close to 1. Each pilot flies one
seed through the family's own runtime, the way a validator does: the seed checks approve it, the field is built, the
pilot decides ten times a second, and the last step's outcome is scored by the family's evaluate_rollout. The pilots
that search read the simulator's own object map of every frame the model is shown, so their eyes are perfect; what
they prove is what the rules accept, not how well a real detector would do.

    silent       the reference search, never a report
    everything   the reference search, a report on every new frame it is shown, the whole frame boxed and the point
                 under the drone given, until the patrol's 20 reports are spent
    boxes_only   the reference search, each person reported once with a perfect box but the point under the drone
                 for his position: what the position rule turns away
    hover        takes off, hovers over the dock, returns home with 40 s left
    random       uniform random actions
    starter      the starter agent, miner/src/submission_template/test_field_v2_drone_agent.py
    warm_bodies  the reference search, every warm body reported once with its box and where it stands: each person
                 and each animal, the people first; what reporting animals costs
    perfect      the seed checks' reference pilot: the search, each person reported once with his box and position
    all_people   the perfect pilot with eyes that take the people passing outside the field for people to find too: it
                 also flies to each one in the scene once the field's people are reported, and reports him

    python validator/scripts/field_v2_reference_models.py fly <pilot> <out.jsonl> <seed> [<seed> ...]
    python validator/scripts/field_v2_reference_models.py tally <out.jsonl> [<out.jsonl> ...]

fly flies the seeds one after another in one process and appends one line per seed, with the process's peak memory
so far; tally gives each pilot's mean seed score, people reported, false alarms, with those on people outside the
field counted apart, and the mean once the cross-seed rule has zeroed the best seeds for every mistake.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from swarm.benchmark.engine_parts.reporting import _cross_seed_penalty
from swarm.challenge_families.solar_patrol import coverage, sensor_noise
from swarm.challenge_families.solar_patrol import reports as sentinel_reports
from swarm.challenge_families.test_field_v2 import animals, outsiders, people, reference_pilot, reports
from swarm.challenge_families.test_field_v2.contract import ACTION_DIM, ACTION_HIGH, ACTION_INDEX, ACTION_LOW, FAMILY_ID
from swarm.challenge_families.test_field_v2.family import TestFieldV2ChallengeFamily
from swarm.constants import SIM_DT
from swarm.domain_model import get_family_seed_mistake_fields, get_family_seeds_zeroed_per_mistake
from swarm.utils.env_factory import make_env_with_initial_obs

STARTER = Path(__file__).resolve().parents[2] / "miner" / "src" / "submission_template" / "test_field_v2_drone_agent.py"
WHOLE_FRAME = (0.5, 0.5, 1.0, 1.0)
HOVER_HOME_LEFT_S = 40.0


class Silent(reference_pilot.Pilot):
    """The reference search, never a report."""

    def __init__(self, env: Any):
        """The search without visits: with nothing to report there is nobody to fly over."""
        super().__init__(env, visit=False)

    def action(self, decision: int, obs: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """The search's flight alone."""
        return self.search.action(decision)


class Everything(Silent):
    """The reference search, a report on every new frame it is shown, the whole frame boxed and the point under the
    drone given."""

    def action(self, decision: int, obs: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """The search's flight, and a report whenever the frame shown is new and the button was not held."""
        a = self.search.action(decision)
        view = reference_pilot.sensor_noise.shown_view(self.ep, "feed")
        if self.ep.phase == "flying" and view is not None and view is not self.eyes.last and decision > self.pressed + 1:
            self.eyes.last = view
            self.press(a, WHOLE_FRAME, self.under())
            self.pressed = decision
        return a

    def under(self) -> np.ndarray:
        """The point under the drone, metres east and north of the dock."""
        return np.asarray(self.env.pos[0][:2], dtype=float) - np.asarray(self.ep.dock_position[:2], dtype=float)


class BoxesOnly(reference_pilot.Pilot):
    """The reference pilot, each person reported with a perfect box but the point under the drone for his position."""

    def point(self, index: int, step: Optional[int] = None) -> np.ndarray:
        """The point under the drone, metres east and north of the dock, whoever the report is about and whenever."""
        return np.asarray(self.env.pos[0][:2], dtype=float) - np.asarray(self.ep.dock_position[:2], dtype=float)


class AllPeopleSearch(reference_pilot.Search):
    """The reference search, which once the people on the field are reported also visits each person passing outside
    it while he is in the scene, from the free point nearest him."""

    def __init__(self, env: Any, ep: Any, reported: set, visit: bool = True):
        """The reference search, and who passes outside the field."""
        super().__init__(env, ep, reported, visit)
        self.outside = [person.index for person in people.people(ep) if not person.inside]
        self.tried.update(dict.fromkeys(self.outside, 0))

    def goal(self, n: int, pos: np.ndarray) -> Optional[tuple]:
        """A person on the field's next place, or for a passer the free point nearest him, tried once."""
        if n < len(self.persons):
            return super().goal(n, pos)
        spot = None if self.tried[n] else self.viewpoint(n, pos)
        return None if spot is None else (spot, None)

    def _next_person(self, pos: np.ndarray) -> Optional[int]:
        """The nearest person on the field still to report, else the nearest passer in the scene still to report."""
        n = super()._next_person(pos)
        if n is not None:
            return n
        everyone = people.people(self.ep)
        left = [i for i in self.outside if i not in self.reported and i not in self.skipped
                and outsiders.in_scene(self.ep, everyone[i].passer)]
        return min(left, key=lambda i: float(np.hypot(*(people.position(self.ep, i)[:2] - pos[:2])))) if left else None

    def viewpoint(self, n: int, pos: np.ndarray) -> Optional[np.ndarray]:
        """A person on the field's viewpoint, or for a passer the free point nearest where he is now."""
        if n < len(self.persons):
            return super().viewpoint(n, pos)
        cell = self.free.nearest_free(people.position(self.ep, n)[:2])
        return None if cell is None else self.free.point(cell)


class AllPeople(reference_pilot.Pilot):
    """The perfect pilot with eyes that take everyone for a person to find, and a search that looks for the people
    passing outside the field too."""

    def __init__(self, env: Any):
        """The reference pilot with the wider search and eyes."""
        super().__init__(env)
        self.search = AllPeopleSearch(env, self.ep, self.reported)
        self.eyes = reference_pilot.Eyes(env, self.ep, outside=True)


class Hover:
    """Takes off, hovers over the dock, and presses return home with HOVER_HOME_LEFT_S left."""

    def __init__(self, env: Any):
        """The flight."""
        self.env, self.ep = env, env._solar

    def action(self, decision: int, obs: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """Take off, hold over the dock, press return home with time to land."""
        a = np.zeros(ACTION_DIM, dtype=np.float32)
        if decision == 0:
            a[ACTION_INDEX["take_off"]] = 1.0
        elif self.ep.phase == "flying" and self.ep.time_s > float(self.env.EP_LEN_SEC) - HOVER_HOME_LEFT_S:
            a[ACTION_INDEX["return_home"]] = 1.0
        return a


class RandomPilot:
    """Uniform random actions inside the contract's bounds."""

    def __init__(self, env: Any):
        """A generator of the seed's own."""
        self.rng = np.random.default_rng(int(env._solar.seed))

    def action(self, decision: int, obs: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """One random action."""
        return self.rng.uniform(ACTION_LOW, ACTION_HIGH).astype(np.float32)


class Starter:
    """The starter agent, loaded from its file as a miner's zip would hold it."""

    def __init__(self, env: Any):
        """A fresh controller."""
        spec = importlib.util.spec_from_file_location("test_field_starter", STARTER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.agent = module.DroneFlightController()

    def action(self, decision: int, obs: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """The agent's own decision on the observation."""
        return np.asarray(self.agent.act(obs), dtype=np.float32)


class WarmBodies(reference_pilot.Pilot):
    """The reference pilot's search and perfect eyes, reporting every warm body a frame shows once, the people before
    the animals, each with its tight box and where it stood when the frame was taken."""

    def __init__(self, env: Any):
        """The search, and nothing reported yet."""
        super().__init__(env)
        self.warm: Dict[tuple, tuple] = {}
        self.told: set = set()

    def action(self, decision: int, obs: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """The search's flight, and a report whenever a body waits to be reported on the frame shown."""
        a = self.search.action(decision)
        if self.ep.phase not in ("flying", "returning", "landing"):
            return a
        view = sensor_noise.shown_view(self.ep, "feed")
        if view is not None and view is not self.eyes.last and view.objects is not None and view.sees:
            self.eyes.last = view
            if coverage.height_above_ground(self.env, self.ep, view.eye) <= reports.MAX_REPORT_HEIGHT_M:
                boxes = sentinel_reports.object_boxes(view, reports.candidates(self.ep))
                self.warm = {key: (reference_pilot.box_shares(pixels, view), int(view.step))
                             for key, pixels in boxes.items() if key not in self.told}
        if decision > self.pressed + 1 and self.warm:
            key = min(self.warm, key=lambda k: (k[0] != "person", k[1]))
            box, step = self.warm.pop(key)
            self.press(a, box, self.where(key, step))
            self.told.add(key)
            if key[0] == "person":
                self.reported.add(key[1])
            self.pressed = decision
        return a

    def where(self, key: tuple, step: int) -> np.ndarray:
        """Where a body stood at a step, metres east and north of the dock."""
        if key[0] == "person":
            return people.reference(self.ep, key[1], step)
        row = [a.uid for a in self.ep.animals.members].index(key[1])
        return animals.positions(self.ep, step)[row, :2] - np.asarray(self.ep.dock_position[:2], dtype=float)


PILOTS = {"silent": Silent, "everything": Everything, "boxes_only": BoxesOnly, "hover": Hover, "random": RandomPilot,
          "starter": Starter, "perfect": reference_pilot.Pilot, "warm_bodies": WarmBodies, "all_people": AllPeople}


def fly(seed: int, pilot_name: str, family: Optional[TestFieldV2ChallengeFamily] = None) -> Dict[str, Any]:
    """Approve, build, fly and score one seed with one pilot."""
    family = family or TestFieldV2ChallengeFamily()
    started = time.perf_counter()
    task = family.build_random_task(sim_dt=SIM_DT, seed=seed)
    with contextlib.redirect_stdout(io.StringIO()):
        env, obs = make_env_with_initial_obs(task)
    build_s = time.perf_counter() - started
    try:
        pilot = PILOTS[pilot_name](env)
        decision = 0
        while True:
            obs, _reward, terminated, truncated, info = env.step(np.asarray(pilot.action(decision, obs))[None, :])
            decision += 1
            if terminated or truncated:
                break
        ep = env._solar
        t_sim = float(info.get("t", env._time_alive))
        evaluation = family.evaluate_rollout(task=task, success=bool(info.get("success", False)), t=t_sim,
                                             horizon=float(task.horizon), min_clearance=None, collision=False,
                                             failure_reason=str(info.get("failure_reason", "NONE")), info=info)
        verdicts: Dict[str, int] = defaultdict(int)
        for _step, _report, verdict in ep.reports["log"]:
            verdicts[verdict] += 1
        return {"seed": seed, "flown_seed": int(task.map_seed), "pilot": pilot_name, "score": float(evaluation.score),
                "terms": {k: float(v) for k, v in evaluation.normalized_metrics.items()},
                "outcome": {k: v for k, v in evaluation.metrics.items() if k in ep.outcome.__dataclass_fields__},
                "poses": [person.pose for person in people.people(ep)], "verdicts": dict(verdicts),
                "passers": len(people.passers(ep)),
                "time_s": round(t_sim, 2), "decisions": decision, "build_s": round(build_s, 1),
                "wall_s": round(time.perf_counter() - started, 1),
                # The process's peak so far: one process flies its seeds one after another.
                "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)}
    finally:
        env.close()


def tally(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Each pilot's seeds, mean seed score, people reported out of those on the field, false alarms and those of them
    on people outside the field, how its patrols ended, and its mean once every mistake has zeroed its share of the
    best seeds."""
    by_pilot: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_pilot[row["pilot"]].append(row)
    table = {}
    for name, rows in by_pilot.items():
        ends: Dict[str, int] = defaultdict(int)
        for row in rows:
            ends[row["outcome"]["end_reason"] or "none"] += 1
        table[name] = {"seeds": len(rows), "mean_score": round(float(np.mean([r["score"] for r in rows])), 4),
                       "people": sum(int(r["outcome"]["threats"]) for r in rows),
                       "reported": sum(int(r["outcome"]["valid_reports"]) for r in rows),
                       "false_alarms": sum(int(r["outcome"]["false_alarms"]) for r in rows),
                       "outside_reports": sum(int(r["outcome"].get("outside_reports", 0)) for r in rows),
                       "ends": dict(ends), "penalised_score": round(_penalised(rows), 4)}
    return table


def _penalised(rows: List[Dict[str, Any]]) -> float:
    """A pilot's mean seed score once the cross-seed rule has zeroed its best seeds for every mistake, as the backend
    does."""
    seeds = [dict(r["outcome"], score=r["score"]) for r in rows]
    fields = get_family_seed_mistake_fields(FAMILY_ID)
    return float(_cross_seed_penalty(seeds, get_family_seeds_zeroed_per_mistake(FAMILY_ID), fields)["penalised_score"])


def main(argv: List[str]) -> None:
    """Fly seeds with a pilot, appending a line per seed, or tally the lines of one or more runs."""
    if argv[0] == "fly":
        pilot_name, out, seeds = argv[1], argv[2], [int(s) for s in argv[3:]]
        family = TestFieldV2ChallengeFamily()
        for seed in seeds:
            result = fly(seed, pilot_name, family)
            with open(out, "a") as handle:
                handle.write(json.dumps(result) + "\n")
            print(json.dumps({k: result[k] for k in ("seed", "flown_seed", "pilot", "score", "time_s", "wall_s")}), flush=True)
    else:
        rows = [json.loads(line) for path in argv[1:] for line in open(path) if line.strip()]
        print(json.dumps(tally(rows), indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
