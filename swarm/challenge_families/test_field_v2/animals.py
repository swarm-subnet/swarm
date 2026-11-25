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

"""Animals near the field: bodies a model sees and must not report.

In the seeds whose mix has animals, the seed draws which kinds come (dogs in about seven of ten such seeds, a few
goats in about a third, at least one kind) and puts them on the field or round it, up to BAND_M outside its edge. Goats
come only to a place where a herd makes sense. A field is never crowded: a herd is 2 to 4 goats of different coats,
sizes and paces whose middle drifts slowly across open ground; dogs are 1 or 2 of different breeds, each roaming,
sniffing or resting by its mood. At night the herd lies close together the whole patrol; dogs keep their moods.

In the seeds whose mix has birds, 1 to 4 birds of the place's kinds come down on the field or round it, by day only:
they roost at night. Pigeons and choughs often come together. A bird pecks, looks about and walks or hops a little on
open ground under open sky, and every so often flies low to another spot; it takes off when a dog or goat is about to
come close, never for the drone. Birds are planned after the goats and dogs, so these move the same with birds or
without them.

In the seeds whose mix has boars, a few wild boars root across open ground where the place is wild enough for them:
one alone, two together, or a sow with her striped piglets, which keep close to her. In the seeds whose mix has
rodents, 1 to 3 rats come to a town, a village or a river bank, and 1 or 2 mice anywhere else: each keeps to short
runs near where it started, sniffing, rearing up and washing. Both are rare by day and come to one night in five, and
both stay up at night. They are planned together with the goats and dogs, so every animal on the ground keeps clear
of every other, and a seed whose mix holds none of them is planned the same with or without them, because they have
their own
streams and plan after the goats and dogs; a bird keeps clear of them as of any animal.

The ground they may use is found with rays from the sky on a GRID_M grid: where the terrain is met first, off the
dock, off every person's ground (a walker's whole patch, a passer-by's whole lane), off every trunk and house the
survey lists, in pieces big enough to live in. Rays meet only solid shapes, so a site that knows more (open sky,
paths, what lies beyond its solid shapes) hands it over as `open_at` in its park: a function of east and north
arrays that says where an animal may stand. Every animal's patrol is planned at reset on that ground (the
animal_moves module), so where it stands and what pose it holds is a function of time alone, and a patrol can be
judged at any past step.

An animal has no collision shape, never ends a patrol, and the 5 m safe-distance rule ignores it. Its one drawn body is
moved only
on a step a frame is taken that can see it, where it is or where it was last drawn, on the ground or in the air, and
rewritten only when its new pose would move on that frame by STILL_PIXELS or more.

Every animal on the ground lives on one side of the field's edge for the whole patrol: on the field, INSIDE_CLEAR_M
and more from the edge, or off it, OUTSIDE_CLEAR_M and more away; only a bird crosses it. The goats, the dogs and
the grown boars that live on the field are the seed's animals to report (the reports module); on the field an
animal keeps out from under the tree crowns, so one to report shows from above. A patrol holds at most
MOST_TO_REPORT people and animals to report, and a group that would pass that lives off the field. A seed
holds at most MOST_ANIMALS animals in all: its goats and dogs, then as many of its birds as fit, then its boars and
then its rats or mice, each of those two whole or not at all. None of them is a person to find: a report that calls
one a person is a false alarm, judged `animal`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pybullet as p
from scipy import ndimage

from swarm.constants import SIM_DT

from ..solar_patrol import camera, laser, zoom
from . import animal_skins as skins
from . import mix, people, site
from .animal_moves import COMPASS, DOG_MOODS, MIN_LAUNCHES, Bird, Dog, Ground, Herd, Mover, Rodent, Shadow, Track, pick, place_start, plan_movers, uniform
from .episode import FieldEpisode

ANIMALS_SEED_STREAM = 0xA41A           # the animals' own stream, so no other draw of the seed moves them
KIND_SHARES = (("dogs", 0.65), ("goats", 0.28), ("both", 0.07))   # dogs in 72 %, goats in 35 %, never neither
DOG_COUNT = ((1, 0.70), (2, 0.30))     # a field is never crowded: one dog, sometimes two
HERD_SIZE = (2, 4)                     # a few goats, not a flock
# Kinds of place a herd of goats is kept on; a field with no kind given, the surveyed meadow, is one.
HERD_PLACES = frozenset({"alpine", "forest", "lake", "river", "scree", "ski", "valley", "village"})
BIRDS_SEED_STREAM = 0xB1D5             # the birds' own stream, so adding them moves no other animal's draw
BIRD_COUNT = ((1, 0.40), (2, 0.30), (3, 0.20), (4, 0.10))   # a few birds, never a crowd
# The birds of each kind of place, and how often each comes; a field with no kind given takes the valley's. Every bird
# named here must be one the skins draw (animal_skins.DRAWN).
BIRD_PLACES = {
    **dict.fromkeys(("town", "village"), (("pigeon", 0.60), ("magpie", 0.20), ("crow", 0.20))),
    **dict.fromkeys(("forest", "valley", "river"), (("woodpigeon", 0.55), ("magpie", 0.25), ("crow", 0.20))),
    **dict.fromkeys(("alpine", "scree", "ski", "lake"), (("chough", 0.70), ("crow", 0.30))),
}
BIRD_PLACE_DEFAULT = "valley"
FLOCKING = frozenset({"pigeon", "chough"})   # birds that often come down together
FLOCK_SHARE = 0.7                      # each further bird joins a flocking first bird this often
FLOCK_M = 5.0                          # a bird that joins starts this close to the first
FLOCK_GAP_M = 1.0
START_TRIES = 12                       # start spots a bird draws at most to find one it can fly from
BOARS_SEED_STREAM = 0xB0A2             # the boars' own stream
SOUNDERS = (("one", 0.45), ("two", 0.20), ("sow", 0.35))   # a boar alone, two together, or a sow with her piglets
PIGLETS = (2, 4)                       # a sow brings this many
# Kinds of place wild boars come to; a field with no kind given, the surveyed meadow, is one. Never a town or a village.
BOAR_PLACES = frozenset({"alpine", "forest", "lake", "river", "ski", "valley"})
RODENTS_SEED_STREAM = 0x2A75           # the rats' and mice's own stream
RAT_PLACES = frozenset({"river", "town", "village"})   # rats live by people and water; mice everywhere else
RAT_COUNT = ((1, 0.50), (2, 0.30), (3, 0.20))
MOUSE_COUNT = ((1, 0.65), (2, 0.35))
RODENT_GROUP_M = 6.0                   # a second rat or mouse starts this close to the first where it can
MOST_ANIMALS = 6                       # a seed holds at most this many animals of every kind together
MOST_TO_REPORT = 10                    # people on the field and animals to report together, as many as people alone may be
INSIDE_CLEAR_M = 3.0                   # an animal that lives on the field keeps this far inside its edge
OUTSIDE_CLEAR_M = 6.0                  # and one that lives off it this far outside, as a passer-by does
REPORTED = {"goat": None, "dog": None, "boar": ("boar",)}   # the kinds to report, and of a kind only these bodies
ON_TERRAIN = frozenset({"bird", "boar", "rat"})   # bodies stood on the terrain itself: a few centimetres tall, or a piglet
GRID_M = 0.5
BAND_M = 40.0                          # animals live on the field or up to this far outside its edge
DOCK_CLEAR_M = 8.0
PERSON_CLEAR_M = 3.0                   # past a person's ground, so a box never sits on a person and an animal
TRUNK_CLEAR_M = 0.8
HOUSE_CLEAR_M = 1.0
LIVABLE_M2 = 150.0                     # an open piece smaller than this is a scrap between obstacles
START_GAP_M = 8.0                      # dogs start at least this far from every other animal where they can
PLACE_TRIES = 200
INSIDE_SHARE = 0.5                     # each dog, and the herd, starts on the field this often
PLAN_MARGIN_S = 10.0
STILL_PIXELS = 0.5
RAY_CHUNK = 8192


@dataclass
class Animal:
    """One animal of this seed: its kind, body and looks, its planned track and the pose its mesh last showed."""

    kind: str
    body: str
    coat: str
    size: float
    pace: float
    track: Optional[Track]
    on_field: bool = False                    # the side of the field's edge it lives on, all patrol long
    uid: int = -1
    shown: Optional[np.ndarray] = None        # the drawn vertices its mesh last received
    drawn_at: Optional[tuple] = None          # where its body was last stood

    def skin(self) -> skins.Body:
        """The body's skin and clips."""
        return skins.load(self.kind).bodies[self.body]

    @property
    def asked(self) -> bool:
        """Whether this animal's kind and body are one a report may name (goat, dog, grown boar)."""
        return self.kind in REPORTED and (REPORTED[self.kind] is None or self.body in REPORTED[self.kind])

    @property
    def to_report(self) -> bool:
        """Whether this animal is one to report: of a kind that is asked for, and living on the field."""
        return self.asked and self.on_field


@dataclass
class Sides:
    """The two grounds a seed's animals live on, one each side of the field's edge, and how many animals to report
    the field still has room for."""

    on_field: Ground
    off_field: Ground
    room: int

    def order(self, drawn_on_field: bool, asked: int) -> List[tuple]:
        """The sides a group tries, as (on the field, its ground): the drawn one first, then the other; never the
        field for a group that brings more animals to report than it has room for."""
        fits = asked <= self.room
        first = drawn_on_field and fits
        tried = [(first, self.on_field if first else self.off_field)]
        if first or fits:
            tried.append((not first, self.off_field if first else self.on_field))
        return tried

    def settle(self, on_field: bool, asked: int) -> None:
        """Take the room a group that settled on the field uses."""
        if on_field:
            self.room -= asked


class Animals(dict):
    """Every drawn animal body mapped to its kind, carrying the animals and the ground they live on."""

    def __init__(self, members: Sequence[Animal] = (), ground: Optional[Ground] = None):
        """The animals of a patrol, each body mapped to its kind."""
        super().__init__({animal.uid: animal.kind for animal in members})
        self.members = list(members)
        self.ground = ground


# The part's hooks
def reset(env: Any, ep: FieldEpisode) -> None:
    """Find the ground, draw and plan this seed's animals, build their bodies, and stand them where they are at step
    zero."""
    ep.animals = Animals()
    place_kind = ep.park.get("kind") if isinstance(ep.park, dict) else None
    counts = wanted(ep.seed, herd_place(ep), place_kind)
    if not counts:
        return
    ground = open_ground(env.CLIENT, ep)
    room = MOST_TO_REPORT - sum(1 for person in people.people(ep) if person.inside)
    planned = plan(ep.seed, ground, counts, float(env.EP_LEN_SEC) + PLAN_MARGIN_S, site.night(ep.seed), place_kind, room)
    textures: Dict[str, int] = {}
    for animal in planned:
        kind = skins.load(animal.kind)
        animal.uid = skins.spawn(env.CLIENT, kind, kind.bodies[animal.body], animal.coat, animal.size, textures)
    ep.animals = Animals(planned, ground)
    ep.outcome.animals = sum(1 for animal in planned if animal.to_report)
    place(env, ep, 0)


def advance(env: Any, ep: FieldEpisode) -> None:
    """Stand the animals where they are on the step physics is about to run, when a frame will be taken on it."""
    if not ep.animals:
        return
    step = ep.step + 1
    shots = [shot for shot in (camera.upcoming(env, ep, step), zoom.upcoming(env, ep)) if shot is not None]
    if shots:
        place(env, ep, step, shots)


def place(env: Any, ep: FieldEpisode, step: int, shots: Optional[Sequence[camera.View]] = None, *,
          beam: Optional[camera.View] = None) -> None:
    """Stand every animal where it is at a step. With no shots every body is moved and posed; with shots, only an
    animal one of them can see where it is now or where it was last drawn is moved, and its mesh rewritten only when
    its new pose would move on one of those frames by STILL_PIXELS or more. With a beam, only an animal the laser
    can reach where it is now or where it was last drawn is moved, and its mesh is always rewritten."""
    t = step * SIM_DT
    ground = ep.animals.ground
    sights = None if shots is None else [Sight.of(shot) for shot in shots]
    beam_sight = Sight.of(beam) if beam is not None else None
    for animal in ep.animals.members:
        x, y, hx, hy = animal.track.place(t)
        skin = animal.skin()
        half = skin.length_m * animal.size / 2.0
        # Whether a frame can see it is asked of a ball round the whole body, head, tail and spread wings included.
        reach = max(skin.length_m, skin.extent_m) * animal.size
        lifted = animal.track.lift(t)
        # Wings beat under the body: a bird is held up by as much as its pose reaches under it.
        up = max(lifted, sink(animal, t)) if animal.kind == "bird" else lifted
        z = float(ground.height_at(np.array([[x, y]]))[0]) + up
        per_metre = None
        if beam_sight is not None and animal.shown is not None:
            if not (beam_sight.near_beam((x, y, z), reach) or beam_sight.near_beam(animal.drawn_at, reach)):
                continue
        if sights is not None and animal.shown is not None:
            per_metre = max(sight.per_metre((x, y, z), reach) for sight in sights)
            if not per_metre and not any(sight.per_metre(animal.drawn_at, reach) for sight in sights):
                continue
        front, back = ground.height_at(np.array([[x + hx * half, y + hy * half], [x - hx * half, y - hy * half]]))
        # A body in the air holds itself level; one on the ground lies along the slope under it.
        pitch = 0.0 if lifted > 0.0 else math.atan2(front - back, 2.0 * half)
        if animal.kind in ON_TERRAIN:
            # Small animals stand on the terrain itself, which the grid's blend can miss by their height.
            under = site.ground_at(env.CLIENT, ep.terrain_uids, x, y)
            if under is not None:
                z = under + up
        p.resetBasePositionAndOrientation(animal.uid, [x, y, z], p.getQuaternionFromEuler([0.0, -pitch, math.atan2(hy, hx)]),
                                          physicsClientId=env.CLIENT)
        animal.drawn_at = (x, y, z)
        if per_metre is not None and not per_metre:
            continue
        vertices = shape(animal, t)
        if per_metre is not None and float(np.abs(vertices - animal.shown).max()) * per_metre < STILL_PIXELS:
            continue
        p.resetMeshData(animal.uid, vertices, physicsClientId=env.CLIENT)
        animal.shown = vertices


def bodies(ep: FieldEpisode) -> Dict[int, str]:
    """Every drawn body of every animal, mapped to what it is: dog, goat, bird, boar or rat (a mouse is a rat's kind)."""
    return dict(ep.animals or {})


def to_report(ep: FieldEpisode) -> Dict[int, Animal]:
    """The drawn body of every animal to report, mapped to the animal."""
    return {animal.uid: animal for animal in getattr(getattr(ep, "animals", None), "members", ()) if animal.to_report}


def member(ep: FieldEpisode, uid: int) -> Optional[Animal]:
    """The animal a drawn body belongs to, None for a body that is no animal's."""
    return next((animal for animal in getattr(getattr(ep, "animals", None), "members", ()) if animal.uid == uid), None)


def where(animal: Animal, step: int) -> tuple:
    """East and north of the middle of an animal at a step of its patrol."""
    x, y, _hx, _hy = animal.track.place(step * SIM_DT)
    return float(x), float(y)


def positions(ep: FieldEpisode, step: int) -> np.ndarray:
    """East, north and heading in radians of every animal at a step, without moving anything."""
    rows = []
    for animal in getattr(ep.animals, "members", ()):
        x, y, hx, hy = animal.track.place(step * SIM_DT)
        rows.append((x, y, math.atan2(hy, hx)))
    return np.array(rows, dtype=float).reshape(-1, 3)


# Which animals a seed holds
def herd_place(ep: FieldEpisode) -> bool:
    """Whether the place is one where a herd of goats is kept: by its kind, when the site gives one."""
    kind = ep.park.get("kind") if isinstance(ep.park, dict) else None
    return kind is None or kind in HERD_PLACES


def _draws(seed: int) -> tuple:
    """The seed's three draws for its animals: which kinds, how many dogs, how many goats."""
    rng = np.random.default_rng([ANIMALS_SEED_STREAM, int(seed), 0])
    return pick(rng, KIND_SHARES), int(pick(rng, DOG_COUNT)), int(rng.integers(HERD_SIZE[0], HERD_SIZE[1] + 1))


def sounder(seed: int) -> List[str]:
    """The bodies of the seed's wild boars: one alone, two together, or a sow with her piglets."""
    rng = np.random.default_rng([BOARS_SEED_STREAM, int(seed), 0])
    group, piglets = pick(rng, SOUNDERS), int(rng.integers(PIGLETS[0], PIGLETS[1] + 1))
    return ["boar"] + {"one": [], "two": ["boar"], "sow": ["piglet"] * piglets}[group]


def rodents(seed: int, place: Optional[str] = None) -> List[str]:
    """The bodies of the seed's rats or mice, by the kind of place: rats by people and water, mice anywhere else."""
    rng = np.random.default_rng([RODENTS_SEED_STREAM, int(seed), 0])
    rats, mice = int(pick(rng, RAT_COUNT)), int(pick(rng, MOUSE_COUNT))
    return ["rat"] * rats if place in RAT_PLACES else ["mouse"] * mice


def wanted(seed: int, herd_fits: bool = True, place: Optional[str] = None) -> Dict[str, int]:
    """How many animals of each kind the seed asks for; reads only the seed and the place, never what is installed,
    so every machine counts alike."""
    present = mix.present(seed, site.night(seed))
    counts = {}
    if present.animals:
        kinds, dogs, goats = _draws(seed)
        if kinds in ("goats", "both") and herd_fits:
            counts["goat"] = goats
        if kinds in ("dogs", "both") or not counts:
            counts["dog"] = dogs
    # Goats and dogs come first, then birds while they fit, then a sounder and the rodents, each whole or not at all.
    held = sum(counts.values())
    birds = 0
    if present.birds and not site.night(seed):
        birds = min(int(pick(np.random.default_rng([BIRDS_SEED_STREAM, int(seed), 0]), BIRD_COUNT)), MOST_ANIMALS - held)
    held += birds
    if present.boars and (place is None or place in BOAR_PLACES) and held + len(sounder(seed)) <= MOST_ANIMALS:
        counts["boar"] = len(sounder(seed))
        held += counts["boar"]
    if present.rodents and held + len(rodents(seed, place)) <= MOST_ANIMALS:
        counts["rat"] = len(rodents(seed, place))
    if birds:
        counts["bird"] = birds
    return counts


def flock(seed: int, place: Optional[str], count: int) -> List[tuple]:
    """Which bird each of the seed's birds is, by the kind of place, and whether it came down with the first: a
    flocking first bird is joined by its own kind most often, other birds come on their own."""
    rng = np.random.default_rng([BIRDS_SEED_STREAM, int(seed), 1])
    shares = BIRD_PLACES.get(place, BIRD_PLACES[BIRD_PLACE_DEFAULT])
    first = str(pick(rng, shares))
    found = [(first, False)]
    for _ in range(count - 1):
        if first in FLOCKING and rng.random() < FLOCK_SHARE:
            found.append((first, True))
        else:
            found.append((str(pick(rng, shares)), False))
    return found


def plan(seed: int, ground: Ground, counts: Dict[str, int], horizon_s: float, night: bool = False,
         place: Optional[str] = None, room: int = MOST_TO_REPORT) -> List[Animal]:
    """The seed's animals on a ground, each with its planned track, the herd resting at night; the engine is not
    touched. Every animal on the ground lives on one side of the field's edge, and the field takes at most `room`
    animals to report. Dogs come in place of a herd that finds no room. Boars and rats or mice are placed after the
    goats and dogs and planned together with them, so each keeps clear of the others. Birds, of the kind of place's
    kinds, come last and keep clear of every animal planned before."""
    movers: List[Mover] = []
    animals: List[Animal] = []
    # On the field an animal also keeps out from under the tree crowns, so one to report is never hidden from above.
    sides = Sides(ground.side(True, INSIDE_CLEAR_M).for_birds(), ground.side(False, OUTSIDE_CLEAR_M), int(room))
    if counts.get("goat"):
        _herd(seed, sides, counts["goat"], horizon_s, movers, animals, night)
    dogs = counts.get("dog", 0)
    if not dogs and not movers and counts.get("goat"):
        dogs = _draws(seed)[1]
    if dogs:
        _dogs(seed, sides, dogs, movers, animals)
    if counts.get("boar"):
        _boars(seed, sides, sounder(seed), horizon_s, movers, animals)
    if counts.get("rat"):
        _rodents(seed, sides, rodents(seed, place), movers, animals)
    for animal, track in zip(animals, plan_movers(movers, horizon_s)):
        animal.track = track
    if counts.get("bird"):
        shadows = [Shadow(animal.track, mover.reach) for animal, mover in zip(animals, movers)]
        _birds(seed, ground.for_birds(), flock(seed, place, counts["bird"]), horizon_s, shadows, animals)
    return animals


def _herd(seed: int, sides: Sides, count: int, horizon_s: float, movers: List[Mover], animals: List[Animal],
          resting: bool = False) -> None:
    """The herd: its middle's start on or off the field, then each goat round it on that side, clear of the others,
    close together when it rests; no herd when fewer than the smallest herd fit."""
    kind = skins.load("goat")
    rng = np.random.default_rng([ANIMALS_SEED_STREAM, int(seed), 1])
    names = sorted(kind.bodies)
    widest = max(body.reach_m for body in kind.bodies.values()) * kind.size_spread[1]
    level = sides.on_field.level_for(widest)
    start, on_field, ground = None, False, sides.off_field
    for on_field, ground in sides.order(rng.random() < INSIDE_SHARE, count):
        found = np.argwhere(ground.level(level + 4))
        if len(found):
            row, col = found[int(rng.integers(len(found)))]
            start = ground.centre(int(row), int(col))
            break
    if start is None:
        return
    herd = Herd(rng, ground, level, start, count, horizon_s, resting)
    coats = [str(c) for c in rng.permutation(sorted({c for b in kind.bodies.values() for c in b.coats}))]
    goats: List[Mover] = []
    for n in range(count):
        own = np.random.default_rng([ANIMALS_SEED_STREAM, int(seed), 2, n])
        body = kind.bodies[names[int(own.integers(len(names)))]]
        coat = next((c for c in coats[n % len(coats):] + coats if c in body.coats), sorted(body.coats)[0])
        goat = Mover(kind=kind, body=body, size=uniform(own, *kind.size_spread), pace=uniform(own, *kind.pace_spread),
                     x=0.0, y=0.0, hx=1.0, hy=0.0, rng=own, ground=ground, mind=herd)
        for _ in range(PLACE_TRIES):
            u, v = uniform(own, -1.0, 1.0), uniform(own, -1.0, 1.0)
            x, y = start[0] + herd.radius * u, start[1] + herd.radius * v
            if u * u + v * v <= 1.0 and goat.can_stand(x, y, goats):
                goat.x, goat.y = x, y
                goat.hx, goat.hy = COMPASS[int(own.integers(len(COMPASS)))]
                goats.append(goat)
                animals.append(Animal(kind="goat", body=body.name, coat=coat, size=goat.size, pace=goat.pace, track=None,
                                      on_field=on_field))
                break
    if len(goats) < HERD_SIZE[0]:
        del animals[len(animals) - len(goats):]
        return
    sides.settle(on_field, len(goats))
    movers.extend(goats)


def _dogs(seed: int, sides: Sides, count: int, movers: List[Mover], animals: List[Animal]) -> None:
    """Each dog a different breed with its coat, size, pace and mood, started on or off the field away from every
    animal already placed, and living on that side."""
    kind = skins.load("dog")
    rng = np.random.default_rng([ANIMALS_SEED_STREAM, int(seed), 3])
    breeds = [str(b) for b in rng.permutation(sorted(kind.bodies))[:count]]
    for n, breed in enumerate(breeds):
        own = np.random.default_rng([ANIMALS_SEED_STREAM, int(seed), 4, n])
        body = kind.bodies[breed]
        coat = sorted(body.coats)[int(own.integers(len(body.coats)))]
        dog = Mover(kind=kind, body=body, size=uniform(own, *kind.size_spread), pace=uniform(own, *kind.pace_spread),
                    x=0.0, y=0.0, hx=1.0, hy=0.0, rng=own, ground=sides.off_field)
        taken = [(m.x, m.y, m.reach) for m in movers]
        spot, on_field = None, False
        for on_field, ground in sides.order(own.random() < INSIDE_SHARE, 1):
            spot = place_start(own, np.argwhere(ground.level(dog.level)), ground, dog.reach, taken, START_GAP_M, PLACE_TRIES)
            if spot is not None:
                dog.ground = ground
                break
        if spot is None:
            continue
        sides.settle(on_field, 1)
        dog.x, dog.y = spot
        dog.hx, dog.hy = COMPASS[int(own.integers(len(COMPASS)))]
        dog.mind = Dog(pick(own, DOG_MOODS), spot)
        movers.append(dog)
        animals.append(Animal(kind="dog", body=breed, coat=coat, size=dog.size, pace=dog.pace, track=None, on_field=on_field))


def _boars(seed: int, sides: Sides, names: List[str], horizon_s: float, movers: List[Mover],
           animals: List[Animal]) -> None:
    """The boars: their middle's start on or off the field, away from the animals already placed, then each one round
    it on that side, the piglets with the sow. They move as a herd does, and are up and rooting by night as by day."""
    kind = skins.load("boar")
    rng = np.random.default_rng([BOARS_SEED_STREAM, int(seed), 1])
    widest = max(kind.bodies[name].reach_m for name in names) * kind.size_spread[1]
    level = sides.on_field.level_for(widest)
    grown = sum(1 for name in names if name in REPORTED["boar"])
    taken = [(m.x, m.y, m.reach) for m in movers]
    start, on_field, ground = None, False, sides.off_field
    for on_field, ground in sides.order(rng.random() < INSIDE_SHARE, grown):
        start = place_start(rng, np.argwhere(ground.level(level + 4)), ground, widest, taken, START_GAP_M, PLACE_TRIES)
        if start is not None:
            break
    if start is None:
        return
    herd = Herd(rng, ground, level, start, len(names), horizon_s)
    boars: List[Mover] = []
    for n, name in enumerate(names):
        if n and not boars and "piglet" in names:
            # Piglets never come without their sow: where she found no room, the group stays away.
            return
        own = np.random.default_rng([BOARS_SEED_STREAM, int(seed), 2, n])
        body = kind.bodies[name]
        coat = sorted(body.coats)[int(own.integers(len(body.coats)))]
        boar = Mover(kind=kind, body=body, size=uniform(own, *kind.size_spread), pace=uniform(own, *kind.pace_spread),
                     x=0.0, y=0.0, hx=1.0, hy=0.0, rng=own, ground=ground, mind=herd)
        for _ in range(PLACE_TRIES):
            u, v = uniform(own, -1.0, 1.0), uniform(own, -1.0, 1.0)
            x, y = start[0] + herd.radius * u, start[1] + herd.radius * v
            if u * u + v * v <= 1.0 and boar.can_stand(x, y, [*movers, *boars]):
                boar.x, boar.y = x, y
                boar.hx, boar.hy = COMPASS[int(own.integers(len(COMPASS)))]
                boars.append(boar)
                animals.append(Animal(kind="boar", body=name, coat=coat, size=boar.size, pace=boar.pace, track=None,
                                      on_field=on_field))
                break
    sides.settle(on_field, sum(1 for boar in boars if boar.body.name in REPORTED["boar"]))
    movers.extend(boars)


def _rodents(seed: int, sides: Sides, names: List[str], movers: List[Mover], animals: List[Animal]) -> None:
    """Each rat or mouse with its coat, size and pace, started on or off the field away from every animal and living
    on that side, a second one near the first where it can."""
    kind = skins.load("rat")
    home, home_side = None, None
    for n, name in enumerate(names):
        own = np.random.default_rng([RODENTS_SEED_STREAM, int(seed), 2, n])
        body = kind.bodies[name]
        coat = sorted(body.coats)[int(own.integers(len(body.coats)))]
        rodent = Mover(kind=kind, body=body, size=uniform(own, *kind.size_spread), pace=uniform(own, *kind.pace_spread),
                       x=0.0, y=0.0, hx=1.0, hy=0.0, rng=own, ground=sides.off_field)
        taken = [(m.x, m.y, m.reach) for m in movers]
        spot, on_field = None, False
        if home is not None:
            on_field, ground = home_side
            cells = _within(ground, np.argwhere(ground.level(rodent.level)), home, RODENT_GROUP_M)
            spot = place_start(own, cells, ground, rodent.reach, taken, 1.0, PLACE_TRIES)
            rodent.ground = ground
        if spot is None:
            for on_field, ground in sides.order(own.random() < INSIDE_SHARE, 0):
                spot = place_start(own, np.argwhere(ground.level(rodent.level)), ground, rodent.reach, taken, START_GAP_M,
                                   PLACE_TRIES)
                if spot is not None:
                    rodent.ground = ground
                    break
        if spot is None:
            continue
        if home is None:
            home, home_side = spot, (on_field, rodent.ground)
        rodent.x, rodent.y = spot
        rodent.hx, rodent.hy = COMPASS[int(own.integers(len(COMPASS)))]
        rodent.mind = Rodent(spot)
        movers.append(rodent)
        animals.append(Animal(kind="rat", body=name, coat=coat, size=rodent.size, pace=rodent.pace, track=None,
                              on_field=on_field))


def _birds(seed: int, ground: Ground, species: List[tuple], horizon_s: float, shadows: List[Shadow],
           animals: List[Animal]) -> None:
    """Each bird with its coat, size and pace, started on or off the field away from every animal, a bird that came
    with the first close to it; then their patrol played out beside the goats and dogs."""
    kind = skins.load("bird")
    coats = {name: [str(c) for c in np.random.default_rng([BIRDS_SEED_STREAM, int(seed), 3, n]).permutation(
        sorted(kind.bodies[name].coats))] for n, name in enumerate(sorted({name for name, _ in species}))}
    movers: List[Mover] = []
    leader = None
    for n, (name, joins) in enumerate(species):
        own = np.random.default_rng([BIRDS_SEED_STREAM, int(seed), 2, n])
        body = kind.bodies[name]
        coat = coats[name][sum(1 for other, _ in species[:n] if other == name) % len(coats[name])]
        bird = Mover(kind=kind, body=body, size=uniform(own, *kind.size_spread), pace=uniform(own, *kind.pace_spread),
                     x=0.0, y=0.0, hx=1.0, hy=0.0, rng=own, ground=ground)
        taken = [(m.x, m.y, m.reach) for m in [*shadows, *movers]]
        cells = np.argwhere(ground.level(bird.level))
        spot = None
        if joins and leader is not None:
            spot = _launch_spot(own, _within(ground, cells, leader, FLOCK_M), bird, taken, FLOCK_GAP_M)
        if spot is None:
            side = ground.inside if own.random() < INSIDE_SHARE else ~ground.inside
            for where in (side, ~side):
                spot = _launch_spot(own, np.argwhere(ground.level(bird.level) & where), bird, taken, START_GAP_M)
                if spot is not None:
                    break
        if spot is None:
            continue
        if n == 0:
            leader = spot
        bird.x, bird.y = spot
        bird.hx, bird.hy = COMPASS[int(own.integers(len(COMPASS)))]
        bird.mind = Bird(own)
        movers.append(bird)
        animals.append(Animal(kind="bird", body=name, coat=coat, size=bird.size, pace=bird.pace, track=None))
    for animal, track in zip(animals[len(animals) - len(movers):], plan_movers(movers, horizon_s, shadows)):
        animal.track = track


def _launch_spot(rng: np.random.Generator, cells: np.ndarray, bird: Mover, taken: list, apart_m: float) -> Optional[tuple]:
    """A start spot for a bird drawn from a list of cells, clear of the bodies already placed, from which it can fly:
    the first drawn with MIN_LAUNCHES flight lines open, else the one drawn with the most; None when no clear spot drawn
    has a single line open, as a bird there could never fly."""
    best, most = None, 0
    for _ in range(START_TRIES):
        spot = place_start(rng, cells, bird.ground, bird.reach, taken, apart_m, PLACE_TRIES)
        if spot is None:
            break
        lines = bird.ground.launches(bird.level, *spot)
        if lines >= MIN_LAUNCHES:
            return spot
        if lines > most:
            best, most = spot, lines
    return best


def _within(ground: Ground, cells: np.ndarray, centre: tuple, radius: float) -> np.ndarray:
    """The cells of a list whose centres lie within a radius of a point."""
    x = ground.origin[0] + cells[:, 1] * ground.step - centre[0]
    y = ground.origin[1] + cells[:, 0] * ground.step - centre[1]
    return cells[x * x + y * y <= radius * radius]


# The ground the animals may use
def open_ground(cli: int, ep: FieldEpisode) -> Ground:
    """The grid of open ground round the field: in the band, off everything the animals keep away from, where a ray
    from the sky meets the terrain first, in pieces big enough to live in."""
    outline = np.asarray(ep.fence, dtype=float)
    low = outline.min(axis=0) - BAND_M - 1.0
    cols = int(math.floor((float(outline[:, 0].max()) + BAND_M + 1.0 - low[0]) / GRID_M)) + 1
    rows = int(math.floor((float(outline[:, 1].max()) + BAND_M + 1.0 - low[1]) / GRID_M)) + 1
    xs = low[0] + np.arange(cols) * GRID_M
    ys = low[1] + np.arange(rows) * GRID_M
    gx, gy = np.meshgrid(xs, ys)
    inside = inside_outline(outline, gx, gy)
    usable = (inside | (edge_distance2(outline, gx, gy) <= BAND_M * BAND_M)) & ~kept_off(ep, gx, gy)
    open_at = ep.park.get("open_at") if isinstance(ep.park, dict) else None
    if open_at is not None:
        usable &= np.asarray(open_at(gx, gy), dtype=bool)
    height = np.full(gx.shape, np.nan)
    cells = np.flatnonzero(usable)
    for chunk in np.array_split(cells, max(1, math.ceil(len(cells) / RAY_CHUNK))):
        tops = np.column_stack([gx.flat[chunk], gy.flat[chunk], np.full(len(chunk), site.RAY_REACH_M)]).tolist()
        bottoms = np.column_stack([gx.flat[chunk], gy.flat[chunk], np.full(len(chunk), -site.RAY_REACH_M)]).tolist()
        for index, hit in zip(chunk.tolist(), p.rayTestBatch(tops, bottoms, numThreads=0, physicsClientId=cli)):
            if int(hit[0]) in ep.terrain_uids:
                height.flat[index] = float(hit[3][2])
    open_ = livable(~np.isnan(height))
    sky = ~np.isnan(height) & ~under_crowns(ep, gx, gy)
    return Ground(origin=(float(low[0]), float(low[1])), step=GRID_M, open_=open_, inside=inside, height=filled(height),
                  sky=sky)


def inside_outline(outline: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Whether each point lies inside an outline, by counting the edges a ray east of it crosses."""
    inside = np.zeros(x.shape, dtype=bool)
    for (ax, ay), (bx, by) in zip(outline, np.roll(outline, -1, axis=0)):
        if ay == by:
            continue
        spans = (ay > y) != (by > y)
        cross = ax + (y - ay) * (bx - ax) / (by - ay)
        inside ^= spans & (x < cross)
    return inside


def edge_distance2(outline: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Each point's squared distance to the nearest point of an outline's edge."""
    best = np.full(x.shape, np.inf)
    for (ax, ay), (bx, by) in zip(outline, np.roll(outline, -1, axis=0)):
        best = np.minimum(best, segment_distance2(ax, ay, bx, by, x, y))
    return best


def segment_distance2(ax: float, ay: float, bx: float, by: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Each point's squared distance to a line segment."""
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    share = np.zeros(x.shape) if length2 == 0.0 else np.clip(((x - ax) * dx + (y - ay) * dy) / length2, 0.0, 1.0)
    ex, ey = x - (ax + share * dx), y - (ay + share * dy)
    return ex * ex + ey * ey


def kept_off(ep: FieldEpisode, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The points an animal keeps away from: near the dock, a person's ground, a trunk or a house the survey lists."""
    off = _near_point(x, y, float(ep.dock_position[0]), float(ep.dock_position[1]), DOCK_CLEAR_M)
    for person in people.people(ep):
        lane = getattr(person, "lane", None)
        if lane is not None and len(lane) > 1:
            lane = np.asarray(lane, dtype=float)
            for (ax, ay), (bx, by) in zip(lane[:-1], lane[1:]):
                off |= segment_distance2(ax, ay, bx, by, x, y) < PERSON_CLEAR_M * PERSON_CLEAR_M
            continue
        reach = person.patch_m
        off |= _near_point(x, y, float(person.xy[0]), float(person.xy[1]), reach + PERSON_CLEAR_M)
    for tx, ty, _radius in site.passable_outlines(ep):
        off |= _near_point(x, y, float(tx), float(ty), TRUNK_CLEAR_M)
    for cx, cy, length, width, _heading in site.buildings(ep):
        off |= _near_point(x, y, float(cx), float(cy), math.sqrt(length * length + width * width) / 2.0 + HOUSE_CLEAR_M)
    return off


def under_crowns(ep: FieldEpisode, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The points under a crown circle the site lists, merged or each tree's own."""
    under = np.zeros(x.shape, dtype=bool)
    for cx, cy, radius in [*site.passable_outlines(ep), *site.trees(ep)]:
        under |= _near_point(x, y, float(cx), float(cy), float(radius))
    return under


def _near_point(x: np.ndarray, y: np.ndarray, cx: float, cy: float, radius: float) -> np.ndarray:
    """Whether each point lies closer than a radius to a centre."""
    dx, dy = x - cx, y - cy
    return dx * dx + dy * dy < radius * radius


def livable(open_: np.ndarray) -> np.ndarray:
    """The open cells in pieces of at least LIVABLE_M2, side neighbours joining a piece."""
    pieces, count = ndimage.label(open_)
    if not count:
        return open_
    sizes = np.bincount(pieces.ravel(), minlength=count + 1)
    keep = sizes * GRID_M * GRID_M >= LIVABLE_M2
    keep[0] = False
    return keep[pieces]


def filled(height: np.ndarray, rings: int = 4) -> np.ndarray:
    """The height grid with the cells round the measured ones given the mean of their measured neighbours, ring by
    ring, as far as a body reaches past the open ground; cells further out are never read."""
    height = height.copy()
    for _ in range(rings):
        missing = np.isnan(height)
        if not missing.any():
            break
        padded = np.pad(height, 1, constant_values=np.nan)
        near = np.stack([padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:]])
        counts = (~np.isnan(near)).sum(axis=0)
        grow = missing & (counts > 0)
        height[grow] = np.nansum(near, axis=0)[grow] / counts[grow]
    return np.nan_to_num(height, nan=0.0)


# What an animal looks like at a time
def sink(animal: Animal, t: float) -> float:
    """How far under its own ground an animal's pose reaches at a time, at its size, eased out of the pose its clip
    before held as its shape is; nothing for a pose that stands on its ground."""
    body = animal.skin()
    segment, frame, run = animal.track.pose(t)
    lowest = skins.low(body, segment.clip, frame)
    blend = skins.load(animal.kind).blend_s
    if segment.held is not None and run < blend:
        share = run / blend
        share = share * share * (3.0 - 2.0 * share)
        lowest = skins.low(body, *segment.held) * (1.0 - share) + lowest * share
    return max(0.0, -lowest * animal.size)


def shape(animal: Animal, t: float) -> np.ndarray:
    """The drawn vertices of an animal at a time, eased out of the pose its clip before held."""
    body = animal.skin()
    segment, frame, run = animal.track.pose(t)
    pose = skins.points(body, segment.clip, frame)
    blend = skins.load(animal.kind).blend_s
    if segment.held is not None and run < blend:
        share = run / blend
        share = np.float32(share * share * (3.0 - 2.0 * share))
        pose = skins.points(body, *segment.held) * (np.float32(1.0) - share) + pose * share
    return skins.drawn(body, pose, animal.size)


@dataclass(frozen=True)
class Sight:
    """A frame's eye, its square axes and the tangents of its half angles, to ask cheaply what it shows."""

    eye: tuple
    forward: tuple
    right: tuple
    up: tuple
    half_h: float
    half_v: float
    height: int
    lean_h: float = 1.0                # sqrt(1 + half_h^2): a side plane's distance over the tangent's gap, per depth
    lean_v: float = 1.0

    @classmethod
    def of(cls, shot: camera.View) -> "Sight":
        """The sight of a frame's view."""
        forward = np.asarray(shot.forward, dtype=float)
        right = np.cross(forward, np.asarray(shot.up, dtype=float))
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        half_v = math.tan(math.radians(shot.vertical_fov_deg) / 2.0)
        half_h = half_v * shot.width / shot.height
        return cls(tuple(float(v) for v in shot.eye), tuple(forward.tolist()), tuple(right.tolist()), tuple(up.tolist()),
                   half_h, half_v, int(shot.height), math.sqrt(1.0 + half_h * half_h), math.sqrt(1.0 + half_v * half_v))

    def per_metre(self, point: Sequence[float], margin: float) -> float:
        """How many pixels a metre at a point spans on the frame; 0 when the frame does not show the point, a margin
        round the frame's edges included."""
        ox, oy, oz = point[0] - self.eye[0], point[1] - self.eye[1], point[2] - self.eye[2]
        depth = ox * self.forward[0] + oy * self.forward[1] + oz * self.forward[2]
        if depth <= 0.0:
            return 0.0
        across = abs(ox * self.right[0] + oy * self.right[1] + oz * self.right[2]) / depth
        rise = abs(ox * self.up[0] + oy * self.up[1] + oz * self.up[2]) / depth
        # A ball of the margin's radius reaches a frame side while its centre is within the margin of that side's plane.
        if across > self.half_h + margin * self.lean_h / depth or rise > self.half_v + margin * self.lean_v / depth:
            return 0.0
        return self.height / (2.0 * self.half_v * depth)

    def near_beam(self, point: Sequence[float], reach: float) -> bool:
        """Check whether an animal's bounding sphere can reach the laser or its slope samples."""
        ox, oy, oz = point[0] - self.eye[0], point[1] - self.eye[1], point[2] - self.eye[2]
        depth = ox * self.forward[0] + oy * self.forward[1] + oz * self.forward[2]
        margin = reach + laser.FINE_SPACING_M * laser.FINE_SIZE / 2.0
        if not -margin <= depth <= laser.SEARCH_M + margin:
            return False
        across = abs(ox * self.right[0] + oy * self.right[1] + oz * self.right[2])
        rise = abs(ox * self.up[0] + oy * self.up[1] + oz * self.up[2])
        return max(across, rise) <= margin + max(depth, 0.0) * self.half_v


def pixels_per_metre(shot: camera.View, point: Sequence[float], margin: float) -> float:
    """How many pixels a metre at a point spans on a frame; 0 when the frame does not show the point, a margin round
    the frame's edges included."""
    return Sight.of(shot).per_metre(point, margin)
