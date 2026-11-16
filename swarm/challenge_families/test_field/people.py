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

"""People on the field: how many a seed holds, who they are, where they stay and how they hold themselves.

A seed holds 0 to 6 people, about one seed in seven none. They are Sentinel's bodies in Sentinel's everyday clothes
(solar's intruders module), with no mask, tool, glove or pack, and as warm to the thermal camera. Each one stays where
the seed puts him for the whole patrol, in one still pose: standing, crouching or kneeling, or lying on the ground, and
nobody reacts to the drone.

Each person stands on the field PERSON_EDGE_M inside its outline, at least DOCK_CLEAR_M from the dock and
APART_M from everyone else, on open ground and clear of every tree crown the survey lists. His reference point, the
place a report must name, is the centre of the box his body covers seen from above: between the feet of someone
standing, the middle of the body of someone lying.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pybullet as p

from ..solar_patrol import intruders, theft_moves
from ..solar_patrol.flight_limit import _distance, _inside, _sides_of
from . import site
from .episode import FieldEpisode

PEOPLE_SEED_STREAM = 0x9E091E          # the people's own stream, so no other draw of the seed moves them
COUNT_SHARES = {0: 0.15, 1: 0.20, 2: 0.20, 3: 0.18, 4: 0.12, 5: 0.08, 6: 0.07}
POSE_SHARES = {"standing": 0.4, "crouching": 0.3, "lying": 0.3}
POSE_HUBS = {"standing": ("stand",), "crouching": ("crouch", "kneel"), "lying": ("prone",)}
# Everyday clothes: Sentinel's wardrobe without masks, gloves, tools, packs or coils.
TOPS = intruders.TOPS
BOTTOMS = intruders.BOTTOMS
SHOES = intruders.SHOES
HEADS = intruders.HEADS[""]
PERSON_EDGE_M = 1.5                    # the reference point this far inside the outline, so a lying body stays on the field
DOCK_CLEAR_M = 8.0
APART_M = 3.0
CROWN_CLEAR_M = 0.5                    # off the edge of every crown the survey lists
OPEN_M = 1.0                           # only the ground meets a ray this close to a person's spot
CONTOUR_TURN_DEG = 20.0                # someone lying lies along the slope's contour, give or take this
SLOPE_PROBE_M = 1.0
PLACE_TRIES = 4000
AWAY_Z = -500.0                        # where a body waits while it is being measured
GROUND_RAYS = 16
RAY_REACH_M = 1000.0


@dataclass
class Person:
    """One person on the field: his pose, where his reference point stands, his body and its drawn ids."""

    index: int
    pose: str
    xy: Tuple[float, float]            # the reference point, world metres
    ground_z: float
    yaw: float
    body: Any = None


class NoRoom(RuntimeError):
    """The field has no room left for the seed's people: the seed's own world fails, not the machine."""


def count(seed: int) -> int:
    """How many people the seed puts on the field."""
    rng = np.random.default_rng([PEOPLE_SEED_STREAM, int(seed), 0])
    return _pick(rng, COUNT_SHARES)


def _pick(rng: np.random.Generator, weights: Dict[Any, float]) -> Any:
    """One key drawn with its weight."""
    keys = list(weights)
    share = np.array([weights[k] for k in keys], dtype=float)
    return keys[int(rng.choice(len(keys), p=share / share.sum()))]


def outfit(rng: np.random.Generator) -> Dict[str, Any]:
    """One person's build, everyday clothes and colours, drawn from the seed's generator."""
    cat = intruders.catalogue()
    garments = [_pick(rng, TOPS), _pick(rng, BOTTOMS), _pick(rng, SHOES)]
    head = _pick(rng, HEADS)
    if garments[0] != "hoodie_up" and head:
        garments.append(head)
    colours = {}
    for name in garments:
        for piece in cat.pieces[name]:
            options = piece["colours"]
            colours[piece["file"]] = options[int(rng.integers(len(options)))]
    body = cat.manifest["body"]
    return {"build": cat.builds[int(rng.integers(len(cat.builds)))], "garments": garments, "colours": colours,
            "skin": body["skin_tones"][int(rng.integers(len(body["skin_tones"])))],
            "undershirt": body["undershirt_colours"][int(rng.integers(len(body["undershirt_colours"])))]}


def pose_frame(rng: np.random.Generator, pose: str) -> Tuple[int, bool]:
    """A still frame of the motion library for a pose: the first frame of a block that starts from one of its hubs,
    and whether it is mirrored."""
    lib = theft_moves.library()
    names = sorted(name for hub in POSE_HUBS[pose] for kind in lib.by_start.get(("hub", hub), {}).values()
                   for name in kind)
    name = names[int(rng.integers(len(names)))]
    return int(lib.blocks[name]["first"]), bool(rng.random() < 0.5)


def pose_args(index: int, mirrored: bool, place: Tuple[float, float, float], yaw: float) -> tuple:
    """The intruder pose for a library frame standing on a place: local rotations, root height, place and yaw."""
    lib = theft_moves.library()
    rot = lib.rotations[int(mirrored)][index]
    root = lib.root[int(mirrored)][index] * np.array([0.0, 1.0, 0.0])
    return rot, root, place, yaw


def ground_at(cli: int, terrain_uids: frozenset, x: float, y: float) -> Optional[float]:
    """Terrain height under a point, past anything standing on it; None when no terrain lies there."""
    top = RAY_REACH_M
    for _ in range(GROUND_RAYS):
        hit = p.rayTest([x, y, top], [x, y, -RAY_REACH_M], physicsClientId=cli)[0]
        if hit[0] < 0:
            return None
        if hit[0] in terrain_uids:
            return float(hit[3][2])
        top = float(hit[3][2]) - 0.01
    return None


def _open(cli: int, terrain_uids: frozenset, xy: np.ndarray) -> bool:
    """Whether only the ground meets rays dropped on a spot and around it, so nobody stands in a wall or on a stone."""
    ring = [(0.0, 0.0)] + [(OPEN_M * math.cos(k * math.pi / 4.0), OPEN_M * math.sin(k * math.pi / 4.0)) for k in range(8)]
    points = [(float(xy[0]) + dx, float(xy[1]) + dy) for dx, dy in ring]
    hits = p.rayTestBatch([[x, y, RAY_REACH_M] for x, y in points], [[x, y, -RAY_REACH_M] for x, y in points],
                          physicsClientId=cli)
    return all(hit[0] in terrain_uids for hit in hits)


def _contour_yaw(cli: int, terrain_uids: frozenset, xy: np.ndarray, rng: np.random.Generator) -> float:
    """An engine yaw along the slope's contour at a spot, turned a little either way."""
    x, y = float(xy[0]), float(xy[1])
    heights = [ground_at(cli, terrain_uids, x + dx, y + dy) for dx, dy in ((SLOPE_PROBE_M, 0.0), (-SLOPE_PROBE_M, 0.0),
                                                                             (0.0, SLOPE_PROBE_M), (0.0, -SLOPE_PROBE_M))]
    turn = math.radians(float(rng.uniform(-CONTOUR_TURN_DEG, CONTOUR_TURN_DEG)))
    if any(h is None for h in heights):
        return float(rng.uniform(-math.pi, math.pi))
    rise = np.array([heights[0] - heights[1], heights[2] - heights[3]])
    contour = np.array([-rise[1], rise[0]]) if np.hypot(*rise) > 1e-9 else np.array([1.0, 0.0])
    return theft_moves.heading_to(contour) + turn


def spots(cli: int, seed: int, outline: np.ndarray, terrain_uids: frozenset, dock_xy: np.ndarray,
          crowns: np.ndarray, wanted: int) -> List[np.ndarray]:
    """Where the seed's people stand, drawn from its own stream: each inside the outline by PERSON_EDGE_M, clear of the
    dock, of each other and of every crown, on open ground."""
    rng = np.random.default_rng([PEOPLE_SEED_STREAM, int(seed), 1])
    low, high = outline.min(axis=0), outline.max(axis=0)
    sides = _sides_of(outline)
    placed: List[np.ndarray] = []
    for _ in range(PLACE_TRIES):
        if len(placed) == wanted:
            break
        spot = rng.uniform(low, high)
        if not _inside(sides, spot) or _distance(outline, spot) < PERSON_EDGE_M:
            continue
        if np.hypot(*(spot - dock_xy)) < DOCK_CLEAR_M or any(np.hypot(*(spot - other)) < APART_M for other in placed):
            continue
        if len(crowns) and np.any(np.hypot(*(crowns[:, :2] - spot).T) < crowns[:, 2] + CROWN_CLEAR_M):
            continue
        if _open(cli, terrain_uids, spot):
            placed.append(spot)
    if len(placed) < wanted:
        raise NoRoom(f"room for {len(placed)} of {wanted} people in seed {seed}")
    return placed


def reset(env: Any, ep: FieldEpisode) -> None:
    """Put the seed's people on the field, each dressed, posed and standing on the ground at his spot."""
    cli = env.CLIENT
    wanted = count(ep.seed)
    ep.people = []
    ep.outcome.threats = wanted
    if not wanted:
        return
    where = spots(cli, ep.seed, ep.fence, ep.terrain_uids, ep.dock_position[:2], site.passable_outlines(ep), wanted)
    rng = np.random.default_rng([PEOPLE_SEED_STREAM, int(ep.seed), 2])
    for n, spot in enumerate(where):
        pose = _pick(rng, POSE_SHARES)
        index, mirrored = pose_frame(rng, pose)
        dress = outfit(rng)
        yaw = _contour_yaw(cli, ep.terrain_uids, spot, rng) if pose == "lying" else float(rng.uniform(-math.pi, math.pi))
        ep.people.append(_stand(cli, ep, n, pose, dress, index, mirrored, spot, yaw))


def _stand(cli: int, ep: FieldEpisode, n: int, pose: str, dress: Dict[str, Any], index: int, mirrored: bool,
           spot: np.ndarray, yaw: float) -> Person:
    """One person's body, posed so his reference point lies on the spot and his lowest point on the highest ground
    under him, so no limb sinks into the slope."""
    body = intruders.Intruder(dress, pose_args(index, mirrored, (0.0, 0.0, AWAY_Z), yaw), cli)
    v = body.vertices(*pose_args(index, mirrored, (0.0, 0.0, 0.0), yaw))
    low, high = v[:, :2].min(axis=0), v[:, :2].max(axis=0)
    centre = (low + high) / 2.0
    corners = [(float(spot[0] + dx), float(spot[1] + dy)) for dx in (low[0] - centre[0], 0.0, high[0] - centre[0])
               for dy in (low[1] - centre[1], 0.0, high[1] - centre[1])]
    heights = [h for h in (ground_at(cli, ep.terrain_uids, x, y) for x, y in corners) if h is not None]
    ground = max(heights) if heights else float(ep.dock_position[2])
    place = (float(spot[0] - centre[0]), float(spot[1] - centre[1]), ground)
    body.pose(*pose_args(index, mirrored, place, yaw))
    return Person(index=n, pose=pose, xy=(float(spot[0]), float(spot[1])), ground_z=ground, yaw=yaw, body=body)


def people(ep: FieldEpisode) -> List[Person]:
    """Everyone on the field this patrol."""
    return list(ep.people or ())


def bodies(ep: FieldEpisode) -> Dict[int, int]:
    """Every drawn body of every person, mapped to the person's index."""
    return {int(uid): person.index for person in people(ep) if person.body for uid in person.body.bodies}


def reference(ep: FieldEpisode, index: int) -> np.ndarray:
    """A person's reference point, metres east and north of the dock."""
    return np.asarray(ep.people[index].xy, dtype=float) - np.asarray(ep.dock_position[:2], dtype=float)
