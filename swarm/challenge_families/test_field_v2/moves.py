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

"""The everyday motion library the field's people move by, and what each kind of person does with it.

The library (maps/custom/testsite/motions in swarm-worlds) is in the format of Sentinel's intruder motions
(`solar_patrol/theft_moves.py`) and is played by the same actor: short blocks that start and end on a few base poses
(stand, crouch, kneel, sit), loops walked or
worked on the spot, and the moves that join them. It holds ways of walking, things people do standing (shifting their
weight, talking, looking at a phone, arms crossed, hands on the hips, looking up, waving, pointing, stretching, taking a
picture) and low (squatting, kneeling, sitting).

A library need not hold every kind: whoever picks from it picks among the kinds it has.

Nobody stands as a statue: standing() fills any wait with the small moves that start and end on the standing pose
(a shift of the weight, a look round, a hand to the head), and bout() is one of the things done standing, played
whole.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import swarm_worlds

from ..solar_patrol.theft_moves import Library, Step

ASSET_DIR = os.path.join("custom", "testsite", "motions")
MOVES_DIR = os.path.join(swarm_worlds.maps_dir(), ASSET_DIR)
STAND = "stand"
# Ways of walking, each a loop with a move that sets off into it and one that stops out of it, and how often it is
# drawn.
GAITS = {"walk": 0.40, "stroll": 0.30, "walk_brisk": 0.10, "walk_phone": 0.10, "walk_pockets": 0.10}
# Things done standing on the spot, each a loop with its way in from standing and its way back, and how often.
HABITS = {"idle": 0.30, "idle_talk": 0.12, "idle_phone": 0.14, "idle_crossed": 0.12, "idle_hips": 0.10, "idle_photo": 0.06,
          "idle_stretch": 0.05, "idle_point": 0.05, "idle_look_up": 0.06}
# What two or more who stand together do: mostly talk.
COMPANY_HABITS = {"idle_talk": 0.50, "idle": 0.18, "idle_crossed": 0.10, "idle_hips": 0.10, "idle_point": 0.07,
                  "idle_phone": 0.05}
# The small moves that start and end on the standing pose, which fill a wait.
FIDGETS = ("stand_shift", "stand_look", "stand_fidget", "stand_breathe")
FIDGET_FITS = 0.6                      # a small move is played while at least this share of it fits the wait left
# What someone standing may do when the drone comes near: look up at it, or wave.
GREETINGS = {"idle_look_up": 0.7, "idle_wave": 0.3}
# The low poses: the base pose each is held from, and the loop worked in it.
LOW = {"crouch": "crouch_idle", "kneel": "kneel_idle", "sit": "sit_idle"}


@lru_cache(maxsize=2)
def _library(folder: str) -> Library:
    """The library at a folder, read once per process."""
    if not os.path.isfile(os.path.join(folder, "motions.json")):
        raise FileNotFoundError(f"everyday motion library missing: {folder}")
    return Library(folder)


def library() -> Library:
    """The shipped everyday library; raises FileNotFoundError when the installed swarm-worlds predates it."""
    return _library(MOVES_DIR)


def problems(folder: Optional[str] = None) -> List[str]:
    """What is wrong with an everyday motion library whose files are both there, one line each: a library that cannot
    be read, no standing pose, or no plain walk that can be played whole (setting off, the loop, stopping). Empty for
    a library people can live by: every other kind is used only where the library holds it."""
    try:
        lib = Library(MOVES_DIR if folder is None else folder)
        found = []
        if not lib.by_start.get(("hub", STAND)):
            found.append("no block starts on the standing pose")
        if "walk" not in gaits(lib):
            found.append("the plain walk cannot be played whole")
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        return [f"the everyday motion library cannot be read: {type(error).__name__}: {error}"]
    return found


def asset_files() -> List[str]:
    """The library's two files."""
    return [os.path.join(MOVES_DIR, leaf) for leaf in ("motions.json", "motions.npz")]


def loop_of(lib: Library, kind: str) -> List[str]:
    """Every take of a kind of loop the library holds, in order."""
    return sorted(name for name, block in lib.blocks.items() if block["kind"] == kind and block["loop"])


def gaits(lib: Library) -> Dict[str, float]:
    """The ways of walking the library can play whole: the loop, setting off into it from standing and stopping out
    of it."""
    found = {}
    for gait, share in GAITS.items():
        takes = loop_of(lib, gait)
        if takes and f"stand_to_{gait}" in lib.kinds({"hub": STAND}) and f"{gait}_to_stand" in lib.kinds({"loop": takes[0]}):
            found[gait] = share
    return found


def habits(lib: Library, wanted: Dict[str, float]) -> Dict[str, float]:
    """The things done standing, of a list, that the library can play whole: the way in from standing, the loop and
    the way back."""
    found = {}
    for habit, share in wanted.items():
        takes = loop_of(lib, habit)
        if takes and f"{habit}_in" in lib.kinds({"hub": STAND}) and f"{habit}_to_stand" in lib.kinds({"loop": takes[0]}):
            found[habit] = share
    return found


def low(lib: Library) -> Dict[str, str]:
    """The low poses the library can hold and work in: base pose -> the kind of its way into the loop worked there;
    a base pose the library has no loop for is simply held."""
    found = {}
    for hub, loop in LOW.items():
        if lib.by_start.get(("hub", hub)):
            found[hub] = f"{loop}_in" if f"{loop}_in" in lib.kinds({"hub": hub}) else ""
    return found


def hub_frame(lib: Library, hub: str) -> Tuple[int, str]:
    """The library frame of a base pose and the block it opens: the first frame of the first block that starts on it."""
    kinds = lib.by_start[("hub", hub)]
    name = kinds[sorted(kinds)[0]][0]
    return int(lib.blocks[name]["first"]), name


def pick(rng: np.random.Generator, weights: Dict[str, float]) -> str:
    """One key drawn with its weight."""
    keys = list(weights)
    share = np.array([weights[k] for k in keys], dtype=float)
    return keys[int(rng.choice(len(keys), p=share / share.sum()))]


def fidgets(lib: Library) -> List[str]:
    """The small moves from standing back to standing that the library holds."""
    held = lib.kinds({"hub": STAND})
    return [kind for kind in FIDGETS if kind in held and lib.blocks[held[kind][0]]["next"] == {"hub": STAND}]


def anchored(track: Any, steps: Iterable[Step]) -> Iterator[Step]:
    """Steps done on one spot, after which the body stands exactly where and as it stood before them: a clip's own
    few centimetres of travel and turn never add up over a patrol."""
    pos, yaw = track.pos.copy(), track.yaw
    yield from steps
    track.pos, track.yaw = pos, yaw


def standing(lib: Library, rng: np.random.Generator, seconds: float, pace: float, stage: str,
             track: Optional[Any] = None) -> Iterator[Step]:
    """Standing for about so many seconds without being a statue: the library's small moves one after another, drawn
    from rng, while FIDGET_FITS of the next still fits; held still for what is left, or where it has none. Given the
    track they are played on, each move leaves the body exactly where it stood."""
    kinds = fidgets(lib)
    left = float(seconds)
    while kinds:
        kind = kinds[int(rng.integers(len(kinds)))]
        length = lib.seconds(lib.kinds({"hub": STAND})[kind][0]) / pace
        if left < FIDGET_FITS * length:
            break
        step = [Step(kind, stage=stage)]
        yield from (step if track is None else anchored(track, step))
        left -= length
    if left > 1e-9:
        yield Step("hold", hold=left, stage=stage)


def crouch(lib: Library) -> List[str]:
    """The kinds a standing body plays to squat for a moment and get up again, in order, with "hold" where it simply
    holds the squat: through the loop worked there when the library has it whole, else down, held and up; empty when
    the library cannot do either."""
    from_stand = lib.kinds({"hub": STAND})
    if "stand_to_crouch" not in from_stand:
        return []
    loops = loop_of(lib, LOW["crouch"])
    busy, back = f"{LOW['crouch']}_in", f"{LOW['crouch']}_to_stand"
    if loops and busy in lib.kinds({"hub": "crouch"}) and back in lib.kinds({"loop": loops[0]}):
        return ["stand_to_crouch", busy, "loop", back]
    return ["stand_to_crouch", "hold", "crouch_to_stand"] if "crouch_to_stand" in lib.kinds({"hub": "crouch"}) else []


def bout(habit: str, cycles: int, stage: str) -> Iterator[Step]:
    """One thing done standing, played whole: into it, some cycles of it, and back to standing."""
    yield Step(f"{habit}_in", stage=stage)
    yield Step("loop", cycles=cycles, stage=stage)
    yield Step(f"{habit}_to_stand", stage=stage)


def bout_seconds(lib: Library, habit: str, cycles: int, pace: float) -> float:
    """How long a bout of a habit lasts at a pace."""
    loop = loop_of(lib, habit)[0]
    way_in = lib.kinds({"hub": STAND})[f"{habit}_in"][0]
    way_out = lib.kinds({"loop": loop})[f"{habit}_to_stand"][0]
    return (lib.seconds(way_in) + cycles * lib.seconds(loop) + lib.seconds(way_out)) / pace
