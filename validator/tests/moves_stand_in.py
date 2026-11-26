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

"""A stand-in everyday motion library, so the tests need no shipped one.

It is Sentinel's own library, which the installed worlds carry, under the everyday library's names: every everyday
block is one of the thieves' blocks played under another name, with the start and the end the everyday block has. A
walk is a walk; whatever is done on the spot is the thieves' looking around or squatting on watch, and a small move
from standing back to standing is their turning to look, with no travel.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from swarm.challenge_families.solar_patrol import theft_moves

WALKS = {"walk": ("walk", 2), "stroll": ("walk_wary", 2), "walk_brisk": ("walk", 1), "walk_phone": ("walk_wary", 1),
         "walk_pockets": ("walk", 1)}
HABITS = ("idle", "idle_talk", "idle_phone", "idle_crossed", "idle_hips", "idle_look_up", "idle_wave", "idle_point",
          "idle_stretch", "idle_photo")
LOW = {"crouch": "crouch_idle", "kneel": "kneel_idle", "sit": "sit_idle"}
FIDGETS = ("stand_shift", "stand_look", "stand_fidget", "stand_breathe")


def write_library(folder: Path) -> str:
    """Write the stand-in library into a folder and return the folder."""
    folder = Path(folder)
    if (folder / "motions.json").is_file():
        return str(folder)
    folder.mkdir(parents=True, exist_ok=True)
    source = Path(theft_moves.swarm_worlds.maps_dir()) / theft_moves.ASSET_DIR
    shutil.copyfile(source / "motions.npz", folder / "motions.npz")
    manifest = json.loads((source / "motions.json").read_text())
    old = manifest["blocks"]
    blocks: Dict[str, Dict[str, Any]] = {}

    def add(name: str, kind: str, donor: str, start: Dict[str, str], after: Optional[Dict[str, str]]) -> None:
        """One everyday block: a thieves' block's frames and travel under an everyday name, start and end."""
        blocks[name] = {"kind": kind, "loop": bool(old[donor]["loop"]), "first": old[donor]["first"], "frames": old[donor]["frames"],
                        "start": start, "next": after, "travel": old[donor]["travel"]}

    stand = {"hub": "stand"}
    for gait, (donor, takes) in WALKS.items():
        for k in range(takes):
            loop = {"loop": f"{gait}.{k}"}
            add(f"{gait}.{k}", gait, f"{donor}.{k}", loop, loop)
            add(f"{gait}_to_stand.{k}", f"{gait}_to_stand", f"{donor}_to_stand.{k}", loop, stand)
        add(f"stand_to_{gait}.0", f"stand_to_{gait}", f"stand_to_{donor}.0", stand, {"loop": f"{gait}.0"})
    add("walk_look_up.0", "walk_look_up", "walk_freeze_resume.0", {"loop": "walk.0"}, {"loop": "walk.0"})
    for habit in HABITS:
        loop = {"loop": f"{habit}.0"}
        add(f"{habit}.0", habit, "look_around.0", loop, loop)
        add(f"{habit}.0.in", f"{habit}_in", "look_around.0.in", stand, loop)
        add(f"{habit}_to_stand.0", f"{habit}_to_stand", "cut_fence_to_stand.0", loop, stand)
    for hub, low in LOW.items():
        loop = {"loop": f"{low}.0"}
        add(f"{low}.0", low, "crouch_watch.0", loop, loop)
        add(f"{low}.0.in", f"{low}_in", "crouch_watch.0.in", {"hub": hub}, loop)
        add(f"{low}_to_stand.0", f"{low}_to_stand", "crouch_to_stand.0", loop, stand)
        donor = hub if hub != "sit" else "crouch"
        add(f"stand_to_{hub}.0", f"stand_to_{hub}", f"stand_to_{donor}.0", stand, {"hub": hub})
        add(f"{hub}_to_stand.0", f"{hub}_to_stand", f"{donor}_to_stand.0", {"hub": hub}, stand)
    for fidget in FIDGETS:
        add(f"{fidget}.0", fidget, "look_around.0.in", stand, stand)
        blocks[f"{fidget}.0"]["travel"] = {"x": 0.0, "z": 0.0, "turn": 0.0}
    manifest["blocks"] = blocks
    manifest["source"] = "stand-in: Sentinel's library under the everyday names"
    (folder / "motions.json").write_text(json.dumps(manifest))
    return str(folder)
