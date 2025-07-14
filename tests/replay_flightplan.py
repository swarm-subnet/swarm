#!/usr/bin/env python3
"""
Replay the best‑scoring FlightPlan stored by the validator.

Usage examples
──────────────
Show the newest stored flight‑plan with GUI:

    python replay_flightplans.py --gui

Pick a specific file (head‑less):

    python -m tests.replay_flightplans --gui --file FILEPATH
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple, List

from swarm.protocol import MapTask, FlightPlan, RPMCmd
from swarm.validator.replay import replay_once


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def _find_flightplan_dir(start: Path) -> Path:
    """
    Walk up the directory tree from *start* until a "flightplans" folder is found.
    Raises FileNotFoundError if none exists.
    """
    for p in [start, *start.parents]:
        candidate = p / "flightplans"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("Could not locate a 'flightplans/' directory.")


def _load_json(fp: Path) -> dict:
    with fp.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _dict_to_task(d: dict) -> MapTask:
    return MapTask(
        map_seed=d["map_seed"],
        start=tuple(d["start"]),
        goal=tuple(d["goal"]),
        sim_dt=d["sim_dt"],
        horizon=d["horizon"],
        version=d.get("version", "1"),
    )


def _dict_to_plan(d: dict) -> FlightPlan:
    cmds: List[RPMCmd] = [RPMCmd(c["t"], tuple(c["rpm"])) for c in d["commands"]]
    return FlightPlan(commands=cmds, sha256=d.get("sha256"))


def _select_best(fp_json: dict) -> dict:
    """Return the flightplan entry with the highest score."""
    return max(fp_json["flightplans"], key=lambda x: x["score"])


# ──────────────────────────────────────────────────────────────────────────────
# Main replay routine
# ──────────────────────────────────────────────────────────────────────────────
def replay_best(
    json_path: Path | None = None,
    *,
    gui: bool = False,
) -> None:
    """
    • json_path == None  ➜  use newest file from flightplans/
    • gui True/False     ➜  replay with or without PyBullet viewer
    """
    if json_path is None:
        fp_dir = _find_flightplan_dir(Path(__file__).resolve().parent)
        files = sorted(
            fp_dir.glob("flightplans_*.json"), key=lambda p: p.stat().st_mtime
        )
        if not files:
            print(f"No JSON files found in {fp_dir}")
            return
        json_path = files[-1]

    if not json_path.is_file():
        print(f"File not found: {json_path}")
        return

    payload = _load_json(json_path)
    best = _select_best(payload)

    task = _dict_to_task(payload["task"])
    plan = _dict_to_plan(best["plan"])

    print("═══════════════════════════════════════════════════════")
    print(f"Replaying best FlightPlan from: {json_path.name}")
    print(f"UID        : {best['uid']}")
    print(f"Score      : {best['score']:.3f}")
    print("═══════════════════════════════════════════════════════\n")

    success, t_sim, energy = replay_once(task, plan, gui=gui)

    print("\n══════════════════ Results ════════════════════════════")
    print(f"Success     : {success}")
    print(f"Time (s)    : {t_sim:.2f}  /  horizon = {task.horizon}")
    print(f"Energy      : {energy:.2f}")
    print("═══════════════════════════════════════════════════════")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Replay the top‑scoring FlightPlan stored by the validator"
    )
    ap.add_argument(
        "--file",
        type=str,
        help="Path to a specific flightplans_*.json file "
        "(defaults to the newest file in flightplans/)",
    )
    ap.add_argument(
        "--gui",
        action="store_true",
        help="Open the PyBullet 3‑D viewer during the replay",
    )
    args = ap.parse_args()

    replay_best(
        json_path=Path(args.file).expanduser().resolve() if args.file else None,
        gui=args.gui,
    )
