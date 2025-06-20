"""
End‑to‑end round‑trip & replay test – **with optional live GUIs**.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Tuple

from swarm.validator.task_gen import random_task
from swarm.validator.forward import SIM_DT, HORIZON_SEC
from neurons.miner import flying_strategy           # reference strategy
from swarm.validator.replay import replay_once
from swarm.validator.reward import flight_reward
from swarm.protocol import MapTask, FlightPlan, ValidationResult

try:
    from loguru import logger
except ImportError:                                 # pragma: no cover
    import logging as logger
    logger.basicConfig(level=logger.INFO)


# ---------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------
def make_task() -> MapTask:
    return random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC)


def make_plan(task: MapTask, gui: bool) -> FlightPlan:
    cmds = flying_strategy(task, gui=gui)
    return FlightPlan(commands=cmds, sha256="")      # hash auto‑computed


# ---------------------------------------------------------------------
# Validation logic
# ---------------------------------------------------------------------
def validate(task: MapTask,
             plan: FlightPlan,
             *,
             sim_gui: bool = False) -> ValidationResult:
    success, t_sim, energy = replay_once(task, plan, gui=sim_gui)
    score = flight_reward(success, t_sim, energy, task.horizon)
    return ValidationResult(uid=-1,
                            success=success,
                            time_sec=t_sim,
                            energy=energy,
                            score=score)


# ---------------------------------------------------------------------
# Tiny Tk dashboard (optional)
# ---------------------------------------------------------------------
def show_gui(res: ValidationResult, plan_hash: str) -> None:  # pragma: no cover
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:
        logger.warning(f"GUI skipped (Tk not available): {exc}")
        return

    root = tk.Tk()
    root.title("Swarm FlightPlan Validation")

    def add(label: str, value: str, row: int) -> None:
        ttk.Label(root, text=label, font=("Arial", 10, "bold"))\
            .grid(sticky="w", padx=6, pady=2, row=row, column=0)
        ttk.Label(root, text=value, font=("Arial", 10))\
            .grid(sticky="w", padx=6, pady=2, row=row, column=1)

    add("FlightPlan SHA‑256", plan_hash, 0)
    for i, (k, v) in enumerate(asdict(res).items(), start=1):
        add(k.replace("_", " ").title(), f"{v}", i)

    ttk.Button(root, text="Close", command=root.destroy)\
        .grid(pady=8, columnspan=2)
    root.mainloop()


# ---------------------------------------------------------------------
# Interactive demo entry point
# ---------------------------------------------------------------------
def run_demo(*,
             show_dashboard: bool = True,
             sim_gui: bool = False   # ←  default is now head‑less
            ) -> ValidationResult:   # pragma: no cover
    task = make_task()
    plan = make_plan(task, gui=sim_gui)

    # round‑trip serialisation check
    plan2 = FlightPlan.unpack(plan.pack())
    assert plan.sha256 == plan2.sha256, "SHA mismatch after msgpack round‑trip"

    res = validate(task, plan2, sim_gui=sim_gui)

    # Console summary
    logger.info("\n═══════════ Validation result ═══════════")
    logger.info(f"success      : {res.success}")
    logger.info(f"time_sec     : {res.time_sec:7.2f} (horizon = {task.horizon})")
    logger.info(f"energy       : {res.energy:7.2f}")
    logger.info(f"reward score : {res.score:7.3f}")
    logger.info(f"plan SHA‑256 : {plan2.sha256}")
    logger.info("═════════════════════════════════════════\n")

    if show_dashboard:
        show_gui(res, plan2.sha256)
    return res


# ---------------------------------------------------------------------
# Pytest entry point (unchanged)
# ---------------------------------------------------------------------
def test_flightplan_roundtrip_and_replay():          # pragma: no cover
    task = make_task()
    plan = make_plan(task, gui=False)

    plan2 = FlightPlan.unpack(plan.pack())
    assert plan.sha256 == plan2.sha256

    success, t_sim, energy = replay_once(task, plan2, gui=False)
    assert success
    assert t_sim <= task.horizon + 1e-6
    assert energy >= 0

    score = flight_reward(success, t_sim, energy, task.horizon)
    assert score > 0


# ---------------------------------------------------------------------
# CLI wrapper – choose GUI or not
# ---------------------------------------------------------------------
if __name__ == "__main__":                            # pragma: no cover
    ap = argparse.ArgumentParser(
        description="Run a single Swarm FlightPlan validation demo")
    ap.add_argument("--gui",  action="store_true",
                    help="open the PyBullet 3‑D viewer")
    ap.add_argument("--nodash", action="store_true",
                    help="suppress the Tkinter dashboard")
    args = ap.parse_args()

    run_demo(show_dashboard=not args.nodash,
             sim_gui=args.gui)
