from __future__ import annotations

# --------------------------------------------------------------------------- #
# Swarm imports (✱ nothing is re‑implemented ✱)
# --------------------------------------------------------------------------- #

from swarm.validator.task_gen import random_task
from swarm.validator.forward import SIM_DT, HORIZON_SEC            # same constants
from neurons.miner import flying_strategy                      # reference planner
from swarm.validator.replay import replay_once                     # physics engine
from swarm.validator.reward import flight_reward                   # canonical scorer
from swarm.protocol import MapTask, FlightPlan, ValidationResult   # data models

# Optional but nice: pretty log colours if loguru is installed
try:
    from loguru import logger
except ImportError:                                                # pragma: no cover
    import logging as logger
    logger.basicConfig(level=logger.INFO)


# --------------------------------------------------------------------------- #
# Helper: Build task ➜ plan ➜ validate
# --------------------------------------------------------------------------- #
def make_task() -> MapTask:
    """Generate the same kind of task the live validator issues."""
    return random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC)


def make_plan(task: MapTask) -> FlightPlan:
    """Run the miners reference strategy and wrap in a FlightPlan."""
    cmds = flying_strategy(task)
    # sha256 is auto‑calculated in __post_init__ when sha256=""
    return FlightPlan(commands=cmds, sha256="")


def validate(task: MapTask, plan: FlightPlan) -> ValidationResult:
    """Replay a plan and produce the exact structure the validator stores."""
    success, t_sim, energy = replay_once(task, plan)
    score                  = flight_reward(success, t_sim, energy, task.horizon)
    return ValidationResult(
        uid      = -1,          # not relevant in an offline test
        success  = success,
        time_sec = t_sim,
        energy   = energy,
        score    = score,
    )


# --------------------------------------------------------------------------- #
# GUI (minimal Tkinter – safe in headless CI; will just not show a window)
# --------------------------------------------------------------------------- #
def show_gui(res: ValidationResult, plan_hash: str) -> None:       # pragma: no cover
    """
    Render a very small read‑only dashboard.

    In a CI environment with no display, the call is harmless because
    `tk.Tk()` throws a `tk.TclError` only when *used*, not when created.
    """
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:                                        # nocv
        logger.warning(f"GUI skipped (Tk not available): {exc}")
        return

    root = tk.Tk()
    root.title("Swarm FlightPlan Validation")

    def add_row(label: str, value: str, row: int):
        ttk.Label(root, text=label, font=("Arial", 10, "bold")).grid(sticky="w", padx=6, pady=2, row=row, column=0)
        ttk.Label(root, text=value, font=("Arial", 10)).grid(sticky="w", padx=6, pady=2, row=row, column=1)

    add_row("FlightPlan SHA‑256", plan_hash,         0)
    add_row("⇧  Success",          str(res.success), 1)
    add_row("⇢  Simulated time",   f"{res.time_sec:.2f} s", 2)
    add_row("⚡ Energy",            f"{res.energy:.2f}",     3)
    add_row("★ Score",             f"{res.score:.3f}",      4)

    ttk.Button(root, text="Close", command=root.destroy).grid(pady=8, columnspan=2)
    root.mainloop()


# --------------------------------------------------------------------------- #
# Main entry point ‑ makes the script executable AND import‑safe for pytest
# --------------------------------------------------------------------------- #
def run_demo(show_window: bool = True) -> ValidationResult:        # pragma: no cover
    task: MapTask        = make_task()
    plan: FlightPlan     = make_plan(task)

    # Wire‑format round‑trip: pack ➜ unpack ➜ compare hash
    blob          = plan.pack()
    unpacked_plan = FlightPlan.unpack(blob)
    assert plan.sha256 == unpacked_plan.sha256, "SHA‑256 changed after (de)serialisation"

    res = validate(task, unpacked_plan)

    # Console report
    logger.info("")
    logger.info("╭─ Validation result ────────────────────────────────────────╮")
    logger.info(f"│  success      : {res.success}")
    logger.info(f"│  time_sec     : {res.time_sec:7.2f}  (task.horizon = {task.horizon})")
    logger.info(f"│  energy       : {res.energy:7.2f}")
    logger.info(f"│  reward score : {res.score:7.3f}")
    logger.info(f"│  plan SHA‑256 : {unpacked_plan.sha256}")
    logger.info("╰───────────────────────────────────────────────────────────╯")

    if show_window:
        show_gui(res, unpacked_plan.sha256)

    return res


# --------------------------------------------------------------------------- #
# PyTest‑compatible test function
# --------------------------------------------------------------------------- #
def test_flightplan_roundtrip_and_replay():
    """
    This is what pytest will run.

    It only *asserts* the invariants – success and hash conservation –
    to keep automated CI strict yet lightweight.
    """
    task  = make_task()
    plan  = make_plan(task)

    # Round‑trip integrity
    plan2 = FlightPlan.unpack(plan.pack())
    assert plan.sha256 == plan2.sha256, "SHA mismatch after msgpack round‑trip"

    # Replay must not error and should finish within the horizon
    success, t_sim, energy = replay_once(task, plan2)
    assert success, "Replay reports failure to reach goal"
    assert t_sim <= task.horizon + 1e-6, "Replay exceeded task horizon"
    assert energy >= 0, "Negative energy makes no sense"

    # Optional extra: reward non‑zero when success is True
    score = flight_reward(success, t_sim, energy, task.horizon)
    assert score > 0, "Expected positive reward for successful flight"


# --------------------------------------------------------------------------- #
# If executed directly: run an interactive demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":                                         # nocv
    run_demo(show_window=True)
