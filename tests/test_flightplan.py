"""
Miners! You can use this script to validate your FlightPlan!!
End-to-end round-trip & replay test – **with optional live GUIs**.
"""
from __future__ import annotations

import argparse
import statistics
from dataclasses import asdict
from typing import List, Optional

from swarm.validator.task_gen import random_task
from swarm.validator.forward import SIM_DT, HORIZON_SEC
from swarm.core.flying_strategy import flying_strategy           # reference strategy
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
    result = flying_strategy(task, gui=gui)
    # Extract commands from tuple result (commands, strategy_name, score)
    if isinstance(result, tuple):
        cmds = result[0]  # Get the List[RPMCmd] from the tuple
    else:
        cmds = result  # Handle case where it returns just the list
    return FlightPlan(commands=cmds, sha256="")      # hash auto-computed


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
# Interactive / batch demo entry point
# ---------------------------------------------------------------------
def run_demo(*,
             sim_gui: bool = False,   # ← default is head-less batch
             num_runs: Optional[int] = None
            ) -> Optional[List[ValidationResult]]:   # pragma: no cover

    if sim_gui:
        # Single GUI run, exactly as before
        task = make_task()
        plan = make_plan(task, gui=True)
        print(f"Plan created: {plan.commands}")

        res = validate(task, plan, sim_gui=True)

        # Console summary
        logger.info("\n═══════════ Validation result ═══════════")
        logger.info(f"success      : {res.success}")
        logger.info(f"time_sec     : {res.time_sec:7.2f} (horizon = {task.horizon})")
        logger.info(f"energy       : {res.energy:7.2f}")
        logger.info(f"reward score : {res.score:7.3f}")
        logger.info("═════════════════════════════════════════\n")

        return [res]

    # ── batch mode ──
    results: List[ValidationResult] = []
    if num_runs is None:
        num_runs = int(input("Enter the number of test runs: "))
    for i in range(1, num_runs + 1):
        task = make_task()
        plan = make_plan(task, gui=False)
        plan2 = FlightPlan.unpack(plan.pack())
        assert plan.sha256 == plan2.sha256, "SHA mismatch after msgpack round-trip"

        res = validate(task, plan2, sim_gui=False)
        results.append(res)
        logger.info(f"Run {i:3d}/{num_runs} : success={res.success}, "
                    f"time={res.time_sec:6.2f}, energy={res.energy:6.2f}, "
                    f"score={res.score:6.3f}")

    # Now compute statistics
    successes = sum(1 for r in results if r.success)
    times   = [r.time_sec for r in results]
    energies= [r.energy   for r in results]
    scores  = [r.score    for r in results]

    logger.info(f"\n═══════════ Batch statistics ({num_runs} runs) ═══════════")
    logger.info(f"Success rate  : {successes}/{num_runs} = {successes/num_runs:.1%}")
    logger.info(f"Time   (s)    : mean={statistics.mean(times):6.2f}, "
                f"min={min(times):6.2f}, max={max(times):6.2f}")
    logger.info(f"Energy        : mean={statistics.mean(energies):6.2f}, "
                f"min={min(energies):6.2f}, max={max(energies):6.2f}")
    logger.info(f"Score         : mean={statistics.mean(scores):6.3f}, "
                f"min={min(scores):6.3f}, max={max(scores):6.3f}")
    logger.info("═══════════════════════════════════════════════════\n")

    return results


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
# CLI wrapper – choose GUI or batch
# ---------------------------------------------------------------------
if __name__ == "__main__":                            # pragma: no cover
    ap = argparse.ArgumentParser(
        description="Run Swarm FlightPlan validation – single GUI or batch")
    ap.add_argument("--gui",  action="store_true",
                    help="open the PyBullet 3-D viewer (single run)")
    ap.add_argument("--runs", type=int, help="Number of test runs for batch mode")
    args = ap.parse_args()

    run_demo(sim_gui=args.gui, num_runs=args.runs)
