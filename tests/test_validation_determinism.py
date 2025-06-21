"""
Validation determinism matrix test
==================================

Purpose
-------
Verify that *multiple* (task, flight‑plan) pairs are deterministic by
replaying each pair several times and checking that every replay returns
identical results.

Parameters (CLI)
----------------
  -t / --tasks     : Number of distinct tasks / flight‑plans to generate (N).
  -r / --replays   : Replays per task / plan (M).
  --tol            : Absolute tolerance for comparing floating values.
  --gui            : Enable the PyBullet GUI for **all** replays (OFF by default).

Example
-------
    # 20 random tasks × 50 replays each (no GUI)
    python test_validation_determinism_matrix.py -t 20 -r 50

    #  5 tasks × 10 replays with a very tight tolerance
    python test_validation_determinism_matrix.py -t 5 -r 10 --tol 1e-12
"""

from __future__ import annotations

import argparse
import math
import statistics
from collections import Counter
from typing import List, Dict, Tuple

# ── Swarm‑sim imports (identical to your original script) ────────────────────
from swarm.validator.task_gen import random_task
from swarm.validator.forward import SIM_DT, HORIZON_SEC
from swarm.flying_strategy.flying_strategy import flying_strategy
from swarm.validator.replay import replay_once
from swarm.validator.reward import flight_reward
from swarm.protocol import MapTask, FlightPlan, ValidationResult

try:
    from loguru import logger
except ImportError:  # pragma: no cover
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO)
    logger = _logging.getLogger(__name__)

# ──────────────────────────── Factories ──────────────────────────────────────
def make_task() -> MapTask:
    """Generate a random MapTask using the global SIM settings."""
    return random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC)


def make_plan(task: MapTask, *, gui: bool = False) -> FlightPlan:
    """Create a FlightPlan for *task* with the reference strategy."""
    cmds = flying_strategy(task, gui=gui)
    return FlightPlan(commands=cmds, sha256="")   # hash is computed automatically


# ───────────────────────── Determinism helpers ───────────────────────────────
def validate(task: MapTask,
             plan: FlightPlan,
             *,
             sim_gui: bool = False) -> ValidationResult:
    """Run one simulation and wrap the raw output into ValidationResult."""
    success, t_sim, energy = replay_once(task, plan, gui=sim_gui)
    score = flight_reward(success, t_sim, energy, task.horizon)
    return ValidationResult(uid=-1,
                            success=success,
                            time_sec=t_sim,
                            energy=energy,
                            score=score)


def results_equal(a: ValidationResult,
                  b: ValidationResult,
                  *,
                  tol: float) -> bool:
    """Compare two ValidationResult objects within *tol*."""
    return (a.success == b.success and
            math.isclose(a.time_sec, b.time_sec, abs_tol=tol) and
            math.isclose(a.energy,   b.energy,   abs_tol=tol) and
            math.isclose(a.score,    b.score,    abs_tol=tol))


def rounded_key(r: ValidationResult, decimals: int = 12) -> Tuple:
    """
    Hashable key for grouping identical outcomes
    (floats rounded to *decimals* places).
    """
    return (
        r.success,
        round(r.time_sec, decimals),
        round(r.energy,   decimals),
        round(r.score,    decimals),
    )


# ──────────────────────── Core matrix test logic ─────────────────────────────
def determinism_matrix(n_tasks: int,
                       m_replays: int,
                       tol: float,
                       *,
                       gui: bool = False) -> List[Dict]:
    """
    Run *n_tasks* different (task, plan) pairs, repeating each *m_replays* times.

    Returns a list with one dictionary per task containing:
        {
            "identical" : int,                 # count of identical replays
            "different" : int,                 # count of divergent replays
            "variants"  : Counter,             # frequency of every distinct result
            "ref"       : ValidationResult,    # reference result (first replay)
        }
    """
    summaries: List[Dict] = []

    for idx in range(1, n_tasks + 1):
        task = make_task()
        plan = make_plan(task, gui=False)        # always generate plan head‑less
        plan2 = FlightPlan.unpack(plan.pack())   # round‑trip integrity check
        assert plan.sha256 == plan2.sha256, "SHA mismatch after pack/unpack"

        # First replay establishes the reference result
        ref = validate(task, plan2, sim_gui=gui)
        variants = Counter({rounded_key(ref): 1})

        identical = 1  # reference run counts as identical to itself
        for _ in range(m_replays - 1):
            res = validate(task, plan2, sim_gui=gui)
            if results_equal(ref, res, tol=tol):
                identical += 1
            variants[rounded_key(res)] += 1

        different = m_replays - identical
        summaries.append({
            "identical": identical,
            "different": different,
            "variants": variants,
            "ref": ref,
        })

        # ── per‑task log line ───────────────────────────────────────────────
        logger.info(
            f"[Task {idx:03d}] identical={identical:>{len(str(m_replays))}d} / "
            f"{m_replays} | diverged={different}"
            + (" – NON‑DETERMINISTIC" if different else "")
        )

    return summaries


# ───────────────────────────── CLI entry point ───────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run N random tasks × M replays each to verify "
                    "determinism of the validation pipeline."
    )
    parser.add_argument("-t", "--tasks",   type=int, default=200,
                        help="Number of distinct tasks / flight‑plans to test (default: 10)")
    parser.add_argument("-r", "--replays", type=int, default=5,
                        help="Number of replays per task (default: 100)")
    parser.add_argument("--tol", type=float, default=1e-9,
                        help="Absolute tolerance for float comparisons (default: 1e-9)")
    parser.add_argument("--gui", action="store_true",
                        help="Show the PyBullet GUI for every replay")
    args = parser.parse_args()

    summaries = determinism_matrix(
        n_tasks=args.tasks,
        m_replays=args.replays,
        tol=args.tol,
        gui=args.gui
    )

    # ── Aggregate statistics ────────────────────────────────────────────────
    deterministic_tasks = sum(1 for s in summaries if s["different"] == 0)
    non_deterministic   = args.tasks - deterministic_tasks
    identical_counts    = [s["identical"] for s in summaries]

    logger.info("\n════════════ Overall summary ════════════")
    logger.info(f"Tasks tested         : {args.tasks}")
    logger.info(f"Deterministic tasks  : {deterministic_tasks}")
    logger.info(f"Non‑deterministic    : {non_deterministic}")
    logger.info(f"Identical replays    : "
                f"mean={statistics.mean(identical_counts):.1f}, "
                f"min={min(identical_counts)}, "
                f"max={max(identical_counts)} "
                f"(out of {args.replays})")
    logger.info("════════════════════════════════════════\n")

    # ── Details for non‑deterministic tasks (if any) ────────────────────────
    if non_deterministic:
        logger.info("Distinct outcome distributions per NON‑DETERMINISTIC task:")
        for idx, s in enumerate(summaries, start=1):
            if s["different"] == 0:
                continue
            logger.info(f"\n--- Task {idx:03d} --- "
                        f"{len(s['variants'])} distinct variants")
            for (succ, t_sec, energy, score), freq in s["variants"].most_common():
                logger.info(f"{freq:>{len(str(args.replays))}d} × "
                            f"success={succ}, "
                            f"time={t_sec:.9f}, "
                            f"energy={energy:.9f}, "
                            f"score={score:.9f}")
    logger.info("End of report.\n")


if __name__ == "__main__":  # pragma: no cover
    main()
