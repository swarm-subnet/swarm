#!/usr/bin/env python3
#TODO - not yet in use
"""
feedback_report.py  (Swarm-subnet edition)
──────────────────────────────────────────
Reads a JSON list of TaskFeedbackSynapse records produced by the Swarm
validator and prints:

    • overall distribution of final_score
    • pivot tables (average reward_score, average time_factor)
      with rows = validator_id   columns = miner_id
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, DefaultDict, Any

# optional prettier tables
try:
    from tabulate import tabulate
except ImportError:            # pragma: no cover
    tabulate = None


# ─────────────────── helpers ──────────────────────────────────────
def _ids(rec: Dict[str, Any]) -> Tuple[str, str]:
    """Return (validator_id, miner_id) as **str**."""
    return str(rec.get("validator_id", "unknown_validator")), str(
        rec.get("miner_id", "unknown_miner")
    )


def _scores(rec: Dict[str, Any]) -> Tuple[float | None, float | None]:
    """
    Extract (final_score, time_factor) from the evaluation_result block.
    reward_score is taken from *top-level* `score`.
    """
    er        = rec.get("evaluation_result", {})
    final     = er.get("final_score")
    time_fact = er.get("time_factor")
    reward    = rec.get("score")  # already EMA-adjusted reward

    return final, reward, time_fact


def _build_matrix(
    validators: List[str],
    miners: List[str],
    source: Dict[Tuple[str, str], List[float]],
) -> List[List[float | None]]:
    mat: List[List[float | None]] = []
    for v in validators:
        row: List[float | None] = []
        for m in miners:
            lst = source.get((v, m), [])
            row.append(round(statistics.mean(lst), 3) if lst else None)
        mat.append(row)
    return mat


def _print_table(title: str, validators: List[str], miners: List[str], matrix):
    print(f"\n=== {title} ===")
    if tabulate:
        hdr  = ["Validator \\ Miner"] + miners
        rows = [[v] + r for v, r in zip(validators, matrix)]
        print(tabulate(rows, headers=hdr, tablefmt="pretty", missingval="—"))
    else:  # simple fallback
        print("     " + " ".join(f"{m:>10}" for m in miners))
        for v, row in zip(validators, matrix):
            cells = [(f"{x:.3f}" if x is not None else "   —   ").rjust(10) for x in row]
            print(f"{v:>5} " + " ".join(cells))


# ─────────────────── main ─────────────────────────────────────────
def main(path: str | Path) -> None:
    path = Path(path).expanduser()
    if not path.exists():
        sys.exit(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        recs: List[Dict[str, Any]] = json.load(f)

    # global containers
    final_scores: List[float] = []
    reward_map: DefaultDict[Tuple[str, str], List[float]] = defaultdict(list)
    time_map: DefaultDict[Tuple[str, str], List[float]] = defaultdict(list)

    for rec in recs:
        v_id, m_id = _ids(rec)
        final, reward, time_fact = _scores(rec)

        if final  is not None:
            final_scores.append(final)
        if reward is not None:
            reward_map[(v_id, m_id)].append(reward)
        if time_fact is not None:
            time_map[(v_id, m_id)].append(time_fact)

    # ─ global stats ─
    print("\n=== Global Overview of final_score ===")
    if not final_scores:
        print("No final_score data present.")
    else:
        print(f"Count : {len(final_scores)}")
        print(f"Mean  : {statistics.mean(final_scores):.3f}")
        print(f"Median: {statistics.median(final_scores):.3f}")
        print(f"Min   : {min(final_scores):.3f}")
        print(f"Max   : {max(final_scores):.3f}")

    # unique IDs
    validators = sorted({k[0] for k in reward_map} | {k[0] for k in time_map})
    miners     = sorted({k[1] for k in reward_map} | {k[1] for k in time_map})

    # matrices
    reward_matrix = _build_matrix(validators, miners, reward_map)
    time_matrix   = _build_matrix(validators, miners, time_map)

    _print_table("Average reward_score", validators, miners, reward_matrix)
    _print_table("Average time_factor", validators, miners, time_matrix)


# ─────────────────── CLI ──────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python feedback_report.py path/to/feedback.json")
        sys.exit(1)
    main(sys.argv[1])
