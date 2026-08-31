"""Paths under `miner/` that are written relative to the file that uses them.

Moving a file changes what `parents[n]` means, and the code that walks up to the
repository root is where that goes wrong quietly: a trainer inserts the wrong
directory on `sys.path` and fails at import, long after the move.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

MINER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MINER_ROOT.parent

PARENTS = re.compile(r"Path\(__file__\)\.resolve\(\)\.parents\[(\d+)\]")

# These walk up to the repository root so they can import `swarm`. Named rather than
# globbed: a glob that stops matching takes its own coverage with it and stays green.
REACH_THE_REPO_ROOT = [
    MINER_ROOT / "src" / "miner.py",
    MINER_ROOT / "src" / "RL" / "test_RL.py",
    MINER_ROOT / "src" / "RL" / "cf_autopilot" / "train.py",
    MINER_ROOT / "src" / "RL" / "cf_interceptor" / "train.py",
    MINER_ROOT / "src" / "RL" / "cf_search_and_rescue" / "train.py",
    MINER_ROOT / "src" / "RL" / "cf_swarm_autopilot" / "train.py",
    MINER_ROOT / "src" / "RL" / "cf_swarm_sar" / "train.py",
]


def test_every_trainer_is_listed_here():
    """A new trainer has to be added above, or it goes unchecked."""
    found = set(MINER_ROOT.glob("src/RL/cf_*/train.py"))
    assert found == {p for p in REACH_THE_REPO_ROOT if p.name == "train.py"}


@pytest.mark.parametrize(
    "source", REACH_THE_REPO_ROOT, ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_the_walk_up_still_lands_on_the_repo_root(source):
    depths = {int(d) for d in PARENTS.findall(source.read_text())}
    assert depths, f"{source} no longer computes a path from its own location"
    landed = {source.resolve().parents[d] for d in depths}
    assert REPO_ROOT in landed, (
        f"{source.relative_to(REPO_ROOT)} walks up to {sorted(landed)}, "
        f"none of which is the repository root {REPO_ROOT}"
    )
