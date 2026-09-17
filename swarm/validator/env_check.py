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

"""The versions this validator is actually running, checked against the ones it pins.

A validator whose packages drift from `requirements.txt` scores the same flight
differently from the rest of the fleet, and nothing says so: the run looks healthy
and only the numbers disagree. Installing from a range leaves this in place for
good, because an already-satisfied range is never upgraded.

The repair installs the pins and re-executes, since a module already imported
cannot be swapped underneath the running process.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Dict, List, Tuple

import bittensor as bt

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = REPO_ROOT / "requirements.txt"
# Set before re-executing, so a pin that cannot be satisfied costs one restart and not
# an endless loop of them.
REPAIRED_MARKER = "SWARM_ENV_REPAIRED"
# `name==version`, ignoring extras, markers and anything that is not an exact pin.
_PIN = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*([^\s;#]+)")


def required_versions(requirements: Path = REQUIREMENTS) -> Dict[str, str]:
    """Every exactly pinned package in the requirements file, by distribution name."""
    pins: Dict[str, str] = {}
    try:
        lines = requirements.read_text().splitlines()
    except OSError:
        return pins
    for line in lines:
        match = _PIN.match(line)
        if match:
            pins[match.group(1).lower()] = match.group(2)
    return pins


def mismatches(requirements: Path = REQUIREMENTS) -> List[Tuple[str, str, str]]:
    """Pinned packages whose installed version differs, as (name, wanted, installed).

    A package that is pinned but absent is reported with "missing" as its installed
    version, because that breaks a run just as surely as a wrong one.
    """
    found: List[Tuple[str, str, str]] = []
    for name, wanted in required_versions(requirements).items():
        try:
            have = version(name)
        except PackageNotFoundError:
            found.append((name, wanted, "missing"))
            continue
        if have != wanted:
            found.append((name, wanted, have))
    return found


def _install(pins: List[Tuple[str, str, str]]) -> bool:
    """Install the wanted version of every mismatched package, reporting success."""
    targets = [f"{name}=={wanted}" for name, wanted, _ in pins]
    bt.logging.warning(f"Installing {', '.join(targets)}")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", *targets],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        bt.logging.error(f"Could not repair the environment: {result.stderr.strip()[-500:]}")
        return False
    return True


def ensure_environment(requirements: Path = REQUIREMENTS) -> None:
    """Bring the running environment onto its pins, restarting the process to load them.

    Does nothing when everything already matches, which is the normal case. After one
    repair the process re-executes; a mismatch that survives that is logged and the
    validator carries on, because refusing to start would take the fleet down over a
    pin that cannot be satisfied on that host.
    """
    found = mismatches(requirements)
    if not found:
        return

    for name, wanted, have in found:
        bt.logging.warning(f"{name} {have} is installed, this validator pins {wanted}")

    if os.environ.get(REPAIRED_MARKER) == "1":
        bt.logging.error(
            "The environment still does not match its pins after a repair; "
            "this validator will score differently from the rest of the fleet"
        )
        return

    if not _install(found):
        return

    bt.logging.warning("Restarting to load the repaired environment")
    os.environ[REPAIRED_MARKER] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
