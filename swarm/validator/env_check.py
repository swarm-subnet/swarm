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

A package pinned to a release tag of its git repository, as swarm-worlds is, has
no version to compare: its version can stay the same across commits that add or
drop files. The tag the installer recorded is compared instead.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from typing import Dict, List, Tuple

import bittensor as bt

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = REPO_ROOT / "requirements.txt"
# Set before re-executing, so a pin that cannot be satisfied costs one restart and not
# an endless loop of them.
REPAIRED_MARKER = "SWARM_ENV_REPAIRED"
# Room for a wheel built from source, while a stalled index or git fetch still ends as a failed repair.
REPAIR_TIMEOUT_S = 1800
# What a package installed editable from a folder reports: someone's working copy, which the repair leaves alone.
LOCAL_CHECKOUT = "a local checkout"
# `name==version`, ignoring extras, markers and anything that is not an exact pin.
_PIN = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*([^\s;#]+)")
# `name @ git+<repository>@<tag>`: a release tag of a git repository.
_TAG_PIN = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*@\s*(git\+[^\s;#]+)@([^\s;#@/]+)\s*(?:[;#].*)?$")


def _lines(requirements: Path) -> List[str]:
    """The requirements file's lines, or none when it cannot be read."""
    try:
        return requirements.read_text().splitlines()
    except OSError:
        return []


def required_versions(requirements: Path = REQUIREMENTS) -> Dict[str, str]:
    """Every exactly pinned package in the requirements file, by distribution name."""
    pins: Dict[str, str] = {}
    for line in _lines(requirements):
        match = _PIN.match(line)
        if match:
            pins[match.group(1).lower()] = match.group(2)
    return pins


def required_tags(requirements: Path = REQUIREMENTS) -> Dict[str, Tuple[str, str]]:
    """Every package pinned to a release tag of a git repository, by distribution name, as (requirement, tag)."""
    pins: Dict[str, Tuple[str, str]] = {}
    for line in _lines(requirements):
        match = _TAG_PIN.match(line)
        if match:
            name, repository, tag = match.groups()
            pins[name.lower()] = (f"{name} @ {repository}@{tag}", tag)
    return pins


def installed_revision(name: str) -> str:
    """Which revision of a git-pinned package is installed: the tag (or, without one, the commit) its installer
    recorded, LOCAL_CHECKOUT for an editable folder, its version when it came from an index, or "missing"."""
    try:
        dist = distribution(name)
    except PackageNotFoundError:
        return "missing"
    origin = json.loads(dist.read_text("direct_url.json") or "{}")
    vcs = origin.get("vcs_info") or {}
    if vcs:
        return vcs.get("requested_revision") or vcs.get("commit_id") or "unknown"
    if (origin.get("dir_info") or {}).get("editable"):
        return LOCAL_CHECKOUT
    return dist.version


def mismatches(requirements: Path = REQUIREMENTS) -> List[Tuple[str, str, str]]:
    """Pinned packages whose installed version or tag differs, as (name, wanted, installed).

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
        if not _satisfies(have, wanted):
            found.append((name, wanted, have))
    for name, (_requirement, tag) in required_tags(requirements).items():
        have = installed_revision(name)
        if have not in (tag, tag.removeprefix("v")):
            found.append((name, tag, have))
    return found


def _satisfies(have: str, wanted: str) -> bool:
    """Whether an installed version is the pinned one, allowing a build label the pin does not name.

    ``torch==2.10.0`` is met by ``2.10.0+cpu``: the local segment says which wheel was
    chosen, and the CPU wheel is the same release as the default one.
    """
    if "+" in wanted:
        return have == wanted
    return have.split("+", 1)[0] == wanted


def _install(pins: List[Tuple[str, str, str]], requirements: Path = REQUIREMENTS) -> bool:
    """Install the wanted version of every mismatched package, reporting success.

    A git-pinned package is installed afresh from its tag and alone: an install that
    already reports the same version would otherwise be kept as it is.
    """
    tags = required_tags(requirements)
    exact = [f"{name}=={wanted}" for name, wanted, _ in pins if name not in tags]
    tagged = [tags[name][0] for name, _, _ in pins if name in tags]
    for targets, options in ((exact, []), (tagged, ["--force-reinstall", "--no-deps"])):
        if not targets:
            continue
        bt.logging.warning(f"Installing {', '.join(targets)}")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet", *options, *targets],
                capture_output=True,
                text=True,
                timeout=REPAIR_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            bt.logging.error(f"Could not repair the environment: pip gave no answer in {REPAIR_TIMEOUT_S} s")
            return False
        if result.returncode != 0:
            bt.logging.error(f"Could not repair the environment: {result.stderr.strip()[-500:]}")
            return False
    return True


def ensure_environment(requirements: Path = REQUIREMENTS) -> None:
    """Bring the running environment onto its pins, restarting the process to load them.

    Does nothing when everything already matches, which is the normal case. After one
    repair the process re-executes; a mismatch that survives that is logged and the
    validator carries on, because refusing to start would take the fleet down over a
    pin that cannot be satisfied on that host. A local checkout is reported and left
    as it is: it is someone's working copy, not a stale install.
    """
    found = mismatches(requirements)
    if not found:
        return

    for name, wanted, have in found:
        if have == LOCAL_CHECKOUT:
            bt.logging.warning(f"{name} is {LOCAL_CHECKOUT}, this validator pins {wanted}; leaving it as it is")
        else:
            bt.logging.warning(f"{name} {have} is installed, this validator pins {wanted}")
    found = [pin for pin in found if pin[2] != LOCAL_CHECKOUT]
    if not found:
        return

    if os.environ.get(REPAIRED_MARKER) == "1":
        bt.logging.error(
            "The environment still does not match its pins after a repair; "
            "this validator will score differently from the rest of the fleet"
        )
        return

    if not _install(found, requirements):
        return

    bt.logging.warning("Restarting to load the repaired environment")
    os.environ[REPAIRED_MARKER] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
