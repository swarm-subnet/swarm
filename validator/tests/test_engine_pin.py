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
"""The engine pin appears in both requirement files and the lock. Nothing ties them
together, so they are compared here, and the installed engine is checked against the pin."""

from __future__ import annotations

import re
from importlib.metadata import version
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REQUIREMENT = re.compile(r"^swarm-bullet3==(\S+)\s*$", re.M)
_LOCK_ENTRY = re.compile(r"^  name: swarm-bullet3\n  version: (\S+)$", re.M)
_LOCK_SOURCE = re.compile(r"^- pypi: (\S+)\n  name: swarm-bullet3$", re.M)


def _pin(relative: str) -> str:
    """The engine version a requirement file pins, or a failure when it pins none."""
    match = _REQUIREMENT.search((_REPO_ROOT / relative).read_text(encoding="utf-8"))
    assert match, f"{relative} must pin swarm-bullet3 to one version"
    return match.group(1)


def test_both_requirement_files_pin_the_same_engine():
    """Proves the validator's own list runs the engine the repository pins."""
    assert _pin("validator/requirements.txt") == _pin("requirements.txt")


def test_pixi_lock_records_the_pinned_engine():
    """Proves the lock was refreshed after the engine pin moved."""
    lock = (_REPO_ROOT / "pixi.lock").read_text(encoding="utf-8")
    assert set(_LOCK_ENTRY.findall(lock)) == {_pin("requirements.txt")}, "run `pixi lock` after changing the engine pin"


def test_pixi_lock_takes_the_engine_from_pypi():
    """Proves the lock names the published wheel, not a file on the machine that wrote it."""
    lock = (_REPO_ROOT / "pixi.lock").read_text(encoding="utf-8")
    sources = _LOCK_SOURCE.findall(lock)
    assert sources and all(source.startswith("https://files.pythonhosted.org/") for source in sources), sources


def test_installed_engine_matches_the_pin():
    """Proves the environment actually holds the pinned engine, not a stale install."""
    assert version("swarm-bullet3") == _pin("requirements.txt")
