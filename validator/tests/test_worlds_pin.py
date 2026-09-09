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

"""The swarm-worlds pin is declared once, in requirements.txt. pixi resolves it
through the editable install and records the commit in pixi.lock, but it does
not notice when the tag in requirements.txt moves, so the two are compared here.
The installed package is checked too, so a stale environment fails loudly."""

from __future__ import annotations

import re
from pathlib import Path

import swarm_worlds

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REQUIREMENT = re.compile(r"^swarm-worlds @ git\+https://github\.com/swarm-subnet/swarm-worlds\.git@(v\d+\.\d+\.\d+)\s*$", re.M)
_LOCK_ENTRY = re.compile(r"git\+https://github\.com/swarm-subnet/swarm-worlds\.git\?rev=(v\d+\.\d+\.\d+)#[0-9a-f]{40}")


def _pinned_tag() -> str:
    match = _REQUIREMENT.search((_REPO_ROOT / "requirements.txt").read_text(encoding="utf-8"))
    assert match, "requirements.txt must pin swarm-worlds to a release tag"
    return match.group(1)


def test_requirements_pin_is_a_release_tag():
    assert _pinned_tag().startswith("v")


def test_pixi_lock_records_the_same_tag():
    lock = (_REPO_ROOT / "pixi.lock").read_text(encoding="utf-8")
    tags = set(_LOCK_ENTRY.findall(lock))
    assert tags == {_pinned_tag()}, "run `pixi lock` after changing the swarm-worlds tag"


def test_installed_package_matches_the_pin():
    assert f"v{swarm_worlds.__version__}" == _pinned_tag()
