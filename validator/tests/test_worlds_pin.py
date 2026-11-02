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

"""The swarm-worlds and swarm-description pins are each declared once, in requirements.txt. pixi
resolves them through the editable install and records the commits in pixi.lock, but it does not
notice when a tag in requirements.txt moves, so the two are compared here. The installed packages
are checked too, so a stale environment fails loudly."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import swarm_description
import swarm_worlds

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGES = {"swarm-worlds": swarm_worlds, "swarm-description": swarm_description}


def _pinned_tag(name: str) -> str:
    """The release tag requirements.txt pins the named package to, or a failure when it pins none."""
    pattern = rf"^{name} @ git\+https://github\.com/swarm-subnet/{name}\.git@(v\d+\.\d+\.\d+)\s*$"
    match = re.search(pattern, (_REPO_ROOT / "requirements.txt").read_text(encoding="utf-8"), re.M)
    assert match, f"requirements.txt must pin {name} to a release tag"
    return match.group(1)


@pytest.mark.parametrize("name", _PACKAGES)
def test_requirements_pin_is_a_release_tag(name):
    """Proves the dependency points at a release tag rather than a branch or a bare commit."""
    assert _pinned_tag(name).startswith("v")


@pytest.mark.parametrize("name", _PACKAGES)
def test_pixi_lock_records_the_same_tag(name):
    """Proves the lock was refreshed after the tag moved, so both name one commit."""
    lock = (_REPO_ROOT / "pixi.lock").read_text(encoding="utf-8")
    entry = rf"git\+https://github\.com/swarm-subnet/{name}\.git\?rev=(v\d+\.\d+\.\d+)#[0-9a-f]{{40}}"
    assert set(re.findall(entry, lock)) == {_pinned_tag(name)}, f"run `pixi lock` after changing the {name} tag"


@pytest.mark.parametrize("name", _PACKAGES)
def test_installed_package_matches_the_pin(name):
    """Proves the environment actually holds the pinned release, not a stale install."""
    assert f"v{_PACKAGES[name].__version__}" == _pinned_tag(name)
