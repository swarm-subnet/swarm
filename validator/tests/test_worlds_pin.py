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
are checked too, so a stale environment fails loudly. A tag may carry a `-pre.N` suffix while the
release it names is still under review."""

from __future__ import annotations

import json
import re
from importlib.metadata import distribution
from pathlib import Path

import pytest
import swarm_description
import swarm_worlds

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGES = {"swarm-worlds": swarm_worlds, "swarm-description": swarm_description}
_TAG = r"v\d+\.\d+\.\d+(?:-pre\.\d+)?"


def _pinned_tag(name: str) -> str:
    """The release tag requirements.txt pins the named package to, or a failure when it pins none."""
    pattern = rf"^{name} @ git\+https://github\.com/swarm-subnet/{name}\.git@({_TAG})\s*$"
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
    entry = rf"git\+https://github\.com/swarm-subnet/{name}\.git\?rev=({_TAG})#[0-9a-f]{{40}}"
    assert set(re.findall(entry, lock)) == {_pinned_tag(name)}, f"run `pixi lock` after changing the {name} tag"


def test_evaluator_image_pins_the_same_description_tag():
    """Proves the model's evaluation container installs the swarm-description that swarm.constants imports."""
    text = (_REPO_ROOT / "swarm" / "validator" / "docker" / "docker-requirements.txt").read_text(encoding="utf-8")
    pattern = rf"^swarm-description @ git\+https://github\.com/swarm-subnet/swarm-description\.git@({_TAG})\s*$"
    assert re.findall(pattern, text, re.M) == [_pinned_tag("swarm-description")]


@pytest.mark.parametrize("name", _PACKAGES)
def test_installed_package_matches_the_pin(name):
    """Proves the environment actually holds the pinned release, not a stale install."""
    assert f"v{_PACKAGES[name].__version__}" == _pinned_tag(name).split("-pre.")[0]
    # Two pre-releases share one version number, so a git install is also checked by the tag it came from.
    origin = distribution(name).read_text("direct_url.json")
    requested = json.loads(origin).get("vcs_info", {}).get("requested_revision") if origin else None
    assert requested in (None, _pinned_tag(name)), f"{name} was installed from {requested}, not {_pinned_tag(name)}"


@pytest.mark.parametrize("tag", ["v1.0.0", "v1.0.0-pre.1", "v12.3.4-pre.10"])
def test_release_and_pre_release_tags_are_accepted(tag):
    """Proves a final release tag and a `-pre.N` tag both count as a fixed release."""
    assert re.fullmatch(_TAG, tag)


@pytest.mark.parametrize("ref", ["main", "feature/ali/swarm-description", "aceee7b", "v1.0.0rc1", "v1.0.0-pre"])
def test_branches_commits_and_other_suffixes_are_rejected(ref):
    """Proves a moving branch, a bare commit or an unnumbered suffix never passes as a release."""
    assert not re.fullmatch(_TAG, ref)
