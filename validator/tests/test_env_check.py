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

"""The startup check that keeps a validator on the versions it pins."""
from __future__ import annotations

import json
import sys
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace

import pytest

from swarm.validator import env_check

REPO_ROOT = Path(__file__).resolve().parents[2]
WORLDS = "git+https://github.com/swarm-subnet/swarm-worlds.git"
WORLDS_PIN = f"swarm-worlds @ {WORLDS}@v1.0.0"


def _requirements(tmp_path: Path, body: str) -> Path:
    """A requirements file holding exactly the given lines."""
    path = tmp_path / "requirements.txt"
    path.write_text(body)
    return path


def test_only_exact_pins_are_read(tmp_path: Path) -> None:
    """Proves a range is not treated as a pin, so it is never 'repaired' to its floor."""
    path = _requirements(tmp_path, "\n".join([
        "# a comment",
        "numpy==2.4.6",
        "loguru>=0.7.0",
        "torch==2.10.0",
        "swarm-worlds @ git+https://example.invalid/x.git@v1.0.0",
        "",
    ]))
    assert env_check.required_versions(path) == {"numpy": "2.4.6", "torch": "2.10.0"}


def test_a_missing_package_counts_as_a_mismatch(tmp_path: Path) -> None:
    """Proves an absent pin is reported, since it breaks a run as surely as a wrong one."""
    path = _requirements(tmp_path, "swarm-package-that-does-not-exist==1.0.0\n")
    assert env_check.mismatches(path) == [
        ("swarm-package-that-does-not-exist", "1.0.0", "missing")
    ]


def test_a_cpu_wheel_meets_the_plain_pin(tmp_path: Path, monkeypatch) -> None:
    """Proves torch 2.10.0+cpu, the wheel the image ships, is not reported as drift.

    A pin that names the label itself is still exact, so a host on the default
    wheel cannot pass as the CPU one.
    """
    monkeypatch.setattr(env_check, "version", lambda name: "2.10.0+cpu")
    assert env_check.mismatches(_requirements(tmp_path, "torch==2.10.0\n")) == []
    assert env_check.mismatches(_requirements(tmp_path, "torch==2.10.0+cpu\n")) == []
    assert env_check.mismatches(_requirements(tmp_path, "torch==2.10.1\n")) == [
        ("torch", "2.10.1", "2.10.0+cpu")
    ]
    monkeypatch.setattr(env_check, "version", lambda name: "2.10.0")
    assert env_check.mismatches(_requirements(tmp_path, "torch==2.10.0+cpu\n")) == [
        ("torch", "2.10.0+cpu", "2.10.0")
    ]


def test_a_matching_environment_reports_nothing(tmp_path: Path) -> None:
    """Proves the normal case is silent, so the log only speaks when something is wrong."""
    installed = env_check.version("pytest")
    path = _requirements(tmp_path, f"pytest=={installed}\n")
    assert env_check.mismatches(path) == []


def test_a_clean_environment_does_not_install_or_restart(tmp_path: Path, monkeypatch) -> None:
    """Proves a healthy validator is never restarted by this check."""
    installed = env_check.version("pytest")
    path = _requirements(tmp_path, f"pytest=={installed}\n")

    def _fail(*args, **kwargs):
        """Stand in for the repair and the restart, and fail if either is reached."""
        raise AssertionError("a matching environment must not be touched")

    monkeypatch.setattr(env_check, "_install", _fail)
    monkeypatch.setattr(env_check.os, "execv", _fail)
    env_check.ensure_environment(path)


def test_a_drifted_environment_installs_the_pin_and_restarts(tmp_path: Path, monkeypatch) -> None:
    """Proves the repair runs and the process re-execs, which is what loads the new version."""
    path = _requirements(tmp_path, "pytest==0.0.1\n")
    installed: list = []
    restarted: list = []

    monkeypatch.setattr(env_check, "_install", lambda pins, requirements: installed.append(pins) or True)
    monkeypatch.setattr(env_check.os, "execv", lambda exe, argv: restarted.append(argv))
    monkeypatch.delenv(env_check.REPAIRED_MARKER, raising=False)

    env_check.ensure_environment(path)

    assert installed == [[("pytest", "0.0.1", env_check.version("pytest"))]]
    assert restarted, "the process must re-exec, a loaded module cannot be swapped in place"


def test_a_second_pass_does_not_restart_again(tmp_path: Path, monkeypatch) -> None:
    """Proves a pin that cannot be satisfied costs one restart, not an endless loop."""
    path = _requirements(tmp_path, "pytest==0.0.1\n")
    monkeypatch.setenv(env_check.REPAIRED_MARKER, "1")

    def _fail(*args, **kwargs):
        """Stand in for the repair and the restart, and fail if either is reached."""
        raise AssertionError("the repair must not run twice")

    monkeypatch.setattr(env_check, "_install", _fail)
    monkeypatch.setattr(env_check.os, "execv", _fail)
    env_check.ensure_environment(path)


def test_a_failed_install_does_not_restart(tmp_path: Path, monkeypatch) -> None:
    """Proves a validator whose repair failed keeps running rather than looping on exec."""
    path = _requirements(tmp_path, "pytest==0.0.1\n")
    monkeypatch.setattr(env_check, "_install", lambda pins, requirements: False)
    monkeypatch.setattr(env_check.os, "execv", lambda *a: pytest.fail("must not restart"))
    monkeypatch.delenv(env_check.REPAIRED_MARKER, raising=False)
    env_check.ensure_environment(path)


def test_the_shipped_requirements_pin_numpy() -> None:
    """Proves the one dependency that drifted across the fleet is pinned exactly.

    A range leaves an already-satisfied install untouched for good, which is how one
    validator ran a different numpy from the rest while every update reported success.
    """
    pins = env_check.required_versions(REPO_ROOT / "requirements.txt")
    assert "numpy" in pins, "numpy must be pinned exactly, not given as a range"


def _installed_as(monkeypatch, version: str = "", direct_url=None) -> None:
    """Stand swarm-worlds in as installed at that version with that record of where it came from; no version means
    it is not installed at all."""
    def distribution(name):
        """The stand-in install, or none."""
        if not version:
            raise PackageNotFoundError(name)
        record = None if direct_url is None else json.dumps(direct_url)
        return SimpleNamespace(version=version, read_text=lambda file: record if file == "direct_url.json" else None)

    monkeypatch.setattr(env_check, "distribution", distribution)


def test_a_git_tag_pin_is_read_with_its_tag(tmp_path: Path) -> None:
    """Proves a package pinned to a git tag is read with the tag and the line pip installs it from, and that a git
    requirement without a tag is not taken for a pin."""
    path = _requirements(tmp_path, "\n".join([
        "numpy==2.4.6",
        f"{WORLDS_PIN}  # assets",
        "loose @ git+https://example.invalid/loose.git",
        "",
    ]))
    assert env_check.required_tags(path) == {"swarm-worlds": (WORLDS_PIN, "v1.0.0")}
    assert env_check.required_versions(path) == {"numpy": "2.4.6"}


def test_the_shipped_requirements_pin_swarm_worlds_to_a_tag() -> None:
    """Proves the check covers the swarm-worlds the validator actually ships with."""
    _requirement, tag = env_check.required_tags(REPO_ROOT / "requirements.txt")["swarm-worlds"]
    assert tag.startswith("v")


@pytest.mark.parametrize(("version", "direct_url", "found"), [
    ("1.0.0", {"url": WORLDS[4:], "vcs_info": {"vcs": "git", "requested_revision": "v1.0.0", "commit_id": "4084a9f"}},
     []),
    ("1.0.0", None, []),
    ("1.1.0", {"url": WORLDS[4:], "vcs_info": {"vcs": "git", "requested_revision": "v1.1.0", "commit_id": "51f7890"}},
     [("swarm-worlds", "v1.0.0", "v1.1.0")]),
    ("1.1.0", {"url": WORLDS[4:], "vcs_info": {"vcs": "git", "commit_id": "51f7890"}},
     [("swarm-worlds", "v1.0.0", "51f7890")]),
    ("1.1.0", None, [("swarm-worlds", "v1.0.0", "1.1.0")]),
    ("1.1.0", {"url": "file:///home/dev/swarm-worlds", "dir_info": {"editable": True}},
     [("swarm-worlds", "v1.0.0", env_check.LOCAL_CHECKOUT)]),
    ("", None, [("swarm-worlds", "v1.0.0", "missing")]),
], ids=["tag", "index-release", "other-tag", "branch-commit", "other-release", "local-checkout", "missing"])
def test_swarm_worlds_is_compared_by_the_tag_it_was_installed_from(tmp_path, monkeypatch, version, direct_url, found):
    """Proves a stale swarm-worlds is noticed however it was installed: from another tag, from a branch's commit,
    from another release, or not at all, while the pinned tag or its release is left alone."""
    _installed_as(monkeypatch, version, direct_url)
    assert env_check.mismatches(_requirements(tmp_path, f"{WORLDS_PIN}\n")) == found


def test_a_stale_swarm_worlds_is_reinstalled_from_its_tag_and_the_validator_restarts(tmp_path, monkeypatch) -> None:
    """Proves the repair reinstalls the tagged release afresh, alone, in its own pip run after the exact pins, and
    then re-executes the validator to load it."""
    _installed_as(monkeypatch, "1.1.0", {"url": WORLDS[4:], "vcs_info": {"vcs": "git", "commit_id": "51f7890"}})
    monkeypatch.setattr(env_check, "version", lambda name: "0.0.1")
    runs, restarted = [], []
    monkeypatch.setattr(env_check.subprocess, "run",
                        lambda argv, **kwargs: runs.append(argv) or SimpleNamespace(returncode=0, stderr=""))
    monkeypatch.setattr(env_check.os, "execv", lambda exe, argv: restarted.append(argv))
    monkeypatch.delenv(env_check.REPAIRED_MARKER, raising=False)

    env_check.ensure_environment(_requirements(tmp_path, f"pytest==1.0.0\n{WORLDS_PIN}\n"))

    pip = [sys.executable, "-m", "pip", "install", "--quiet"]
    assert runs == [pip + ["pytest==1.0.0"], pip + ["--force-reinstall", "--no-deps", WORLDS_PIN]]
    assert restarted


def test_a_failed_tag_repair_does_not_restart(tmp_path, monkeypatch) -> None:
    """Proves a tag that cannot be installed leaves the validator running, as a failed version pin does."""
    _installed_as(monkeypatch, "1.1.0", None)
    monkeypatch.setattr(env_check.subprocess, "run",
                        lambda argv, **kwargs: SimpleNamespace(returncode=1, stderr="could not reach the repository"))
    monkeypatch.setattr(env_check.os, "execv", lambda *a: pytest.fail("must not restart"))
    monkeypatch.delenv(env_check.REPAIRED_MARKER, raising=False)
    env_check.ensure_environment(_requirements(tmp_path, f"{WORLDS_PIN}\n"))


def test_a_local_checkout_is_reported_and_left_alone(tmp_path, monkeypatch) -> None:
    """Proves a swarm-worlds someone installed from their own folder is named in the log but never overwritten."""
    _installed_as(monkeypatch, "1.1.0", {"url": "file:///home/dev/swarm-worlds", "dir_info": {"editable": True}})
    warned = []
    monkeypatch.setattr(env_check.bt.logging, "warning", warned.append)
    monkeypatch.setattr(env_check, "_install", lambda *a: pytest.fail("a local checkout must not be reinstalled"))
    monkeypatch.setattr(env_check.os, "execv", lambda *a: pytest.fail("must not restart"))
    monkeypatch.delenv(env_check.REPAIRED_MARKER, raising=False)
    env_check.ensure_environment(_requirements(tmp_path, f"{WORLDS_PIN}\n"))
    assert warned == [f"swarm-worlds is {env_check.LOCAL_CHECKOUT}, this validator pins v1.0.0; leaving it as it is"]
