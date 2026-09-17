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

from pathlib import Path

import pytest

from swarm.validator import env_check

REPO_ROOT = Path(__file__).resolve().parents[2]


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

    monkeypatch.setattr(env_check, "_install", lambda pins: installed.append(pins) or True)
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
    monkeypatch.setattr(env_check, "_install", lambda pins: False)
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
