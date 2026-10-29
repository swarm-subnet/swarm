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

"""The start gate holds every file of a submission back until the validator has locked the network."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from swarm.validator.docker.docker_evaluator_parts import _shared, batch

# a submission file named like a module the runner imports; it only leaves a mark
_STAND_IN = 'import pathlib\npathlib.Path(__file__).with_name("ran").touch()\nraise SystemExit(0)\n'


@pytest.fixture
def staged(tmp_path, monkeypatch):
    """A submission folder holding the trusted runner beside a stand-in standard module, with ``python`` on PATH."""
    submission = tmp_path / "submission"
    submission.mkdir()
    for name in ("main.py", "runtime_caps.py"):
        shutil.copy(_shared._submission_template_dir() / name, submission)
    (submission / "dataclasses.py").write_text(_STAND_IN)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    python.chmod(0o755)
    gate = tmp_path / "gate"
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("SWARM_START_GATE", str(gate))
    return SimpleNamespace(main=submission / "main.py", mark=submission / "ran", gate=gate)


def test_no_submission_file_runs_before_the_gate(staged):
    """A submission module shadowing a runner import stays unloaded until the gate opens, then loads."""
    proc = subprocess.Popen(
        batch._gated_runner_command(str(staged.main), gate=str(staged.gate)),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(0.3)
        assert proc.poll() is None
        assert not staged.mark.exists()
        staged.gate.touch()
        proc.wait(timeout=10)
        assert staged.mark.exists()
    finally:
        proc.kill()
        proc.wait()


def test_the_runner_never_starts_when_the_gate_stays_shut(staged):
    """With no gate before the deadline the shell exits with an error and Python is never started."""
    result = subprocess.run(
        batch._gated_runner_command(str(staged.main), gate=str(staged.gate), timeout_sec=0.2),
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 1
    assert "start gate never opened" in result.stderr
    assert not staged.mark.exists()


def test_a_failed_preparation_stops_the_start(staged):
    """When the step before the wait fails, the shell exits and Python never starts, even with the gate open."""
    staged.gate.touch()
    result = subprocess.run(
        batch._gated_runner_command(str(staged.main), gate=str(staged.gate), prepare="false"),
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 1
    assert not staged.mark.exists()
