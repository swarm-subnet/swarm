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

import asyncio
import io
import os
import shutil
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

import pytest

from swarm.core.faults import ReasonCode
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


def _copy_ctx(tmp_path, removed: list) -> SimpleNamespace:
    """A started container's context over a staged folder holding a runner and a nested weights file."""
    submission = tmp_path / "submission"
    (submission / "models").mkdir(parents=True)
    (submission / "main.py").write_text("print('runner')\n")
    (submission / "models" / "weights.bin").write_bytes(bytes(1024))
    return SimpleNamespace(
        submission_dir=submission, container_name="swarm_eval_7_w0_1", current_uid=1000, current_gid=1001,
        launch_sec=0.0, failure_error="", helpers=SimpleNamespace(run_docker_cmd_quiet=removed.append),
    )


def test_the_staged_files_are_streamed_into_the_folder(tmp_path, monkeypatch):
    """Every staged file reaches the container's in-memory folder through docker exec, as the model's own user."""
    seen = {}

    def _exec(cmd, **kwargs):
        """Take the stream the way the container's tar would."""
        seen["cmd"] = list(cmd)
        seen["stream"] = kwargs["stdin"].read()
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(batch.subprocess, "run", _exec)
    removed: list = []
    assert batch._copy_submission_in(_copy_ctx(tmp_path, removed)) is None
    assert seen["cmd"] == [
        "docker", "exec", "-i", "--user", "1000:1001", "swarm_eval_7_w0_1",
        "tar", "-xf", "-", "-C", "/workspace/submission",
    ]
    with tarfile.open(fileobj=io.BytesIO(seen["stream"])) as archive:
        assert {"./main.py", "./models/weights.bin"} <= set(archive.getnames())
    assert removed == []


@pytest.mark.parametrize("outcome", ["refused", "timeout"])
def test_a_failed_copy_removes_the_container_as_an_infrastructure_fault(tmp_path, monkeypatch, outcome):
    """A copy that fails or hangs removes the container and is charged to the validator, never to the miner."""

    def _exec(cmd, **kwargs):
        """Fail the copy the scripted way."""
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
        return SimpleNamespace(returncode=2, stdout="", stderr="tar: Cannot write: No space left on device")

    monkeypatch.setattr(batch.subprocess, "run", _exec)
    removed: list = []
    ctx = _copy_ctx(tmp_path, removed)
    assert batch._copy_submission_in(ctx) is ReasonCode.INFRA_DOCKER
    assert removed == [["docker", "rm", "-f", "swarm_eval_7_w0_1"]]
    assert ctx.failure_error.startswith("copy into the container failed")


def _record_start(monkeypatch, copy_result=None) -> list:
    """Replace each container start step with one that logs its name; the copy returns ``copy_result``."""
    steps: list = []

    async def _lockdown_and_gate(ctx):
        """Log the lockdown, which is what opens the start gate."""
        steps.append("lockdown_and_gate")

    monkeypatch.setattr(batch, "_setup_workspace", lambda ctx: steps.append("setup"))
    monkeypatch.setattr(batch, "_launch_container", lambda ctx, warm=False: steps.append("launch"))
    monkeypatch.setattr(batch, "_copy_submission_in", lambda ctx: steps.append("copy") or copy_result)
    monkeypatch.setattr(batch, "_prepare_network_and_rpc", _lockdown_and_gate)
    return steps


@pytest.mark.parametrize("warm", [False, True])
def test_files_land_after_the_launch_and_before_the_gate_opens(monkeypatch, warm):
    """Fresh and spare containers alike get their files after the launch and before the lockdown opens the gate."""
    steps = _record_start(monkeypatch)
    assert asyncio.run(batch._prepare_container(SimpleNamespace(), warm=warm)) is None
    assert steps == ["setup", "launch", "copy", "lockdown_and_gate"]


def test_a_failed_copy_never_opens_the_gate(monkeypatch):
    """When the files do not land, the start ends there and the gate stays shut."""
    steps = _record_start(monkeypatch, copy_result=ReasonCode.INFRA_DOCKER)
    assert asyncio.run(batch._prepare_container(SimpleNamespace())) is ReasonCode.INFRA_DOCKER
    assert steps == ["setup", "launch", "copy"]
