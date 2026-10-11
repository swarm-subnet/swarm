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

"""What the first-time model check hands the host's Docker daemon to mount."""
from __future__ import annotations

import asyncio
from pathlib import Path

from swarm.core import model_verify
from swarm.validator.docker import docker_evaluator


class _ReadyEvaluator:
    """Stand in for the evaluator with its base image already built."""

    _base_ready = True
    base_image = "swarm_evaluator_base:latest"


class _FinishedProcess:
    """Stand in for a container that exited cleanly without printing anything."""

    returncode = 0

    async def communicate(self):
        """Return empty stdout and stderr."""
        return b"", b""


def _bind_sources(cmd: list[str]) -> dict[str, str]:
    """Map each container path in a `docker run` command to the host path mounted there."""
    sources = {}
    for flag, value in zip(cmd, cmd[1:]):
        if flag == "-v":
            source, target = value.split(":")[:2]
            sources[target] = source
    return sources


def test_the_model_is_mounted_from_the_shared_temp_directory(monkeypatch, tmp_path: Path) -> None:
    """Proves the check mounts the model from the directory it shares with the daemon, not from the model cache.

    Inside the validator image the cache sits at a path the host does not have, so the
    daemon would mount an empty directory there and the check would call the model fake.
    """
    cache = tmp_path / "miner_models"
    cache.mkdir()
    model = cache / "UID_7.zip"
    model.write_bytes(b"model bytes")
    seen = {}

    async def run_container(*cmd, **_kwargs):
        """Record what the daemon would mount, while the temp directory still exists."""
        sources = _bind_sources(list(cmd))
        seen["shared"] = Path(sources["/workspace/shared"])
        seen["model"] = Path(sources["/workspace/model.zip"])
        seen["bytes"] = seen["model"].read_bytes()
        return _FinishedProcess()

    monkeypatch.setattr(docker_evaluator, "DockerSecureEvaluator", _ReadyEvaluator)
    monkeypatch.setattr(model_verify.asyncio, "create_subprocess_exec", run_container)

    asyncio.run(model_verify.verify_new_model_with_docker(model, "a" * 64, "hotkey", 7))

    assert seen["model"].parent == seen["shared"], "every mount must come from the shared temp directory"
    assert seen["bytes"] == b"model bytes"
    assert model.is_file(), "the cached model stays where it was"
