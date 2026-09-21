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

"""Docker cleanup: scoped to its owner, bounded in time, and kept off the validator event loop."""

from __future__ import annotations

import asyncio
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from swarm.utils import docker_instance
from swarm.validator import forward as forward_mod
from swarm.validator.docker import docker_evaluator as de
from swarm.validator.docker.docker_evaluator_parts import batch
from swarm.validator.runtime_telemetry import ValidatorRuntimeTracker

_OWNER = "5OwnerHotkey"


class _FakeDocker:
    """Records every docker command and answers the list calls from canned output."""

    def __init__(self, containers: str = "", images: str = ""):
        """Hold the text the container and image listings hand back."""
        self.containers = containers
        self.images = images
        self.calls: list[tuple[list[str], dict]] = []

    def run(self, cmd, **kwargs):
        """Stand in for subprocess.run: log the call and return the matching listing."""
        self.calls.append((list(cmd), kwargs))
        stdout = ""
        if cmd[:2] == ["docker", "ps"] and "name=swarm_eval_" in cmd:
            stdout = self.containers
        elif cmd[:2] == ["docker", "images"]:
            stdout = self.images
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    def removed(self) -> list[str]:
        """Names of the containers a cleanup force-removed."""
        return [cmd[3] for cmd, _ in self.calls if cmd[:3] == ["docker", "rm", "-f"]]

    def rmi(self) -> list[str]:
        """Tags of the images a cleanup removed."""
        return [cmd[2] for cmd, _ in self.calls if cmd[:2] == ["docker", "rmi"]]

    def commands(self) -> list[str]:
        """Every recorded command as one string."""
        return [" ".join(cmd) for cmd, _ in self.calls]


@pytest.fixture
def evaluator(monkeypatch, tmp_path):
    """An evaluator owned by a fixed hotkey, with an empty model directory."""
    monkeypatch.setenv(docker_instance._INSTANCE_ID_ENV, _OWNER)
    monkeypatch.setattr(batch, "MODEL_DIR", tmp_path)
    return de.DockerSecureEvaluator.__new__(de.DockerSecureEvaluator)


def test_instance_label_follows_the_owner_set_at_startup(monkeypatch):
    """The label names the owner once one is set, and falls back to a shared default before that."""
    monkeypatch.delenv(docker_instance._INSTANCE_ID_ENV, raising=False)
    assert docker_instance.instance_label() == "swarm.instance=local"
    docker_instance.set_instance_id(_OWNER)
    assert docker_instance.instance_label() == f"swarm.instance={_OWNER}"


def test_cleanup_leaves_another_validators_containers_alone(monkeypatch, evaluator):
    """On a shared daemon only containers labelled with this owner are removed."""
    docker = _FakeDocker(
        containers=f"swarm_eval_1_w0_1\t{_OWNER}\nswarm_eval_2_w0_1\t5OtherHotkey\nswarm_eval_3_w0_1\t\n"
    )
    monkeypatch.setattr(subprocess, "run", docker.run)
    evaluator.cleanup()
    assert docker.removed() == ["swarm_eval_1_w0_1"]


def test_startup_cleanup_adopts_containers_from_before_the_label(monkeypatch, evaluator):
    """Unlabelled leftovers of an older release are taken only when asked, never another owner's."""
    docker = _FakeDocker(
        containers=f"swarm_eval_1_w0_1\t{_OWNER}\nswarm_eval_2_w0_1\t5OtherHotkey\nswarm_eval_3_w0_1\t\n"
    )
    monkeypatch.setattr(subprocess, "run", docker.run)
    evaluator.cleanup(adopt_unlabelled=True)
    assert docker.removed() == ["swarm_eval_1_w0_1", "swarm_eval_3_w0_1"]


def test_every_cleanup_docker_call_is_bounded(monkeypatch, evaluator):
    """No docker call in a full pass may wait forever on a wedged daemon."""
    docker = _FakeDocker(
        containers=f"swarm_eval_1_w0_1\t{_OWNER}\n",
        images=f"swarm_eval_model_aaaaaaaaaaaa:latest\t{_OWNER}\n",
    )
    monkeypatch.setattr(subprocess, "run", docker.run)
    monkeypatch.setattr(batch.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    evaluator.cleanup(prune=True, adopt_unlabelled=True)
    assert any("builder prune" in c for c in docker.commands())
    unbounded = [" ".join(cmd) for cmd, kwargs in docker.calls if not kwargs.get("timeout")]
    assert unbounded == []


def test_task_end_cleanup_runs_no_prune(monkeypatch, evaluator):
    """The pass after every task sweeps containers and model images and leaves the prunes out."""
    docker = _FakeDocker()
    monkeypatch.setattr(subprocess, "run", docker.run)
    evaluator.cleanup()
    assert not any("prune" in c for c in docker.commands())


def test_prune_touches_only_swarm_images(monkeypatch, evaluator):
    """The image prune is filtered to the Swarm label, and volumes and foreign containers are never pruned."""
    docker = _FakeDocker()
    monkeypatch.setattr(subprocess, "run", docker.run)
    monkeypatch.setattr(batch.shutil, "disk_usage", lambda _: SimpleNamespace(free=1 << 40))
    evaluator.cleanup(prune=True)
    prunes =[c for c in docker.commands() if "prune" in c]
    assert prunes == ["docker image prune -f --filter label=swarm.code_hash"]


_IMAGES = (
    f"swarm_eval_model_aaaaaaaaaaaa:latest\t{_OWNER}\n"
    "swarm_eval_model_bbbbbbbbbbbb:latest\t5OtherHotkey\n"
    "swarm_eval_model_cccccccccccc:latest\t\n"
    "swarm_eval_model_dddddddddddd:latest\tlocal\n"
)


def test_model_images_are_reaped_by_owner_without_their_zip(monkeypatch, evaluator):
    """Only this owner's model image without a zip on disk is removed; the rest are left alone."""
    docker = _FakeDocker(images=_IMAGES)
    monkeypatch.setattr(subprocess, "run", docker.run)
    evaluator.cleanup()
    assert docker.rmi() == ["swarm_eval_model_aaaaaaaaaaaa:latest"]


def test_startup_cleanup_adopts_images_from_before_the_label(monkeypatch, evaluator):
    """Images left by a release before the label, unlabelled or under the default name, are reaped at startup.

    Without this every per-model image built before the release stays on disk forever,
    because nothing else removes a tagged image.
    """
    docker = _FakeDocker(images=_IMAGES)
    monkeypatch.setattr(subprocess, "run", docker.run)
    evaluator.cleanup(adopt_unlabelled=True)
    assert docker.rmi() == [
        "swarm_eval_model_aaaaaaaaaaaa:latest",
        "swarm_eval_model_cccccccccccc:latest",
        "swarm_eval_model_dddddddddddd:latest",
    ]


def test_startup_cleanup_adopts_containers_under_the_default_name(monkeypatch, evaluator):
    """A container started by a process that never named itself carries 'local', and is an orphan too."""
    docker = _FakeDocker(containers="swarm_eval_1_w0_1\tlocal\nswarm_eval_2_w0_1\t5OtherHotkey\n")
    monkeypatch.setattr(subprocess, "run", docker.run)
    evaluator.cleanup(adopt_unlabelled=True)
    assert docker.removed() == ["swarm_eval_1_w0_1"]


def test_a_base_rebuild_leaves_another_validators_model_cache_alone(monkeypatch, evaluator):
    """The sweep before a base rebuild takes this owner's images and the ownerless ones, never a co-tenant's."""
    docker = _FakeDocker(images=_IMAGES)
    monkeypatch.setattr(subprocess, "run", docker.run)
    batch.remove_all_model_images(adopt_unlabelled=True)
    assert "swarm_eval_model_bbbbbbbbbbbb:latest" not in docker.rmi()
    assert len(docker.rmi()) == 3


def test_a_wedged_daemon_costs_one_wait_even_while_reaping_images(monkeypatch, evaluator):
    """A timeout inside an image removal ends the pass, instead of one wait per image."""
    calls = []

    def _list_then_hang(cmd, **kwargs):
        """Answer the listings, then never answer a removal."""
        calls.append(cmd)
        if cmd[:2] == ["docker", "rmi"]:
            raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
        stdout = _IMAGES if cmd[:2] == ["docker", "images"] else ""
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", _list_then_hang)
    evaluator.cleanup(adopt_unlabelled=True)
    assert [c for c in calls if c[:2] == ["docker", "rmi"]] == [
        ["docker", "rmi", "swarm_eval_model_aaaaaaaaaaaa:latest"]
    ]


def test_the_observation_buffer_sweep_is_scoped_to_its_owner(monkeypatch, evaluator, tmp_path):
    """Only this owner's buffers go, plus the ownerless port-only ones at startup; a co-tenant's stay."""
    mine = tmp_path / f"swarm_obs_{_OWNER}_18000.bin"
    theirs = tmp_path / "swarm_obs_5OtherHotkey_18001.bin"
    old = tmp_path / "swarm_obs_18002.bin"
    for p in (mine, theirs, old):
        p.write_bytes(b"")
    monkeypatch.setattr(batch, "Path", lambda _: tmp_path)
    monkeypatch.setattr(subprocess, "run", _FakeDocker().run)

    evaluator.cleanup()
    assert not mine.exists() and theirs.exists() and old.exists()
    evaluator.cleanup(adopt_unlabelled=True)
    assert theirs.exists() and not old.exists()


def test_the_observation_buffer_path_carries_the_owner(evaluator):
    """The buffer a container is handed is the one the sweep will recognise as this owner's."""
    assert batch._obs_shm_host_path(18000) == f"/dev/shm/swarm_obs_{_OWNER}_18000.bin"


def test_cleanup_gives_up_at_the_first_timeout(monkeypatch, evaluator):
    """A wedged daemon costs one bounded wait: the pass ends there and nothing is raised."""
    calls = []

    def _hang(cmd, **kwargs):
        """Time out the way a docker client does when the daemon never answers."""
        calls.append(cmd)
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", _hang)
    evaluator.cleanup(prune=True)
    assert len(calls) == 1


def test_launched_containers_carry_the_owner_label(monkeypatch, evaluator):
    """An evaluation container is started with this owner's label, so a later sweep can tell it apart."""
    docker = _FakeDocker()
    monkeypatch.setattr(subprocess, "run", docker.run)
    monkeypatch.setattr(batch, "_create_obs_shm", lambda port: None)
    ctx = SimpleNamespace(
        host_port=18000, container_name="swarm_eval_7_w0_1", current_uid=1000, current_gid=1000,
        worker_limits={"memory": "4g", "cpus": "1", "cpuset_cpus": ""},
        submission_dir=Path("/tmp/submission"), docker_envs={}, run_image="swarm_evaluator_base:latest",
    )
    assert batch._launch_container(ctx) is None
    cmd = docker.calls[0][0]
    assert cmd[cmd.index("--label") + 1] == f"swarm.instance={_OWNER}"


def test_event_loop_keeps_running_while_cleanup_works(tmp_path):
    """A slow cleanup no longer freezes the loop: other coroutines tick throughout, and the duration is recorded."""
    tracker = ValidatorRuntimeTracker(state_dir=tmp_path)
    seen = {}

    def _slow_cleanup(prune=False):
        """Block the calling thread the way a slow docker daemon would."""
        seen["prune"] = prune
        time.sleep(0.3)

    validator = SimpleNamespace(
        docker_evaluator=SimpleNamespace(cleanup=_slow_cleanup), runtime_tracker=tracker,
    )

    async def _scenario() -> int:
        """Count loop ticks that land while the cleanup is still running."""
        ticks = 0
        job = asyncio.create_task(
            forward_mod._cleanup_off_loop(validator, reason="epoch_change", prune=True)
        )
        while not job.done():
            await asyncio.sleep(0.01)
            ticks += 1
        await job
        return ticks

    ticks = asyncio.run(_scenario())
    docker = tracker.snapshot_copy()["docker"]
    assert ticks >= 10
    assert seen["prune"] is True
    assert docker["cleanup_count"] == 1
    assert docker["last_cleanup_reason"] == "epoch_change"
    assert docker["last_cleanup_duration_sec"] >= 0.3
