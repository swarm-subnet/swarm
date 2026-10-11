"""What a miner's model may write: entries per submission zip, and the size of the folder it writes to."""

from __future__ import annotations

import queue
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from swarm.benchmark.engine_parts import workers
from swarm.constants import SUBMISSION_DIR_MAX_BYTES, SUBMISSION_DIR_MAX_FILES
from swarm.core.submission_policy import MAX_ARCHIVE_ENTRIES, check_safety, validate_submission_zip
from swarm.protocol import FailureReason, ValidationResult
from swarm.validator.docker import docker_evaluator as de
from swarm.validator.docker.docker_evaluator_parts import batch
from swarm.validator.utils_parts.model_fetch import _admitted
from validator.tests.test_container_prewarm import _request
from validator.tests.test_container_prewarm import fake_docker as _shared_fake_docker  # noqa: F401


def _zip_with_entries(path: Path, count: int) -> Path:
    """A valid agent zip holding ``count`` entries, the rest empty files."""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("drone_agent.py", "class DroneFlightController: pass\n")
        for i in range(count - 1):
            zf.writestr(f"data/{i}.txt", "")
    return path


def test_a_zip_at_the_entry_limit_is_accepted(tmp_path):
    """Exactly the entry limit passes the full submission check."""
    assert validate_submission_zip(_zip_with_entries(tmp_path / "ok.zip", MAX_ARCHIVE_ENTRIES)) == (True, "ok")


def test_a_zip_past_the_entry_limit_is_refused(tmp_path):
    """One entry past the limit is refused, though the archive is only kilobytes."""
    path = _zip_with_entries(tmp_path / "many.zip", MAX_ARCHIVE_ENTRIES + 1)
    assert path.stat().st_size < 200 * 1024
    assert check_safety(path) == (
        False, f"Too many files ({MAX_ARCHIVE_ENTRIES + 1}, limit {MAX_ARCHIVE_ENTRIES})"
    )


def test_the_validator_refuses_a_fetched_zip_past_the_entry_limit(tmp_path):
    """A downloaded model past the entry limit is never admitted for evaluation."""
    assert _admitted(_zip_with_entries(tmp_path / "UID_7.zip", MAX_ARCHIVE_ENTRIES + 1)) is False
    assert _admitted(_zip_with_entries(tmp_path / "UID_8.zip", MAX_ARCHIVE_ENTRIES)) is True


def test_the_entry_limit_sits_far_above_a_real_model():
    """The largest crowned model holds 53 files, so honest submissions keep wide room."""
    assert MAX_ARCHIVE_ENTRIES >= 10 * 53


def _launch_cmd(monkeypatch) -> list[str]:
    """The docker command an evaluation container is started with."""
    calls = []
    monkeypatch.setattr(
        batch.subprocess, "run",
        lambda cmd, **kw: calls.append(list(cmd)) or SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(batch, "_create_obs_shm", lambda port: None)
    ctx = SimpleNamespace(
        host_port=18000, container_name="swarm_eval_7_w0_1", current_uid=1000, current_gid=1001,
        worker_limits={"memory": "6g", "cpus": "2", "cpuset_cpus": ""},
        submission_dir=Path("/tmp/submission"), docker_envs={}, run_image="swarm_evaluator_base:latest",
    )
    assert batch._launch_container(ctx) is None
    return calls[0]


def _mounts(cmd: list[str], flag: str) -> list[str]:
    """Every value given to ``flag`` in a docker command."""
    return [cmd[i + 1] for i, part in enumerate(cmd) if part == flag]


def test_the_model_folder_is_a_capped_memory_folder(monkeypatch):
    """The folder the model writes to is held in memory, capped in bytes and files, and owned by the model's user."""
    cmd = _launch_cmd(monkeypatch)
    folder = next(m for m in _mounts(cmd, "--tmpfs") if m.startswith("/workspace/submission:"))
    options = folder.split(":", 1)[1].split(",")
    assert f"size={SUBMISSION_DIR_MAX_BYTES}" in options
    assert f"nr_inodes={SUBMISSION_DIR_MAX_FILES}" in options
    assert {"uid=1000", "gid=1001", "mode=0700", "nosuid"} <= set(options)


def test_the_host_copy_of_the_code_is_mounted_read_only(monkeypatch):
    """No folder on the validator's disk is writable from inside the container."""
    cmd = _launch_cmd(monkeypatch)
    assert "--read-only" in cmd
    assert _mounts(cmd, "-v") == ["/tmp/submission:/workspace/bundle:ro"]


def test_the_code_is_copied_into_the_folder_before_the_agent_starts(monkeypatch):
    """The container copies the code into its capped folder, then runs the agent from there."""
    cmd = _launch_cmd(monkeypatch)
    assert cmd[-4:] == [
        "swarm_evaluator_base:latest", "sh", "-c",
        "cp -R /workspace/bundle/. /workspace/submission/ && exec python /workspace/submission/main.py",
    ]


def test_the_folder_limits_hold_the_largest_allowed_submission():
    """The capped folder fits a full 50 MiB, 1,000-entry submission with room to spare."""
    assert SUBMISSION_DIR_MAX_BYTES >= 4 * 50 * 1024 * 1024
    assert SUBMISSION_DIR_MAX_FILES >= 2 * MAX_ARCHIVE_ENTRIES


@pytest.mark.parametrize(
    ("bytes_free", "files_free", "full"),
    [(10, 10, False), (0, 10, True), (10, 0, True), (0, 0, True)],
)
def test_the_folder_counts_as_full_when_bytes_or_files_run_out(monkeypatch, bytes_free, files_free, full):
    """No free space or no free file slot both count as a full folder, read through the container's root."""
    seen = []

    def _statvfs(path):
        """Report the scripted free space for the path asked about."""
        seen.append(path)
        return SimpleNamespace(f_bavail=bytes_free, f_favail=files_free)

    monkeypatch.setattr(batch.os, "statvfs", _statvfs)
    assert batch._submission_dir_full(4242) is full
    assert seen == ["/proc/4242/root/workspace/submission"]


def test_an_unreadable_folder_is_not_called_full(monkeypatch):
    """A container already gone, or never started, is left to the RPC to report."""

    def _gone(path):
        """Fail the way a vanished process does."""
        raise FileNotFoundError(path)

    monkeypatch.setattr(batch.os, "statvfs", _gone)
    assert batch._submission_dir_full(4242) is False
    assert batch._submission_dir_full(None) is False


@pytest.fixture
def fake(request):
    """The shared fake daemon under a local name."""
    return request.getfixturevalue("_shared_fake_docker")


def _fly_one_seed(monkeypatch, fake, fly, folder_full) -> tuple:
    """Fly one seed through the worker body; return its packed result and the daemon's events."""
    monkeypatch.setattr(de.DockerSecureEvaluator, "_run_multi_seed_rpc_sync", fly)
    monkeypatch.setattr(batch, "_submission_dir_full", folder_full)
    task_queue: queue.Queue = queue.Queue()
    result_queue: queue.Queue = queue.Queue()
    task_queue.put(_request(0, 7, fake.model_path, prewarm_next=False))
    task_queue.put(None)
    workers._benchmark_worker_main(0, task_queue, result_queue, queue.Queue())
    return result_queue.get_nowait().results[0], fake.events


def _fly_until_killed(fake, outcome: ValidationResult):
    """A flight that ends only when its container is killed, reporting ``outcome``."""

    def fly(self, tasks, uid, rpc_port, *args, **kwargs):
        """Wait for the kill of this flight's container, then report."""
        _ = self, args, kwargs
        name = fake.owner[int(rpc_port)]
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if any(kind == "kill" and subject == name for _, kind, subject in fake.events):
                break
            time.sleep(0.05)
        return [outcome for _ in tasks]

    return fly


def test_a_model_that_fills_its_folder_is_stopped_and_fails_the_seed(fake, monkeypatch):
    """Once the folder fills, the container is killed and the seed ends as the model's own exit, whatever the flight said."""
    started = time.monotonic()
    infra_outcome = ValidationResult(7, False, 3.0, 0.0, failure_reason=FailureReason.INFRA.value)
    packed, events = _fly_one_seed(
        monkeypatch, fake, _fly_until_killed(fake, infra_outcome),
        lambda pid: time.monotonic() - started > 1.5,
    )
    assert packed[1:5] == (False, 3.0, 0.0, FailureReason.AGENT_EXITED.value)
    assert [kind for _, kind, _ in events].count("kill") >= 1


def _land(landed: list):
    """A flight that lands with a score, noting in ``landed`` when it returns."""

    def fly(self, tasks, uid, *args, **kwargs):
        """Fly briefly, then land every seed with a score."""
        _ = self, args, kwargs
        time.sleep(0.5)
        landed.append(time.monotonic())
        return [ValidationResult(uid, True, 1.0, 0.5) for _ in tasks]

    return fly


def test_a_folder_filled_at_the_last_moment_still_fails_the_seed(fake, monkeypatch):
    """A seed that lands but leaves its folder full is charged to the model, with no score."""
    landed: list = []
    packed, _ = _fly_one_seed(monkeypatch, fake, _land(landed), lambda pid: bool(landed))
    assert packed[1:5] == (False, 1.0, 0.0, FailureReason.AGENT_EXITED.value)


def test_a_model_within_its_limits_keeps_its_score(fake, monkeypatch):
    """A folder with room left changes nothing: the container is never killed early and the score stands."""
    landed: list = []
    packed, events = _fly_one_seed(monkeypatch, fake, _land(landed), lambda pid: False)
    assert packed[1:5] == (True, 1.0, 0.5, FailureReason.NONE.value)
    assert all(t >= landed[0] for t, kind, _ in events if kind == "kill")
