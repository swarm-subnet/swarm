from __future__ import annotations

import swarm.constants as constants


def test_available_vcpu_count_prefers_sched_getaffinity(monkeypatch):
    monkeypatch.setattr(constants.os, "sched_getaffinity", lambda _pid: {0, 1, 2, 3})
    monkeypatch.setattr(constants.os, "cpu_count", lambda: 16)

    assert constants.available_vcpu_count() == 4


def test_cpus_per_docker_worker_parses_constant(monkeypatch):
    monkeypatch.setattr(constants, "DOCKER_WORKER_CPUS", "2")
    assert constants.cpus_per_docker_worker() == 2

    monkeypatch.setattr(constants, "DOCKER_WORKER_CPUS", "4")
    assert constants.cpus_per_docker_worker() == 4


def test_cpus_per_docker_worker_handles_invalid(monkeypatch):
    monkeypatch.setattr(constants, "DOCKER_WORKER_CPUS", "not-a-number")
    assert constants.cpus_per_docker_worker() == 1


def test_default_docker_worker_count_uses_all_complete_cpu_groups(monkeypatch):
    monkeypatch.delenv("SWARM_MAX_DOCKER_WORKERS", raising=False)
    monkeypatch.setattr(constants.os, "sched_getaffinity", lambda _pid: set(range(64)))
    monkeypatch.setattr(constants, "DOCKER_WORKER_CPUS", "2")

    assert constants.default_docker_worker_count() == 32


def test_default_docker_worker_count_honors_configured_maximum(monkeypatch):
    monkeypatch.setenv("SWARM_MAX_DOCKER_WORKERS", "12")
    monkeypatch.setattr(constants.os, "sched_getaffinity", lambda _pid: set(range(64)))
    monkeypatch.setattr(constants, "DOCKER_WORKER_CPUS", "2")

    assert constants.default_docker_worker_count() == 12


def test_default_docker_worker_count_ignores_invalid_maximum(monkeypatch):
    monkeypatch.setenv("SWARM_MAX_DOCKER_WORKERS", "invalid")
    monkeypatch.setattr(constants.os, "sched_getaffinity", lambda _pid: set(range(32)))
    monkeypatch.setattr(constants, "DOCKER_WORKER_CPUS", "2")

    assert constants.default_docker_worker_count() == 16


def test_default_docker_worker_count_partitions_by_cpus_per_worker(monkeypatch):
    monkeypatch.delenv("SWARM_MAX_DOCKER_WORKERS", raising=False)
    monkeypatch.delattr(constants.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(constants.os, "cpu_count", lambda: 12)
    monkeypatch.setattr(constants, "DOCKER_WORKER_CPUS", "2")

    assert constants.available_vcpu_count() == 12
    assert constants.default_docker_worker_count() == 6


def test_default_docker_worker_count_handles_small_hosts(monkeypatch):
    monkeypatch.delenv("SWARM_MAX_DOCKER_WORKERS", raising=False)
    monkeypatch.delattr(constants.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(constants.os, "cpu_count", lambda: 1)
    monkeypatch.setattr(constants, "DOCKER_WORKER_CPUS", "2")

    assert constants.default_docker_worker_count() == 1
