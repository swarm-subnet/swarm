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

"""How many docker workers a box is given: the CPU count it trusts and the ceiling an operator can impose."""
from __future__ import annotations

import swarm.constants as constants


def test_available_vcpu_count_prefers_sched_getaffinity(monkeypatch):
    """The affinity mask wins over os.cpu_count, so a pinned container is not overcounted."""
    monkeypatch.setattr(constants.os, "sched_getaffinity", lambda _pid: {0, 1, 2, 3})
    monkeypatch.setattr(constants.os, "cpu_count", lambda: 16)

    assert constants.available_vcpu_count() == 4


def test_cpus_per_docker_worker_parses_constant(monkeypatch):
    """DOCKER_WORKER_CPUS is read at call time, so it is not frozen at import."""
    monkeypatch.setattr(constants, "DOCKER_WORKER_CPUS", "2")
    assert constants.cpus_per_docker_worker() == 2

    monkeypatch.setattr(constants, "DOCKER_WORKER_CPUS", "4")
    assert constants.cpus_per_docker_worker() == 4


def test_cpus_per_docker_worker_handles_invalid(monkeypatch):
    """An unparsable setting falls back to one CPU rather than raising."""
    monkeypatch.setattr(constants, "DOCKER_WORKER_CPUS", "not-a-number")
    assert constants.cpus_per_docker_worker() == 1


def test_default_docker_worker_count_uses_all_complete_cpu_groups(monkeypatch):
    """Every whole CPU group becomes a slot: 64 visible CPUs at two each give 32 slots."""
    monkeypatch.delenv("SWARM_MAX_DOCKER_WORKERS", raising=False)
    monkeypatch.setattr(constants.os, "sched_getaffinity", lambda _pid: set(range(64)))
    monkeypatch.setattr(constants, "DOCKER_WORKER_CPUS", "2")

    assert constants.default_docker_worker_count() == 32


def test_default_docker_worker_count_honors_configured_maximum(monkeypatch):
    """SWARM_MAX_DOCKER_WORKERS caps the result below what the CPUs would otherwise allow."""
    monkeypatch.setenv("SWARM_MAX_DOCKER_WORKERS", "12")
    monkeypatch.setattr(constants.os, "sched_getaffinity", lambda _pid: set(range(64)))
    monkeypatch.setattr(constants, "DOCKER_WORKER_CPUS", "2")

    assert constants.default_docker_worker_count() == 12


def test_default_docker_worker_count_ignores_invalid_maximum(monkeypatch):
    """A ceiling that is not a number is discarded and the CPU capacity stands."""
    monkeypatch.setenv("SWARM_MAX_DOCKER_WORKERS", "invalid")
    monkeypatch.setattr(constants.os, "sched_getaffinity", lambda _pid: set(range(32)))
    monkeypatch.setattr(constants, "DOCKER_WORKER_CPUS", "2")

    assert constants.default_docker_worker_count() == 16


def test_default_docker_worker_count_partitions_by_cpus_per_worker(monkeypatch):
    """Without an affinity mask the os.cpu_count value is divided by the CPUs a worker takes."""
    monkeypatch.delenv("SWARM_MAX_DOCKER_WORKERS", raising=False)
    monkeypatch.delattr(constants.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(constants.os, "cpu_count", lambda: 12)
    monkeypatch.setattr(constants, "DOCKER_WORKER_CPUS", "2")

    assert constants.available_vcpu_count() == 12
    assert constants.default_docker_worker_count() == 6


def test_default_docker_worker_count_handles_small_hosts(monkeypatch):
    """A single-CPU box still gets one slot rather than none."""
    monkeypatch.delenv("SWARM_MAX_DOCKER_WORKERS", raising=False)
    monkeypatch.delattr(constants.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(constants.os, "cpu_count", lambda: 1)
    monkeypatch.setattr(constants, "DOCKER_WORKER_CPUS", "2")

    assert constants.default_docker_worker_count() == 1
