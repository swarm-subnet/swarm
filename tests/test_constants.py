from __future__ import annotations

import swarm.constants as constants


def test_available_vcpu_count_prefers_sched_getaffinity(monkeypatch):
    monkeypatch.setattr(constants.os, "sched_getaffinity", lambda _pid: {0, 1, 2, 3})
    monkeypatch.setattr(constants.os, "cpu_count", lambda: 16)

    assert constants.available_vcpu_count() == 4


def test_default_docker_worker_count_caps_at_twelve(monkeypatch):
    monkeypatch.setattr(constants.os, "sched_getaffinity", lambda _pid: set(range(32)))

    assert constants.default_docker_worker_count() == 12


def test_default_docker_worker_count_falls_back_to_cpu_count(monkeypatch):
    monkeypatch.delattr(constants.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(constants.os, "cpu_count", lambda: 6)

    assert constants.available_vcpu_count() == 6
    assert constants.default_docker_worker_count() == 6
