"""Typed runtime settings loaded from environment variables.

Keep env parsing centralized so validator, CLI, and packaging entrypoints use
the same rules and defaults.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

from swarm.constants import DOCKER_WORKER_CPUS, DOCKER_WORKER_MEMORY

_TRUTHY = {"1", "true", "yes", "on"}


def env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return bool(default)
    return value.strip().lower() in _TRUTHY


def env_int(name: str, default: int, *, minimum: Optional[int] = None) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = int(default)
    if minimum is not None:
        value = max(int(minimum), value)
    return value


def env_float(name: str, default: float, *, minimum: Optional[float] = None) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = float(default)
    if minimum is not None:
        value = max(float(minimum), value)
    return value


@dataclass(frozen=True)
class RpcTraceSettings:
    enabled: bool
    trace_every: int
    heartbeat_sec: float

    @classmethod
    def from_env(cls) -> "RpcTraceSettings":
        return cls(
            enabled=env_bool("SWARM_LOG_RPC_TRACE", False),
            trace_every=env_int("SWARM_LOG_RPC_TRACE_EVERY", 25, minimum=1),
            heartbeat_sec=env_float("SWARM_LOG_RPC_HEARTBEAT_SEC", 15.0, minimum=0.0),
        )


@dataclass(frozen=True)
class DockerBatchTimeoutSettings:
    multiplier: float
    hard_cap_sec: float
    extend_on_progress: bool
    extend_by_sec: float
    progress_stale_sec: float
    progress_min_sim_advance: float
    max_total_timeout_sec: float

    @classmethod
    def from_env(cls) -> "DockerBatchTimeoutSettings":
        max_total_timeout_sec = env_float("SWARM_BATCH_TIMEOUT_MAX_TOTAL_SEC", 0.0)
        if max_total_timeout_sec < 0:
            max_total_timeout_sec = 0.0
        return cls(
            multiplier=env_float("SWARM_BATCH_TIMEOUT_MULT", 1.0, minimum=1.0),
            hard_cap_sec=env_float("SWARM_BATCH_TIMEOUT_HARD_CAP_SEC", 0.0),
            extend_on_progress=env_bool("SWARM_BATCH_TIMEOUT_EXTEND_ON_PROGRESS", False),
            extend_by_sec=env_float("SWARM_BATCH_TIMEOUT_EXTEND_SEC", 30.0, minimum=1.0),
            progress_stale_sec=env_float(
                "SWARM_BATCH_TIMEOUT_PROGRESS_STALE_SEC", 3.0, minimum=0.5
            ),
            progress_min_sim_advance=env_float(
                "SWARM_BATCH_TIMEOUT_PROGRESS_MIN_SIM_ADVANCE", 0.02, minimum=0.0
            ),
            max_total_timeout_sec=max_total_timeout_sec,
        )


@dataclass(frozen=True)
class DockerWorkerLimits:
    cpus: str
    memory: str
    cpuset_cpus: Optional[str]


@dataclass(frozen=True)
class DockerRuntimeSettings:
    thread_caps_enabled: bool
    torch_num_threads: Optional[str]
    torch_interop_threads: Optional[str]
    cpus_override: Optional[str]
    memory_override: Optional[str]
    cpuset_map: Optional[str]

    @classmethod
    def from_env(cls) -> "DockerRuntimeSettings":
        return cls(
            thread_caps_enabled=env_bool("SWARM_DOCKER_THREAD_CAPS", False),
            torch_num_threads=env_str("SWARM_TORCH_NUM_THREADS"),
            torch_interop_threads=env_str("SWARM_TORCH_INTEROP_THREADS"),
            cpus_override=env_str("SWARM_DOCKER_WORKER_CPUS_OVERRIDE"),
            memory_override=env_str("SWARM_DOCKER_WORKER_MEMORY_OVERRIDE"),
            cpuset_map=env_str("SWARM_DOCKER_WORKER_CPUSETS"),
        )

    @staticmethod
    def split_cpuset_map(raw: str) -> list[str]:
        return [entry.strip() for entry in re.split(r"[;|]", raw) if entry.strip()]

    def docker_env_overrides(self, thread_cap_env_vars: tuple[str, ...]) -> dict[str, str]:
        envs: dict[str, str] = {}
        if self.thread_caps_enabled:
            for key in thread_cap_env_vars:
                envs[key] = "1"
            envs["SWARM_TORCH_NUM_THREADS"] = "1"
            envs["SWARM_TORCH_INTEROP_THREADS"] = "1"

        if self.torch_num_threads is not None:
            envs["SWARM_TORCH_NUM_THREADS"] = self.torch_num_threads
        if self.torch_interop_threads is not None:
            envs["SWARM_TORCH_INTEROP_THREADS"] = self.torch_interop_threads

        for key in thread_cap_env_vars:
            value = env_str(key)
            if value is not None:
                envs[key] = value
        return envs

    def resolve_worker_limits(self, worker_id: int) -> DockerWorkerLimits:
        cpuset = env_str(f"SWARM_DOCKER_WORKER_CPUSET_CPUS_{worker_id}")
        if cpuset is None and self.cpuset_map:
            entries = self.split_cpuset_map(self.cpuset_map)
            if worker_id < len(entries):
                cpuset = entries[worker_id]
        return DockerWorkerLimits(
            cpus=self.cpus_override or DOCKER_WORKER_CPUS,
            memory=self.memory_override or DOCKER_WORKER_MEMORY,
            cpuset_cpus=cpuset,
        )


@dataclass(frozen=True)
class BackendApiSettings:
    base_url: Optional[str]

    @classmethod
    def from_env(cls) -> "BackendApiSettings":
        return cls(base_url=env_str("SWARM_BACKEND_API_URL"))
