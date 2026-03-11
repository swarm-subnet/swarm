import asyncio
import gc
import hashlib
import os
import re
import shutil
import socket
import statistics
import subprocess
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Callable, Optional

import bittensor as bt
import capnp
import numpy as np
from gym_pybullet_drones.utils.enums import ActionType

from swarm.constants import (
    CALIBRATION_BENCHMARK_REF_NS,
    CALIBRATION_CPU_FACTOR_CAP,
    CALIBRATION_MARGIN_SEC,
    CALIBRATION_OVERHEAD_CAP_SEC,
    CALIBRATION_ROUNDS,
    CALIBRATION_TIMEOUT_SEC,
    DOCKER_PIP_WHITELIST,
    DOCKER_WORKER_CPUS,
    DOCKER_WORKER_MEMORY,
    GLOBAL_EVAL_BASE_SEC,
    GLOBAL_EVAL_CAP_SEC,
    GLOBAL_EVAL_PER_SEED_SEC,
    MINER_COMPUTE_BUDGET_SEC,
    N_DOCKER_WORKERS,
    RPC_FIRST_STEP_TIMEOUT_SEC,
    RPC_MAX_STRIKES_PER_SEED,
    RPC_PING_TIMEOUT_SEC,
    RPC_RESET_TIMEOUT_SEC,
    RPC_STEP_TIMEOUT_SEC,
    SIM_DT,
    SPEED_LIMIT,
)
from swarm.core.model_verify import add_to_blacklist
from swarm.protocol import ValidationResult
from swarm.utils.env_factory import make_env
from swarm.utils.hash import sha256sum
from swarm.validator.reward import flight_reward

_HEAVY_CHALLENGE_TYPES = frozenset({3, 5})

_THREAD_CAP_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _swarm_package_dir() -> Path:
    return Path(__file__).resolve().parents[3]


def _submission_template_dir() -> Path:
    return _swarm_package_dir() / "submission_template"


def _run_multi_seed_rpc_sync_isolated_payload(
    tasks: list, uid: int, rpc_port: int
) -> list[tuple[int, bool, float, float]]:
    """Run RPC sync in an isolated subprocess and return primitive tuples."""
    from swarm.validator.docker import docker_evaluator as docker_evaluator_mod

    evaluator_cls = docker_evaluator_mod.DockerSecureEvaluator
    evaluator = evaluator_cls.__new__(evaluator_cls)
    results = evaluator_cls._run_multi_seed_rpc_sync(
        evaluator,
        tasks,
        uid,
        rpc_port,
        None,
        None,
        None,
        0,
        None,
    )
    return [
        (int(r.uid), bool(r.success), float(r.time_sec), float(r.score))
        for r in results
    ]


def _docker_evaluator_facade():
    from swarm.validator.docker import docker_evaluator as docker_evaluator_mod

    return docker_evaluator_mod


def _heavy_aware_chunk(
    tasks: list, num_chunks: int,
) -> tuple[list[list], list[list[int]]]:
    """Distribute tasks into chunks with heavy maps spread evenly.

    Returns (chunks, index_map) where index_map[i][j] is the original
    position of chunks[i][j] in the input *tasks* list.
    """
    heavy = [
        (i, t)
        for i, t in enumerate(tasks)
        if getattr(t, "challenge_type", 0) in _HEAVY_CHALLENGE_TYPES
    ]
    light = [
        (i, t)
        for i, t in enumerate(tasks)
        if getattr(t, "challenge_type", 0) not in _HEAVY_CHALLENGE_TYPES
    ]

    buckets: list[list[tuple[int, object]]] = [[] for _ in range(num_chunks)]

    for k, item in enumerate(heavy):
        buckets[k % num_chunks].append(item)

    for item in light:
        target = min(range(num_chunks), key=lambda w: len(buckets[w]))
        buckets[target].append(item)

    buckets = [b for b in buckets if b]

    chunks = [[t for _, t in bucket] for bucket in buckets]
    index_map = [[idx for idx, _ in bucket] for bucket in buckets]
    return chunks, index_map


def _cleanup_env_quietly(env: object) -> None:
    try:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()
    except Exception:
        pass
    gc.collect()


__all__ = [name for name in globals() if not name.startswith("__")]
