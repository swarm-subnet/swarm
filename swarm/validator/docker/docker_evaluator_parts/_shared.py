import asyncio
import gc
import hashlib
import os
import shutil
import socket
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import bittensor as bt
import capnp
import numpy as np

from swarm.challenge_families import runtime_profile_for_tasks
from swarm.challenge_families.base import ChallengeFamilyRuntimeProfile
from swarm.constants import (
    CALIBRATION_BENCHMARK_REF_NS,
    CALIBRATION_MARGIN_SEC,
    CALIBRATION_OVERHEAD_CAP_SEC,
    CALIBRATION_ROUNDS,
    CALIBRATION_TIMEOUT_SEC,
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
)
from swarm.protocol import ValidationResult
from swarm.utils.env_factory import make_env


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


def _graph_runtime_template_dir() -> Path:
    return _swarm_package_dir() / "graph_runtime_template"


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


def _cleanup_env_quietly(env: object) -> None:
    try:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()
    except Exception:
        pass
    gc.collect()


def _runtime_profile_from_payload(
    payload: Optional[dict[str, object]],
    tasks: list,
) -> ChallengeFamilyRuntimeProfile:
    if isinstance(payload, dict) and payload:
        return ChallengeFamilyRuntimeProfile.from_mapping(payload)
    return runtime_profile_for_tasks(tasks)


def _runtime_profile_env(profile: ChallengeFamilyRuntimeProfile) -> dict[str, str]:
    env = dict(profile.docker_env)
    env.setdefault("SWARM_CHALLENGE_FAMILY_ID", str(profile.family_id))
    env.setdefault("SWARM_RUNTIME_PROFILE", str(profile.profile_name))
    env.setdefault("SWARM_RUNTIME_RESOURCE_CLASS", str(profile.resource_class))
    env.setdefault("SWARM_RUNTIME_IMAGE_KEY", str(profile.image_key))
    for key, value in dict(profile.env_bootstrap).items():
        env_key = f"SWARM_BOOTSTRAP_{str(key).upper()}"
        if isinstance(value, bool):
            env[env_key] = "1" if value else "0"
        else:
            env[env_key] = str(value)
    return env


__all__ = [name for name in globals() if not name.startswith("__")]
