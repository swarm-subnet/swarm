from __future__ import annotations

import asyncio
import json
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import bittensor as bt
import numpy as np

from swarm.constants import (
    BATCH_DELAY_SEC,
    BENCHMARK_VERSION,
    BURN_EMISSIONS,
    BURN_FRACTION,
    KEEP_FRACTION,
    MAP_CACHE_ENABLED,
    MAP_CACHE_WARMUP_BATCH_SIZE,
    MAP_CACHE_WARMUP_MAX_LOGGED_FAILURES,
    MAX_CONCURRENT_CONNECTIONS,
    MAX_MODEL_BYTES,
    MODEL_DIR,
    PARALLEL_BATCH_SIZE,
    QUERY_REF_TIMEOUT,
    SCREENING_BOOTSTRAP_THRESHOLD,
    SCREENING_TOP_MODEL_FACTOR,
    SIM_DT,
    UID_ZERO,
)
from swarm.core.env_builder import prebuild_static_world_cache
from swarm.core.model_verify import (
    load_blacklist,
    verify_new_model_with_docker,
    zip_is_safe,
)
from swarm.protocol import PolicyRef, PolicySynapse
from swarm.utils.github import (
    build_raw_urls,
    check_readme_matches,
    download_from_github,
    validate_github_url,
)
from swarm.utils.hash import sha256sum
from swarm.validator.backend_api import BackendApiClient, classify_backend_failure
from swarm.validator.task_gen import random_task

STATE_DIR = Path(__file__).resolve().parent.parent.parent / "state"
NORMAL_MODEL_QUEUE_FILE = STATE_DIR / "normal_model_queue.json"
NORMAL_MODEL_QUEUE_PROCESS_LIMIT = 1
MAP_CACHE_WARMUP_STATE_FILE = STATE_DIR / "map_cache_warmup_state.json"
CACHE_FILE = STATE_DIR / "benchmark_cache.json"
CLAIMED_REPOS_FILE = STATE_DIR / "claimed_repos.json"

_claimed_repos_lock = threading.Lock()
_readme_ok_cache: set[str] = set()


__all__ = [name for name in globals() if not name.startswith("__")]
