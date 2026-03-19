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
    BENCHMARK_VERSION,
    BURN_EMISSIONS,
    BURN_FRACTION,
    KEEP_FRACTION,
    MAX_MODEL_BYTES,
    MODEL_DIR,
    SCREENING_BOOTSTRAP_THRESHOLD,
    SCREENING_TOP_MODEL_FACTOR,
    SIM_DT,
    UID_ZERO,
)
from swarm.core.model_verify import (
    load_blacklist,
    verify_new_model_with_docker,
    zip_is_safe,
)
from swarm.protocol import PolicyRef
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
CACHE_FILE = STATE_DIR / "benchmark_cache.json"
CLAIMED_REPOS_FILE = STATE_DIR / "claimed_repos.json"

_readme_ok_cache: set[str] = set()


__all__ = [name for name in globals() if not name.startswith("__")]
