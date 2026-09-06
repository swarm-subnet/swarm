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
    MAX_MODEL_BYTES,
    MODEL_DIR,
    SIM_DT,
    UID_ZERO,
)
from swarm.core.model_verify import load_blacklist, verify_new_model_with_docker
from swarm.policy_interface import PolicyInterfaceError, read_policy_contract_from_zip
from swarm.core.submission_lane import (
    RUNNABLE_INTERFACE_VERSIONS,
    graph_declared_family,
    is_model_graph_artifact,
)
from swarm.core.submission_policy import (
    SUBMISSION_INTERFACE_VERSION,
    check_safety,
    validate_submission_zip,
)
from swarm.protocol import PolicyRef
from swarm.utils.github import (
    build_raw_urls,
    download_from_github,
    validate_github_url,
)
from swarm.utils.hash import sha256sum
from swarm.validator.backend_api import (
    BackendApiClient,
    BackendTransportError,
    authorize_with_retry,
)

STATE_DIR = Path(__file__).resolve().parent.parent.parent / "state"
NORMAL_MODEL_QUEUE_FILE = STATE_DIR / "normal_model_queue.json"
NORMAL_MODEL_QUEUE_PROCESS_LIMIT = 1
CACHE_FILE = STATE_DIR / "benchmark_cache.json"
CLAIMED_REPOS_FILE = STATE_DIR / "claimed_repos.json"

_readme_ok_cache: set[str] = set()


__all__ = [name for name in globals() if not name.startswith("__")]
