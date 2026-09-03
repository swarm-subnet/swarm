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

import argparse
import asyncio
import gc
import io
import json
import multiprocessing as mp
import os
import queue as queue_mod
import random
import re
import statistics
import sys
import threading
import time
import traceback
from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from swarm.domain_model import BENCHMARK_GROUP_TO_CHALLENGE_TYPE

try:
    from tqdm import tqdm as _tqdm
except Exception:  # pragma: no cover - fallback path when tqdm is unavailable.
    _tqdm = None

SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_UID_RE = re.compile(r"uid[_-]?(\d+)", re.IGNORECASE)

BENCH_GROUP_ORDER = list(BENCHMARK_GROUP_TO_CHALLENGE_TYPE)
BENCH_GROUP_TO_TYPE = dict(BENCHMARK_GROUP_TO_CHALLENGE_TYPE)


@dataclass
class _RunOptions:
    heartbeat_sec: float = 30.0
    rpc_trace: bool = False
    rpc_trace_every: int = 250
    rpc_heartbeat_sec: float = 150.0
    max_batch_timeout_sec: float = 900.0
    timeout_multiplier: float = 1.0
    extend_timeout_on_progress: bool = True
    timeout_extend_sec: float = 30.0
    timeout_progress_stale_sec: float = 3.0
    timeout_progress_min_sim_advance: float = 0.02
    max_seed_walltime_sec: float = 0.0
    default_log_out: Optional[str] = None


@dataclass
class _BatchStat:
    worker_id: int
    batch_index: int
    seed_count: int
    elapsed_sec: float
    seed_processing_sec: float
    startup_overhead_sec: float
    seeds: List[int]


@dataclass
class _ProcessBatchRequest:
    batch_index: int
    batch_indices: List[int]
    tasks: List[Any]
    uid: int
    model_path: str
    task_total: int
    runtime_profile: Optional[Dict[str, Any]] = None
    host_speed_factor: Optional[float] = None
    model_image: Optional[str] = None


@dataclass
class _ProcessBatchResult:
    worker_id: int
    batch_index: int
    batch_indices: List[int]
    results: List[Tuple[int, bool, float, float, str]]
    elapsed_sec: float
    error: Optional[str] = None
    traceback_text: Optional[str] = None


@dataclass
class _ProcessSeedEvent:
    worker_id: int
    batch_index: int
    seed_meta: Optional[Dict[str, Any]] = None


@dataclass
class _ProcessWorkerHeartbeat:
    worker_id: int
    batch_index: int
    event_type: str
    ts: float

__all__ = [name for name in globals() if not name.startswith("__")]
