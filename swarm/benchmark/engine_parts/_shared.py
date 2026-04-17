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

try:
    from tqdm import tqdm as _tqdm
except Exception:  # pragma: no cover - fallback path when tqdm is unavailable.
    _tqdm = None

SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_UID_RE = re.compile(r"uid[_-]?(\d+)", re.IGNORECASE)

BENCH_GROUP_ORDER = [
    "type1_city",
    "type2_open",
    "type3_mountain",
    "type4_village",
    "type5_warehouse",
    "type6_forest",
]
BENCH_GROUP_TO_TYPE = {
    "type1_city": 1,
    "type2_open": 2,
    "type3_mountain": 3,
    "type4_village": 4,
    "type5_warehouse": 5,
    "type6_forest": 6,
}


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
    timing_breakdown: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _ProcessBatchRequest:
    batch_index: int
    batch_indices: List[int]
    tasks: List[Any]
    uid: int
    model_path: str
    task_total: int
    model_image: Optional[str] = None


@dataclass
class _ProcessBatchResult:
    worker_id: int
    batch_index: int
    batch_indices: List[int]
    results: List[Tuple[int, bool, float, float]]
    elapsed_sec: float
    batch_meta: Optional[Dict[str, Any]] = None
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
