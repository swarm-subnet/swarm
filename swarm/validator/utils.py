# ---------------------------------------------------------------
#  Swarm validator utilities – extracted from forward.py
# ---------------------------------------------------------------
from __future__ import annotations

import asyncio
import json
import shutil
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Callable

import bittensor as bt
import numpy as np

from swarm.protocol import PolicySynapse, PolicyRef
from swarm.utils.hash import sha256sum
from swarm.core.model_verify import (
    load_blacklist,
    zip_is_safe,
    verify_new_model_with_docker,
)
from swarm.constants import (
    SIM_DT,
    QUERY_REF_TIMEOUT,
    BURN_EMISSIONS,
    MAX_MODEL_BYTES,
    BURN_FRACTION,
    KEEP_FRACTION,
    UID_ZERO,
    MODEL_DIR,
    PARALLEL_BATCH_SIZE,
    MAX_CONCURRENT_CONNECTIONS,
    BATCH_DELAY_SEC,
    BENCHMARK_VERSION,
    SCREENING_BOOTSTRAP_THRESHOLD,
    SCREENING_TOP_MODEL_FACTOR,
    MAP_CACHE_ENABLED,
    MAP_CACHE_WARMUP_BATCH_SIZE,
    MAP_CACHE_WARMUP_MAX_LOGGED_FAILURES,
)
from swarm.core.env_builder import prebuild_static_world_cache
from .task_gen import random_task
from .backend_api import BackendApiClient, classify_backend_failure
from swarm.utils.github import (
    validate_github_url,
    build_raw_urls,
    download_from_github,
    check_readme_matches,
)

# ──────────────────────────────────────────────────────────────────────────
# State file paths
# ──────────────────────────────────────────────────────────────────────────

STATE_DIR = Path(__file__).parent.parent.parent / "state"
NORMAL_MODEL_QUEUE_FILE = STATE_DIR / "normal_model_queue.json"
NORMAL_MODEL_QUEUE_PROCESS_LIMIT = 1
MAP_CACHE_WARMUP_STATE_FILE = STATE_DIR / "map_cache_warmup_state.json"
CACHE_FILE = STATE_DIR / "benchmark_cache.json"
CLAIMED_REPOS_FILE = STATE_DIR / "claimed_repos.json"


# ──────────────────────────────────────────────────────────────────────────
# HeartbeatManager – thread-safe progress tracking for backend
# ──────────────────────────────────────────────────────────────────────────

class HeartbeatManager:
    """Thread-safe heartbeat progress manager for evaluation tracking.

    Sends throttled heartbeat updates to the backend during seed evaluation.
    Designed to be called from worker threads while safely dispatching async
    heartbeat calls to the main event loop.
    """

    def __init__(self, backend_api: BackendApiClient, main_loop: asyncio.AbstractEventLoop):
        self.backend_api = backend_api
        self.main_loop = main_loop
        self._progress = 0
        self._total = 0
        self._last_sent = 0
        self._lock = threading.Lock()
        self._status = "idle"
        self._uid: Optional[int] = None
        self._session_id = 0
        self._active = False

    def start(self, status: str, uid: int, total: int) -> None:
        with self._lock:
            self._session_id += 1
            self._status = status
            self._uid = uid
            self._total = total
            self._progress = 0
            self._last_sent = 0
            self._active = True

        asyncio.run_coroutine_threadsafe(
            self._safe_heartbeat(0, self._session_id),
            self.main_loop
        )

    def on_seed_complete(self) -> None:
        """Called from worker thread after each seed completes (throttled)."""
        with self._lock:
            if not self._active:
                return
            self._progress += 1
            progress = self._progress
            session_id = self._session_id
            if progress - self._last_sent < 10:
                return
            self._last_sent = progress

        self.main_loop.call_soon_threadsafe(
            lambda p=progress, s=session_id: asyncio.create_task(self._safe_heartbeat(p, s))
        )

    def finish(self) -> None:
        with self._lock:
            final_progress = self._progress
            session_id = self._session_id
            self._active = False

        asyncio.run_coroutine_threadsafe(
            self._finish_async(final_progress, session_id),
            self.main_loop
        )

    async def _finish_async(self, final_progress: int, session_id: int) -> None:
        if final_progress > 0:
            await self._safe_heartbeat(final_progress, session_id, allow_inactive=True)
        await self._send_idle()

    async def _safe_heartbeat(
        self, progress: int, session_id: int, allow_inactive: bool = False
    ) -> None:
        with self._lock:
            if session_id != self._session_id:
                return
            if not allow_inactive and not self._active:
                return
            status = self._status
            uid = self._uid
            total = self._total

        try:
            await asyncio.wait_for(
                self.backend_api.post_heartbeat(
                    status=status,
                    current_uid=uid,
                    progress=progress,
                    total_seeds=total
                ),
                timeout=2.0
            )
        except Exception:
            pass

    async def _send_idle(self) -> None:
        try:
            await asyncio.wait_for(
                self.backend_api.post_heartbeat(status="idle"),
                timeout=2.0
            )
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────
# Dendrite RPC helper
# ──────────────────────────────────────────────────────────────────────────

async def send_with_fresh_uuid(
    wallet: "bt.Wallet",
    synapse: "bt.Synapse",
    axon,
    *,
    timeout: float,
    deserialize: bool = True,
):
    """Create a transient Dendrite client so every RPC gets a unique UUID."""
    async with bt.Dendrite(wallet=wallet) as dend:
        responses = await dend(
            axons=[axon],
            synapse=synapse,
            deserialize=deserialize,
            timeout=timeout,
        )

    bt.logging.warning(
        f"➡️  sending: nonce={synapse.dendrite.nonce} "
        f"timeout={synapse.timeout} uuid={synapse.dendrite.uuid} "
        f"computed_body_hash={synapse.computed_body_hash} "
        f"axon={axon} dendrite"
    )
    return responses


# ──────────────────────────────────────────────────────────────────────────
# Model hash tracker (UID → hash persistence)
# ──────────────────────────────────────────────────────────────────────────

def load_model_hash_tracker() -> dict:
    hash_tracker_file = STATE_DIR / "uid_model_hashes.json"
    try:
        if hash_tracker_file.exists():
            with open(hash_tracker_file, 'r') as f:
                return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return {}


def save_model_hash_tracker(tracker: dict) -> None:
    STATE_DIR.mkdir(exist_ok=True)
    hash_tracker_file = STATE_DIR / "uid_model_hashes.json"
    temp_file = hash_tracker_file.with_suffix(".tmp")
    try:
        with open(temp_file, 'w') as f:
            json.dump(tracker, f)
        temp_file.replace(hash_tracker_file)
    except IOError as e:
        bt.logging.error(f"Failed to save model hash tracker: {e}")
        temp_file.unlink(missing_ok=True)


def mark_model_hash_processed(uid: int, model_hash: str) -> None:
    tracker = load_model_hash_tracker()
    tracker[str(uid)] = model_hash
    save_model_hash_tracker(tracker)


# ──────────────────────────────────────────────────────────────────────────
# GitHub repo ownership (one repo per hotkey)
# ──────────────────────────────────────────────────────────────────────────

_claimed_repos_lock = threading.Lock()


def load_claimed_repos() -> dict:
    try:
        if CLAIMED_REPOS_FILE.exists():
            with open(CLAIMED_REPOS_FILE, 'r') as f:
                return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return {}


def save_claimed_repos(claimed: dict) -> bool:
    STATE_DIR.mkdir(exist_ok=True)
    temp_file = CLAIMED_REPOS_FILE.with_suffix(".tmp")
    try:
        with open(temp_file, 'w') as f:
            json.dump(claimed, f)
        temp_file.replace(CLAIMED_REPOS_FILE)
        return True
    except IOError as e:
        bt.logging.error(f"Failed to save claimed repos: {e}")
        temp_file.unlink(missing_ok=True)
        return False


def check_repo_ownership(github_url: str, hotkey: str, uid: int) -> bool:
    normalized = validate_github_url(github_url)
    if not normalized:
        return False
    key = normalized.lower()
    with _claimed_repos_lock:
        claimed = load_claimed_repos()
        owner = claimed.get(key)
        if owner is None:
            claimed[key] = hotkey
            if not save_claimed_repos(claimed):
                return False
            return True
        owner = str(owner)
        if owner == hotkey:
            return True
    bt.logging.warning(
        f"UID {uid}: repo {normalized} already claimed by hotkey {owner[:16]}..."
    )
    return False


# ──────────────────────────────────────────────────────────────────────────
# Normal-model processing queue (persistent)
# ──────────────────────────────────────────────────────────────────────────

def load_normal_model_queue() -> dict:
    try:
        if NORMAL_MODEL_QUEUE_FILE.exists():
            with open(NORMAL_MODEL_QUEUE_FILE, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict) and isinstance(data.get("items", {}), dict):
                    return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        bt.logging.warning(f"Normal queue load failed, starting fresh: {e}")
    return {"items": {}}


def save_normal_model_queue(queue: dict) -> None:
    STATE_DIR.mkdir(exist_ok=True)
    temp_file = NORMAL_MODEL_QUEUE_FILE.with_suffix(".tmp")
    try:
        with open(temp_file, 'w') as f:
            json.dump(queue, f)
        temp_file.replace(NORMAL_MODEL_QUEUE_FILE)
    except IOError as e:
        bt.logging.error(f"Normal queue save failed: {e}")
        temp_file.unlink(missing_ok=True)


def _queue_key(uid: int, model_hash: str) -> str:
    return f"{uid}:{model_hash}"


def _schedule_queue_retry(item: Dict[str, Any], reason: str) -> None:
    now = time.time()
    attempts = int(item.get("retry_attempts", 0)) + 1
    backoff_sec = min(300, 2 ** min(attempts, 8))
    item["status"] = "retry"
    item["retry_attempts"] = attempts
    item["next_retry_at"] = now + backoff_sec
    item["last_error"] = reason
    item["updated_at"] = now


def _refresh_normal_model_queue(new_models: Dict[int, Tuple[Path, str]]) -> dict:
    queue = load_normal_model_queue()
    items = queue.setdefault("items", {})
    now = time.time()

    for uid, (model_path, model_hash) in new_models.items():
        key = _queue_key(uid, model_hash)

        stale_keys = [
            k for k, v in items.items()
            if int(v.get("uid", -1)) == uid and v.get("model_hash") != model_hash
        ]
        for stale_key in stale_keys:
            stale_item = items.get(stale_key, {})
            if stale_item.get("status") != "terminal_rejected":
                del items[stale_key]

        if key in items:
            items[key]["model_path"] = str(model_path)
            items[key]["updated_at"] = now
            continue

        items[key] = {
            "uid": uid,
            "model_hash": model_hash,
            "model_path": str(model_path),
            "status": "pending",
            "registered": False,
            "screening_recorded": False,
            "screening_passed": None,
            "score_recorded": False,
            "retry_attempts": 0,
            "next_retry_at": 0,
            "last_error": "",
            "created_at": now,
            "updated_at": now,
        }

    queue["items"] = items
    save_normal_model_queue(queue)
    return queue


def _get_processable_queue_keys(queue: dict, limit: int) -> List[str]:
    now = time.time()
    items = queue.get("items", {})
    ready = []

    for key, item in items.items():
        status = item.get("status", "pending")
        if status in ("completed", "terminal_rejected"):
            continue

        next_retry_at = float(item.get("next_retry_at", 0) or 0)
        if next_retry_at > now:
            continue

        ready.append((float(item.get("created_at", 0) or 0), key))

    ready.sort(key=lambda pair: pair[0])
    return [key for _, key in ready[:limit]]


# ──────────────────────────────────────────────────────────────────────────
# Map-cache warmup state
# ──────────────────────────────────────────────────────────────────────────

def load_map_cache_warmup_state() -> Dict[str, Any]:
    default_state = {
        "benchmark_version": BENCHMARK_VERSION,
        "screening_index": 0,
        "public_index": 0,
        "completed": False,
        "failed_count": 0,
        "last_update": 0,
    }
    try:
        if MAP_CACHE_WARMUP_STATE_FILE.exists():
            with open(MAP_CACHE_WARMUP_STATE_FILE, 'r') as f:
                data = json.load(f)
            if data.get("benchmark_version") == BENCHMARK_VERSION:
                default_state.update(data)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        bt.logging.warning(f"Map cache warmup state load failed, resetting: {e}")
    return default_state


def save_map_cache_warmup_state(state: Dict[str, Any]) -> None:
    STATE_DIR.mkdir(exist_ok=True)
    temp_file = MAP_CACHE_WARMUP_STATE_FILE.with_suffix(".tmp")
    try:
        with open(temp_file, 'w') as f:
            json.dump(state, f)
        temp_file.replace(MAP_CACHE_WARMUP_STATE_FILE)
    except IOError as e:
        bt.logging.warning(f"Map cache warmup state save failed: {e}")
        temp_file.unlink(missing_ok=True)


async def _run_map_cache_prebuild_all_once(self) -> None:
    """Prebuild map cache for all benchmark seeds once before scoring."""
    if not MAP_CACHE_ENABLED:
        return

    if not hasattr(self, 'seed_manager'):
        return

    state = load_map_cache_warmup_state()
    if state.get("completed") and state.get("epoch") == self.seed_manager.epoch_number:
        return

    if state.get("epoch") != self.seed_manager.epoch_number:
        state = {"epoch": self.seed_manager.epoch_number}

    all_seeds = self.seed_manager.get_all_seeds()
    seed_index = int(state.get("seed_index", 0) or 0)
    total_seeds = len(all_seeds)

    if seed_index >= total_seeds:
        state["completed"] = True
        state["last_update"] = time.time()
        save_map_cache_warmup_state(state)
        return

    bt.logging.info(
        f"Map cache prebuild: building remaining seeds "
        f"({seed_index}/{total_seeds} already warmed, epoch={self.seed_manager.epoch_number})"
    )

    failed_now = 0
    logged_failures = 0

    while seed_index < total_seeds:
        seed = int(all_seeds[seed_index])
        seed_index += 1

        try:
            task = random_task(sim_dt=SIM_DT, seed=seed)
            prebuild_static_world_cache(
                seed=task.map_seed,
                challenge_type=task.challenge_type,
                start=task.start,
                goal=task.goal,
            )
        except Exception as e:
            failed_now += 1
            if logged_failures < MAP_CACHE_WARMUP_MAX_LOGGED_FAILURES:
                bt.logging.warning(f"Map cache prebuild seed failed ({seed}): {e}")
                logged_failures += 1

        if seed_index % 100 == 0:
            bt.logging.info(f"Map cache prebuild progress: {seed_index}/{total_seeds}")
            await asyncio.sleep(0)

    state["seed_index"] = seed_index
    state["failed_count"] = int(state.get("failed_count", 0) or 0) + failed_now
    state["completed"] = True
    state["last_update"] = time.time()
    save_map_cache_warmup_state(state)

    bt.logging.info(
        f"Map cache prebuild complete: {seed_index}/{total_seeds} seeds "
        f"(failures={state['failed_count']})"
    )


async def _run_map_cache_warmup_step(self) -> None:
    """Warm map cache incrementally (small batch per forward cycle)."""
    if not MAP_CACHE_ENABLED:
        return

    if not hasattr(self, 'seed_manager'):
        return

    state = load_map_cache_warmup_state()

    if state.get("epoch") != self.seed_manager.epoch_number:
        state = {"epoch": self.seed_manager.epoch_number}

    if state.get("completed"):
        return

    all_seeds = self.seed_manager.get_all_seeds()
    seed_index = int(state.get("seed_index", 0) or 0)
    total_seeds = len(all_seeds)

    if seed_index >= total_seeds:
        state["completed"] = True
        state["last_update"] = time.time()
        save_map_cache_warmup_state(state)
        return

    warmed_now = 0
    failed_now = 0

    while warmed_now < MAP_CACHE_WARMUP_BATCH_SIZE and seed_index < total_seeds:
        seed = int(all_seeds[seed_index])
        seed_index += 1

        try:
            task = random_task(sim_dt=SIM_DT, seed=seed)
            prebuild_static_world_cache(
                seed=task.map_seed,
                challenge_type=task.challenge_type,
                start=task.start,
                goal=task.goal,
            )
        except Exception as e:
            failed_now += 1
            if failed_now <= MAP_CACHE_WARMUP_MAX_LOGGED_FAILURES:
                bt.logging.warning(f"Map cache warmup seed failed ({seed}): {e}")

        warmed_now += 1

    state["seed_index"] = seed_index
    state["failed_count"] = int(state.get("failed_count", 0) or 0) + failed_now
    state["last_update"] = time.time()

    if seed_index >= total_seeds:
        state["completed"] = True
        bt.logging.info(
            f"Map cache warmup complete: {seed_index}/{total_seeds} seeds "
            f"(failures={state['failed_count']})"
        )
    else:
        bt.logging.info(
            f"Map cache warmup progress: +{warmed_now} this cycle, "
            f"{seed_index}/{total_seeds} total"
        )

    save_map_cache_warmup_state(state)


# ──────────────────────────────────────────────────────────────────────────
# Benchmark score cache (by model_hash + benchmark_version)
# ──────────────────────────────────────────────────────────────────────────

def load_benchmark_cache() -> dict:
    try:
        if CACHE_FILE.exists():
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        bt.logging.warning(f"Cache load failed, starting fresh: {e}")
    return {}


def save_benchmark_cache(cache: dict) -> None:
    STATE_DIR.mkdir(exist_ok=True)
    temp_file = CACHE_FILE.with_suffix(".tmp")
    try:
        with open(temp_file, 'w') as f:
            json.dump(cache, f)
        temp_file.replace(CACHE_FILE)
    except IOError as e:
        bt.logging.error(f"Cache save failed: {e}")
        temp_file.unlink(missing_ok=True)


def _score_cache_key(model_hash: str, epoch: int) -> str:
    return f"{model_hash}_{epoch}_{BENCHMARK_VERSION}"


def get_cached_score(model_hash: str, epoch: int) -> Optional[Dict[str, Any]]:
    cache = load_benchmark_cache()
    key = _score_cache_key(model_hash, epoch)
    result = cache.get(key)
    if result:
        bt.logging.debug(f"Cache hit for {model_hash[:16]}...")
    return result


def set_cached_score(model_hash: str, epoch: int, result: Dict[str, Any]) -> None:
    cache = load_benchmark_cache()
    key = _score_cache_key(model_hash, epoch)
    result["cached_at"] = time.time()
    result["benchmark_version"] = BENCHMARK_VERSION
    result["epoch_number"] = epoch
    cache[key] = result
    save_benchmark_cache(cache)
    bt.logging.info(f"Cached score for {model_hash[:16]}... (epoch={epoch})")


def has_cached_score(model_hash: str, epoch: int) -> bool:
    cache = load_benchmark_cache()
    key = _score_cache_key(model_hash, epoch)
    return key in cache


# ──────────────────────────────────────────────────────────────────────────
# Model download & retrieval
# ──────────────────────────────────────────────────────────────────────────

_readme_ok_cache: set[str] = set()


async def _download_model_from_github(
    github_url: str, ref: PolicyRef, dest: Path, uid: int
) -> bool:
    """Download model ZIP from a miner's public GitHub repository."""
    validated = validate_github_url(github_url, uid=uid)
    if not validated:
        bt.logging.warning(f"UID {uid}: invalid github_url: {github_url}")
        return False

    cache_key = f"{validated}:{ref.sha256}"
    if cache_key not in _readme_ok_cache:
        if not await check_readme_matches(validated, uid=uid):
            bt.logging.warning(
                f"UID {uid}: README.md missing or does not match template, skipping"
            )
            return False
        _readme_ok_cache.add(cache_key)

    candidate_urls = build_raw_urls(validated)
    downloaded = False
    for raw_url in candidate_urls:
        if await download_from_github(raw_url, dest, max_bytes=MAX_MODEL_BYTES):
            downloaded = True
            break

    if not downloaded:
        bt.logging.warning(f"UID {uid}: GitHub download failed from {validated}")
        dest.unlink(missing_ok=True)
        return False

    if not zip_is_safe(dest, max_uncompressed=MAX_MODEL_BYTES):
        bt.logging.error(f"UID {uid}: unsafe ZIP from GitHub")
        dest.unlink(missing_ok=True)
        return False

    downloaded_hash = sha256sum(dest)
    if downloaded_hash != ref.sha256:
        bt.logging.error(
            f"UID {uid}: SHA256 mismatch — "
            f"expected {ref.sha256[:16]}..., got {downloaded_hash[:16]}..."
        )
        dest.unlink(missing_ok=True)
        return False

    bt.logging.info(f"Stored model for UID {uid} from GitHub at {dest}")
    await verify_new_model_with_docker(dest, ref.sha256, f"github-uid-{uid}", uid)
    return True


async def _process_single_uid(self, uid: int) -> Tuple[int, Optional[Path]]:
    """Fetch and verify a single miner's model."""
    try:
        axon = self.metagraph.axons[uid]

        try:
            responses = await send_with_fresh_uuid(
                wallet=self.wallet,
                synapse=PolicySynapse.request_ref(),
                axon=axon,
                timeout=QUERY_REF_TIMEOUT,
            )

            if not responses:
                return (uid, None)

            syn = responses[0]

            if not syn.ref:
                return (uid, None)

            ref = PolicyRef(**syn.ref)
        except Exception:
            return (uid, None)

        blacklist = load_blacklist()
        if ref.sha256 in blacklist:
            bt.logging.warning(f"Skipping blacklisted model {ref.sha256[:16]}... from UID {uid}")
            return (uid, None)

        model_fp = MODEL_DIR / f"UID_{uid}.zip"
        if model_fp.exists() and model_fp.is_dir():
            shutil.rmtree(model_fp)

        up_to_date = False
        if model_fp.is_file():
            try:
                up_to_date = sha256sum(model_fp) == ref.sha256
            except Exception:
                up_to_date = False

        if up_to_date:
            if (
                model_fp.stat().st_size <= MAX_MODEL_BYTES
                and zip_is_safe(model_fp, max_uncompressed=MAX_MODEL_BYTES)
            ):
                return (uid, model_fp)
            else:
                model_fp.unlink(missing_ok=True)

        if not ref.github_url:
            bt.logging.warning(f"UID {uid}: no github_url in PolicyRef, skipping")
            return (uid, None)

        hotkey = self.metagraph.hotkeys[uid]
        if not check_repo_ownership(ref.github_url, hotkey, uid):
            return (uid, None)

        ok = await _download_model_from_github(ref.github_url, ref, model_fp, uid)
        if ok and model_fp.is_file():
            return (uid, model_fp)
        else:
            model_fp.unlink(missing_ok=True)
            return (uid, None)

    except Exception:
        return (uid, None)


async def _ensure_models(self, uids: List[int]) -> Dict[int, Path]:
    """Fetch models from all given UIDs in parallel batches."""
    MODEL_DIR.mkdir(exist_ok=True)
    paths: Dict[int, Path] = {}
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CONNECTIONS)
    total_batches = (len(uids) + PARALLEL_BATCH_SIZE - 1) // PARALLEL_BATCH_SIZE

    bt.logging.info(f"Starting model fetch for {len(uids)} UIDs in {total_batches} batches")

    async def _limited_process(uid: int) -> Tuple[int, Optional[Path]]:
        async with semaphore:
            return await _process_single_uid(self, uid)

    for batch_start in range(0, len(uids), PARALLEL_BATCH_SIZE):
        batch = uids[batch_start:batch_start + PARALLEL_BATCH_SIZE]
        batch_num = batch_start // PARALLEL_BATCH_SIZE + 1

        results = await asyncio.gather(
            *[_limited_process(uid) for uid in batch],
            return_exceptions=True
        )

        batch_found = 0
        for result in results:
            if isinstance(result, Exception):
                continue
            uid, path = result
            if path is not None:
                paths[uid] = path
                batch_found += 1

        if batch_num % 5 == 0 or batch_found > 0:
            bt.logging.debug(
                f"Batch {batch_num}/{total_batches}: "
                f"found {batch_found} models, total so far: {len(paths)}"
            )

        if batch_start + PARALLEL_BATCH_SIZE < len(uids):
            await asyncio.sleep(BATCH_DELAY_SEC)

    bt.logging.info(f"Model fetch complete: found {len(paths)} models from {len(uids)} UIDs")
    return paths


# ──────────────────────────────────────────────────────────────────────────
# Benchmark evaluation
# ──────────────────────────────────────────────────────────────────────────

async def _evaluate_seeds(
    self,
    uid: int,
    model_path: Path,
    seeds: List[int],
    description: str = "benchmark",
    on_seed_complete: Optional[Callable[[], None]] = None,
) -> Tuple[List[float], Dict[str, List[float]]]:
    """Evaluate a model on multiple seeds using parallel Docker containers."""
    all_scores = []
    per_type_scores = {
        "city": [], "open": [], "mountain": [],
        "village": [], "warehouse": [], "moving_platform": [],
    }

    challenge_type_to_name = {
        1: "city",
        2: "open",
        3: "mountain",
        4: "village",
        5: "warehouse",
    }

    bt.logging.info(f"🔬 Starting {description} for UID {uid}: {len(seeds)} seeds (parallel)")

    tasks = []
    for seed in seeds:
        try:
            task = random_task(sim_dt=SIM_DT, seed=seed)
            tasks.append(task)
        except Exception as e:
            bt.logging.warning(f"Failed to create task for seed {seed}: {e}")
            tasks.append(None)

    valid_tasks = [t for t in tasks if t is not None]
    if not valid_tasks:
        bt.logging.warning(f"No valid tasks created for UID {uid}")
        return [], per_type_scores

    results = await self.docker_evaluator.evaluate_seeds_parallel(
        tasks=valid_tasks,
        uid=uid,
        model_path=model_path,
        on_seed_complete=on_seed_complete,
    )

    task_idx = 0
    for i, task in enumerate(tasks):
        if task is None:
            all_scores.append(0.0)
            continue

        if task_idx < len(results):
            result = results[task_idx]
            score = result.score if result else 0.0
            all_scores.append(score)

            type_name = challenge_type_to_name.get(task.challenge_type, "unknown")
            if getattr(task, 'moving_platform', False):
                per_type_scores["moving_platform"].append(score)
            elif type_name in per_type_scores:
                per_type_scores[type_name].append(score)

            task_idx += 1
        else:
            all_scores.append(0.0)

    bt.logging.info(f"✅ {description} complete for UID {uid}: {len(all_scores)} seeds evaluated")
    return all_scores, per_type_scores


async def _run_screening(self, uid: int, model_path: Path) -> Tuple[float, List[float]]:
    """Run screening benchmark with private seeds."""
    screening_seeds = self.seed_manager.get_screening_seeds()

    hb = HeartbeatManager(self.backend_api, asyncio.get_running_loop())
    hb.start("evaluating_screening", uid, len(screening_seeds))

    try:
        all_scores, _ = await _evaluate_seeds(
            self, uid, model_path, screening_seeds, "screening",
            on_seed_complete=hb.on_seed_complete
        )
    finally:
        hb.finish()

    if all_scores:
        median_score = float(np.median(all_scores))
    else:
        median_score = 0.0

    bt.logging.info(f"📊 Screening result for UID {uid}: median={median_score:.4f}")
    return median_score, all_scores


async def _run_full_benchmark(
    self, uid: int, model_path: Path
) -> Tuple[float, Dict[str, float], List[float]]:
    """Run full benchmark with public seeds."""
    benchmark_seeds = self.seed_manager.get_benchmark_seeds()

    hb = HeartbeatManager(self.backend_api, asyncio.get_running_loop())
    hb.start("evaluating_benchmark", uid, len(benchmark_seeds))

    try:
        all_scores, per_type_scores = await _evaluate_seeds(
            self, uid, model_path, benchmark_seeds, "full benchmark",
            on_seed_complete=hb.on_seed_complete
        )
    finally:
        hb.finish()

    if all_scores:
        median_score = float(np.median(all_scores))
    else:
        median_score = 0.0

    per_type_medians = {}
    for type_name, scores in per_type_scores.items():
        if scores:
            per_type_medians[type_name] = float(np.median(scores))
        else:
            per_type_medians[type_name] = 0.0

    bt.logging.info(f"📊 Full benchmark result for UID {uid}: median={median_score:.4f}")
    return median_score, per_type_medians, all_scores


# ──────────────────────────────────────────────────────────────────────────
# Scoring & detection helpers
# ──────────────────────────────────────────────────────────────────────────

def _passes_screening(self, screening_score: float) -> bool:
    """Check if screening score meets the threshold."""
    current_top = getattr(self, '_current_top', None)

    if not current_top or not current_top.get('score'):
        threshold = SCREENING_BOOTSTRAP_THRESHOLD
        passed = screening_score >= threshold
        bt.logging.info(
            f"Screening (bootstrap mode): {screening_score:.4f} >= {threshold} = {passed}"
        )
        return passed

    top_score = current_top.get('score', 0.0)
    threshold = top_score * SCREENING_TOP_MODEL_FACTOR
    passed = screening_score >= threshold
    bt.logging.info(
        f"Screening: {screening_score:.4f} >= {threshold:.4f} "
        f"(80% of top {top_score:.4f}) = {passed}"
    )
    return passed


def _detect_new_models(self, model_paths: Dict[int, Path]) -> Dict[int, Tuple[Path, str]]:
    """Detect models that have changed (new hash) since last check."""
    tracker = load_model_hash_tracker()
    new_models = {}

    for uid, path in model_paths.items():
        try:
            current_hash = sha256sum(path)
            uid_str = str(uid)
            old_hash = tracker.get(uid_str)

            if old_hash != current_hash:
                if old_hash:
                    bt.logging.info(
                        f"🔄 Model changed for UID {uid}: "
                        f"{old_hash[:16]}... → {current_hash[:16]}..."
                    )
                else:
                    bt.logging.info(f"🆕 New model for UID {uid}: {current_hash[:16]}...")
                new_models[uid] = (path, current_hash)

        except Exception as e:
            bt.logging.warning(f"Failed to check model hash for UID {uid}: {e}")

    return new_models


def _get_validator_stake(self) -> float:
    """Get this validator's stake from metagraph."""
    try:
        my_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        stake = float(self.metagraph.S[my_uid])
        return stake
    except Exception as e:
        bt.logging.warning(f"Failed to get validator stake: {e}")
        return 0.0


def _get_miner_coldkey(self, uid: int) -> str:
    """Get miner's coldkey from metagraph."""
    try:
        return self.metagraph.coldkeys[uid]
    except Exception:
        return ""


# ──────────────────────────────────────────────────────────────────────────
# Backend submit wrappers (with retry classification)
# ──────────────────────────────────────────────────────────────────────────

async def _register_new_model_with_ack(
    self,
    uid: int,
    model_hash: str,
    validator_hotkey: str,
) -> Tuple[bool, bool, str]:
    coldkey = _get_miner_coldkey(self, uid)
    response = await self.backend_api.post_new_model(
        uid=uid,
        model_hash=model_hash,
        coldkey=coldkey,
        validator_hotkey=validator_hotkey,
    )

    if response.get("accepted", False):
        model_path = MODEL_DIR / f"UID_{uid}.zip"
        if model_path.is_file():
            try:
                upload_resp = await self.backend_api.upload_model_file(uid, model_path)
                bt.logging.info(f"Model upload for UID {uid}: {upload_resp}")
            except Exception as e:
                bt.logging.warning(f"Model upload failed for UID {uid} (non-fatal): {e}")
        return True, False, ""

    terminal, reason = classify_backend_failure(response, "new_model")
    return False, terminal, reason


async def _submit_screening_with_ack(
    self,
    uid: int,
    validator_hotkey: str,
    validator_stake: float,
    screening_score: float,
    passed: bool,
) -> Tuple[bool, bool, str]:
    response = await self.backend_api.post_screening(
        uid=uid,
        validator_hotkey=validator_hotkey,
        validator_stake=validator_stake,
        screening_score=screening_score,
        passed=passed,
    )

    if response.get("recorded", False):
        return True, False, ""

    terminal, reason = classify_backend_failure(response, "screening")
    return False, terminal, reason


async def _submit_score_with_ack(
    self,
    uid: int,
    validator_hotkey: str,
    validator_stake: float,
    model_hash: str,
    total_score: float,
    per_type_scores: Dict[str, float],
    seeds_evaluated: int,
    epoch_number: Optional[int] = None,
) -> Tuple[bool, bool, str]:
    response = await self.backend_api.post_score(
        uid=uid,
        validator_hotkey=validator_hotkey,
        validator_stake=validator_stake,
        model_hash=model_hash,
        total_score=total_score,
        per_type_scores=per_type_scores,
        seeds_evaluated=seeds_evaluated,
        epoch_number=epoch_number,
    )

    if response.get("recorded", False):
        return True, False, ""

    terminal, reason = classify_backend_failure(response, "score")
    return False, terminal, reason


# ──────────────────────────────────────────────────────────────────────────
# Queue worker – processes a single normal-model pipeline item
# ──────────────────────────────────────────────────────────────────────────

async def _process_normal_queue_item(
    self,
    queue: dict,
    key: str,
    validator_hotkey: str,
    validator_stake: float,
) -> None:
    items = queue.get("items", {})
    item = items.get(key)
    if not item:
        return

    try:
        uid = int(item.get("uid", -1))
        model_hash = str(item.get("model_hash", ""))
        model_path = Path(str(item.get("model_path", "")))

        if uid < 0 or not model_hash:
            item["status"] = "terminal_rejected"
            item["last_error"] = "invalid queue item"
            item["updated_at"] = time.time()
            return

        item["status"] = "processing"
        item["updated_at"] = time.time()

        if not model_path.exists():
            _schedule_queue_retry(item, "model file missing")
            return

        current_hash = sha256sum(model_path)
        if current_hash != model_hash:
            item["status"] = "terminal_rejected"
            item["last_error"] = "model hash changed before processing"
            item["updated_at"] = time.time()
            return

        if not item.get("registered", False):
            accepted, terminal, reason = await _register_new_model_with_ack(
                self,
                uid=uid,
                model_hash=model_hash,
                validator_hotkey=validator_hotkey,
            )
            if not accepted:
                if terminal:
                    item["status"] = "terminal_rejected"
                    item["last_error"] = reason
                    item["updated_at"] = time.time()
                    mark_model_hash_processed(uid, model_hash)
                else:
                    _schedule_queue_retry(item, f"register failed: {reason}")
                return

            item["registered"] = True
            item["status"] = "registered"
            item["retry_attempts"] = 0
            item["next_retry_at"] = 0
            item["last_error"] = ""
            item["updated_at"] = time.time()

        epoch = self.seed_manager.epoch_number
        cached = get_cached_score(model_hash, epoch) if has_cached_score(model_hash, epoch) else None

        if item.get("screening_score") is None:
            if cached:
                item["screening_score"] = float(cached.get("screening_score", 0.0))
            else:
                screening_score, screening_scores = await _run_screening(self, uid, model_path)
                item["screening_score"] = float(screening_score)
                item["screening_scores"] = screening_scores
            item["updated_at"] = time.time()

        if item.get("screening_passed") is None:
            item["screening_passed"] = _passes_screening(
                self, float(item.get("screening_score", 0.0))
            )

        screening_passed = bool(item.get("screening_passed", False))

        if not item.get("screening_recorded", False):
            recorded, terminal, reason = await _submit_screening_with_ack(
                self,
                uid=uid,
                validator_hotkey=validator_hotkey,
                validator_stake=validator_stake,
                screening_score=float(item.get("screening_score", 0.0)),
                passed=screening_passed,
            )
            if not recorded:
                if terminal:
                    item["status"] = "terminal_rejected"
                    item["last_error"] = reason
                    item["updated_at"] = time.time()
                    mark_model_hash_processed(uid, model_hash)
                else:
                    _schedule_queue_retry(item, f"screening submit failed: {reason}")
                return

            item["screening_recorded"] = True
            item["status"] = "screening_recorded"
            item["retry_attempts"] = 0
            item["next_retry_at"] = 0
            item["last_error"] = ""
            item["updated_at"] = time.time()

        if not screening_passed:
            item["status"] = "completed"
            item["updated_at"] = time.time()
            item.pop("screening_scores", None)
            mark_model_hash_processed(uid, model_hash)
            return

        missing_score_payload = (
            item.get("total_score") is None
            or not isinstance(item.get("per_type_scores"), dict)
            or item.get("seeds_evaluated") is None
        )

        if missing_score_payload:
            if cached:
                per_type_scores = cached.get("per_type_scores", {})
                if not isinstance(per_type_scores, dict):
                    per_type_scores = {}
                item["full_score"] = float(
                    cached.get("full_score", cached.get("total_score", 0.0))
                )
                item["total_score"] = float(cached.get("total_score", 0.0))
                item["per_type_scores"] = per_type_scores
                item["seeds_evaluated"] = int(cached.get("seeds_evaluated", 1200))
            else:
                full_score, per_type_scores, full_scores = await _run_full_benchmark(
                    self, uid, model_path
                )
                screening_scores = item.get("screening_scores", [])
                if not isinstance(screening_scores, list):
                    screening_scores = []
                combined_scores = screening_scores + full_scores
                total_score = float(np.median(combined_scores)) if combined_scores else 0.0
                item["full_score"] = float(full_score)
                item["total_score"] = total_score
                item["per_type_scores"] = per_type_scores
                item["seeds_evaluated"] = len(combined_scores)
            item["updated_at"] = time.time()

        if not item.get("score_recorded", False):
            recorded, terminal, reason = await _submit_score_with_ack(
                self,
                uid=uid,
                validator_hotkey=validator_hotkey,
                validator_stake=validator_stake,
                model_hash=model_hash,
                total_score=float(item.get("total_score", 0.0)),
                per_type_scores=dict(item.get("per_type_scores", {})),
                seeds_evaluated=int(item.get("seeds_evaluated", 0) or 0),
                epoch_number=self.seed_manager.epoch_number,
            )
            if not recorded:
                if terminal:
                    item["status"] = "terminal_rejected"
                    item["last_error"] = reason
                    item["updated_at"] = time.time()
                    mark_model_hash_processed(uid, model_hash)
                else:
                    _schedule_queue_retry(item, f"score submit failed: {reason}")
                return

            item["score_recorded"] = True
            item["status"] = "completed"
            item["retry_attempts"] = 0
            item["next_retry_at"] = 0
            item["last_error"] = ""
            item["updated_at"] = time.time()

        set_cached_score(model_hash, epoch, {
            "uid": uid,
            "total_score": float(item.get("total_score", 0.0)),
            "screening_score": float(item.get("screening_score", 0.0)),
            "full_score": float(item.get("full_score", item.get("total_score", 0.0))),
            "per_type_scores": dict(item.get("per_type_scores", {})),
            "seeds_evaluated": int(item.get("seeds_evaluated", 0) or 0),
        })

        item.pop("screening_scores", None)
        mark_model_hash_processed(uid, model_hash)

    except Exception as e:
        _schedule_queue_retry(item, f"queue worker exception: {e}")


# ──────────────────────────────────────────────────────────────────────────
# Weight application
# ──────────────────────────────────────────────────────────────────────────

def _apply_backend_weights_to_scores(self, backend_weights: Dict[Any, Any]) -> None:
    """Apply backend weights to validator scores with deterministic reset."""
    self.scores = np.zeros(self.metagraph.n, dtype=np.float32)

    if not backend_weights:
        if BURN_EMISSIONS and 0 <= UID_ZERO < self.metagraph.n:
            self.scores[UID_ZERO] = 1.0
        return

    uids_list = []
    weights_list = []

    for uid_str, weight in backend_weights.items():
        try:
            uid = int(uid_str)
            parsed_weight = float(weight)
            if uid < 0 or uid >= self.metagraph.n:
                continue
            uids_list.append(uid)
            weights_list.append(parsed_weight)
        except (ValueError, TypeError):
            continue

    if not uids_list:
        if BURN_EMISSIONS and 0 <= UID_ZERO < self.metagraph.n:
            self.scores[UID_ZERO] = 1.0
        return

    uids_np = np.array(uids_list, dtype=np.int64)
    weights_np = np.array(weights_list, dtype=np.float32)

    if BURN_EMISSIONS and UID_ZERO not in uids_np and 0 <= UID_ZERO < self.metagraph.n:
        total_weight = weights_np.sum()
        if total_weight > 0:
            weights_np *= KEEP_FRACTION / total_weight
        uids_np = np.concatenate(([UID_ZERO], uids_np))
        weights_np = np.concatenate(([BURN_FRACTION], weights_np))

    self.scores[uids_np] = weights_np
