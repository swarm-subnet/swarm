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

"""On-disk validator state: the UID model-hash tracker, the normal-model queue and the score cache."""

import sys

from ._shared import *


def _runtime_setting(name: str):
    """Read a setting off the swarm.validator.utils facade when it is imported, falling back to the local globals, so a patched path takes effect."""
    facade = sys.modules.get("swarm.validator.utils")
    if facade is not None and hasattr(facade, name):
        return getattr(facade, name)
    return globals()[name]


def _state_dir() -> Path:
    """Return the directory holding the validator's JSON files."""
    return _runtime_setting("STATE_DIR")


def _normal_model_queue_file() -> Path:
    """Return the path of the JSON document backing the persistent normal-model queue."""
    return _runtime_setting("NORMAL_MODEL_QUEUE_FILE")

def _cache_file() -> Path:
    """Return the path of the JSON document holding benchmark scores."""
    return _runtime_setting("CACHE_FILE")


def load_model_hash_tracker() -> dict:
    """Return the UID to hash map from uid_model_hashes.json, empty when absent or unparsable."""
    hash_tracker_file = _state_dir() / "uid_model_hashes.json"
    try:
        if hash_tracker_file.exists():
            with open(hash_tracker_file, 'r') as f:
                return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return {}


def save_model_hash_tracker(tracker: dict) -> None:
    """Write the UID to hash map into uid_model_hashes.json through a temp file rename."""
    state_dir = _state_dir()
    state_dir.mkdir(exist_ok=True)
    hash_tracker_file = state_dir / "uid_model_hashes.json"
    temp_file = hash_tracker_file.with_suffix(".tmp")
    try:
        with open(temp_file, 'w') as f:
            json.dump(tracker, f)
        temp_file.replace(hash_tracker_file)
    except IOError as e:
        bt.logging.error(f"Failed to save model hash tracker: {e}")
        temp_file.unlink(missing_ok=True)


def mark_model_hash_processed(uid: int, model_hash: str) -> None:
    """Store model_hash as the latest hash seen for uid and persist the tracker."""
    tracker = load_model_hash_tracker()
    tracker[str(uid)] = model_hash
    save_model_hash_tracker(tracker)


# ──────────────────────────────────────────────────────────────────────────
# Normal-model processing queue (persistent)
# ──────────────────────────────────────────────────────────────────────────

def clear_normal_model_queue() -> None:
    """Empty the persisted queue on disk, logging it as an epoch transition."""
    save_normal_model_queue({"items": {}})
    bt.logging.info("Cleared normal model queue (epoch transition)")


def clear_benchmark_cache() -> None:
    """Drop every stored score so the new epoch re-runs each model."""
    save_benchmark_cache({})
    bt.logging.info("Cleared benchmark cache (epoch transition)")


def load_normal_model_queue() -> dict:
    """Return the persisted queue, or an empty items map when the file is absent or malformed."""
    normal_model_queue_file = _normal_model_queue_file()
    try:
        if normal_model_queue_file.exists():
            with open(normal_model_queue_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict) and isinstance(data.get("items", {}), dict):
                    return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        bt.logging.warning(f"Normal queue load failed, starting fresh: {e}")
    return {"items": {}}


def save_normal_model_queue(queue: dict) -> None:
    """Write the queue to disk atomically through a temp file, logging an IO failure instead of raising."""
    state_dir = _state_dir()
    normal_model_queue_file = _normal_model_queue_file()
    state_dir.mkdir(exist_ok=True)
    temp_file = normal_model_queue_file.with_suffix(".tmp")
    try:
        with open(temp_file, 'w') as f:
            json.dump(queue, f)
        temp_file.replace(normal_model_queue_file)
    except IOError as e:
        bt.logging.error(f"Normal queue save failed: {e}")
        temp_file.unlink(missing_ok=True)


def _queue_key(uid: int, model_hash: str) -> str:
    """Return the uid:model_hash string that identifies one entry."""
    return f"{uid}:{model_hash}"


def _schedule_queue_retry(item: Dict[str, Any], reason: str) -> None:
    """Push an item's next attempt out by an exponential backoff that tops out at 256 seconds, recording why."""
    now = time.time()
    attempts = int(item.get("retry_attempts", 0)) + 1
    backoff_sec = min(300, 2 ** min(attempts, 8))
    item["status"] = "retry"
    item["retry_attempts"] = attempts
    item["next_retry_at"] = now + backoff_sec
    item["last_error"] = reason
    item["updated_at"] = now


def _refresh_normal_model_queue(new_models: Dict[int, Tuple[Path, str, str]]) -> dict:
    """Merge freshly discovered submissions into the persisted queue, save it and return it.

    Earlier entries for the same UID are dropped unless they were terminally rejected, and an entry
    the backend cancelled goes back to pending when the same hash turns up again.
    """
    queue = load_normal_model_queue()
    items = queue.setdefault("items", {})
    now = time.time()

    for uid, (model_path, model_hash, github_url) in new_models.items():
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
            existing = items[key]
            if existing.get("status") == "cancelled":
                existing["status"] = "pending"
                existing["last_error"] = ""
                existing["retry_attempts"] = 0
                existing["next_retry_at"] = 0
                existing["backend_authorized"] = True
                existing["backend_reason"] = ""
            existing["model_path"] = str(model_path)
            existing["github_url"] = github_url
            existing["updated_at"] = now
            continue

        items[key] = {
            "uid": uid,
            "model_hash": model_hash,
            "model_path": str(model_path),
            "github_url": github_url,
            "status": "pending",
            "registered": False,
            "from_backend": True,
            "screening_recorded": False,
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
    """Return up to limit keys of entries that are neither finished nor still inside a retry window, oldest first."""
    now = time.time()
    items = queue.get("items", {})
    ready = []

    for key, item in items.items():
        status = item.get("status", "pending")
        if status in ("completed", "terminal_rejected", "cancelled"):
            continue

        next_retry_at = float(item.get("next_retry_at", 0) or 0)
        if next_retry_at > now:
            continue

        ready.append((float(item.get("created_at", 0) or 0), key))

    ready.sort(key=lambda pair: pair[0])
    return [key for _, key in ready[:limit]]


def _format_queue_timestamp(ts: float | int | None) -> str | None:
    """Render an epoch seconds value as a UTC ISO-8601 string, None when it is zero or absent."""
    if not ts:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(ts)))


def _queue_phase(item: Dict[str, Any]) -> str:
    """Return screening or benchmark, whichever stage an item has reached."""
    if item.get("screening_recorded"):
        return "benchmark"
    if str(item.get("status", "")).startswith("benchmark"):
        return "benchmark"
    return "screening"


def build_heartbeat_queue_snapshot(queue: dict) -> List[Dict[str, Any]]:
    """Return one dict per unfinished entry for the heartbeat payload, ordered by enqueue time and saying what blocks each."""
    now = time.time()
    items = list(queue.get("items", {}).items())
    items.sort(key=lambda pair: (float(pair[1].get("created_at", 0) or 0), pair[0]))

    snapshot: List[Dict[str, Any]] = []
    for position, (key, item) in enumerate(items, start=1):
        status = str(item.get("status", "pending"))
        if status in ("completed", "terminal_rejected"):
            continue
        next_retry_at = float(item.get("next_retry_at", 0) or 0)
        processable = status not in ("retry", "cancelled") and next_retry_at <= now
        blocked_reason = ""
        if status == "cancelled":
            blocked_reason = str(item.get("last_error", "cancelled by backend"))
        elif next_retry_at > now:
            blocked_reason = str(item.get("last_error", "waiting for retry window"))
        elif status == "retry":
            blocked_reason = str(item.get("last_error", "retry pending"))

        snapshot.append(
            {
                "key": str(key),
                "uid": int(item.get("uid", -1)),
                "phase": _queue_phase(item),
                "status": status,
                "queue_position": position,
                "enqueue_time": _format_queue_timestamp(item.get("created_at")),
                "updated_at": _format_queue_timestamp(item.get("updated_at")),
                "assignment_id": item.get("assignment_id"),
                "processable": processable,
                "blocked_reason": blocked_reason or None,
                "retry_attempts": int(item.get("retry_attempts", 0) or 0),
                "backend_authorized": item.get("backend_authorized"),
                "backend_reason": item.get("backend_reason"),
                "backend_decision_version": item.get("backend_decision_version"),
                "model_hash": str(item.get("model_hash", ""))[:12],
            }
        )
    return snapshot


# ──────────────────────────────────────────────────────────────────────────
# Benchmark score cache (by model_hash + epoch + benchmark_version)
# ──────────────────────────────────────────────────────────────────────────

def load_benchmark_cache() -> dict:
    """Return the stored scores keyed by hash, epoch and version, empty when the file is unreadable."""
    cache_file = _cache_file()
    try:
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        bt.logging.warning(f"Cache load failed, starting fresh: {e}")
    return {}


def save_benchmark_cache(cache: dict) -> None:
    """Write the scores to disk atomically through a temp file, logging an IO failure instead of raising."""
    state_dir = _state_dir()
    cache_file = _cache_file()
    state_dir.mkdir(exist_ok=True)
    temp_file = cache_file.with_suffix(".tmp")
    try:
        with open(temp_file, 'w') as f:
            json.dump(cache, f)
        temp_file.replace(cache_file)
    except IOError as e:
        bt.logging.error(f"Cache save failed: {e}")
        temp_file.unlink(missing_ok=True)


def _score_cache_key(model_hash: str, epoch: int) -> str:
    """Return the key joining a model hash, an epoch number and BENCHMARK_VERSION."""
    return f"{model_hash}_{epoch}_{BENCHMARK_VERSION}"


def get_cached_score(model_hash: str, epoch: int) -> Optional[Dict[str, Any]]:
    """Return the stored benchmark result for a hash in an epoch, None when there is none."""
    cache = load_benchmark_cache()
    key = _score_cache_key(model_hash, epoch)
    result = cache.get(key)
    if result:
        bt.logging.debug(f"Cache hit for {model_hash[:16]}...")
    return result


def set_cached_score(model_hash: str, epoch: int, result: Dict[str, Any]) -> None:
    """Store a benchmark result under its hash and epoch, stamped with the wall clock time and version."""
    cache = load_benchmark_cache()
    key = _score_cache_key(model_hash, epoch)
    result["cached_at"] = time.time()
    result["benchmark_version"] = BENCHMARK_VERSION
    result["epoch_number"] = epoch
    cache[key] = result
    save_benchmark_cache(cache)
    bt.logging.info(f"Cached score for {model_hash[:16]}... (epoch={epoch})")


def has_cached_score(model_hash: str, epoch: int) -> bool:
    """Return True when this hash already holds a result for the epoch at the current version."""
    cache = load_benchmark_cache()
    key = _score_cache_key(model_hash, epoch)
    return key in cache
