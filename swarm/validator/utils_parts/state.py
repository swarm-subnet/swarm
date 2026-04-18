import sys

from ._shared import *


def _runtime_setting(name: str):
    facade = sys.modules.get("swarm.validator.utils")
    if facade is not None and hasattr(facade, name):
        return getattr(facade, name)
    return globals()[name]


def _state_dir() -> Path:
    return _runtime_setting("STATE_DIR")


def _normal_model_queue_file() -> Path:
    return _runtime_setting("NORMAL_MODEL_QUEUE_FILE")

def _cache_file() -> Path:
    return _runtime_setting("CACHE_FILE")


def _claimed_repos_file() -> Path:
    return _runtime_setting("CLAIMED_REPOS_FILE")

def load_model_hash_tracker() -> dict:
    hash_tracker_file = _state_dir() / "uid_model_hashes.json"
    try:
        if hash_tracker_file.exists():
            with open(hash_tracker_file, 'r') as f:
                return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return {}


def save_model_hash_tracker(tracker: dict) -> None:
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
    tracker = load_model_hash_tracker()
    tracker[str(uid)] = model_hash
    save_model_hash_tracker(tracker)


# ──────────────────────────────────────────────────────────────────────────
# GitHub repo ownership (one repo per hotkey)
# ──────────────────────────────────────────────────────────────────────────

_claimed_repos_lock = threading.Lock()


def load_claimed_repos() -> dict:
    claimed_repos_file = _claimed_repos_file()
    try:
        if claimed_repos_file.exists():
            with open(claimed_repos_file, 'r') as f:
                return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return {}


def save_claimed_repos(claimed: dict) -> bool:
    state_dir = _state_dir()
    claimed_repos_file = _claimed_repos_file()
    state_dir.mkdir(exist_ok=True)
    temp_file = claimed_repos_file.with_suffix(".tmp")
    try:
        with open(temp_file, 'w') as f:
            json.dump(claimed, f)
        temp_file.replace(claimed_repos_file)
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

def clear_normal_model_queue() -> None:
    save_normal_model_queue({"items": {}})
    bt.logging.info("Cleared normal model queue (epoch transition)")


def clear_benchmark_cache() -> None:
    save_benchmark_cache({})
    bt.logging.info("Cleared benchmark cache (epoch transition)")


def load_normal_model_queue() -> dict:
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


def _refresh_normal_model_queue(new_models: Dict[int, Tuple[Path, str, str]]) -> dict:
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
            items[key]["model_path"] = str(model_path)
            items[key]["github_url"] = github_url
            items[key]["updated_at"] = now
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
    if not ts:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(ts)))


def _queue_phase(item: Dict[str, Any]) -> str:
    if item.get("screening_recorded"):
        return "benchmark"
    if str(item.get("status", "")).startswith("benchmark"):
        return "benchmark"
    return "screening"


def build_heartbeat_queue_snapshot(queue: dict) -> List[Dict[str, Any]]:
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
    cache_file = _cache_file()
    try:
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        bt.logging.warning(f"Cache load failed, starting fresh: {e}")
    return {}


def save_benchmark_cache(cache: dict) -> None:
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
