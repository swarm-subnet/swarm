# ---------------------------------------------------------------
#  Swarm validator – Policy API v2   (hardened, 10 MiB limits)
# ---------------------------------------------------------------
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Callable

import bittensor as bt
import numpy as np
from zipfile import ZipFile, BadZipFile

from swarm.protocol import PolicySynapse, PolicyRef
from swarm.utils.uids import get_random_uids
from swarm.utils.hash import sha256sum
import base64

from ..core.model_verify import (
    load_blacklist,
    add_to_blacklist,
    save_fake_model_for_analysis,
)
from .task_gen import random_task
from .docker.docker_evaluator import DockerSecureEvaluator
from .seed_manager import BenchmarkSeedManager
from swarm.constants import (
    SIM_DT,
    HORIZON_SEC,
    SAMPLE_K,
    QUERY_REF_TIMEOUT,
    QUERY_BLOB_TIMEOUT,
    FORWARD_SLEEP_SEC,
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
    MAP_CACHE_PREBUILD_ALL_AT_START,
    MAP_CACHE_WARMUP_BATCH_SIZE,
    MAP_CACHE_WARMUP_MAX_LOGGED_FAILURES,
)
from swarm.core.env_builder import prebuild_static_world_cache
from .backend_api import BackendApiClient

STATE_DIR = Path(__file__).parent.parent.parent / "state"
NORMAL_MODEL_QUEUE_FILE = STATE_DIR / "normal_model_queue.json"
NORMAL_MODEL_QUEUE_PROCESS_LIMIT = 1
MAP_CACHE_WARMUP_STATE_FILE = STATE_DIR / "map_cache_warmup_state.json"


# ──────────────────────────────────────────────────────────────────────────
# 0.  HeartbeatManager – thread-safe progress tracking for backend
# ──────────────────────────────────────────────────────────────────────────
class HeartbeatManager:
    """Thread-safe heartbeat progress manager for evaluation tracking.

    This class handles sending heartbeat updates to the backend during seed
    evaluation. It's designed to be called from worker threads while safely
    dispatching async heartbeat calls to the main event loop.

    Usage:
        hb = HeartbeatManager(backend_api, asyncio.get_running_loop())
        hb.start("evaluating_screening", uid=42, total=200)
        # Pass hb.on_seed_complete as callback to evaluator
        # ... evaluation runs ...
        hb.finish()
    """

    def __init__(self, backend_api: "BackendApiClient", main_loop: asyncio.AbstractEventLoop):
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
        """Start tracking a new evaluation session.

        Args:
            status: "evaluating_screening" or "evaluating_benchmark"
            uid: Miner UID being evaluated
            total: Total number of seeds to evaluate
        """
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
        """Called from worker thread after each seed completes.

        Thread-safe. Increments progress and sends throttled heartbeat.
        """
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
        """End the current evaluation session and send idle heartbeat."""
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

    async def _safe_heartbeat(self, progress: int, session_id: int, allow_inactive: bool = False) -> None:
        """Send heartbeat with timeout. Ignores stale sessions and inactive managers."""
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
        """Send idle status heartbeat."""
        try:
            await asyncio.wait_for(
                self.backend_api.post_heartbeat(status="idle"),
                timeout=2.0
            )
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────
# 1.  Helpers – secure ZIP inspection
# ──────────────────────────────────────────────────────────────────────────
def _zip_is_safe(path: Path, *, max_uncompressed: int) -> bool:
    """
    Reject dangerous ZIP files *without* extracting them.

    • Total uncompressed size must not exceed `max_uncompressed`.
    • No absolute paths or ".." traversal sequences.
    """
    try:
        with ZipFile(path) as zf:
            total_uncompressed = 0
            for info in zf.infolist():
                # (1) forbid absolute paths or traversal
                name = info.filename
                if name.startswith(("/", "\\")) or ".." in Path(name).parts:
                    bt.logging.error(f"ZIP path traversal attempt: {name}")
                    return False

                # (2) track size
                total_uncompressed += info.file_size
                if total_uncompressed > max_uncompressed:
                    bt.logging.error(
                        f"ZIP too large when decompressed "
                        f"({total_uncompressed/1e6:.1f} MB > {max_uncompressed/1e6:.1f} MB)"
                    )
                    return False
            return True
    except BadZipFile:
        bt.logging.error("Corrupted ZIP archive.")
        return False
    except Exception as e:
        bt.logging.error(f"ZIP inspection error: {e}")
        return False


# ──────────────────────────────────────────────────────────────────────────
# 2.  Secure, cached model download
# ──────────────────────────────────────────────────────────────────────────
async def _download_model(self, axon, ref: PolicyRef, dest: Path, uid: int) -> None:
    """
    Ask the miner for the full ZIP in one message (base‑64 encoded)
    and save it to *dest*.  All integrity and size checks still apply.
    """
    tmp = dest.with_suffix(".part")
    tmp.unlink(missing_ok=True)

    try:
        # 1 – request the blob
        responses = await send_with_fresh_uuid(
            wallet=self.wallet,
            synapse=PolicySynapse.request_blob(),
            axon=axon,
            timeout=QUERY_BLOB_TIMEOUT,
        )

        if not responses:
            bt.logging.warning(f"Miner {axon.hotkey} sent no reply to blob request")
            return

        syn = responses[0]

        # 2 – make sure we actually got chunk data
        if not syn.chunk or "data" not in syn.chunk:
            bt.logging.warning(f"Miner {axon.hotkey} reply lacked chunk data")
            return

        # 3 – decode base‑64 → raw bytes
        try:
            raw_bytes = base64.b64decode(syn.chunk["data"])
        except Exception as e:
            bt.logging.warning(f"Base‑64 decode failed from miner {axon.hotkey}: {e}")
            return

        if len(raw_bytes) > MAX_MODEL_BYTES:
            bt.logging.error(
                f"Miner {axon.hotkey} sent oversized blob "
                f"({len(raw_bytes)/1e6:.1f} MB > {MAX_MODEL_BYTES/1e6:.0f} MB)"
            )
            return

        # 4 – write to temp file
        with tmp.open("wb") as fh:
            fh.write(raw_bytes)

        # 5 – ZIP sanity check
        if not _zip_is_safe(tmp, max_uncompressed=MAX_MODEL_BYTES):
            bt.logging.error(f"Unsafe ZIP from miner {axon.hotkey}.")
            tmp.unlink(missing_ok=True)
            return

        # 6 – Model is not blacklisted, proceed with storage and verification

        bt.logging.info(f"📦 Downloaded model {ref.sha256[:16]}... from miner {axon.hotkey}")

        # Atomic replacement to prevent corruption
        if dest.exists() and dest.is_dir():
            bt.logging.warning(f"Replacing directory with file: {dest}")
            shutil.rmtree(dest)
        tmp.replace(dest)

        downloaded_hash = sha256sum(dest)
        if downloaded_hash != ref.sha256:
            bt.logging.error(f"SHA256 mismatch for {axon.hotkey}: expected {ref.sha256[:16]}..., got {downloaded_hash[:16]}...")
            dest.unlink(missing_ok=True)
            return

        bt.logging.info(f"Stored model for {axon.hotkey} at {dest}.")

        # 7 – FIRST-TIME VERIFICATION: Run fake model detection in Docker container
        await _verify_new_model_with_docker(dest, ref.sha256, axon.hotkey, uid)

    except Exception as e:
        bt.logging.warning(f"Download error ({axon.hotkey}): {e}")
        tmp.unlink(missing_ok=True)

async def _verify_new_model_with_docker(model_path: Path, model_hash: str, miner_hotkey: str, uid: int):
    """
    FIRST-TIME MODEL VERIFICATION: Run fake model detection in Docker container

    Creates a fresh Docker container from base image, copies the model inside,
    runs the 3-layer fake detection process, and handles fake model blacklisting.
    """

    if not model_path.is_file():
        bt.logging.warning(f"Verification skipped; model file missing: {model_path}")
        return

    bt.logging.info(f"🔍 Starting first-time verification for model {model_hash[:16]}... from {miner_hotkey}")

    # Create Docker evaluator instance
    docker_evaluator = DockerSecureEvaluator()

    if not docker_evaluator._base_ready:
        bt.logging.warning(f"Docker not ready for verification of {model_hash[:16]}...")
        return

    # Create verification container name
    container_name = f"swarm_verify_{model_hash[:8]}_{int(time.time() * 1000)}"

    try:
        # Create temp directory for verification
        with tempfile.TemporaryDirectory() as tmpdir:
            current_uid = os.getuid()
            current_gid = os.getgid()
            os.chown(tmpdir, current_uid, current_gid)
            os.chmod(tmpdir, 0o755)

            verification_result_file = Path(tmpdir) / "verification_result.json"

            # Create minimal task for verification (not used for actual evaluation)
            dummy_task = {
                "start": [0, 0, 1], "goal": [5, 5, 2], "obstacles": [],
                "horizon": HORIZON_SEC, "seed": 12345
            }

            task_file = Path(tmpdir) / "task.json"
            with open(task_file, 'w') as f:
                json.dump(dummy_task, f)

            bt.logging.info(f"🐳 Starting Docker container for verification of UID model {model_hash[:16]}...")

            # Docker run command for verification (copy model inside container)
            cmd = [
                "docker", "run",
                "--rm",
                "--name", container_name,
                "--user", f"{current_uid}:{current_gid}",
                "--memory=4g",
                "--cpus=1",
                "--pids-limit=10",
                "--ulimit", "nofile=256:256",
                "--ulimit", "fsize=524288000:524288000",
                "--security-opt", "no-new-privileges",
                "--network", "none",
                "-v", f"{tmpdir}:/workspace/shared",
                "-v", f"{model_path.absolute()}:/workspace/model.zip:ro",
                docker_evaluator.base_image,
                "python", "/app/swarm/core/evaluator.py",
                "VERIFY_ONLY",
                str(uid),
                "/workspace/model.zip",
                "/workspace/shared/verification_result.json"
            ]

            # Execute verification with timeout
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=60  # 1 minute timeout for verification
                )

                # Enhanced debugging for verification
                stdout_str = stdout.decode() if stdout else ""
                stderr_str = stderr.decode() if stderr else ""

                bt.logging.debug(f"Verification container for {model_hash[:16]}:")
                bt.logging.debug(f"  Return code: {proc.returncode}")
                bt.logging.debug(f"  STDOUT: {stdout_str}")
                bt.logging.debug(f"  STDERR: {stderr_str}")

                if proc.returncode != 0:
                    bt.logging.warning(f"Verification container failed for {model_hash[:16]} with return code {proc.returncode}")
                    bt.logging.warning(f"Error output: {stderr_str}")

            except asyncio.TimeoutError:
                # Kill container if timeout
                subprocess.run(["docker", "kill", container_name], capture_output=True)
                bt.logging.warning(f"⏰ Verification timeout for model {model_hash[:16]}...")
                return

            bt.logging.info(f"🔚 Ending Docker container for verification of model {model_hash[:16]}...")

            # Read verification results
            if verification_result_file.exists():
                try:
                    with open(verification_result_file, 'r') as f:
                        verification_data = json.load(f)

                    # Handle different verification outcomes
                    if verification_data.get('is_fake_model', False):
                        # Actually fake/malicious model → blacklist
                        fake_reason = verification_data.get('fake_reason', 'Unknown')
                        inspection_results = verification_data.get('inspection_results', {})

                        bt.logging.warning(f"🚫 FAKE MODEL DETECTED during verification: {fake_reason}")
                        bt.logging.info(f"Model hash: {model_hash}")
                        bt.logging.debug(f"Inspection details: {inspection_results}")

                        # Save fake model for analysis and add to blacklist
                        save_fake_model_for_analysis(model_path, uid, model_hash, fake_reason, inspection_results)
                        add_to_blacklist(model_hash)

                        # Remove the fake model from cache
                        model_path.unlink(missing_ok=True)
                        bt.logging.info(f"🗑️ Removed fake model {model_hash[:16]}... from cache and blacklisted")

                    elif verification_data.get('missing_drone_agent', False):
                        # Missing drone_agent.py → reject but don't blacklist
                        rejection_reason = verification_data.get('rejection_reason', 'Missing drone_agent.py')

                        bt.logging.warning(f"⚠️ MISSING drone_agent.py during verification: {rejection_reason}")
                        bt.logging.info(f"Model hash: {model_hash}")

                        # Remove model but don't blacklist (allows resubmission)
                        model_path.unlink(missing_ok=True)
                        bt.logging.info(f"🗑️ Removed model {model_hash[:16]}... from cache (missing drone_agent.py - can resubmit)")

                    else:
                        # Legitimate model
                        bt.logging.info(f"✅ Model {model_hash[:16]}... passed verification - legitimate model")

                except Exception as e:
                    bt.logging.warning(f"Failed to parse verification results for {model_hash[:16]}: {e}")
            else:
                bt.logging.warning(f"No verification results found for model {model_hash[:16]}...")

                # Debug: Check what files exist in the temp directory
                try:
                    temp_files = list(Path(tmpdir).glob("*"))
                    bt.logging.debug(f"Files in temp directory: {[f.name for f in temp_files]}")

                    # Check if the result file path is what we expect
                    expected_file = Path(tmpdir) / "verification_result.json"
                    bt.logging.debug(f"Expected result file: {expected_file}")
                    bt.logging.debug(f"Expected file exists: {expected_file.exists()}")

                except Exception as e:
                    bt.logging.debug(f"Error checking temp directory: {e}")

    except Exception as e:
        bt.logging.warning(f"Docker verification failed for model {model_hash[:16]}: {e}")
        # Ensure container is killed
        subprocess.run(["docker", "kill", container_name], capture_output=True)

def load_model_hash_tracker() -> dict:
    """Load UID to model hash mapping."""
    hash_tracker_file = STATE_DIR / "uid_model_hashes.json"
    try:
        if hash_tracker_file.exists():
            with open(hash_tracker_file, 'r') as f:
                return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return {}


def save_model_hash_tracker(tracker: dict) -> None:
    """Save UID to model hash mapping."""
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
    """Persist that a UID/hash pair has been fully handled."""
    tracker = load_model_hash_tracker()
    tracker[str(uid)] = model_hash
    save_model_hash_tracker(tracker)


def load_normal_model_queue() -> dict:
    """Load persistent normal-model processing queue."""
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
    """Save normal-model queue atomically."""
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


def _extract_backend_reason(response: Dict[str, Any]) -> str:
    for key in ("detail", "reason", "message", "error"):
        value = response.get(key)
        if value:
            return str(value)
    return str(response)


def _classify_backend_failure(response: Dict[str, Any], stage: str) -> Tuple[bool, str]:
    reason = _extract_backend_reason(response)
    text = reason.lower()

    if stage == "new_model":
        terminal_patterns = (
            "already submitted",
            "can only submit once",
            "already registered",
            "already exists",
            "duplicate",
            "409",
            "429",
            "conflict",
        )
        if any(pattern in text for pattern in terminal_patterns):
            return True, reason

    if stage in ("screening", "score"):
        terminal_patterns = (
            "no model with uid",
            "pending screening",
            "pending benchmark",
            "not found",
            "404",
        )
        if any(pattern in text for pattern in terminal_patterns):
            return True, reason

    transient_patterns = (
        "timeout",
        "timed out",
        "connection",
        "temporar",
        "unreachable",
        "503",
        "502",
        "500",
        "network",
    )
    if any(pattern in text for pattern in transient_patterns):
        return False, reason

    return False, reason


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


def load_map_cache_warmup_state() -> Dict[str, Any]:
    """Load incremental map-cache warmup state."""
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
    """Save incremental map-cache warmup state atomically."""
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
    if state.get("completed"):
        return

    screening_seeds = self.seed_manager.get_screening_seeds()
    public_seeds = self.seed_manager.get_public_seeds()

    screening_index = int(state.get("screening_index", 0) or 0)
    public_index = int(state.get("public_index", 0) or 0)

    total_seeds = len(screening_seeds) + len(public_seeds)
    warmed_before = screening_index + public_index
    if warmed_before >= total_seeds:
        state["completed"] = True
        state["last_update"] = time.time()
        save_map_cache_warmup_state(state)
        return

    bt.logging.info(
        f"🗺️ Map cache prebuild mode: building all remaining seeds "
        f"({warmed_before}/{total_seeds} already warmed)"
    )

    failed_now = 0
    logged_failures = 0

    while screening_index < len(screening_seeds):
        seed = int(screening_seeds[screening_index])
        screening_index += 1

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

        warmed_total = screening_index + public_index
        if warmed_total % 100 == 0:
            bt.logging.info(f"🗺️ Map cache prebuild progress: {warmed_total}/{total_seeds}")
            await asyncio.sleep(0)

    while public_index < len(public_seeds):
        seed = int(public_seeds[public_index])
        public_index += 1

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

        warmed_total = screening_index + public_index
        if warmed_total % 100 == 0:
            bt.logging.info(f"🗺️ Map cache prebuild progress: {warmed_total}/{total_seeds}")
            await asyncio.sleep(0)

    state["screening_index"] = screening_index
    state["public_index"] = public_index
    state["failed_count"] = int(state.get("failed_count", 0) or 0) + failed_now
    state["completed"] = True
    state["last_update"] = time.time()
    save_map_cache_warmup_state(state)

    bt.logging.info(
        f"🗺️ Map cache prebuild complete: {screening_index + public_index}/{total_seeds} seeds "
        f"(failures={state['failed_count']})"
    )


async def _run_map_cache_warmup_step(self) -> None:
    """Warm map cache incrementally (small batch per forward cycle)."""
    if not MAP_CACHE_ENABLED:
        return

    if not hasattr(self, 'seed_manager'):
        return

    state = load_map_cache_warmup_state()
    if state.get("completed"):
        return

    screening_seeds = self.seed_manager.get_screening_seeds()
    public_seeds = self.seed_manager.get_public_seeds()

    screening_index = int(state.get("screening_index", 0) or 0)
    public_index = int(state.get("public_index", 0) or 0)

    total_seeds = len(screening_seeds) + len(public_seeds)
    warmed_before = screening_index + public_index
    if warmed_before >= total_seeds:
        state["completed"] = True
        state["last_update"] = time.time()
        save_map_cache_warmup_state(state)
        return

    warmed_now = 0
    failed_now = 0

    while warmed_now < MAP_CACHE_WARMUP_BATCH_SIZE:
        phase = "screening" if screening_index < len(screening_seeds) else "public"

        if phase == "screening":
            seed = int(screening_seeds[screening_index])
            screening_index += 1
        else:
            if public_index >= len(public_seeds):
                break
            seed = int(public_seeds[public_index])
            public_index += 1

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

    state["screening_index"] = screening_index
    state["public_index"] = public_index
    state["failed_count"] = int(state.get("failed_count", 0) or 0) + failed_now
    state["last_update"] = time.time()

    warmed_total = screening_index + public_index
    if warmed_total >= total_seeds:
        state["completed"] = True
        bt.logging.info(
            f"🗺️ Map cache warmup complete: {warmed_total}/{total_seeds} seeds "
            f"(failures={state['failed_count']})"
        )
    else:
        bt.logging.info(
            f"🗺️ Map cache warmup progress: +{warmed_now} this cycle, "
            f"{warmed_total}/{total_seeds} total"
        )

    save_map_cache_warmup_state(state)


# ──────────────────────────────────────────────────────────────────────────
# 3.  Benchmark score cache (by model_hash + benchmark_version)
# ──────────────────────────────────────────────────────────────────────────
CACHE_FILE = STATE_DIR / "benchmark_cache.json"

def load_benchmark_cache() -> dict:
    """Load benchmark score cache from disk."""
    try:
        if CACHE_FILE.exists():
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        bt.logging.warning(f"Cache load failed, starting fresh: {e}")
    return {}


def save_benchmark_cache(cache: dict) -> None:
    """Save benchmark cache to disk atomically."""
    STATE_DIR.mkdir(exist_ok=True)
    temp_file = CACHE_FILE.with_suffix(".tmp")
    try:
        with open(temp_file, 'w') as f:
            json.dump(cache, f)
        temp_file.replace(CACHE_FILE)
    except IOError as e:
        bt.logging.error(f"Cache save failed: {e}")
        temp_file.unlink(missing_ok=True)


def get_cached_score(model_hash: str) -> Optional[Dict[str, Any]]:
    """Get cached benchmark score for model.

    Returns:
        Dict with score data if cached, None if not found.
    """
    cache = load_benchmark_cache()
    key = f"{model_hash}_{BENCHMARK_VERSION}"
    result = cache.get(key)
    if result:
        bt.logging.debug(f"Cache hit for {model_hash[:16]}...")
    return result


def set_cached_score(model_hash: str, result: Dict[str, Any]) -> None:
    """Cache benchmark score for model.

    Args:
        model_hash: SHA256 hash of the model
        result: Dict containing score data
    """
    cache = load_benchmark_cache()
    key = f"{model_hash}_{BENCHMARK_VERSION}"
    result["cached_at"] = time.time()
    result["benchmark_version"] = BENCHMARK_VERSION
    cache[key] = result
    save_benchmark_cache(cache)
    bt.logging.info(f"Cached score for {model_hash[:16]}...")


def has_cached_score(model_hash: str) -> bool:
    """Check if model has cached benchmark score."""
    cache = load_benchmark_cache()
    key = f"{model_hash}_{BENCHMARK_VERSION}"
    return key in cache


async def send_with_fresh_uuid(
    wallet: "bt.Wallet",
    synapse: "bt.Synapse",
    axon,
    *,
    timeout: float,
    deserialize: bool = True,
    ):
    """
    Creates a *new* transient Dendrite client for this single RPC so that the
    library stamps a fresh `dendrite.uuid`.  That guarantees every miner sees
    an endpoint_key they have never stored before ⇒ no nonce collisions.
    """

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

async def _process_single_uid(self, uid: int) -> Tuple[int, Optional[Path]]:
    """Process a single UID for model retrieval"""
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
                and _zip_is_safe(model_fp, max_uncompressed=MAX_MODEL_BYTES)
            ):
                return (uid, model_fp)
            else:
                model_fp.unlink(missing_ok=True)

        await _download_model(self, axon, ref, model_fp, uid)
        if (
            model_fp.is_file()
            and sha256sum(model_fp) == ref.sha256
            and model_fp.stat().st_size <= MAX_MODEL_BYTES
            and _zip_is_safe(model_fp, max_uncompressed=MAX_MODEL_BYTES)
        ):
            return (uid, model_fp)
        else:
            model_fp.unlink(missing_ok=True)
            return (uid, None)

    except Exception:
        return (uid, None)


async def _ensure_models(self, uids: List[int]) -> Dict[int, Path]:
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
            bt.logging.debug(f"Batch {batch_num}/{total_batches}: found {batch_found} models, total so far: {len(paths)}")

        if batch_start + PARALLEL_BATCH_SIZE < len(uids):
            await asyncio.sleep(BATCH_DELAY_SEC)

    bt.logging.info(f"Model fetch complete: found {len(paths)} models from {len(uids)} UIDs")
    return paths


# ──────────────────────────────────────────────────────────────────────────
# 3.  Benchmark evaluation helpers
# ──────────────────────────────────────────────────────────────────────────

async def _evaluate_seeds(
    self,
    uid: int,
    model_path: Path,
    seeds: List[int],
    description: str = "benchmark",
    on_seed_complete: Optional[Callable[[], None]] = None,
) -> Tuple[List[float], Dict[str, List[float]]]:
    """
    Evaluate a model on multiple seeds using parallel Docker containers.

    This is the main evaluation entry point for V4 benchmark.
    Uses N_DOCKER_WORKERS parallel containers for speedup.

    Args:
        self: Validator instance
        uid: Miner UID
        model_path: Path to model ZIP
        seeds: List of seeds to evaluate
        description: Log description (e.g., "screening", "full benchmark")
        on_seed_complete: Optional callback called after each seed (thread-safe)

    Returns:
        Tuple of (all_scores, per_type_scores)
        - all_scores: List of scores for each seed
        - per_type_scores: Dict mapping challenge type name to list of scores
    """
    all_scores = []
    per_type_scores = {"city": [], "open": [], "mountain": [], "moving_platform": []}

    challenge_type_to_name = {
        1: "city",
        2: "open",
        3: "mountain",
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
    """
    Run screening benchmark with 200 private seeds.

    Args:
        self: Validator instance
        uid: Miner UID
        model_path: Path to model ZIP

    Returns:
        Tuple of (median_score, all_scores)
    """
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


async def _run_full_benchmark(self, uid: int, model_path: Path) -> Tuple[float, Dict[str, float], List[float]]:
    """
    Run full benchmark with 1000 public seeds.

    Args:
        self: Validator instance
        uid: Miner UID
        model_path: Path to model ZIP

    Returns:
        Tuple of (median_score, per_type_median_scores, all_scores)
    """
    public_seeds = self.seed_manager.get_public_seeds()

    hb = HeartbeatManager(self.backend_api, asyncio.get_running_loop())
    hb.start("evaluating_benchmark", uid, len(public_seeds))

    try:
        all_scores, per_type_scores = await _evaluate_seeds(
            self, uid, model_path, public_seeds, "full benchmark",
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


def _passes_screening(self, screening_score: float) -> bool:
    """
    Check if screening score meets the threshold.

    Bootstrap mode: If no current top model, just need > 0.1
    Normal mode: Must be within 80% of the top model's screening score

    Args:
        self: Validator instance (has backend_api with current_top)
        screening_score: The model's screening score

    Returns:
        True if passed, False if failed
    """
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
        f"Screening: {screening_score:.4f} >= {threshold:.4f} (80% of top {top_score:.4f}) = {passed}"
    )
    return passed


def _detect_new_models(self, model_paths: Dict[int, Path]) -> Dict[int, Tuple[Path, str]]:
    """
    Detect models that have changed (new hash) since last check.

    Args:
        self: Validator instance
        model_paths: Dict of UID -> model path

    Returns:
        Dict of UID -> (model_path, model_hash) for changed models
    """
    tracker = load_model_hash_tracker()
    new_models = {}

    for uid, path in model_paths.items():
        try:
            current_hash = sha256sum(path)
            uid_str = str(uid)
            old_hash = tracker.get(uid_str)

            if old_hash != current_hash:
                if old_hash:
                    bt.logging.info(f"🔄 Model changed for UID {uid}: {old_hash[:16]}... → {current_hash[:16]}...")
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
        return True, False, ""

    terminal, reason = _classify_backend_failure(response, "new_model")
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

    terminal, reason = _classify_backend_failure(response, "screening")
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
) -> Tuple[bool, bool, str]:
    response = await self.backend_api.post_score(
        uid=uid,
        validator_hotkey=validator_hotkey,
        validator_stake=validator_stake,
        model_hash=model_hash,
        total_score=total_score,
        per_type_scores=per_type_scores,
        seeds_evaluated=seeds_evaluated,
    )

    if response.get("recorded", False):
        return True, False, ""

    terminal, reason = _classify_backend_failure(response, "score")
    return False, terminal, reason


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

        cached = get_cached_score(model_hash) if has_cached_score(model_hash) else None

        if item.get("screening_score") is None:
            if cached:
                item["screening_score"] = float(cached.get("screening_score", 0.0))
            else:
                screening_score, screening_scores = await _run_screening(self, uid, model_path)
                item["screening_score"] = float(screening_score)
                item["screening_scores"] = screening_scores
            item["updated_at"] = time.time()

        if item.get("screening_passed") is None:
            item["screening_passed"] = _passes_screening(self, float(item.get("screening_score", 0.0)))

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
                item["full_score"] = float(cached.get("full_score", cached.get("total_score", 0.0)))
                item["total_score"] = float(cached.get("total_score", 0.0))
                item["per_type_scores"] = per_type_scores
                item["seeds_evaluated"] = int(cached.get("seeds_evaluated", 1200))
            else:
                full_score, per_type_scores, full_scores = await _run_full_benchmark(self, uid, model_path)
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

        set_cached_score(model_hash, {
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
# 5.  Public coroutine – called by neurons/validator.py
# ──────────────────────────────────────────────────────────────────────────
def _apply_backend_weights_to_scores(self, backend_weights: Dict[Any, Any]) -> None:
    """Apply backend weights to validator scores with deterministic reset behavior."""
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


async def forward(self) -> None:
    """
    Benchmark-style validator forward.

    Flow:
    1. Sync with backend → get weights + reeval queue
    2. Apply weights from backend (set on-chain)
    3. Process re-eval queue (if any)
    4. Detect new/changed models
    5. For each new model: screening → full benchmark → submit
    """
    try:
        self.forward_count = getattr(self, "forward_count", 0) + 1
        bt.logging.info(f"[Forward #{self.forward_count}] start")

        # ──────────────────────────────────────────────────────────────
        # STEP 0: Initialize components if needed
        # ──────────────────────────────────────────────────────────────
        if not hasattr(self, 'seed_manager'):
            self.seed_manager = BenchmarkSeedManager()

        if not hasattr(self, 'backend_api'):
            try:
                self.backend_api = BackendApiClient(wallet=self.wallet)
            except ValueError as e:
                bt.logging.error(f"Backend API initialization failed: {e}")
                bt.logging.error("Set SWARM_BACKEND_API_URL environment variable")
                await asyncio.sleep(FORWARD_SLEEP_SEC)
                return

        if not hasattr(self, 'docker_evaluator') or not DockerSecureEvaluator._base_ready:
            bt.logging.error("Docker evaluator not ready")
            await asyncio.sleep(FORWARD_SLEEP_SEC)
            return

        validator_hotkey = self.wallet.hotkey.ss58_address
        validator_stake = _get_validator_stake(self)

        # ──────────────────────────────────────────────────────────────
        # STEP 0.5: Map-cache warmup (prebuild-all or incremental)
        # ──────────────────────────────────────────────────────────────
        if MAP_CACHE_PREBUILD_ALL_AT_START:
            await _run_map_cache_prebuild_all_once(self)
        else:
            await _run_map_cache_warmup_step(self)

        # ──────────────────────────────────────────────────────────────
        # STEP 1: Sync with backend
        # ──────────────────────────────────────────────────────────────
        bt.logging.info("📡 Syncing with backend...")
        sync_data = await self.backend_api.sync()

        if sync_data.get("fallback"):
            bt.logging.warning("Using freeze-last weights (backend unavailable)")

        self._current_top = sync_data.get("current_top", {})
        reeval_queue = sync_data.get("reeval_queue", [])

        # ──────────────────────────────────────────────────────────────
        # STEP 2: Apply weights from backend
        # ──────────────────────────────────────────────────────────────
        backend_weights = sync_data.get("weights", {})
        _apply_backend_weights_to_scores(self, backend_weights)
        nonzero_uids = int(np.count_nonzero(self.scores))
        bt.logging.info(f"⚖️ Applied backend weights to local scores: {nonzero_uids} non-zero UID(s)")

        # ──────────────────────────────────────────────────────────────
        # STEP 3: Process re-eval queue
        # ──────────────────────────────────────────────────────────────
        if sync_data.get("fallback") and reeval_queue:
            bt.logging.warning(
                f"Freeze-last active: skipping {len(reeval_queue)} cached re-eval item(s) this cycle"
            )
        else:
            for reeval_item in reeval_queue:
                uid = reeval_item.get("uid")
                reason = reeval_item.get("reason", "unknown")
                bt.logging.info(f"🔄 Re-evaluation requested for UID {uid}: {reason}")

                model_path = MODEL_DIR / f"UID_{uid}.zip"
                if not model_path.exists():
                    bt.logging.warning(f"Model not found for re-eval UID {uid}")
                    continue

                model_hash = sha256sum(model_path)

                screening_score, screening_scores = await _run_screening(self, uid, model_path)
                full_score, per_type_scores, full_scores = await _run_full_benchmark(self, uid, model_path)

                combined_scores = screening_scores + full_scores
                all_seeds_count = len(combined_scores)
                total_score = float(np.median(combined_scores)) if combined_scores else 0.0

                recorded, terminal, ack_reason = await _submit_score_with_ack(
                    self,
                    uid=uid,
                    validator_hotkey=validator_hotkey,
                    validator_stake=validator_stake,
                    model_hash=model_hash,
                    total_score=total_score,
                    per_type_scores=per_type_scores,
                    seeds_evaluated=all_seeds_count,
                )

                if not recorded:
                    if terminal:
                        bt.logging.error(
                            f"Re-eval score rejected permanently for UID {uid}: {ack_reason}"
                        )
                    else:
                        bt.logging.warning(
                            f"Re-eval score submit failed for UID {uid}, will retry next sync: {ack_reason}"
                        )
                    continue

                set_cached_score(model_hash, {
                    "uid": uid,
                    "total_score": total_score,
                    "screening_score": screening_score,
                    "full_score": full_score,
                    "per_type_scores": per_type_scores,
                    "seeds_evaluated": all_seeds_count
                })

                bt.logging.info(f"✅ Re-eval complete for UID {uid}: score={total_score:.4f}")

        # ──────────────────────────────────────────────────────────────
        # STEP 4: Discovery refresh (normal-model queue producer)
        # ──────────────────────────────────────────────────────────────
        uids = get_random_uids(self, k=SAMPLE_K)
        bt.logging.info(f"Checking {len(uids)} miners for model updates...")

        model_paths = await _ensure_models(self, uids)
        if not model_paths:
            bt.logging.info("No models found this cycle")
            new_models = {}
        else:
            new_models = _detect_new_models(self, model_paths)
            if new_models:
                bt.logging.info(f"🆕 Found {len(new_models)} new/changed models for queue")
            else:
                bt.logging.info("No new/changed models detected")

        queue = _refresh_normal_model_queue(new_models) if new_models else load_normal_model_queue()

        # ──────────────────────────────────────────────────────────────
        # STEP 5: Queue worker (normal-model pipeline consumer)
        # ──────────────────────────────────────────────────────────────
        processable_keys = _get_processable_queue_keys(queue, NORMAL_MODEL_QUEUE_PROCESS_LIMIT)
        if not processable_keys:
            bt.logging.info("No normal-model queue items ready this cycle")
        else:
            bt.logging.info(f"📦 Processing {len(processable_keys)} queued model(s)")
            for queue_key in processable_keys:
                await _process_normal_queue_item(
                    self,
                    queue=queue,
                    key=queue_key,
                    validator_hotkey=validator_hotkey,
                    validator_stake=validator_stake,
                )

        items = queue.get("items", {})
        completed_keys = [
            key for key, item in items.items()
            if item.get("status") in ("completed", "terminal_rejected")
        ]
        for key in completed_keys:
            items.pop(key, None)

        queue["items"] = items
        save_normal_model_queue(queue)

        self.docker_evaluator.cleanup()
        bt.logging.info(f"[Forward #{self.forward_count}] complete")

    except Exception as e:
        bt.logging.error(f"Validator forward error: {e}")
        bt.logging.error(traceback.format_exc())

    await asyncio.sleep(FORWARD_SLEEP_SEC)
