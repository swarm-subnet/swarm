# ---------------------------------------------------------------
#  Swarm validator – Policy API v2   (hardened, 10 MiB limits)
# ---------------------------------------------------------------
from __future__ import annotations

import asyncio
import gc
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import bittensor as bt
import numpy as np
from zipfile import ZipFile, BadZipFile

from swarm.protocol import PolicySynapse, PolicyRef, ValidationResult
from swarm.utils.uids import get_random_uids, get_low_performer_uids
from swarm.utils.hash import sha256sum
import base64

from ..core.model_verify import (
    load_blacklist,
    save_blacklist,
    add_to_blacklist,
    save_fake_model_for_analysis,
)
from .task_gen import random_task
from .docker.docker_evaluator import DockerSecureEvaluator
from .seed_manager import SynchronizedSeedManager
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
    WINNER_TAKE_ALL,
    N_RUNS_HISTORY,
    MIN_RUNS_FOR_WEIGHTS,
    MIN_RECENT_RUNS_FOR_WEIGHTS,
    RECENT_RUN_WINDOW_SEC,
    UID_HISTORY_STALE_RESET_SEC,
    AVGS_DIR,
    ENABLE_PER_TYPE_NORMALIZATION,
    CHALLENGE_TYPE_DISTRIBUTION,
    USE_SYNCHRONIZED_SEEDS,
    SEED_WINDOW_MINUTES,
    PARALLEL_BATCH_SIZE,
    MAX_CONCURRENT_CONNECTIONS,
    BATCH_DELAY_SEC,
    PRIORITY_RETRY_TIMEOUT,
    MIN_RESPONSE_STREAK,
    MAX_FAILED_CYCLES,
)


# ──────────────────────────────────────────────────────────────────────────
# 1.  Helpers – secure ZIP inspection
# ──────────────────────────────────────────────────────────────────────────
def _zip_is_safe(path: Path, *, max_uncompressed: int) -> bool:
    """
    Reject dangerous ZIP files *without* extracting them.

    • Total uncompressed size must not exceed `max_uncompressed`.
    • No absolute paths or “..” traversal sequences.
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
                        f"({total_uncompressed/1e6:.1f} MB > {max_uncompressed/1e6:.1f} MB)"
                    )
                    return False
            return True
    except BadZipFile:
        bt.logging.error("Corrupted ZIP archive.")
        return False
    except Exception as e:
        bt.logging.error(f"ZIP inspection error: {e}")
        return False


def _update_response_tracking(uid: int, responded: bool) -> None:
    try:
        uid = int(uid)
        history = load_uid_history(uid)

        if responded:
            history["response_streak"] = history.get("response_streak", 0) + 1
            history["failed_cycles"] = 0
        else:
            failed = history.get("failed_cycles", 0) + 1
            history["failed_cycles"] = failed
            if failed >= MAX_FAILED_CYCLES:
                history["response_streak"] = 0

        save_uid_history(uid, history)
    except Exception as e:
        bt.logging.warning(f"Failed to update response tracking for UID {uid}: {e}")


def _get_priority_uids() -> set:
    ensure_avgs_directory()
    priority = set()
    
    for file_path in AVGS_DIR.glob("uid_*.json"):
        try:
            uid_str = file_path.stem.replace("uid_", "")
            uid = int(uid_str)
            
            with open(file_path, 'r') as f:
                history = json.load(f)
            streak = history.get("response_streak", 0)
            if streak >= MIN_RESPONSE_STREAK:
                priority.add(uid)
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            continue
    
    return priority


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
    hash_tracker_file = Path("/tmp/uid_model_hashes.json")
    try:
        if hash_tracker_file.exists():
            with open(hash_tracker_file, 'r') as f:
                return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return {}


def save_model_hash_tracker(tracker: dict) -> None:
    """Save UID to model hash mapping."""
    hash_tracker_file = Path("/tmp/uid_model_hashes.json")
    with open(hash_tracker_file, 'w') as f:
        json.dump(tracker, f)


def clear_low_performer_status(uid: int) -> None:
    """Clear low-performer flag and start grace period when model is updated."""
    history_file = Path("/tmp/victory_history.json")
    if not history_file.exists():
        return

    try:
        with open(history_file, 'r') as f:
            history = json.load(f)

        uid_str = str(uid)
        if uid_str in history:
            if "is_low_performer" in history[uid_str]:
                del history[uid_str]["is_low_performer"]

            history[uid_str]["grace_period_start"] = len(history[uid_str].get("runs", []))

            with open(history_file, 'w') as f:
                json.dump(history, f)
            bt.logging.info(f"Cleared low-performer flag for UID {uid}, grace period started (runs preserved)")
    except Exception as e:
        bt.logging.debug(f"Failed to clear low-performer status for UID {uid}: {e}")


def check_and_update_model_hash(uid: int, new_hash: str) -> bool:
    """Check if model hash has changed and update tracker.

    Returns:
        True if hash changed (model updated), False otherwise
    """
    tracker = load_model_hash_tracker()
    uid_str = str(uid)
    old_hash = tracker.get(uid_str)

    if old_hash and old_hash != new_hash:
        bt.logging.info(f"Model update detected for UID {uid}: {old_hash[:16]}... → {new_hash[:16]}...")
        clear_low_performer_status(uid)
        tracker[uid_str] = new_hash
        save_model_hash_tracker(tracker)
        return True
    elif not old_hash:
        tracker[uid_str] = new_hash
        save_model_hash_tracker(tracker)

    return False


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
                check_and_update_model_hash(uid, ref.sha256)
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
            check_and_update_model_hash(uid, ref.sha256)
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

    queried_uids = set()
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
            queried_uids.add(uid)
            if path is not None:
                paths[uid] = path
                batch_found += 1
        
        if batch_num % 5 == 0 or batch_found > 0:
            bt.logging.debug(f"Batch {batch_num}/{total_batches}: found {batch_found} models, total so far: {len(paths)}")

        if batch_start + PARALLEL_BATCH_SIZE < len(uids):
            await asyncio.sleep(BATCH_DELAY_SEC)

    priority_uids = _get_priority_uids()
    failed_priority = [u for u in priority_uids if u in queried_uids and u not in paths]
    
    if failed_priority:
        bt.logging.info(f"Retrying {len(failed_priority)} priority UIDs: {failed_priority}")
        
        for uid in failed_priority:
            try:
                result = await _process_single_uid_retry(self, uid)
                if result[1] is not None:
                    paths[uid] = result[1]
                    bt.logging.info(f"Priority retry success for UID {uid}")
            except Exception:
                pass

    for uid in queried_uids:
        _update_response_tracking(uid, uid in paths)

    bt.logging.info(f"Model fetch complete: found {len(paths)} models from {len(uids)} UIDs")
    return paths


async def _process_single_uid_retry(self, uid: int) -> Tuple[int, Optional[Path]]:
    try:
        axon = self.metagraph.axons[uid]

        try:
            responses = await send_with_fresh_uuid(
                wallet=self.wallet,
                synapse=PolicySynapse.request_ref(),
                axon=axon,
                timeout=PRIORITY_RETRY_TIMEOUT,
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
                check_and_update_model_hash(uid, ref.sha256)
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
            check_and_update_model_hash(uid, ref.sha256)
            return (uid, model_fp)
        else:
            model_fp.unlink(missing_ok=True)
            return (uid, None)

    except Exception:
        return (uid, None)




# ──────────────────────────────────────────────────────────────────────────
# 3.  Performance history tracking system
# ──────────────────────────────────────────────────────────────────────────

def _log_normalized_score(uid: int) -> None:
    history = load_uid_history(uid)
    normalized_score = calculate_normalized_score(history)

    if ENABLE_PER_TYPE_NORMALIZATION:
        all_runs = history.get("all_runs", [])

        type_runs = {}
        for run in all_runs:
            t = str(run.get("challenge_type", 1))
            if t not in type_runs:
                type_runs[t] = []
            type_runs[t].append(run["score"])

        type_info = []
        for type_id in sorted(CHALLENGE_TYPE_DISTRIBUTION.keys()):
            type_str = str(type_id)
            if type_str not in type_runs:
                continue
            scores = type_runs[type_str]
            weight = CHALLENGE_TYPE_DISTRIBUTION[type_id]
            avg = sum(scores) / len(scores)
            type_info.append(f"T{type_id}({weight:.0%}):{len(scores)}runs/{avg:.3f}avg")

        bt.logging.info(
            f"UID {uid:3d} | normalized: {normalized_score:.4f} | {' | '.join(type_info)}"
        )

def load_victory_history() -> dict:
    """Load victory history from temp file."""
    try:
        with open("/tmp/victory_history.json", "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_victory_history(history: dict) -> None:
    """Save victory history to temp file."""
    with open("/tmp/victory_history.json", "w") as f:
        json.dump(history, f)

def update_victory_history(history: dict, uid: int, won: bool, score: float) -> None:
    """Update victory history for a UID with rolling window."""
    uid_str = str(uid)
    if uid_str not in history:
        history[uid_str] = {"runs": []}
    
    history[uid_str]["runs"].append({"won": won, "score": float(score)})
    
    if len(history[uid_str]["runs"]) > N_RUNS_HISTORY:
        history[uid_str]["runs"] = history[uid_str]["runs"][-N_RUNS_HISTORY:]

def calculate_score_metrics(history: dict, uids: np.ndarray) -> List[tuple]:
    """Calculate average score and victory rate for each UID."""
    metrics = []
    for uid in uids:
        uid_str = str(uid)
        if uid_str not in history or not history[uid_str]["runs"]:
            continue
            
        runs = history[uid_str]["runs"]
        wins = [run for run in runs if run["won"]]
        
        avg_score = sum(run["score"] for run in runs) / len(runs)
        victory_rate = len(wins) / len(runs)
        
        metrics.append((uid, avg_score, victory_rate))
    
    return metrics

# ──────────────────────────────────────────────────────────────────────────
# 3.5.  Per-Type Normalization System
# ──────────────────────────────────────────────────────────────────────────

def ensure_avgs_directory():
    AVGS_DIR.mkdir(parents=True, exist_ok=True)


def _migrate_to_shared_pool(old_history: dict) -> dict:
    all_runs = []
    for type_id, type_data in old_history.get("runs_by_type", {}).items():
        for run in type_data.get("runs", []):
            run_copy = run.copy()
            run_copy["challenge_type"] = int(type_id)
            all_runs.append(run_copy)

    all_runs.sort(key=lambda x: x.get("timestamp", 0))
    all_runs = all_runs[-N_RUNS_HISTORY:]

    return {
        "uid": old_history.get("uid", 0),
        "total_runs": old_history.get("total_runs", 0),
        "last_updated": time.time(),
        "all_runs": all_runs,
        "normalized_score": 0.0
    }


def _default_uid_history(uid: int) -> dict:
    return {
        "uid": uid,
        "total_runs": 0,
        "last_updated": 0.0,
        "all_runs": [],
        "normalized_score": 0.0,
        "response_streak": 0,
        "failed_cycles": 0,
    }


def _apply_inactivity_reset(uid: int, history: dict) -> Tuple[dict, bool]:
    last_updated = float(history.get("last_updated", 0.0) or 0.0)
    all_runs = history.get("all_runs", [])

    if not all_runs or last_updated <= 0.0:
        return history, False

    idle_seconds = time.time() - last_updated
    if idle_seconds < UID_HISTORY_STALE_RESET_SEC:
        return history, False

    bt.logging.info(
        f"Resetting stale history for UID {uid} after {idle_seconds:.0f}s of inactivity"
    )
    history["total_runs"] = 0
    history["all_runs"] = []
    history["normalized_score"] = 0.0
    history["response_streak"] = 0
    history["failed_cycles"] = 0
    return history, True


def _count_recent_runs(history: dict) -> int:
    cutoff = time.time() - RECENT_RUN_WINDOW_SEC
    return sum(
        1 for run in history.get("all_runs", [])
        if float(run.get("timestamp", 0.0) or 0.0) >= cutoff
    )


def _uid_weight_eligibility(history: dict) -> Tuple[bool, str]:
    total_runs = int(history.get("total_runs", 0))
    if total_runs < MIN_RUNS_FOR_WEIGHTS:
        return False, f"{total_runs}/{MIN_RUNS_FOR_WEIGHTS} total runs"

    recent_runs = _count_recent_runs(history)
    if recent_runs < MIN_RECENT_RUNS_FOR_WEIGHTS:
        return (
            False,
            f"{recent_runs}/{MIN_RECENT_RUNS_FOR_WEIGHTS} recent runs in {RECENT_RUN_WINDOW_SEC}s",
        )

    return True, "eligible"


def load_uid_history(uid: int) -> dict:
    ensure_avgs_directory()
    uid = int(uid)
    file_path = AVGS_DIR / f"uid_{uid}.json"

    defaults = _default_uid_history(uid)

    if file_path.exists():
        try:
            with open(file_path, 'r') as f:
                history = json.load(f)

            if "all_runs" not in history and "runs_by_type" in history:
                bt.logging.info(f"Migrating UID {uid} history to shared pool format")
                history = _migrate_to_shared_pool(history)
                save_uid_history(uid, history)

            for key, value in defaults.items():
                if key not in history:
                    history[key] = value

            history, was_reset = _apply_inactivity_reset(uid, history)
            if was_reset:
                save_uid_history(uid, history)

            return history
        except (FileNotFoundError, json.JSONDecodeError) as e:
            bt.logging.warning(f"Failed to load history for UID {uid}: {e}")

    return defaults

def save_uid_history(uid: int, history: dict):
    ensure_avgs_directory()
    file_path = AVGS_DIR / f"uid_{uid}.json"
    temp_path = file_path.with_suffix(".tmp")
    history["last_updated"] = time.time()

    try:
        with open(temp_path, 'w') as f:
            json.dump(history, f, indent=2)
        temp_path.replace(file_path)
    except Exception as e:
        bt.logging.error(f"Failed to save history for UID {uid}: {e}")
        temp_path.unlink(missing_ok=True)

def update_per_type_history(uid: int, challenge_type: int, score: float, success: bool, time_sec: float):
    history = load_uid_history(uid)

    run_data = {
        "challenge_type": int(challenge_type),
        "score": float(score),
        "success": success,
        "time_sec": float(time_sec),
        "timestamp": time.time()
    }

    history["all_runs"].append(run_data)
    history["total_runs"] += 1

    if len(history["all_runs"]) > N_RUNS_HISTORY:
        history["all_runs"] = history["all_runs"][-N_RUNS_HISTORY:]

    history["normalized_score"] = calculate_normalized_score(history)
    save_uid_history(uid, history)


def _record_zero_run(uid: int, challenge_type: int) -> None:
    update_per_type_history(uid, challenge_type, 0.0, False, 0.0)
    _log_normalized_score(uid)

def calculate_normalized_score(history: dict) -> float:
    all_runs = history.get("all_runs", [])

    if not all_runs:
        return 0.0

    if not ENABLE_PER_TYPE_NORMALIZATION:
        scores = [r["score"] for r in all_runs]
        return sum(scores) / len(scores)

    type_scores = {}
    for run in all_runs:
        t = str(run.get("challenge_type", 1))
        if t not in type_scores:
            type_scores[t] = []
        type_scores[t].append(run["score"])

    normalized = 0.0
    total_weight = 0.0

    for type_id, weight in CHALLENGE_TYPE_DISTRIBUTION.items():
        type_str = str(type_id)
        if type_str in type_scores and type_scores[type_str]:
            avg = sum(type_scores[type_str]) / len(type_scores[type_str])
            normalized += weight * avg
            total_weight += weight

    return normalized / total_weight if total_weight > 0 else 0.0

def calculate_all_normalized_scores(uids: List[int]) -> Dict[int, float]:
    scores = {}
    for uid in uids:
        history = load_uid_history(uid)
        normalized_score = calculate_normalized_score(history)
        scores[uid] = normalized_score
    return scores

# ──────────────────────────────────────────────────────────────────────────
# 4.  Winner-take-all reward system
# ──────────────────────────────────────────────────────────────────────────

def compute_winner_take_all_weights(score_metrics: List[tuple]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Compute winner-take-all weights using 3-level tiebreaking system.
    
    Args:
        score_metrics: List of (uid, avg_score, victory_rate) tuples
        
    Returns:
        Tuple containing:
        - sorted_uids: UIDs ordered by performance (winner first)
        - weights: Winner gets 1.0, all others get 0.0
        - debug_info: Dictionary with allocation statistics
    """
    if not score_metrics:
        bt.logging.warning("Winner-take-all: No score metrics provided")
        return np.array([]), np.array([]), {"error": "No score metrics"}
    
    # Sort by: avg_score (desc), victory_rate (desc), uid (asc)
    sorted_metrics = sorted(score_metrics, key=lambda x: (-x[1], -x[2], x[0]))
    
    sorted_uids = np.array([m[0] for m in sorted_metrics], dtype=np.int64)
    weights = np.zeros(len(sorted_uids), dtype=np.float32)
    
    if sorted_metrics[0][1] > 0:
        weights[0] = 1.0
        winner_uid = sorted_metrics[0][0]
        winner_avg_score = sorted_metrics[0][1]
        winner_victory_rate = sorted_metrics[0][2]
    else:
        bt.logging.warning("Winner-take-all: No miners with positive average score")
    
    zero_score_count = sum(1 for m in sorted_metrics if m[1] <= 0.0)
    non_zero_count = len(sorted_metrics) - zero_score_count
    
    debug_info = {
        "n_total": len(sorted_uids),
        "n_rewarded": 1 if weights[0] > 0 else 0,
        "n_excluded": len(sorted_uids) - (1 if weights[0] > 0 else 0),
        "zero_score_miners": zero_score_count,
        "non_zero_miners": non_zero_count,
        "zero_redistribution_amount": 0.0,
        "top_tier_allocation": 1.0,
        "winner_percentage": weights[0] * 100,
        "winner_uid": sorted_uids[0] if len(sorted_uids) > 0 and weights[0] > 0 else None,
        "winner_score": sorted_metrics[0][1] if sorted_metrics and weights[0] > 0 else None,
    }
    
    return sorted_uids, weights, debug_info


# ──────────────────────────────────────────────────────────────────────────
# 4.  Winner-Take-All reward system (only system used)
# ──────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────
# 4.  Public coroutine – called by neurons/validator.py
# ──────────────────────────────────────────────────────────────────────────
async def _check_single_low_performer(self, uid: int) -> None:
    """Check if a single low performer has updated their model"""
    try:
        axon = self.metagraph.axons[uid]
        if not axon.is_serving:
            return

        synapse = PolicySynapse()
        resp = await send_with_fresh_uuid(
            wallet=self.wallet,
            synapse=synapse,
            axon=axon,
            timeout=QUERY_REF_TIMEOUT,
        )

        if resp and len(resp) > 0 and resp[0].ref:
            ref_data = resp[0].ref
            sha256_hash = None
            
            if isinstance(ref_data, dict):
                sha256_hash = ref_data.get('sha256')
            else:
                sha256_hash = getattr(ref_data, 'sha256', None)
            
            if sha256_hash:
                updated = check_and_update_model_hash(uid, sha256_hash)
                if updated:
                    bt.logging.info(f"Low performer UID {uid} submitted new model, grace period granted")
    except Exception as e:
        bt.logging.debug(f"Failed to check model update for low performer UID {uid}: {e}")


async def _check_low_performer_model_updates(self) -> None:
    """Check if low performer miners have updated their models.

    Queries low performers in parallel batches with connection limiting.
    If hash differs from stored, clears low performer status and grants grace period.
    """
    low_performer_uids = get_low_performer_uids()
    if not low_performer_uids:
        return

    bt.logging.info(f"Checking {len(low_performer_uids)} low performers for model updates")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CONNECTIONS)

    async def _limited_check(uid: int) -> None:
        async with semaphore:
            await _check_single_low_performer(self, uid)

    for batch_start in range(0, len(low_performer_uids), PARALLEL_BATCH_SIZE):
        batch = low_performer_uids[batch_start:batch_start + PARALLEL_BATCH_SIZE]

        await asyncio.gather(
            *[_limited_check(uid) for uid in batch],
            return_exceptions=True
        )

        if batch_start + PARALLEL_BATCH_SIZE < len(low_performer_uids):
            await asyncio.sleep(BATCH_DELAY_SEC)


async def forward(self) -> None:
    """Full validator tick with boosted weighting + optional burn."""
    try:
        self.forward_count = getattr(self, "forward_count", 0) + 1
        bt.logging.info(f"[Forward #{self.forward_count}] start")

        if hasattr(self, "reconcile_hotkeys_with_metagraph"):
            self.reconcile_hotkeys_with_metagraph(reason="pre-forward guard")

        if USE_SYNCHRONIZED_SEEDS:
            if not hasattr(self, 'seed_manager'):
                secret_key = os.getenv("VALIDATOR_SECRET_KEY")
                if not secret_key:
                    bt.logging.error("VALIDATOR_SECRET_KEY not set in environment")
                    raise ValueError("VALIDATOR_SECRET_KEY required for synchronized seeds")
                self.seed_manager = SynchronizedSeedManager(secret_key, SEED_WINDOW_MINUTES)

        await _check_low_performer_model_updates(self)

        uids = get_random_uids(self, k=SAMPLE_K)
        bt.logging.info(f"Sampled miners: {uids}")
        sampled_uids = [int(uid) for uid in np.asarray(uids, dtype=np.int64).tolist()]

        model_paths = await _ensure_models(self, uids)
        bt.logging.info(f"Verified models: {list(model_paths)}")

        if USE_SYNCHRONIZED_SEEDS:
            seed, window_start, window_end = self.seed_manager.generate_seed()
            task = random_task(sim_dt=SIM_DT, seed=seed)
        else:
            task = random_task(sim_dt=SIM_DT)

        if not model_paths:
            bt.logging.warning("No models available this cycle")
            history = load_victory_history()

            # Penalize sampled non-responders even if they never reached evaluation.
            for uid in sampled_uids:
                _record_zero_run(uid, task.challenge_type)
                update_victory_history(history, uid, False, 0.0)
            save_victory_history(history)

            if BURN_EMISSIONS:
                all_uids = np.array(list(range(self.metagraph.n)), dtype=np.int64)
                score_metrics = calculate_score_metrics(history, all_uids)
                
                if score_metrics:
                    sorted_metrics = sorted(score_metrics, key=lambda x: (-x[1], -x[2], x[0]))
                    winner_uid = sorted_metrics[0][0]
                    winner_avg_score = sorted_metrics[0][1]
                    
                    uids_np = np.array([UID_ZERO, winner_uid], dtype=np.int64)
                    boosted = np.array([BURN_FRACTION, KEEP_FRACTION], dtype=np.float32)
                    
                    bt.logging.info(
                        f"No models: {BURN_FRACTION:.0%} to UID 0, "
                        f"{KEEP_FRACTION:.0%} to top miner UID {winner_uid} (avg_score: {winner_avg_score:.4f})"
                    )
                else:
                    uids_np = np.array([UID_ZERO], dtype=np.int64)
                    boosted = np.array([1.0], dtype=np.float32)
                    bt.logging.info("No models and no history: 100% to UID 0")
                
                self.update_scores(boosted, uids_np)
                if hasattr(self, 'wandb_helper') and self.wandb_helper:
                    try:
                        self.wandb_helper.log_weight_update(
                            uids=[int(uid) for uid in uids_np],
                            scores=[float(score) for score in boosted]
                        )
                    except Exception:
                        pass
            if USE_SYNCHRONIZED_SEEDS and hasattr(self, 'seed_manager'):
                self.seed_manager.wait_for_next_window()
            else:
                await asyncio.sleep(FORWARD_SLEEP_SEC)
            return

        start_pos = np.array(task.start)
        goal_pos = np.array(task.goal)
        distance = np.linalg.norm(goal_pos - start_pos)

        challenge_type_names = {
            1: "Type 1 (City navigation)",
            2: "Type 2 (Higher obstacles)",
            3: "Type 3 (Easy)",
            4: "Type 4 (No obstacles)",
            5: "Type 5 (Moving platform)",
        }
        type_name = challenge_type_names.get(task.challenge_type, f"Type {task.challenge_type}")

        bt.logging.info(f"Seed: {task.map_seed}, Distance: {distance:.2f}m, Challenge: {type_name}")

        print(f"🚀 DEBUG: Starting Docker evaluation for {len(model_paths)} models")

        # Use pre-initialized Docker evaluator
        history = load_victory_history()
        evaluated_uids = set()

        if not hasattr(self, 'docker_evaluator') or not DockerSecureEvaluator._base_ready:
            bt.logging.error("Docker evaluator not ready - falling back to no evaluation")
            results = [ValidationResult(uid, False, 0.0, 0.0) for uid in model_paths.keys()]
            for result in results:
                evaluated_uids.add(int(result.uid))
                update_per_type_history(
                    result.uid,
                    task.challenge_type,
                    result.score,
                    result.success,
                    result.time_sec,
                )
                _log_normalized_score(result.uid)
        else:
            
            # Evaluate models sequentially in Docker containers
            results = []
            fake_models_detected = []
            
            for uid, fp in model_paths.items():
                print(f"🔄 DEBUG: Evaluating UID {uid}...")
                try:
                    result = await self.docker_evaluator.evaluate_model(task, uid, fp)
                    
                    # Check if fake model was detected
                    if self.docker_evaluator.last_fake_model_info and self.docker_evaluator.last_fake_model_info['uid'] == uid:
                        # Get model hash for blacklisting
                        model_hash = sha256sum(fp)
                        fake_models_detected.append({
                            'uid': uid,
                            'hash': model_hash,
                            'reason': self.docker_evaluator.last_fake_model_info['reason'],
                            'inspection_results': self.docker_evaluator.last_fake_model_info['inspection_results']
                        })
                        
                        # Save fake model for analysis
                        try:
                            save_fake_model_for_analysis(
                                fp, uid, model_hash,
                                self.docker_evaluator.last_fake_model_info['reason'],
                                self.docker_evaluator.last_fake_model_info['inspection_results']
                            )
                        except Exception as e:
                            bt.logging.warning(f"Failed to save fake model for analysis: {e}")

                except Exception as e:
                    bt.logging.warning(f"Docker evaluation failed for UID {uid}: {e}")
                    result = ValidationResult(uid, False, 0.0, 0.0)

                results.append(result)
                evaluated_uids.add(int(uid))
                update_per_type_history(uid, task.challenge_type, result.score, result.success, result.time_sec)
                _log_normalized_score(uid)
            
            # Add detected fake models to blacklist
            if fake_models_detected:
                blacklist = load_blacklist()
                for fake_model in fake_models_detected:
                    bt.logging.info(f"🚫 Adding fake model to blacklist: UID {fake_model['uid']}, hash {fake_model['hash'][:16]}...")
                    blacklist.add(fake_model['hash'])
                save_blacklist(blacklist)
            
            # Cleanup orphaned containers
            self.docker_evaluator.cleanup()

        missing_sampled_uids = [
            uid for uid in sampled_uids if uid not in evaluated_uids
        ]
        if missing_sampled_uids:
            bt.logging.info(
                f"Applying explicit zero-score penalties to {len(missing_sampled_uids)} sampled non-responders"
            )
            for uid in missing_sampled_uids:
                _record_zero_run(uid, task.challenge_type)

        print(f"✅ DEBUG: Docker evaluation completed, got {len(results)} results")
        if not results:
            bt.logging.warning("No valid results this round.")
            for uid in sampled_uids:
                update_victory_history(history, uid, False, 0.0)
            save_victory_history(history)
            if BURN_EMISSIONS:
                uids_np = np.array([UID_ZERO], dtype=np.int64)
                boosted = np.array([1.0], dtype=np.float32)
                self.update_scores(boosted, uids_np)
                if hasattr(self, 'wandb_helper') and self.wandb_helper:
                    try:
                        self.wandb_helper.log_weight_update(
                            uids=[int(UID_ZERO)],
                            scores=[1.0]
                        )
                    except Exception:
                        pass
            if hasattr(self, 'wandb_helper') and self.wandb_helper:
                try:
                    self.wandb_helper.log_forward_results(
                        forward_count=self.forward_count,
                        task=task,
                        results=[],
                        timestamp=time.time()
                    )
                except Exception as e:
                    bt.logging.debug(f"Wandb empty forward logging failed: {e}")
            if USE_SYNCHRONIZED_SEEDS and hasattr(self, 'seed_manager'):
                self.seed_manager.wait_for_next_window()
            else:
                await asyncio.sleep(FORWARD_SLEEP_SEC)
            return

        raw_scores = np.asarray([r.score for r in results], dtype=np.float32)
        uids_np    = np.asarray([r.uid   for r in results], dtype=np.int64)

        # ------------------------------------------------------------------
        # 4. performance history tracking and reward weight allocation

        max_score = float(raw_scores.max()) if len(raw_scores) > 0 else 0.0
        round_scores = {uid: 0.0 for uid in sampled_uids}
        for i, uid in enumerate(uids_np):
            round_scores[int(uid)] = float(raw_scores[i])

        if round_scores:
            max_score = max(round_scores.values())
            for uid, score in round_scores.items():
                won = score > 0.0 and score == max_score
                update_victory_history(history, uid, won, score)
            save_victory_history(history)

        normalized_scores_dict = calculate_all_normalized_scores(uids_np.tolist())

        if WINNER_TAKE_ALL:
            if ENABLE_PER_TYPE_NORMALIZATION and normalized_scores_dict:
                eligible_scores = {}
                ineligible_uids = []

                for uid, score in normalized_scores_dict.items():
                    uid_history = load_uid_history(uid)
                    is_eligible, reason = _uid_weight_eligibility(uid_history)
                    if is_eligible:
                        eligible_scores[uid] = score
                    else:
                        ineligible_uids.append(uid)
                        bt.logging.info(f"UID {uid} ineligible: {reason}")

                if eligible_scores:
                    sorted_items = sorted(eligible_scores.items(), key=lambda x: (-x[1], x[0]))
                    uids_out = np.array([uid for uid, _ in sorted_items], dtype=np.int64)

                    if sorted_items[0][1] > 0:
                        boosted = np.zeros(len(sorted_items), dtype=np.float32)
                        boosted[0] = 1.0
                    else:
                        boosted = np.zeros(len(sorted_items), dtype=np.float32)

                    if ineligible_uids:
                        uids_out = np.concatenate([uids_out, np.array(ineligible_uids, dtype=np.int64)])
                        boosted = np.concatenate([boosted, np.zeros(len(ineligible_uids), dtype=np.float32)])

                    debug_info = {
                        "winner_uid": sorted_items[0][0] if sorted_items[0][1] > 0 else None,
                        "winner_score": sorted_items[0][1] if sorted_items else 0.0,
                    }
                else:
                    bt.logging.warning(
                        f"No miners meet eligibility requirements (min_total={MIN_RUNS_FOR_WEIGHTS}, "
                        f"min_recent={MIN_RECENT_RUNS_FOR_WEIGHTS} in {RECENT_RUN_WINDOW_SEC}s)"
                    )
                    uids_out = np.array(list(normalized_scores_dict.keys()), dtype=np.int64)
                    boosted = np.zeros(len(uids_out), dtype=np.float32)
                    debug_info = {
                        "winner_uid": None,
                        "winner_score": 0.0,
                    }
            else:
                score_metrics = calculate_score_metrics(history, uids_np)

                if score_metrics:
                    eligible_metrics = []
                    ineligible_uids = []

                    for uid, avg_score, victory_rate in score_metrics:
                        uid_history = load_uid_history(uid)
                        is_eligible, reason = _uid_weight_eligibility(uid_history)
                        if is_eligible:
                            eligible_metrics.append((uid, avg_score, victory_rate))
                        else:
                            ineligible_uids.append(uid)
                            bt.logging.info(f"UID {uid} ineligible: {reason}")

                    if eligible_metrics:
                        uids_out, boosted, debug_info = compute_winner_take_all_weights(eligible_metrics)

                        if ineligible_uids:
                            uids_out = np.concatenate([uids_out, np.array(ineligible_uids, dtype=np.int64)])
                            boosted = np.concatenate([boosted, np.zeros(len(ineligible_uids), dtype=np.float32)])
                    else:
                        bt.logging.warning(
                            f"No miners meet eligibility requirements (min_total={MIN_RUNS_FOR_WEIGHTS}, "
                            f"min_recent={MIN_RECENT_RUNS_FOR_WEIGHTS} in {RECENT_RUN_WINDOW_SEC}s)"
                        )
                        all_uids = [uid for uid, _, _ in score_metrics]
                        uids_out = np.array(all_uids, dtype=np.int64)
                        boosted = np.zeros(len(uids_out), dtype=np.float32)
                        debug_info = {
                            "winner_uid": None,
                            "winner_score": 0.0,
                        }
                else:
                    eligible_metrics = []
                    ineligible_uids = []

                    for i, uid in enumerate(uids_np):
                        if raw_scores[i] > 0:
                            uid_history = load_uid_history(uid)
                            is_eligible, reason = _uid_weight_eligibility(uid_history)
                            if is_eligible:
                                eligible_metrics.append((uid, raw_scores[i], 1.0 if raw_scores[i] == max_score else 0.0))
                            else:
                                ineligible_uids.append(uid)
                                bt.logging.info(f"UID {uid} ineligible: {reason}")

                    if eligible_metrics:
                        uids_out, boosted, debug_info = compute_winner_take_all_weights(eligible_metrics)

                        if ineligible_uids:
                            uids_out = np.concatenate([uids_out, np.array(ineligible_uids, dtype=np.int64)])
                            boosted = np.concatenate([boosted, np.zeros(len(ineligible_uids), dtype=np.float32)])
                    else:
                        bt.logging.warning(
                            f"No miners meet eligibility requirements (min_total={MIN_RUNS_FOR_WEIGHTS}, "
                            f"min_recent={MIN_RECENT_RUNS_FOR_WEIGHTS} in {RECENT_RUN_WINDOW_SEC}s)"
                        )
                        uids_out = np.array([uid for i, uid in enumerate(uids_np) if raw_scores[i] > 0], dtype=np.int64)
                        boosted = np.zeros(len(uids_out), dtype=np.float32)
                        debug_info = {
                            "winner_uid": None,
                            "winner_score": 0.0,
                        }

        uid_to_score = dict(zip(uids_np, raw_scores))
        uids_np = uids_out

        winner_uid = debug_info.get('winner_uid')
        if winner_uid is not None:
            winner_normalized = normalized_scores_dict.get(winner_uid, 0.0)
            current_raw = uid_to_score.get(winner_uid, 0.0)

            bt.logging.info(f"🏆 ROUND {self.forward_count}: Winner UID {winner_uid} | Normalized: {winner_normalized:.4f}, Current Raw: {current_raw:.4f}")

            top_5 = sorted(normalized_scores_dict.items(), key=lambda x: (-x[1], x[0]))[:5]
            top_5_str = ", ".join([f"UID {uid} ({score:.4f})" for uid, score in top_5])
            bt.logging.info(f"📊 TOP 5 (Normalized): {top_5_str}")
        else:
            bt.logging.info(f"ROUND {self.forward_count}: No winner (all scores 0.0)")

        # ------------------------------------------------------------------
        # 5. (NEW) optional burn logic
        if BURN_EMISSIONS:
            # ensure UID 0 is present once
            if UID_ZERO in uids_np:
                # remove it from the evaluation list – we’ll set it manually
                mask      = uids_np != UID_ZERO
                boosted   = boosted[mask]
                uids_np   = uids_np[mask]

            # rescale miner weights so they consume only the KEEP_FRACTION
            total_boost = boosted.sum()
            if total_boost > 0.0:
                boosted *= KEEP_FRACTION / total_boost
            else:
                # edge‑case: nobody returned a score > 0
                boosted = np.zeros_like(boosted)

            # prepend UID 0 with the burn weight
            uids_np   = np.concatenate(([UID_ZERO], uids_np))
            boosted   = np.concatenate(([BURN_FRACTION], boosted))
            non_zero_miners = np.count_nonzero(boosted[1:])

            bt.logging.info(
                f"Burn enabled → {BURN_FRACTION:.0%} to UID 0, "
                f"{KEEP_FRACTION:.0%} to {non_zero_miners} winner{'s' if non_zero_miners != 1 else ''}"
            )
        else:
            # burn disabled – weights are raw boosted scores
            bt.logging.info("Burn disabled – using boosted weights as is.")

        # ------------------------------------------------------------------
        # 6. log results to wandb before updating scores
        if hasattr(self, 'wandb_helper') and self.wandb_helper:
            try:
                self.wandb_helper.log_forward_results(
                    forward_count=self.forward_count,
                    task=task,
                    results=results,
                    timestamp=time.time()
                )
            except Exception as e:
                bt.logging.debug(f"Wandb forward logging failed: {e}")

        print(f"🎯 DEBUG: Setting weights - UIDs: {uids_np}, Scores: {boosted}")
        self.update_scores(boosted, uids_np)
        if hasattr(self, 'wandb_helper') and self.wandb_helper:
            try:
                self.wandb_helper.log_weight_update(
                    uids=[int(uid) for uid in uids_np],
                    scores=[float(score) for score in boosted]
                )
            except Exception as e:
                bt.logging.debug(f"Wandb weight logging failed: {e}")

        print(f"✅ DEBUG: Weights updated successfully! Forward cycle complete.")

    except Exception as e:
        bt.logging.error(f"Validator forward error: {e}")
        # Log error to wandb
        if hasattr(self, 'wandb_helper') and self.wandb_helper:
            try:
                self.wandb_helper.log_error(
                    error_message=str(e),
                    error_type="forward_error"
                )
            except Exception:
                pass

    # ----------------------------------------------------------------------
    # 7. pace the main loop
    if USE_SYNCHRONIZED_SEEDS and hasattr(self, 'seed_manager'):
        if self.seed_manager.should_wait():
            self.seed_manager.wait_for_next_window()
        else:
            await asyncio.sleep(FORWARD_SLEEP_SEC)
    else:
        await asyncio.sleep(FORWARD_SLEEP_SEC)
