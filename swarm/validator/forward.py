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
from typing import Dict, List, Tuple, Any

import bittensor as bt
import numpy as np
from zipfile import ZipFile, BadZipFile

from swarm.protocol import PolicySynapse, PolicyRef, ValidationResult
from swarm.utils.uids import get_random_uids
from swarm.utils.hash import sha256sum
import base64

from ..core.model_verify import (
    load_blacklist,
    save_blacklist, 
    add_to_blacklist,
    save_fake_model_for_analysis,
)
from .task_gen import random_task
from .docker.docker_evaluator import DockerSecureEvaluator  # For _base_ready check
from .rewards_system import compute_tiered_weights
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
    VERIFICATION_ENFORCEMENT_ENABLED,
    VERIFICATION_TOP_N,
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
            # Set ownership and permissions for container user (UID 1000)
            os.chown(tmpdir, 1000, 1000)
            os.chmod(tmpdir, 0o755)
            
            verification_result_file = Path(tmpdir) / "verification_result.json"
            
            # Create minimal task for verification (not used for actual evaluation)
            dummy_task = {
                "start": [0, 0, 1], "goal": [5, 5, 2], "obstacles": [],
                "horizon": 30.0, "seed": 12345
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
                "--user", "1000:1000",
                "--memory=4g",  # Less memory needed for verification
                "--cpus=1",     # Single CPU for verification
                "--pids-limit=10",
                "--ulimit", "nofile=32:32",
                "--ulimit", "fsize=262144000:262144000",  # 250MB file size limit
                "--security-opt", "no-new-privileges",
                "--network", "none",
                "-v", f"{tmpdir}:/workspace/shared",
                "-v", f"{model_path.absolute()}:/workspace/model.zip:ro",
                docker_evaluator.base_image,
                # Use special verification mode
                "VERIFY_ONLY",  # Special flag to run only verification
                str(uid),  # Real UID for verification
                "/workspace/model.zip",  # Model path
                "/workspace/shared/verification_result.json"  # Result file
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
                        
                    elif verification_data.get('missing_metadata', False):
                        # Missing metadata → reject but don't blacklist
                        rejection_reason = verification_data.get('rejection_reason', 'Missing secure metadata')
                        
                        bt.logging.warning(f"⚠️ MISSING METADATA during verification: {rejection_reason}")
                        bt.logging.info(f"Model hash: {model_hash}")
                        
                        # Remove model but don't blacklist (allows resubmission)
                        model_path.unlink(missing_ok=True)
                        bt.logging.info(f"🗑️ Removed model {model_hash[:16]}... from cache (missing metadata - can resubmit)")
                        
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
    
    async with bt.dendrite(wallet=wallet) as dend:
        responses = await dend(
            axons=[axon],
            synapse=synapse,
            deserialize=deserialize,
            timeout=timeout,
        )

    bt.logging.warning(
        f"➡️  sending: nonce={synapse.dendrite.nonce} "
        f"timeout={synapse.timeout} uuid={synapse.dendrite.uuid}"
        f"comcomputed_body_hash={synapse.computed_body_hash}"
        f"axon={axon}"
        f"dendrite"
    )
    return responses

async def _ensure_models(self, uids: List[int]) -> Dict[int, Path]:
    """
    For every UID return the local Path to its latest .zip.
    Downloads if the cached SHA differs from the miner's PolicyRef.
    """
    MODEL_DIR.mkdir(exist_ok=True)
    paths: Dict[int, Path] = {}

    for uid in uids:
        try:
            axon = self.metagraph.axons[uid]

            # 1 – ask for current PolicyRef
            try:
                responses = await send_with_fresh_uuid(
                    wallet=self.wallet,
                    synapse=PolicySynapse.request_ref(),
                    axon=axon,
                    timeout=QUERY_REF_TIMEOUT,
                    )

                if not responses:
                    bt.logging.warning(f"Miner {uid} returned no response.")
                    continue
                print(f"Miner {uid} returned {len(responses)} responses {responses}")

                syn = responses[0]              # <- get the first PolicySynapse

                if not syn.ref:
                    bt.logging.warning(f"Miner {uid} returned no PolicyRef.")
                    continue

                ref = PolicyRef(**syn.ref)
            except Exception as e:
                bt.logging.warning(f"Handshake with miner {uid} failed: {e}")
                continue

            # 2 – FIRST CHECK: Is this hash blacklisted?
            blacklist = load_blacklist()
            if ref.sha256 in blacklist:
                bt.logging.warning(f"Skipping blacklisted fake model {ref.sha256[:16]}... from miner {uid}")
                continue

            # 3 – compare with cache (directory-safe)
            model_fp = MODEL_DIR / f"UID_{uid}.zip"
            if model_fp.exists() and model_fp.is_dir():
                bt.logging.warning(f"Cache path is a directory (fixing): {model_fp}")
                shutil.rmtree(model_fp)

            up_to_date = False
            if model_fp.is_file():
                try:
                    up_to_date = sha256sum(model_fp) == ref.sha256
                except Exception as e:
                    bt.logging.warning(f"Hash check failed for {model_fp}: {e}")
                    up_to_date = False

            if up_to_date:
                # confirm cached file is still within limits
                if (
                    model_fp.stat().st_size <= MAX_MODEL_BYTES
                    and _zip_is_safe(model_fp, max_uncompressed=MAX_MODEL_BYTES)
                ):
                    paths[uid] = model_fp
                    continue
                else:
                    bt.logging.warning(f"Cached model for {uid} violates limits; redownloading.")
                    model_fp.unlink(missing_ok=True)

            # 4 – request payload
            await _download_model(self, axon, ref, model_fp, uid)
            if (
                model_fp.is_file()
                and model_fp.stat().st_size <= MAX_MODEL_BYTES
                and _zip_is_safe(model_fp, max_uncompressed=MAX_MODEL_BYTES)
            ):
                paths[uid] = model_fp
            else:
                bt.logging.warning(f"Failed to obtain valid model for miner {uid}.")
                model_fp.unlink(missing_ok=True)

        except Exception as e:
            bt.logging.warning(f"UID {uid}: model preparation failed: {e}")
            continue

    return paths




# ──────────────────────────────────────────────────────────────────────────
# 3.  Performance history tracking system
# ──────────────────────────────────────────────────────────────────────────

def _log_uid_performance(uid: int, current_score: float, history: dict) -> None:
    """Log simple UID performance summary after evaluation"""
    uid_str = str(uid)
    
    if uid_str in history and history[uid_str]["runs"]:
        runs = history[uid_str]["runs"]
        # Include current score and enforce 10-run rolling window
        all_scores = [run["score"] for run in runs] + [current_score]
        # Keep only last 10 runs (same as N_RUNS_HISTORY)
        if len(all_scores) > 10:
            all_scores = all_scores[-10:]
        
        avg_score = sum(all_scores) / len(all_scores)
        total_runs = len(all_scores)
        
        bt.logging.info(f"📊 UID {uid:3d} | avg: {avg_score:.4f} ({total_runs} runs)")
    else:
        bt.logging.info(f"📊 UID {uid:3d} | current: {current_score:.4f} (first evaluation)")

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
# 4.  Tiered reward system (see rewards_system.py)
# ──────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────
# 4.  Public coroutine – called by neurons/validator.py
# ──────────────────────────────────────────────────────────────────────────
async def forward(self) -> None:
    """Full validator tick with boosted weighting + optional burn."""
    try:
        self.forward_count = getattr(self, "forward_count", 0) + 1
        bt.logging.info(f"[Forward #{self.forward_count}] start")

        # ------------------------------------------------------------------
        # 1. build a secret task
        task = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC)
        
        # ------------------------------------------------------------------
        # 2. sample miners & secure their models
        uids = get_random_uids(self, k=SAMPLE_K)
        bt.logging.info(f"Sampled miners: {uids}")

        model_paths = await _ensure_models(self, uids)
        bt.logging.info(f"Verified models: {list(model_paths)}")
        print(f"🔍 DEBUG: Verified models: {list(model_paths.keys())}")
        
        # Calculate distance from start to goal  
        start_pos = np.array(task.start)
        goal_pos = np.array(task.goal)
        distance = np.linalg.norm(goal_pos - start_pos)
        
        # Map challenge type to name
        challenge_type_names = {
            1: "Type 1 (Standard)",
            2: "Type 2 (higher obstacles)",
            3: "Type 3 (Easy)"
        }
        type_name = challenge_type_names.get(task.challenge_type, f"Type {task.challenge_type}")
        
        bt.logging.info(f"Cycle seed: {task.map_seed}, Distance: {distance:.2f}m, Challenge: {type_name}")

        # ------------------------------------------------------------------
        # 3. Docker-based secure evaluation (sequential)
        print(f"🚀 DEBUG: Starting Docker evaluation for {len(model_paths)} models")
        
        # Use pre-initialized Docker evaluator
        if not hasattr(self, 'docker_evaluator') or not DockerSecureEvaluator._base_ready:
            bt.logging.error("Docker evaluator not ready - falling back to no evaluation")
            results = [ValidationResult(uid, False, 0.0, 0.0) for uid in model_paths.keys()]
        else:
            # Load history for UID performance tracking
            history = load_victory_history()
            
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
                    
                    results.append(result)
                    
                    # Log UID historical performance summary
                    _log_uid_performance(uid, result.score, history)
                    
                except Exception as e:
                    bt.logging.warning(f"Docker evaluation failed for UID {uid}: {e}")
                    results.append(ValidationResult(uid, False, 0.0, 0.0))
            
            # Add detected fake models to blacklist
            if fake_models_detected:
                blacklist = load_blacklist()
                for fake_model in fake_models_detected:
                    bt.logging.info(f"🚫 Adding fake model to blacklist: UID {fake_model['uid']}, hash {fake_model['hash'][:16]}...")
                    blacklist.add(fake_model['hash'])
                save_blacklist(blacklist)
            
            # Cleanup orphaned containers
            self.docker_evaluator.cleanup()
        
        print(f"✅ DEBUG: Docker evaluation completed, got {len(results)} results")
        if not results:
            bt.logging.warning("No valid results this round.")
            # Log empty forward to wandb
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
            await asyncio.sleep(FORWARD_SLEEP_SEC)
            return

        raw_scores = np.asarray([r.score for r in results], dtype=np.float32)
        uids_np    = np.asarray([r.uid   for r in results], dtype=np.int64)
        
        print(f"📊 DEBUG: Raw scores: {raw_scores}, UIDs: {uids_np}")

        # ------------------------------------------------------------------
        # 4. performance history tracking and reward weight allocation
        
        # Determine winners from current run
        if len(raw_scores) > 0:
            max_score = raw_scores.max()
            current_winners = uids_np[raw_scores == max_score]
            
            # Update victory history for all evaluated UIDs
            for i, uid in enumerate(uids_np):
                won = uid in current_winners
                score = raw_scores[i]
                update_victory_history(history, uid, won, score)
            
            save_victory_history(history)
        
        if WINNER_TAKE_ALL:
            # Use average score-based winner-take-all system
            score_metrics = calculate_score_metrics(history, uids_np)
            if score_metrics:
                uids_out, boosted, debug_info = compute_winner_take_all_weights(score_metrics)
            else:
                # Fallback to single-run if no history
                uids_out, boosted, debug_info = compute_winner_take_all_weights([(uid, raw_scores[i], 1.0 if raw_scores[i] == max_score else 0.0) for i, uid in enumerate(uids_np) if raw_scores[i] > 0])
            reward_system = "Winner-Take-All (Avg Score-Based)"

            # Training verification enforcement
            if VERIFICATION_ENFORCEMENT_ENABLED and hasattr(self, 'training_verifier'):
                winner_uid = debug_info.get('winner_uid')
                if winner_uid is not None:
                    sorted_metrics = sorted(score_metrics, key=lambda x: (-x[1], -x[2], x[0])) if score_metrics else []
                    top_n_uids = [m[0] for m in sorted_metrics[:VERIFICATION_TOP_N]]

                    if winner_uid in top_n_uids:
                        is_verified = self.training_verifier.is_miner_verified(winner_uid)
                        if not is_verified:
                            winner_idx = np.where(uids_out == winner_uid)[0]
                            if len(winner_idx) > 0:
                                boosted[winner_idx[0]] = 0.0
                                bt.logging.warning(f"⚠️ ENFORCEMENT: Winner UID {winner_uid} NOT VERIFIED - setting weight to 0")
                                debug_info['enforcement_applied'] = True
                                debug_info['winner_unverified'] = True
        else:
            # Use tiered reward system: balanced distribution across top performers
            uids_out, boosted, debug_info = compute_tiered_weights(uids_np, raw_scores)
            reward_system = "Tiered"
        
        # Create UID to current score mapping BEFORE reordering
        uid_to_score = dict(zip(uids_np, raw_scores))
        
        uids_np = uids_out  # use reordered UIDs from reward system
        
        # Professional round summary
        winner_uid = debug_info.get('winner_uid')
        if winner_uid is not None:
            winner_avg_score = debug_info.get('winner_score', 0.0)
            current_score = uid_to_score.get(winner_uid, 0.0)
            
            bt.logging.info(f"ROUND {self.forward_count}: Winner UID {winner_uid} (score: {current_score:.4f}, avg: {winner_avg_score:.4f})")
            
            # Top 5 performers by average
            if score_metrics:
                sorted_metrics = sorted(score_metrics, key=lambda x: (-x[1], -x[2], x[0]))
                top_5 = sorted_metrics[:5]
                top_5_str = ", ".join([f"UID {uid} ({avg:.4f})" for uid, avg, _ in top_5])
                bt.logging.info(f"TOP 5: {top_5_str}")
        else:
            bt.logging.info(f"ROUND {self.forward_count}: No winner (all miners scored 0.0)")

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

            bt.logging.info(
                f"Burn enabled → {BURN_FRACTION:.0%} to UID 0, "
                f"{KEEP_FRACTION:.0%} distributed over {len(boosted)-1} miners."
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

        # ------------------------------------------------------------------
        # 7. push weights on‑chain (store locally then call set_weights later)
        print(f"🎯 DEBUG: Setting weights - UIDs: {uids_np}, Scores: {boosted}")  # Temporary debug
        self.update_scores(boosted, uids_np)
        
        # ------------------------------------------------------------------
        # 8. log weight updates to wandb
        if hasattr(self, 'wandb_helper') and self.wandb_helper:
            try:
                self.wandb_helper.log_weight_update(
                    uids=[int(uid) for uid in uids_np],
                    scores=[float(score) for score in boosted]
                )
            except Exception as e:
                bt.logging.debug(f"Wandb weight logging failed: {e}")
                
        print(f"✅ DEBUG: Weights updated successfully! Forward cycle complete.")  # Temporary debug

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
    await asyncio.sleep(FORWARD_SLEEP_SEC)