# ---------------------------------------------------------------
#  Swarm validator – Policy API v2   (hardened, 50 MiB limits)
# ---------------------------------------------------------------
from __future__ import annotations

import asyncio
import gc
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List

import bittensor as bt
import numpy as np
from zipfile import ZipFile, BadZipFile

from swarm.protocol import PolicySynapse, PolicyRef, ValidationResult
from swarm.utils.uids import get_random_uids
from swarm.utils.hash import sha256sum
import base64

from ..core.Model_verify import (
    load_blacklist,
    save_blacklist, 
    add_to_blacklist,
    save_fake_model_for_analysis,
)
from .task_gen import random_task
from .docker.docker_evaluator import DockerSecureEvaluator  # For _base_ready check
from swarm.constants import (
    SIM_DT,
    HORIZON_SEC,
    SAMPLE_K,
    QUERY_TIMEOUT,
    FORWARD_SLEEP_SEC,
    BURN_EMISSIONS,
    MAX_MODEL_BYTES,
    BURN_FRACTION,
    KEEP_FRACTION,
    UID_ZERO,
    MODEL_DIR,
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
            timeout=QUERY_TIMEOUT,
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
    from .docker.docker_evaluator import DockerSecureEvaluator
    
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
            import os
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
        axon = self.metagraph.axons[uid]

        # 1 – ask for current PolicyRef
        try:
            responses = await send_with_fresh_uuid(
                wallet=self.wallet,
                synapse=PolicySynapse.request_ref(),
                axon=axon,
                timeout=QUERY_TIMEOUT,
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

        # 2 – compare with cache
        model_fp = MODEL_DIR / f"UID_{uid}.zip"
        up_to_date = model_fp.exists() and sha256sum(model_fp) == ref.sha256
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

        # 3 – request payload
        await _download_model(self, axon, ref, model_fp, uid)
        if (
            model_fp.exists()
            and model_fp.stat().st_size <= MAX_MODEL_BYTES
            and _zip_is_safe(model_fp, max_uncompressed=MAX_MODEL_BYTES)
        ):
            paths[uid] = model_fp
        else:
            bt.logging.warning(f"Failed to obtain valid model for miner {uid}.")
            model_fp.unlink(missing_ok=True)

    return paths




# ──────────────────────────────────────────────────────────────────────────
# 3.  Weight boosting
# ──────────────────────────────────────────────────────────────────────────
def _boost_scores(raw: np.ndarray, *, beta: float = 5.0) -> np.ndarray:
    """
    Exponential boost driven by absolute gap to the best score,
    scaled by batch standard deviation.
    """
    if raw.size == 0:
        return raw

    s_max = float(raw.max())
    sigma = float(raw.std())
    if sigma < 1e-9:                          # all miners identical
        weights = (raw == s_max).astype(np.float32)
    else:
        weights = np.exp(beta * (raw - s_max) / sigma)
        weights /= weights.max()              # normalise so best → 1

    return weights.astype(np.float32)


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
        bt.logging.info(f"Cycle seed: {task.map_seed}")

        # ------------------------------------------------------------------
        # 2. sample miners & secure their models
        uids = get_random_uids(self, k=SAMPLE_K)
        bt.logging.info(f"Sampled miners: {uids}")

        model_paths = await _ensure_models(self, uids)
        bt.logging.info(f"Verified models: {list(model_paths)}")
        print(f"🔍 DEBUG: Verified models: {list(model_paths.keys())}")

        # ------------------------------------------------------------------
        # 3. Docker-based secure evaluation (sequential)
        print(f"🚀 DEBUG: Starting Docker evaluation for {len(model_paths)} models")
        
        # Use pre-initialized Docker evaluator
        if not hasattr(self, 'docker_evaluator') or not DockerSecureEvaluator._base_ready:
            bt.logging.error("Docker evaluator not ready - falling back to no evaluation")
            results = [ValidationResult(uid, False, 0.0, 0.0, 0.0) for uid in model_paths.keys()]
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
                        from swarm.utils.hash import sha256sum
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
                except Exception as e:
                    bt.logging.warning(f"Docker evaluation failed for UID {uid}: {e}")
                    results.append(ValidationResult(uid, False, 0.0, 0.0, 0.0))
            
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
        
        print(f"📊 DEBUG: Raw scores: {raw_scores}, UIDs: {uids_np}")  # Temporary debug

        # ------------------------------------------------------------------
        # 4. adaptive boost
        boosted = _boost_scores(raw_scores, beta=5.0)
        print(f"⚡ DEBUG: Boosted scores: {boosted}")  # Temporary debug

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
                    uids=uids_np.tolist(),
                    scores=boosted.tolist()
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