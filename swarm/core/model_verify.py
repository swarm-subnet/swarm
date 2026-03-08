#!/usr/bin/env python3
"""
Model Verification Module for Swarm Subnet.

Blacklist management, ZIP safety inspection, Docker-based model verification,
and forensic storage of detected fake models.
"""

import asyncio
import json
import os
import subprocess
import tempfile
import time
import shutil
from pathlib import Path
from typing import Dict, Tuple, Set, Optional
from zipfile import ZipFile, BadZipFile

import bittensor as bt

from swarm.constants import MODEL_DIR, BLACKLIST_FILE, HORIZON_SEC


# ──────────────────────────────────────────────────────────────────────────
# Blacklist Management
# ──────────────────────────────────────────────────────────────────────────

def load_blacklist(file_path: Path = None) -> Set[str]:
    """Load blacklisted fake model hashes from file."""
    try:
        target_file = file_path if file_path is not None else BLACKLIST_FILE
        if target_file.exists():
            with open(target_file, 'r') as f:
                return {line.strip() for line in f if line.strip()}
        return set()
    except Exception as e:
        bt.logging.warning(f"Error loading blacklist: {e}")
        return set()


def save_blacklist(blacklist: Set[str], file_path: Path = None) -> None:
    """Save blacklisted fake model hashes to file."""
    try:
        # Ensure the directory exists
        target_file = file_path if file_path is not None else BLACKLIST_FILE
        target_file.parent.mkdir(parents=True, exist_ok=True)
        with open(target_file, 'w') as f:
            for hash_val in sorted(blacklist):
                f.write(f"{hash_val}\n")
    except Exception as e:
        bt.logging.error(f"Error saving blacklist: {e}")


def add_to_blacklist(model_hash: str, file_path: Path = None) -> None:
    """Add a single model hash to the blacklist."""
    try:
        blacklist = load_blacklist(file_path)
        blacklist.add(model_hash)
        save_blacklist(blacklist, file_path)
        bt.logging.info(f"🚫 Added {model_hash[:16]}... to blacklist")
    except Exception as e:
        bt.logging.error(f"Error adding to blacklist: {e}")


# ──────────────────────────────────────────────────────────────────────────
# Model Structure Analysis
# ──────────────────────────────────────────────────────────────────────────

def inspect_model_structure(zip_path: Path) -> Dict:
    """
    Inspect RPC agent submission structure.
    Miners submit only drone_agent.py + model files.
    Template files (main.py, agent.capnp, agent_server.py) are injected automatically.
    """
    try:
        with ZipFile(zip_path, 'r') as zf:
            file_list = zf.namelist()
            
            if "drone_agent.py" not in file_list:
                return {
                    "error": "Missing drone_agent.py - RPC agent submission required",
                    "missing_drone_agent": True
                }
            
            dangerous_files = [f for f in file_list 
                              if f.endswith(('.exe', '.so', '.dll', '.sh', '.bat'))]
            if dangerous_files:
                return {"error": f"Dangerous executable files detected: {dangerous_files}"}
            
            return {
                "submission_type": "rpc",
                "has_mlp_extractor": True,
                "suspicious_patterns": [],
                "class_names": ["RPC Custom Agent"]
            }
            
    except Exception as e:
        return {"error": f"ZIP inspection failed: {e}"}


def classify_model_validity(inspection_results: Dict) -> Tuple[str, str]:
    """
    Classify RPC agent validity:
    - "legitimate": RPC agent passes all checks
    - "missing_drone_agent": Missing drone_agent.py (reject but don't blacklist)
    - "fake": Dangerous files detected (reject and blacklist)
    
    Returns (status, reason)
    """
    if inspection_results.get("missing_drone_agent", False):
        return "missing_drone_agent", "Missing drone_agent.py - RPC agent submission required"
    
    if "malicious_findings" in inspection_results:
        return "fake", "Security violation: Malicious code detected"
    
    if "error" in inspection_results:
        if "Security violation" in inspection_results["error"]:
            return "fake", inspection_results["error"]
        if "Missing drone_agent.py" in inspection_results["error"]:
            return "missing_drone_agent", inspection_results["error"]
        if "Dangerous executable" in inspection_results["error"]:
            return "fake", inspection_results["error"]
        return "fake", f"Inspection error: {inspection_results['error']}"
    
    if inspection_results.get("submission_type") == "rpc":
        return "legitimate", "RPC submission validated"
    
    return "legitimate", "RPC agent appears legitimate"


# ──────────────────────────────────────────────────────────────────────────
# Forensic Storage
# ──────────────────────────────────────────────────────────────────────────

def save_fake_model_for_analysis(model_path: Path, uid: int, model_hash: str, reason: str, inspection_results: Dict) -> None:
    """
    Save fake model for forensic analysis. Keep max 3 fake models per UID.
    Creates: miner_models_v2/UID_X_fake_Y/
    """
    try:
        # Create base directory for this UID's fake models
        uid_fake_dir = MODEL_DIR / f"UID_{uid}_fake"
        uid_fake_dir.mkdir(parents=True, exist_ok=True)
        
        # Find existing fake models for this UID
        existing_fakes = []
        if uid_fake_dir.exists():
            existing_fakes = [d for d in uid_fake_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            existing_fakes.sort(key=lambda x: int(x.name))
        
        # Determine next fake number
        if len(existing_fakes) >= 3:
            # Remove oldest fake model (fake_1) and shift others
            for i, fake_dir in enumerate(existing_fakes):
                if i == 0:  # Remove first (oldest)
                    shutil.rmtree(fake_dir, ignore_errors=True)
                else:  # Rename others: fake_2 -> fake_1, fake_3 -> fake_2
                    new_name = fake_dir.parent / str(i)
                    fake_dir.rename(new_name)
            next_fake_num = 3
        else:
            next_fake_num = len(existing_fakes) + 1
        
        # Create directory for this fake model
        fake_model_dir = uid_fake_dir / str(next_fake_num)
        fake_model_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy the fake model
        fake_model_file = fake_model_dir / "model.zip"
        shutil.copy2(model_path, fake_model_file)
        
        # Save analysis report
        report_file = fake_model_dir / "analysis_report.json"
        report_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "uid": uid,
            "model_hash": model_hash,
            "detection_reason": reason,
            "file_size_bytes": model_path.stat().st_size,
            "inspection_results": inspection_results
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        bt.logging.info(f"📁 Saved fake model UID_{uid}_fake/{next_fake_num}/ for analysis")
        bt.logging.info(f"   Size: {model_path.stat().st_size} bytes")
        bt.logging.info(f"   Hash: {model_hash[:16]}...")
        
    except Exception as e:
        bt.logging.error(f"Failed to save fake model for analysis: {e}")


# ──────────────────────────────────────────────────────────────────────────
# ZIP Safety Inspection
# ──────────────────────────────────────────────────────────────────────────

def zip_is_safe(path: Path, *, max_uncompressed: int) -> bool:
    """Reject dangerous ZIP files without extracting them.

    Checks total uncompressed size and forbids path traversal.
    """
    try:
        with ZipFile(path) as zf:
            total_uncompressed = 0
            for info in zf.infolist():
                name = info.filename
                if name.startswith(("/", "\\")) or ".." in Path(name).parts:
                    bt.logging.error(f"ZIP path traversal attempt: {name}")
                    return False

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
# Docker-based First-Time Model Verification
# ──────────────────────────────────────────────────────────────────────────

async def verify_new_model_with_docker(
    model_path: Path, model_hash: str, miner_hotkey: str, uid: int
) -> None:
    """Run fake model detection in a Docker container for first-time verification."""
    from swarm.validator.docker.docker_evaluator import DockerSecureEvaluator

    if not model_path.is_file():
        bt.logging.warning(f"Verification skipped; model file missing: {model_path}")
        return

    bt.logging.info(
        f"🔍 Starting first-time verification for model {model_hash[:16]}... from {miner_hotkey}"
    )

    docker_evaluator = DockerSecureEvaluator()

    if not docker_evaluator._base_ready:
        bt.logging.warning(f"Docker not ready for verification of {model_hash[:16]}...")
        return

    container_name = f"swarm_verify_{model_hash[:8]}_{int(time.time() * 1000)}"

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            current_uid = os.getuid()
            current_gid = os.getgid()
            os.chown(tmpdir, current_uid, current_gid)
            os.chmod(tmpdir, 0o755)

            verification_result_file = Path(tmpdir) / "verification_result.json"

            dummy_task = {
                "start": [0, 0, 1], "goal": [5, 5, 2], "obstacles": [],
                "horizon": HORIZON_SEC, "seed": 12345
            }

            task_file = Path(tmpdir) / "task.json"
            with open(task_file, 'w') as f:
                json.dump(dummy_task, f)

            bt.logging.info(
                f"🐳 Starting Docker container for verification of UID model {model_hash[:16]}..."
            )

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

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=60
                )

                stdout_str = stdout.decode() if stdout else ""
                stderr_str = stderr.decode() if stderr else ""

                bt.logging.debug(f"Verification container for {model_hash[:16]}:")
                bt.logging.debug(f"  Return code: {proc.returncode}")
                bt.logging.debug(f"  STDOUT: {stdout_str}")
                bt.logging.debug(f"  STDERR: {stderr_str}")

                if proc.returncode != 0:
                    bt.logging.warning(
                        f"Verification container failed for {model_hash[:16]} "
                        f"with return code {proc.returncode}"
                    )
                    bt.logging.warning(f"Error output: {stderr_str}")

            except asyncio.TimeoutError:
                subprocess.run(["docker", "kill", container_name], capture_output=True)
                bt.logging.warning(f"⏰ Verification timeout for model {model_hash[:16]}...")
                return

            bt.logging.info(
                f"🔚 Ending Docker container for verification of model {model_hash[:16]}..."
            )

            if verification_result_file.exists():
                try:
                    with open(verification_result_file, 'r') as f:
                        verification_data = json.load(f)

                    if verification_data.get('is_fake_model', False):
                        fake_reason = verification_data.get('fake_reason', 'Unknown')
                        inspection_results = verification_data.get('inspection_results', {})

                        bt.logging.warning(
                            f"🚫 FAKE MODEL DETECTED during verification: {fake_reason}"
                        )
                        bt.logging.info(f"Model hash: {model_hash}")
                        bt.logging.debug(f"Inspection details: {inspection_results}")

                        save_fake_model_for_analysis(
                            model_path, uid, model_hash, fake_reason, inspection_results
                        )
                        add_to_blacklist(model_hash)
                        model_path.unlink(missing_ok=True)
                        bt.logging.info(
                            f"🗑️ Removed fake model {model_hash[:16]}... from cache and blacklisted"
                        )

                    elif verification_data.get('missing_drone_agent', False):
                        rejection_reason = verification_data.get(
                            'rejection_reason', 'Missing drone_agent.py'
                        )
                        bt.logging.warning(
                            f"⚠️ MISSING drone_agent.py during verification: {rejection_reason}"
                        )
                        bt.logging.info(f"Model hash: {model_hash}")
                        model_path.unlink(missing_ok=True)
                        bt.logging.info(
                            f"🗑️ Removed model {model_hash[:16]}... from cache "
                            f"(missing drone_agent.py - can resubmit)"
                        )

                    else:
                        bt.logging.info(
                            f"✅ Model {model_hash[:16]}... passed verification - legitimate model"
                        )

                except Exception as e:
                    bt.logging.warning(
                        f"Failed to parse verification results for {model_hash[:16]}: {e}"
                    )
            else:
                bt.logging.warning(
                    f"No verification results found for model {model_hash[:16]}..."
                )
                try:
                    temp_files = list(Path(tmpdir).glob("*"))
                    bt.logging.debug(
                        f"Files in temp directory: {[f.name for f in temp_files]}"
                    )
                    expected_file = Path(tmpdir) / "verification_result.json"
                    bt.logging.debug(f"Expected result file: {expected_file}")
                    bt.logging.debug(f"Expected file exists: {expected_file.exists()}")
                except Exception as e:
                    bt.logging.debug(f"Error checking temp directory: {e}")

    except Exception as e:
        bt.logging.warning(f"Docker verification failed for model {model_hash[:16]}: {e}")
        subprocess.run(["docker", "kill", container_name], capture_output=True)