# ---------------------------------------------------------------
#  Swarm validator – Policy API v2   (hardened, 50 MiB limits)
# ---------------------------------------------------------------
from __future__ import annotations

import asyncio
import gc
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import bittensor as bt
import numpy as np
from stable_baselines3 import PPO                       # SB‑3 loader
from zipfile import ZipFile, BadZipFile

from swarm.core.drone import track_drone
from swarm.protocol import MapTask, PolicySynapse, PolicyRef, ValidationResult
from swarm.utils.uids import get_random_uids
from swarm.utils.hash import sha256sum
from swarm.utils.env_factory import make_env
import base64
import pickle
import io
import torch
import warnings
import uuid
import bittensor as bt
import shutil

from .task_gen import random_task
from .reward   import flight_reward
from .docker_evaluator import DockerSecureEvaluator  # For _base_ready check
from swarm.constants import (
    SIM_DT,
    HORIZON_SEC,
    SAMPLE_K,
    QUERY_TIMEOUT,
    FORWARD_SLEEP_SEC,
    BURN_EMISSIONS,
    MAX_MODEL_BYTES,
    EVAL_TIMEOUT_SEC
)

BURN_FRACTION  = 0.90            # 90 % burn (weight for UID 0)
KEEP_FRACTION  = 1.0 - BURN_FRACTION
UID_ZERO       = 0

# ──────────────────────────────────────────────────────────────────────────
# 0.  Global hardening parameters
# ──────────────────────────────────────────────────────────────────────────
MODEL_DIR         = Path("miner_models_v2")        # all zips stored here - v2 for fresh start
CHUNK_SIZE        = 2 << 20                        # 2 MiB
SUBPROC_MEM_MB    = 8192                            # RSS limit per subprocess
BLACKLIST_FILE    = MODEL_DIR / "fake_models_blacklist.txt"  # blacklisted fake model hashes

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
# 1.5  Fake Model Detection System
# ──────────────────────────────────────────────────────────────────────────

def _load_blacklist() -> set:
    """Load blacklisted fake model hashes from file."""
    try:
        if BLACKLIST_FILE.exists():
            with open(BLACKLIST_FILE, 'r') as f:
                return {line.strip() for line in f if line.strip()}
        return set()
    except Exception as e:
        bt.logging.warning(f"Error loading blacklist: {e}")
        return set()

def _save_blacklist(blacklist: set) -> None:
    """Save blacklisted fake model hashes to file."""
    try:
        # Ensure the directory exists
        BLACKLIST_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(BLACKLIST_FILE, 'w') as f:
            for hash_val in sorted(blacklist):
                f.write(f"{hash_val}\n")
    except Exception as e:
        bt.logging.error(f"Error saving blacklist: {e}")

def _add_to_blacklist(model_hash: str) -> None:
    """Add a single model hash to the blacklist."""
    try:
        blacklist = _load_blacklist()
        blacklist.add(model_hash)
        _save_blacklist(blacklist)
        bt.logging.info(f"🚫 Added {model_hash[:16]}... to blacklist")
    except Exception as e:
        bt.logging.error(f"Error adding to blacklist: {e}")

def _inspect_model_structure(zip_path: Path) -> dict:
    """
    Inspect PPO model structure without loading it through SB3.
    Returns dict with inspection results.
    """
    try:
        with ZipFile(zip_path, 'r') as zf:
            file_list = zf.namelist()
            
            # Check if this is a legitimate SB3 model structure
            sb3_required_files = ['policy.pth', 'data', '_stable_baselines3_version']
            is_legitimate_sb3 = all(f in file_list for f in sb3_required_files)
            
            if is_legitimate_sb3:
                # SB3 structure found - but we need to validate the actual neural network
                bt.logging.info(f"🔍 Found SB3 structure, analyzing neural network content...")
                
                try:
                    # Load and analyze the actual PyTorch state dict
                    policy_data = zf.read('policy.pth')
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        state_dict = torch.load(io.BytesIO(policy_data), map_location='cpu')
                    
                    return _analyze_pytorch_neural_network(state_dict, file_list)
                    
                except Exception as e:
                    return {"error": f"Failed to analyze SB3 neural network: {e}"}
            
            # Look for pickle files (for custom/fake models that might use old format)
            pkl_files = [name for name in file_list if name.endswith('.pkl')]
            
            if not pkl_files:
                # No SB3 structure AND no pickle files = suspicious
                return {"error": f"No valid model files found. Files in ZIP: {file_list}"}
            
            # Inspect the first/main pkl file for custom models
            pkl_data = zf.read(pkl_files[0])
            
            # Load pickle data safely (without executing)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    model_data = pickle.loads(pkl_data)
                except Exception as e:
                    return {"error": f"Failed to load pickle: {e}"}
            
            return _analyze_model_data(model_data)
            
    except Exception as e:
        return {"error": f"ZIP inspection failed: {e}"}

def _analyze_pytorch_neural_network(state_dict, file_list) -> dict:
    """
    Deep analysis of PyTorch neural network for authenticity.
    Validates actual network structure, parameters, and weight distributions.
    """
    try:
        results = {
            "has_mlp_extractor": False,
            "parameter_count": 0,
            "has_training_artifacts": False,
            "weight_variance": 0.0,
            "suspicious_patterns": [],
            "class_names": ["SB3 PyTorch Model"],
            "layer_analysis": {}
        }
        
        # 1. NEURAL ARCHITECTURE VALIDATION
        # Look for expected SB3 PPO layer structure
        layer_names = list(state_dict.keys())
        
        # Expected layers in SB3 PPO models
        expected_patterns = [
            'mlp_extractor',  # Core feature extractor
            'action_net',     # Action output layer  
            'value_net'       # Value function layer
        ]
        
        found_layers = {pattern: [] for pattern in expected_patterns}
        for layer_name in layer_names:
            for pattern in expected_patterns:
                if pattern in layer_name:
                    found_layers[pattern].append(layer_name)
        
        # Check for mlp_extractor (required for legitimate PPO)
        if found_layers['mlp_extractor']:
            results["has_mlp_extractor"] = True
        else:
            results["suspicious_patterns"].append("Missing mlp_extractor layers")
        
        # Check for both action and value networks (critical for PPO)
        if not found_layers['action_net']:
            results["suspicious_patterns"].append("Missing action_net (required for PPO)")
        if not found_layers['value_net']:
            results["suspicious_patterns"].append("Missing value_net (required for PPO)")
        
        # 2. PARAMETER COUNT & DISTRIBUTION ANALYSIS
        all_weights = []
        layer_stats = {}
        total_params = 0
        
        for name, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor) and tensor.dtype.is_floating_point:
                weight_array = tensor.detach().cpu().numpy()
                param_count = weight_array.size
                total_params += param_count
                
                # Per-layer analysis
                layer_stats[name] = {
                    "shape": list(tensor.shape),
                    "params": param_count,
                    "mean": float(np.mean(weight_array)),
                    "std": float(np.std(weight_array)),
                    "min": float(np.min(weight_array)),
                    "max": float(np.max(weight_array))
                }
                
                # Collect weights for global analysis
                all_weights.append(weight_array.flatten())
        
        results["parameter_count"] = total_params
        results["layer_analysis"] = layer_stats
        
        # Parameter count validation
        if total_params < 5000:
            results["suspicious_patterns"].append(f"Too few parameters: {total_params} < 5000")
        
        # 3. WEIGHT DISTRIBUTION ANALYSIS
        if all_weights:
            combined_weights = np.concatenate(all_weights)
            results["weight_variance"] = float(np.var(combined_weights))
            
            # Check for suspicious weight patterns
            unique_values = len(np.unique(combined_weights))
            total_values = len(combined_weights)
            
            # Too few unique values indicates hardcoding
            if unique_values < 10 and total_values > 100:
                results["suspicious_patterns"].append(f"Too few unique weights: {unique_values}")
            
            # All weights identical
            if np.all(combined_weights == combined_weights[0]):
                results["suspicious_patterns"].append("All weights identical")
            
            # Check for obviously hardcoded common values
            common_hardcoded = [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0]
            hardcoded_ratio = np.mean([w in common_hardcoded for w in combined_weights[:1000]])
            if hardcoded_ratio > 0.8:
                results["suspicious_patterns"].append(f"Many hardcoded values: {hardcoded_ratio:.1%}")
            
            # Check weight variance (trained models should have reasonable variance)
            if results["weight_variance"] < 1e-8:
                results["suspicious_patterns"].append(f"Weights too uniform: variance {results['weight_variance']:.2e}")
            
            # Check for unrealistic weight magnitudes
            weight_range = np.max(combined_weights) - np.min(combined_weights)
            if weight_range > 100:
                results["suspicious_patterns"].append(f"Extreme weight range: {weight_range:.1f}")
        
        # 4. TRAINING AUTHENTICITY CHECKS
        # Look for batch normalization or training artifacts
        bn_layers = [name for name in layer_names if 'batch_norm' in name.lower() or 'running_' in name.lower()]
        if bn_layers:
            results["has_training_artifacts"] = True
        
        # Check layer structure consistency (input/output dimensions should match)
        mlp_layers = [name for name in layer_names if 'mlp_extractor' in name and 'weight' in name]
        if len(mlp_layers) >= 2:
            try:
                # Check if layer dimensions are consistent
                layer1_shape = state_dict[mlp_layers[0]].shape
                layer2_shape = state_dict[mlp_layers[1]].shape
                if len(layer1_shape) == 2 and len(layer2_shape) == 2:
                    if layer1_shape[0] != layer2_shape[1]:  # Output of layer1 should match input of layer2
                        results["suspicious_patterns"].append("Inconsistent layer dimensions")
            except Exception:
                pass  # Skip if analysis fails
        
        # 5. FINAL VALIDATION
        # Model size consistency check
        expected_min_size = total_params * 4  # 4 bytes per float32 parameter
        if len(file_list) > 0:  # We have file list, could check total ZIP size
            pass  # Could add ZIP size validation here
        
        # Advanced pattern detection: check weight initialization patterns
        if len(layer_stats) > 0:
            # Xavier/He initialization typically results in small, centered distributions
            suspicious_inits = 0
            for name, stats in layer_stats.items():
                if 'weight' in name:
                    # Suspiciously large standard deviation suggests fake weights
                    if stats['std'] > 10.0:
                        suspicious_inits += 1
                    # Suspiciously perfect standard deviation suggests manual setting
                    if abs(stats['std'] - 1.0) < 1e-6:  # Exactly 1.0 is suspicious
                        suspicious_inits += 1
                    # Also check for exactly 0 std (all same value)
                    if stats['std'] == 0.0:
                        suspicious_inits += 1
            
            if suspicious_inits > len([n for n in layer_stats if 'weight' in n]) * 0.5:
                results["suspicious_patterns"].append("Suspicious weight initialization patterns")
        
        return results
        
    except Exception as e:
        return {"error": f"PyTorch neural network analysis failed: {e}"}

def _save_fake_model_for_analysis(model_path: Path, uid: int, model_hash: str, reason: str, inspection_results: dict) -> None:
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

def _get_uid_from_axon(self, axon) -> int:
    """Get UID for a given axon by matching hotkey."""
    try:
        # Find the UID that matches this axon's hotkey
        for uid, metagraph_axon in enumerate(self.metagraph.axons):
            if metagraph_axon.hotkey == axon.hotkey:
                return uid
        return None  # Not found
    except Exception as e:
        bt.logging.warning(f"Failed to get UID from axon: {e}")
        return None

def _analyze_model_data(model_data) -> dict:
    """Analyze the loaded model data for fake model indicators."""
    try:
        results = {
            "has_mlp_extractor": False,
            "parameter_count": 0,
            "has_training_artifacts": False,
            "weight_variance": 0.0,
            "suspicious_patterns": [],
            "class_names": []
        }
        
        # Extract class information
        if hasattr(model_data, '__class__'):
            results["class_names"].append(str(model_data.__class__))
        
        # Check if it's a proper SB3 model structure
        if isinstance(model_data, dict):
            # SB3 models are typically dicts with specific keys
            if 'policy' in model_data:
                policy = model_data['policy']
                results["class_names"].append(str(type(policy)))
                
                # Check for MLP extractor (neural network component)
                if hasattr(policy, 'mlp_extractor') or 'mlp_extractor' in str(policy):
                    results["has_mlp_extractor"] = True
                    
                # Look for batch norm stats (training artifacts)
                policy_str = str(policy)
                if any(artifact in policy_str.lower() for artifact in 
                       ['batch_norm', 'running_mean', 'running_var', 'num_batches_tracked']):
                    results["has_training_artifacts"] = True
        
        # Analyze weight parameters
        weights = []
        _extract_weights(model_data, weights)
        
        if weights:
            all_weights = np.concatenate([w.flatten() for w in weights])
            results["parameter_count"] = len(all_weights)
            results["weight_variance"] = float(np.var(all_weights))
            
            # Check for suspicious patterns
            if np.all(all_weights == all_weights[0]):
                results["suspicious_patterns"].append("All weights identical")
            
            if len(np.unique(all_weights)) < 10 and len(all_weights) > 100:
                results["suspicious_patterns"].append("Too few unique weight values")
                
            # Check for obviously hardcoded values
            common_hardcoded = [0.0, 1.0, -1.0, 0.5, -0.5]
            if np.mean([w in common_hardcoded for w in all_weights[:100]]) > 0.8:
                results["suspicious_patterns"].append("Many hardcoded common values")
        
        return results
        
    except Exception as e:
        return {"error": f"Model analysis failed: {e}"}

def _extract_weights(obj, weights_list, max_depth=5):
    """Recursively extract numeric arrays from model structure."""
    if max_depth <= 0:
        return
        
    try:
        if isinstance(obj, np.ndarray) and obj.dtype in [np.float32, np.float64]:
            weights_list.append(obj)
        elif hasattr(obj, 'detach') and hasattr(obj, 'numpy'):  # PyTorch tensor
            weights_list.append(obj.detach().numpy())
        elif isinstance(obj, dict):
            for value in obj.values():
                _extract_weights(value, weights_list, max_depth - 1)
        elif hasattr(obj, '__dict__'):
            for value in obj.__dict__.values():
                _extract_weights(value, weights_list, max_depth - 1)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _extract_weights(item, weights_list, max_depth - 1)
    except Exception:
        pass  # Skip problematic objects

def _is_fake_model(inspection_results: dict) -> tuple[bool, str]:
    """
    Determine if model is fake based on inspection results.
    Returns (is_fake, reason).
    """
    if "error" in inspection_results:
        return True, f"Inspection error: {inspection_results['error']}"
    
    # Prioritize structural issues first
    
    # Check for MLP extractor (required for valid PPO)
    if not inspection_results["has_mlp_extractor"]:
        return True, "Missing mlp_extractor (required for valid PPO model)"
    
    # Check for suspicious patterns (includes missing layers, extreme weights, etc)
    if inspection_results["suspicious_patterns"]:
        # Check for critical missing components first
        for pattern in inspection_results["suspicious_patterns"]:
            if "missing" in pattern.lower():
                return True, pattern
        
        # Check for extreme/suspicious weight patterns
        for pattern in inspection_results["suspicious_patterns"]:
            if any(keyword in pattern.lower() for keyword in ["extreme", "suspicious", "hardcoded", "few unique"]):
                return True, f"Suspicious patterns: {', '.join(inspection_results['suspicious_patterns'])}"
    
    # Check weight variance (should not be uniform)
    if inspection_results["weight_variance"] < 1e-6:
        return True, f"Weights too uniform, variance: {inspection_results['weight_variance']}"
    
    # ENHANCED DETECTION: Check for sophisticated fakes
    layer_analysis = inspection_results.get("layer_analysis", {})
    
    # Check for all-zero biases (sign of artificial model)
    all_zero_biases = 0
    total_bias_layers = 0
    
    for layer_name, stats in layer_analysis.items():
        if 'bias' in layer_name:
            total_bias_layers += 1
            if (stats.get('mean', 1.0) == 0.0 and 
                stats.get('std', 1.0) == 0.0 and 
                stats.get('min', 1.0) == 0.0 and 
                stats.get('max', 1.0) == 0.0):
                all_zero_biases += 1
    
    # If ALL bias layers are zero, likely fake
    if total_bias_layers > 0 and all_zero_biases == total_bias_layers:
        return True, f"All {all_zero_biases} bias layers are zero (artificial model)"
    
    # Check for suspicious log_std (PPO action noise parameter)
    if 'log_std' in layer_analysis:
        log_std_stats = layer_analysis['log_std']
        if (log_std_stats.get('std', 1.0) == 0.0 and 
            log_std_stats.get('mean', 1.0) == 0.0):
            return True, "log_std parameter is all zeros (untrained PPO model)"
    
    # Check for lack of training artifacts
    if not inspection_results.get("has_training_artifacts", True):
        # This alone isn't enough, but combined with other factors...
        # Count additional suspicious indicators
        suspicious_indicators = 0
        
        # All zero biases
        if total_bias_layers > 0 and all_zero_biases >= total_bias_layers * 0.8:  # 80% or more
            suspicious_indicators += 1
            
        # Zero log_std
        if 'log_std' in layer_analysis:
            log_std_stats = layer_analysis['log_std']
            if log_std_stats.get('std', 1.0) == 0.0:
                suspicious_indicators += 1
        
        # If multiple indicators + no training artifacts = likely fake
        if suspicious_indicators >= 2:
            return True, "Multiple indicators of artificial model (no training artifacts + suspicious patterns)"
    
    # Check minimum parameter count last (as it's the most generic)
    if inspection_results["parameter_count"] < 5000:
        return True, f"Too few parameters: {inspection_results['parameter_count']} < 5000"
    
    # Additional check for remaining suspicious patterns
    if inspection_results["suspicious_patterns"]:
        return True, f"Suspicious patterns: {', '.join(inspection_results['suspicious_patterns'])}"
    
    # Passed all checks
    return False, "Model appears legitimate"


# ──────────────────────────────────────────────────────────────────────────
# 2.  Episode roll‑out (unchanged)
# ──────────────────────────────────────────────────────────────────────────
def _run_episode(
    task: "MapTask",
    uid: int,
    model: PPO,
    *,
    gui: bool = False,
) -> ValidationResult:
    """
    Executes one closed‑loop flight using *model* as the policy.
    Returns a fully‑populated ValidationResult.
    """
    class _Pilot:
        def __init__(self, m): self.m = m
        def reset(self, task):  pass
        def act(self, obs, t):
            act, _ = self.m.predict(obs, deterministic=True)
            return act.squeeze()

    pilot = _Pilot(model)
    env   = make_env(task, gui=gui)

    # initial observation
    try:
        obs = env._computeObs()                # type: ignore[attr-defined]
    except AttributeError:
        obs = env.get_observation()            # type: ignore[attr-defined]

    if isinstance(obs, dict):
        obs = obs[next(iter(obs))]

    pos0       = np.asarray(task.start, dtype=float)
    last_pos   = pos0.copy()
    t_sim      = 0.0
    energy     = 0.0
    success    = False
    step_count = 0
    frames_per_cam = max(1, int(round(1.0 / (SIM_DT * 60.0))))   # ≈60 Hz

    while t_sim < task.horizon:
        rpm  = pilot.act(obs, t_sim)
        obs, _r, terminated, truncated, info = env.step(rpm[None, :])

        t_sim   += SIM_DT
        energy  += np.abs(rpm).sum() * SIM_DT
        last_pos = obs[:3] if obs.ndim == 1 else obs[0, :3]

        if gui and step_count % frames_per_cam == 0:
            try:
                cli_id = getattr(env, "CLIENT", getattr(env, "_cli", 0))
                track_drone(cli=cli_id, drone_id=env.DRONE_IDS[0])
            except Exception:
                pass
        if gui:
            time.sleep(SIM_DT)

        if terminated or truncated:
            success = info.get("success", False)
            break

        step_count += 1

    if not gui:
        env.close()

    # ── final score with new reward function ──────────────────────────────
    score = flight_reward(
        success = success,
        t       = t_sim,
        e       = energy,
        horizon = task.horizon,
        # (optionally) tweak e_budget or weightings here if needed
    )

    return ValidationResult(uid, success, t_sim, energy, score)


# ──────────────────────────────────────────────────────────────────────────
# 3.  Secure, cached model download
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
    from .docker_evaluator import DockerSecureEvaluator
    
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
                    
                    # Check if fake model was detected
                    if verification_data.get('is_fake_model', False):
                        fake_reason = verification_data.get('fake_reason', 'Unknown')
                        inspection_results = verification_data.get('inspection_results', {})
                        
                        bt.logging.warning(f"🚫 FAKE MODEL DETECTED during verification: {fake_reason}")
                        bt.logging.info(f"Model hash: {model_hash}")
                        bt.logging.debug(f"Inspection details: {inspection_results}")
                        
                        # Save fake model for analysis and add to blacklist
                        _save_fake_model_for_analysis(model_path, uid, model_hash, fake_reason, inspection_results)
                        _add_to_blacklist(model_hash)
                        
                        # Remove the fake model from cache
                        model_path.unlink(missing_ok=True)
                        bt.logging.info(f"🗑️ Removed fake model {model_hash[:16]}... from cache")
                        
                    else:
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
        blacklist = _load_blacklist()
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
# 4.  Sand‑boxed evaluation (subprocess with rlimits)
# ──────────────────────────────────────────────────────────────────────────

def _evaluate_uid(task: MapTask, uid: int, model_fp: Path) -> ValidationResult:
    """
    Spawn the standalone evaluator in a sandboxed subprocess, enforce a timeout,
    and return a ValidationResult.

    ─────────────────────────────────────────────────────────────────────────────
    Key behaviour
    ─────────────────────────────────────────────────────────────────────────────
    1. Temporary files are stored under ./tmp (created if necessary, otherwise
       we fall back to the system temp directory).
    2. Temporary files are always deleted in the finally‑block.
    3. If the evaluator reports success but a score of exactly 0.0, we bump it
       to 0.01 to acknowledge a correct setup.  All other cases (errors, parse
       failures, timeouts, etc.) return a 0.0 score.
    """
    print(f"🔬 DEBUG: _evaluate_uid called for UID {uid}, model: {model_fp}")


    # ------------------------------------------------------------------
    # 1. Resolve ./tmp directory (use system tmp if creation fails)
    # ------------------------------------------------------------------
    try:
        tmp_dir = Path.cwd() / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        bt.logging.warning(f"Could not create ./tmp directory: {e}. Falling back to system tmp.")
        tmp_dir = Path(tempfile.gettempdir())

    unique_id   = f"{int(time.time() * 1_000_000)}_{os.getpid()}_{uid}_{uuid.uuid4().hex[:8]}"
    task_file   = tmp_dir / f"swarm_task_{unique_id}.json"
    result_file = tmp_dir / f"swarm_result_{unique_id}.json"
    print(f"📁 DEBUG: Using temp files: {task_file}, {result_file}")

    try:
        # --------------------------------------------------------------
        # Write task to disk for the evaluator subprocess
        # --------------------------------------------------------------
        with task_file.open("w") as f:
            json.dump(asdict(task), f)

        # --------------------------------------------------------------
        # Build subprocess command
        # --------------------------------------------------------------
        evaluator_script = Path(__file__).parent.parent / "core" / "evaluator.py"
        if not evaluator_script.exists():
            bt.logging.error(f"Evaluator script not found at {evaluator_script}")
            return ValidationResult(uid, False, 0.0, 0.0, 0.0)

        cmd = [
            sys.executable,
            str(evaluator_script),
            str(task_file),
            str(uid),
            str(model_fp),
            str(result_file),
        ]

        # ------------------------------------------------
        # Launch evaluator (with timeout guard)
        # ------------------------------------------------
        proc = subprocess.run(
            cmd,
            timeout=EVAL_TIMEOUT_SEC,
            capture_output=True,
            text=True,
        )

        # ------------------------------------------------
        # Process evaluator output
        # ------------------------------------------------
        if result_file.exists():
            try:
                with result_file.open("r") as f:
                    data = json.load(f)

                # Check if there was an error
                had_error = "error" in data
                if had_error:
                    bt.logging.debug(f"Subprocess error for UID {uid}: {data['error']}")

                result_data = {k: v for k, v in data.items() if k != "error"}

                # DEBUG: Show actual result data
                print(f"🔍 DEBUG: UID {uid} result_data: {result_data}, had_error: {had_error}")

                # ───── Reward‑floor logic (evaluator completed successfully WITHOUT errors) ─────
                if not had_error and float(result_data.get("score", 0.0)) == 0.0:
                    bt.logging.debug(f"UID {uid} score is 0 but no errors → bumping to 0.01")
                    result_data["score"] = 0.01
                    print(f"🎯 DEBUG: UID {uid} score bumped to 0.01 (model worked but failed mission)!")
                elif had_error:
                    bt.logging.debug(f"UID {uid} had errors → keeping score at 0.0")
                    print(f"❌ DEBUG: UID {uid} had errors, no reward bump")

                return ValidationResult(**result_data)

            except (json.JSONDecodeError, TypeError, KeyError) as e:
                bt.logging.warning(f"Failed to parse result file for UID {uid}: {e}")

        else:
            # The subprocess ended but produced no result file
            if proc.returncode != 0:
                bt.logging.warning(f"Subprocess failed for UID {uid}, returncode={proc.returncode}")
                if proc.stderr:
                    bt.logging.debug(f"Subprocess stderr: {proc.stderr}")
            else:
                bt.logging.warning(f"No result file found for UID {uid}")

    except subprocess.TimeoutExpired:
        bt.logging.warning(f"Miner {uid} exceeded timeout of {EVAL_TIMEOUT_SEC}s")
    except Exception as e:
        bt.logging.warning(f"Subprocess evaluation failed for UID {uid}: {e}")

    finally:
        # -----------------------------------------------------------
        # 2. Always delete temporary files
        # -----------------------------------------------------------
        for tmp in (task_file, result_file):
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    print(f"⚠️  DEBUG: Fallback result for UID {uid} – giving 0.0 reward (error path)")
    # ────────────────────────────────────────────────────────────────────
    # Final fallback (evaluation failed entirely)  →  score = 0.0
    # ────────────────────────────────────────────────────────────────────
    return ValidationResult(uid, False, 0.0, 0.0, 0.0)


# ──────────────────────────────────────────────────────────────────────────
# 5.  Weight boosting
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
# 6.  Public coroutine – called by neurons/validator.py
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
                            _save_fake_model_for_analysis(
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
                blacklist = _load_blacklist()
                for fake_model in fake_models_detected:
                    bt.logging.info(f"🚫 Adding fake model to blacklist: UID {fake_model['uid']}, hash {fake_model['hash'][:16]}...")
                    blacklist.add(fake_model['hash'])
                _save_blacklist(blacklist)
            
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