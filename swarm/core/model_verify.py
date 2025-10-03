#!/usr/bin/env python3
"""
Model Verification Module for Swarm Subnet
==========================================

This module contains the fake model detection system extracted from forward.py.
Provides comprehensive analysis of AI models to detect cheating, exploits, and fake submissions.

Key Components:
- Blacklist management for known fake models
- PyTorch neural network structure analysis
- Model weight distribution validation
- Forensic storage of detected fake models
"""

import json
import math
import time
import warnings
import shutil
import io
import pickle
import base64
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple, Set, List
from zipfile import ZipFile, BadZipFile

import numpy as np
import torch
import bittensor as bt

# Import centralized constants
from swarm.constants import MODEL_DIR, BLACKLIST_FILE


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
    Inspect PPO model structure without loading it through SB3.
    NOW REQUIRES secure loading metadata.
    """
    try:
        # FIRST: Check for secure loading metadata (REQUIRED)
        with ZipFile(zip_path, 'r') as zf:
            file_list = zf.namelist()
            
            # Require safe_policy_meta.json
            if "safe_policy_meta.json" not in file_list:
                return {
                    "error": "Missing safe_policy_meta.json - model not compatible with secure loading",
                    "missing_secure_metadata": True
                }
            
            # Validate JSON structure
            try:
                meta_content = zf.read("safe_policy_meta.json").decode("utf-8")
                meta = json.loads(meta_content)
                
                required_keys = ["activation_fn", "net_arch", "use_sde"]
                for key in required_keys:
                    if key not in meta:
                        return {"error": f"Invalid metadata: missing {key}"}
                        
            except json.JSONDecodeError as e:
                return {"error": f"Invalid JSON in safe_policy_meta.json: {e}"}

        # Model metadata validation complete - proceed to structural analysis
        
        with ZipFile(zip_path, 'r') as zf:
            file_list = zf.namelist()

            # Reject models with executable/code files (CHECK FIRST)
            suspicious_files = [f for f in file_list if f.endswith(('.py', '.pkl', '.pyc', '.so', '.exe', '.sh', '.bat'))]
            if suspicious_files:
                return {"error": f"Executable/code files detected: {suspicious_files}"}

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
                        state_dict = torch.load(io.BytesIO(policy_data), map_location='cpu', weights_only=True)

                    return analyze_pytorch_neural_network(state_dict, file_list)

                except Exception as e:
                    return {"error": f"Failed to analyze SB3 neural network: {e}"}

            # Only accept legitimate SB3 structure
            return {"error": f"Invalid model structure. Files: {file_list}"}
            
    except Exception as e:
        return {"error": f"ZIP inspection failed: {e}"}


def analyze_pytorch_neural_network(state_dict: Dict, file_list: list) -> Dict:
    """
    Deep analysis of PyTorch neural network for authenticity.
    Validates actual network structure, parameters, and weight distributions.
    """
    try:
        # Handle empty or invalid state dict
        if not state_dict or not isinstance(state_dict, dict):
            return {"error": "Invalid or empty state dictionary"}

        results = {
            "has_mlp_extractor": False,
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
        
        # Basic layer analysis for architecture validation
        layer_stats = {}

        for name, tensor in state_dict.items():
            try:
                # Handle both real torch.Tensor and mock objects
                has_dtype = hasattr(tensor, 'dtype') and hasattr(tensor.dtype, 'is_floating_point')
                has_detach = hasattr(tensor, 'detach')
                has_shape = hasattr(tensor, 'shape')

                if (has_dtype and has_detach and has_shape and
                    tensor.dtype.is_floating_point):
                    # Real torch.Tensor or proper mock
                    weight_array = tensor.detach().cpu().numpy()

                    # Basic layer info for structure validation
                    layer_stats[name] = {
                        "shape": list(tensor.shape),
                        "mean": float(np.mean(weight_array)),
                        "std": float(np.std(weight_array))
                    }
            except Exception:
                # Skip tensors that can't be processed
                continue

        results["layer_analysis"] = layer_stats
        
        # Basic validation complete - focusing on essential architecture checks only
        
        return results
        
    except Exception as e:
        return {"error": f"PyTorch neural network analysis failed: {e}"}


def analyze_model_data(model_data) -> Dict:
    """Analyze the loaded model data for fake model indicators."""
    try:
        results = {
            "has_mlp_extractor": False,
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
        extract_weights(model_data, weights)
        
        # Basic structure validation only - weight analysis removed for efficiency
        
        return results
        
    except Exception as e:
        return {"error": f"Model analysis failed: {e}"}


def extract_weights(obj, weights_list: list, max_depth: int = 5) -> None:
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
                extract_weights(value, weights_list, max_depth - 1)
        elif hasattr(obj, '__dict__'):
            for value in obj.__dict__.values():
                extract_weights(value, weights_list, max_depth - 1)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                extract_weights(item, weights_list, max_depth - 1)
    except Exception:
        pass  # Skip problematic objects


def classify_model_validity(inspection_results: Dict) -> Tuple[str, str]:
    """
    Classify model validity into three categories:
    - "legitimate": Model passes all checks
    - "missing_metadata": Model lacks secure metadata (reject but don't blacklist)  
    - "fake": Model is actually fake/malicious (reject and blacklist)
    
    Returns (status, reason)
    """
    # PRIORITY 1: Missing secure metadata (reject but don't blacklist)
    if inspection_results.get("missing_secure_metadata", False):
        return "missing_metadata", "Missing secure loading metadata - model rejected"
    
    # PRIORITY 2: Security violations (fake models)
    if "malicious_findings" in inspection_results:
        return "fake", "Security violation: Malicious code detected"
    
    if "error" in inspection_results:
        if "Security violation" in inspection_results["error"]:
            return "fake", inspection_results["error"]
        if "Missing safe_policy_meta.json" in inspection_results["error"]:
            return "missing_metadata", inspection_results["error"]
        return "fake", f"Inspection error: {inspection_results['error']}"
    
    # PRIORITY 3: Structural issues (fake models)

    # Check for MLP extractor (required for valid PPO)
    if not inspection_results.get("has_mlp_extractor", False):
        return "fake", "Missing mlp_extractor (required for valid PPO model)"
    
    # Check for suspicious patterns (includes missing layers, extreme weights, etc)
    suspicious_patterns = inspection_results.get("suspicious_patterns", [])
    if suspicious_patterns:
        # Check for critical missing components first
        for pattern in suspicious_patterns:
            if "missing" in pattern.lower():
                return "fake", pattern

        # Check for extreme/suspicious weight patterns
        for pattern in suspicious_patterns:
            if any(keyword in pattern.lower() for keyword in ["extreme", "suspicious", "hardcoded", "few unique"]):
                return "fake", f"Suspicious patterns: {', '.join(suspicious_patterns)}"
    
    
    # PRIORITY 4: Enhanced detection for sophisticated fakes
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
        return "fake", f"All {all_zero_biases} bias layers are zero (artificial model)"
    
    # Check for suspicious log_std (PPO action noise parameter)
    if 'log_std' in layer_analysis:
        log_std_stats = layer_analysis['log_std']
        if (log_std_stats.get('std', 1.0) == 0.0 and 
            log_std_stats.get('mean', 1.0) == 0.0):
            return "fake", "log_std parameter is all zeros (untrained PPO model)"
    
    
  
    # Additional check for remaining suspicious patterns
    if suspicious_patterns:
        return "fake", f"Suspicious patterns: {', '.join(suspicious_patterns)}"
    
    # Passed all checks
    return "legitimate", "Model appears legitimate"


def is_fake_model(inspection_results: Dict) -> Tuple[bool, str]:
    """
    Legacy compatibility wrapper for classify_model_validity().
    Returns True for both fake and missing_metadata cases.
    """
    status, reason = classify_model_validity(inspection_results)
    return status != "legitimate", reason


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


def get_uid_from_axon(metagraph, axon) -> int:
    """Get UID for a given axon by matching hotkey."""
    try:
        # Find the UID that matches this axon's hotkey
        for uid, metagraph_axon in enumerate(metagraph.axons):
            if metagraph_axon.hotkey == axon.hotkey:
                return uid
        return None  # Not found
    except Exception as e:
        bt.logging.warning(f"Failed to get UID from axon: {e}")
        return None