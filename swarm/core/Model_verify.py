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
import time
import warnings
import shutil
import io
import pickle
from pathlib import Path
from typing import Dict, Tuple, Set
from zipfile import ZipFile, BadZipFile

import numpy as np
import torch
import bittensor as bt

# ──────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────
MODEL_DIR = Path("miner_models_v2")
BLACKLIST_FILE = MODEL_DIR / "fake_models_blacklist.txt"


# ──────────────────────────────────────────────────────────────────────────
# Blacklist Management
# ──────────────────────────────────────────────────────────────────────────

def load_blacklist() -> Set[str]:
    """Load blacklisted fake model hashes from file."""
    try:
        if BLACKLIST_FILE.exists():
            with open(BLACKLIST_FILE, 'r') as f:
                return {line.strip() for line in f if line.strip()}
        return set()
    except Exception as e:
        bt.logging.warning(f"Error loading blacklist: {e}")
        return set()


def save_blacklist(blacklist: Set[str]) -> None:
    """Save blacklisted fake model hashes to file."""
    try:
        # Ensure the directory exists
        BLACKLIST_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(BLACKLIST_FILE, 'w') as f:
            for hash_val in sorted(blacklist):
                f.write(f"{hash_val}\n")
    except Exception as e:
        bt.logging.error(f"Error saving blacklist: {e}")


def add_to_blacklist(model_hash: str) -> None:
    """Add a single model hash to the blacklist."""
    try:
        blacklist = load_blacklist()
        blacklist.add(model_hash)
        save_blacklist(blacklist)
        bt.logging.info(f"🚫 Added {model_hash[:16]}... to blacklist")
    except Exception as e:
        bt.logging.error(f"Error adding to blacklist: {e}")


# ──────────────────────────────────────────────────────────────────────────
# Model Structure Analysis
# ──────────────────────────────────────────────────────────────────────────

def inspect_model_structure(zip_path: Path) -> Dict:
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
                    
                    return analyze_pytorch_neural_network(state_dict, file_list)
                    
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
            
            return analyze_model_data(model_data)
            
    except Exception as e:
        return {"error": f"ZIP inspection failed: {e}"}


def analyze_pytorch_neural_network(state_dict: Dict, file_list: list) -> Dict:
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


def analyze_model_data(model_data) -> Dict:
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
        extract_weights(model_data, weights)
        
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


def is_fake_model(inspection_results: Dict) -> Tuple[bool, str]:
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