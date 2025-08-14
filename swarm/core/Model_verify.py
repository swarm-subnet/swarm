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
import base64
import re
from pathlib import Path
from typing import Dict, Tuple, Set, List
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
# Malicious Pattern Detection
# ──────────────────────────────────────────────────────────────────────────

def analyze_pickle_structure(pickle_bytes: bytes) -> Tuple[bool, List[str]]:
    """Deep structural analysis of pickle bytecode."""
    findings = []
    
    dangerous_imports = [
        b'cos\nsystem',
        b'cos\nexec',
        b'csubprocess\n',
        b'cbuiltins\neval',
        b'cbuiltins\nexec',
        b'c__builtin__\neval',
        b'c__builtin__\nexec',
    ]
    
    for pattern in dangerous_imports:
        if pattern in pickle_bytes:
            findings.append(f"CRITICAL: Dangerous import detected")
    
    execution_patterns = [
        b'R}q\x00(X',  
        b'cbuiltins\n',
        b'c__builtin__\n',
    ]
    
    reduce_count = pickle_bytes.count(b'R')
    if reduce_count > 30:
        for pattern in execution_patterns:
            if pattern in pickle_bytes:
                findings.append(f"CRITICAL: High REDUCE count with execution pattern")
    
    if len(pickle_bytes) > 10000:
        findings.append("WARNING: Large pickle size")
    
    import math
    from collections import Counter
    byte_counts = Counter(pickle_bytes)
    entropy = -sum((count/len(pickle_bytes)) * math.log2(count/len(pickle_bytes)) 
                   for count in byte_counts.values() if count > 0)
    
    if entropy > 7.8:
        findings.append(f"WARNING: High entropy ({entropy:.2f})")
    
    exec_indicators = [
        b'\x8c\x04exec',
        b'\x8c\x06compile',
        b'builtins\neval',
        b'builtins\nexec',
    ]
    
    for indicator in exec_indicators:
        if indicator in pickle_bytes:
            findings.append(f"CRITICAL: Dangerous execution pattern found")
    
    is_malicious = len([f for f in findings if 'CRITICAL' in f]) > 0
    return is_malicious, findings


def scan_for_malicious_patterns(zip_path: Path) -> Tuple[bool, List[str]]:
    """Pre-execution security scan for RCE and code injection patterns."""
    
    malicious_patterns = [
        # File system manipulation
        b'/workspace/shared/result.json',
        b'/workspace/shared',
        b'/workspace',
        b'result.json',
        b'/tmp/',
        b'/etc/',
        
        # System commands
        b'os.chmod',
        b'os.system',
        b'os.popen',
        b'os.execv',
        b'subprocess.call',
        b'subprocess.run',
        b'subprocess.Popen',
        b'commands.get',
        
        # Code execution
        b'eval(',
        b'exec(',
        b'compile(',
        b'__import__',
        b'importlib.import',
        b'builtins.eval',
        b'builtins.exec',
        b'builtins.compile',
        b'globals()[',
        b'locals()[',
        b'setattr(',
        b'delattr(',
        
        # Pickle exploits
        b'reduce',
        b'apply',
        b'__reduce__',
        b'__reduce_ex__',
        b'find_class',
        b'dispatch_table',
        b'persistent_id',
        b'inst_persistent_id',
        
        # Network operations
        b'socket.',
        b'urllib.request',
        b'requests.get',
        b'requests.post',
        b'http.client',
        b'ftplib',
        b'smtplib',
        
        # File operations
        b'open(',
        b'.write(',
        b'.writelines(',
        b'with open',
        b'file(',
        b'input(',
        b'raw_input(',
        
        # Dangerous functions
        b'do_rewrite',
        b'atexit.register',
        b'signal.signal',
        b'threading.Thread',
        b'multiprocessing.Process',
        
        # String obfuscation techniques
        b'chr(',
        b'ord(',
        b'join([',
        b'.join(',
        b'base64.b64decode',
        b'base64.b32decode',
        b'base64.b16decode',
        b'codecs.decode',
        b'binascii.unhexlify',
        b'bytes.fromhex',
        
        # Pickle-specific dangerous patterns (removed single byte checks)
        b'cbuiltins\n',
        b'cos\nsystem',
        b'csubprocess\n',
        b'c__builtin__\n',
        
        # Score/result manipulation
        b'"score"',
        b"'score'",
        b'"success": true',
        b"'success': True",
        b'"energy": 0',
        b"'energy': 0",
    ]
    
    findings = []
    
    try:
        with ZipFile(zip_path, 'r') as zf:
            for filename in zf.namelist():
                try:
                    # Path traversal check
                    if '..' in filename or filename.startswith('/'):
                        findings.append(f"CRITICAL: Path traversal attempt in filename: {filename}")
                        continue
                    
                    content = zf.read(filename)
                    
                    # Direct pattern scanning
                    for pattern in malicious_patterns:
                        if pattern in content:
                            findings.append(f"Pattern '{pattern.decode('utf-8', errors='ignore')}' in {filename}")
                    
                    # Special handling for data file
                    if filename == 'data':
                        try:
                            data_json = json.loads(content)
                            if isinstance(data_json, dict) and 'policy_class' in data_json:
                                policy = data_json['policy_class']
                                if isinstance(policy, dict) and ':serialized:' in policy:
                                    # Decode base64 pickle
                                    try:
                                        b64_data = policy[':serialized:']
                                        pickle_bytes = base64.b64decode(b64_data)
                                        
                                        # Scan decoded pickle
                                        for pattern in malicious_patterns:
                                            if pattern in pickle_bytes:
                                                findings.append(f"CRITICAL: Hidden pattern '{pattern.decode('utf-8', errors='ignore')}' in base64 pickle")
                                        
                                        # Error suppression detection
                                        if b'try:' in pickle_bytes or b'except:' in pickle_bytes:
                                            if b'pass' in pickle_bytes or b'continue' in pickle_bytes:
                                                findings.append("CRITICAL: Error suppression pattern detected")
                                                
                                        # Deep structural analysis of pickle
                                        struct_malicious, struct_findings = analyze_pickle_structure(pickle_bytes)
                                        findings.extend(struct_findings)
                                        
                                        if struct_malicious:
                                            findings.append("CRITICAL: Pickle structure analysis detected malicious patterns")
                                        
                                        # Check for any executable-like patterns
                                        if len(pickle_bytes) > 5000:
                                            # Very large pickle might be suspicious
                                            findings.append(f"WARNING: Large pickle size ({len(pickle_bytes)} bytes)")
                                        
                                        # Check if pickle contains too much non-printable data
                                        non_printable = sum(1 for b in pickle_bytes if b < 32 or b > 126)
                                        if non_printable > len(pickle_bytes) * 0.7:
                                            findings.append("WARNING: High ratio of non-printable bytes")
                                            
                                    except Exception:
                                        findings.append("WARNING: Failed to decode base64 in data file")
                        except json.JSONDecodeError:
                            pass  # Not JSON, continue
                    
                    # Look for hidden base64 in any text file
                    if filename.endswith(('.json', '.txt', '.xml', '.yaml', '.yml')):
                        base64_pattern = re.compile(b'[A-Za-z0-9+/]{100,}={0,2}')
                        matches = base64_pattern.findall(content)
                        for match in matches[:5]:  # Check first 5 long base64 strings
                            try:
                                decoded = base64.b64decode(match)
                                for pattern in malicious_patterns:
                                    if pattern in decoded:
                                        findings.append(f"CRITICAL: Hidden pattern in base64 string in {filename}")
                                        break
                            except Exception:
                                pass
                                
                except Exception as e:
                    findings.append(f"Error scanning {filename}: {str(e)}")
        
        # Determine if malicious
        critical_count = sum(1 for f in findings if 'CRITICAL' in f)
        is_malicious = critical_count > 0
        
        # Also flag if multiple suspicious patterns
        workspace_patterns = sum(1 for f in findings if 'workspace' in f.lower())
        exec_patterns = sum(1 for f in findings if any(x in f.lower() for x in ['eval', 'exec', 'compile', '__import__']))
        file_patterns = sum(1 for f in findings if any(x in f.lower() for x in ['open(', 'write', 'chmod']))
        obfuscation_patterns = sum(1 for f in findings if any(x in f.lower() for x in ['chr(', 'b64decode', 'fromhex', 'opcode', 'entropy', 'obfuscat']))
        structural_patterns = sum(1 for f in findings if any(x in f.lower() for x in ['reduce', 'stack_global', 'opcode', 'pickle']))
        
        if workspace_patterns >= 2 or exec_patterns >= 2 or file_patterns >= 3 or obfuscation_patterns >= 2 or structural_patterns >= 3:
            is_malicious = True
        
        # Special case: if we find high entropy or unusual structure, be more strict
        if any('entropy' in f.lower() or 'large pickle' in f.lower() for f in findings):
            if critical_count >= 2:  # Lower threshold when obfuscation detected
                is_malicious = True
            
        return is_malicious, findings
        
    except Exception as e:
        return True, [f"Failed to scan ZIP: {str(e)}"]


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

        # Continue with existing security scan...
        is_malicious, findings = scan_for_malicious_patterns(zip_path)
        
        if is_malicious:
            bt.logging.error(f"🚨 Malicious patterns detected in {zip_path.name}")
            critical_findings = [f for f in findings if 'CRITICAL' in f]
            for finding in critical_findings[:5]:  # Show first 5 critical findings
                bt.logging.error(f"  {finding}")
            
            return {
                "error": "Security violation: Malicious code patterns detected",
                "malicious_findings": findings,
                "suspicious_patterns": ["RCE exploit patterns found"]
            }
        
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
                        state_dict = torch.load(io.BytesIO(policy_data), map_location='cpu', weights_only=True)
                    
                    return analyze_pytorch_neural_network(state_dict, file_list)
                    
                except Exception as e:
                    return {"error": f"Failed to analyze SB3 neural network: {e}"}
            
            # Reject models with executable/code files
            suspicious_files = [f for f in file_list if f.endswith(('.py', '.pkl', '.pyc', '.so', '.exe', '.sh', '.bat'))]
            if suspicious_files:
                return {"error": f"Executable/code files detected: {suspicious_files}"}
            
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
    if not inspection_results["has_mlp_extractor"]:
        return "fake", "Missing mlp_extractor (required for valid PPO model)"
    
    # Check for suspicious patterns (includes missing layers, extreme weights, etc)
    if inspection_results["suspicious_patterns"]:
        # Check for critical missing components first
        for pattern in inspection_results["suspicious_patterns"]:
            if "missing" in pattern.lower():
                return "fake", pattern
        
        # Check for extreme/suspicious weight patterns
        for pattern in inspection_results["suspicious_patterns"]:
            if any(keyword in pattern.lower() for keyword in ["extreme", "suspicious", "hardcoded", "few unique"]):
                return "fake", f"Suspicious patterns: {', '.join(inspection_results['suspicious_patterns'])}"
    
    # Check weight variance (should not be uniform)
    if inspection_results["weight_variance"] < 1e-6:
        return "fake", f"Weights too uniform, variance: {inspection_results['weight_variance']}"
    
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
            return "fake", "Multiple indicators of artificial model (no training artifacts + suspicious patterns)"
    
    # Check minimum parameter count last (as it's the most generic)
    if inspection_results["parameter_count"] < 5000:
        return "fake", f"Too few parameters: {inspection_results['parameter_count']} < 5000"
    
    # Additional check for remaining suspicious patterns
    if inspection_results["suspicious_patterns"]:
        return "fake", f"Suspicious patterns: {', '.join(inspection_results['suspicious_patterns'])}"
    
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