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

import hashlib
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
from typing import Dict, Tuple, Set, List, Optional
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


def is_fake_model(inspection_results: Dict) -> Tuple[bool, str]:
    """
    Legacy compatibility wrapper for classify_model_validity().
    Returns True for both fake and missing_drone_agent cases.
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
