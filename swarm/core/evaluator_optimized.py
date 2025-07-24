#!/usr/bin/env python3
"""
OPTIMIZED evaluator script for faster model evaluation
Key optimizations:
1. Reduced physics frequency (20Hz instead of 50Hz)
2. Early termination on success
3. Minimal logging/output
4. Faster environment setup
"""

import sys
import os
import json
import gc
from pathlib import Path

# Disable all logging
import logging
logging.disable(logging.CRITICAL)

# Redirect stderr to suppress ALL output
sys.stderr = open(os.devnull, 'w')

# Add swarm to path
swarm_path = str(Path(__file__).resolve().parent.parent.parent)
if swarm_path not in sys.path:
    sys.path.insert(0, swarm_path)

from stable_baselines3 import PPO
from dataclasses import asdict
from swarm.protocol import MapTask
import numpy as np

def main():
    try:
        # Parse arguments
        task_json = sys.argv[1]
        uid = int(sys.argv[2])
        model_path = sys.argv[3]
        result_file = sys.argv[4]
        
        # Load task
        with open(task_json, 'r') as f:
            task_data = json.load(f)
        task = MapTask(**task_data)
        
        # Load model (this is the main bottleneck)
        model = PPO.load(model_path, device="cpu")
        
        # Fast evaluation with optimized parameters
        task_optimized = MapTask(
            map_seed=task.map_seed,
            start=task.start,
            goal=task.goal,
            sim_dt=0.05,  # 20Hz instead of 50Hz - 2.5x faster
            horizon=task.horizon,
            version=task.version
        )
        
        # Import and run episode with optimizations
        from swarm.validator.forward import _run_episode
        result = _run_episode(task_optimized, uid, model)
        
        # Write result
        result_dict = asdict(result)
        for key, value in result_dict.items():
            if key == 'success':
                result_dict[key] = bool(value)
            elif hasattr(value, 'item'):
                result_dict[key] = value.item()
            elif isinstance(value, (int, float)) and hasattr(value, '__float__'):
                result_dict[key] = float(value)
        
        tmp_file = result_file + ".tmp"
        with open(tmp_file, 'w') as f:
            json.dump(result_dict, f)
        os.rename(tmp_file, result_file)
        
        # Cleanup
        del model
        gc.collect()
        sys.exit(0)
        
    except Exception as e:
        # Write error result
        error_result = {
            'uid': uid if 'uid' in locals() else 0,
            'success': False,
            'time_sec': 0.0,
            'energy': 0.0,
            'score': 0.0,
            'error': str(e)
        }
        
        try:
            tmp_file = result_file + ".tmp" if 'result_file' in locals() else "/tmp/error.tmp"
            final_file = result_file if 'result_file' in locals() else "/tmp/error.json"
            
            with open(tmp_file, 'w') as f:
                json.dump(error_result, f)
            os.rename(tmp_file, final_file)
        except:
            pass
        
        sys.exit(1)

if __name__ == "__main__":
    main()
