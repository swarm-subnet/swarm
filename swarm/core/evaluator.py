#!/usr/bin/env python3
"""
Standalone evaluator script for subprocess execution.
This runs as a separate Python process to completely isolate from bittensor logging.
"""

import sys
import os
import json
import gc
import traceback
from pathlib import Path
from stable_baselines3 import PPO
from dataclasses import asdict
from swarm.protocol import MapTask

# Add swarm to path  
swarm_path = str(Path(__file__).resolve().parent.parent)
if swarm_path not in sys.path:
    sys.path.insert(0, swarm_path)

def main():
    """Main evaluator entry point"""
    
    # Disable all logging to prevent any logging threads
    import logging
    logging.disable(logging.CRITICAL)
    
    # Redirect stderr to suppress output
    original_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    
    try:
        # Parse command line arguments
        if len(sys.argv) != 5:
            raise ValueError("Usage: evaluator.py <task_json> <uid> <model_path> <result_file>")
        
        task_json = sys.argv[1]
        uid = int(sys.argv[2])
        model_path = sys.argv[3]
        result_file = sys.argv[4]
        
        # Set memory limits
        try:
            import resource
            SUBPROC_MEM_MB = 4096
            rss_bytes = SUBPROC_MEM_MB * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (rss_bytes, rss_bytes))
            resource.setrlimit(resource.RLIMIT_DATA, (rss_bytes, rss_bytes))
        except Exception:
            pass
        
        # Parse task from JSON
        with open(task_json, 'r') as f:
            task_data = json.load(f)
        
        task = MapTask(**task_data)
        
        # Load model and run evaluation
        model = PPO.load(model_path, device="cpu")
        
        # Import and run episode  
        from swarm.validator.forward import _run_episode
        result = _run_episode(task, uid, model)
        
        # Write result to file
        tmp_file = result_file + ".tmp"
        with open(tmp_file, 'w') as f:
            json.dump(asdict(result), f)
        
        # Atomic rename
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
            tmp_file = result_file + ".tmp" if 'result_file' in locals() else "/tmp/error_result.tmp"
            final_file = result_file if 'result_file' in locals() else "/tmp/error_result.json"
            
            with open(tmp_file, 'w') as f:
                json.dump(error_result, f)
            os.rename(tmp_file, final_file)
        except Exception:
            pass
        
        sys.exit(1)
        
    finally:
        # Restore stderr and logging
        try:
            sys.stderr.close()
            sys.stderr = original_stderr
            logging.disable(logging.NOTSET)
        except Exception:
            pass

if __name__ == "__main__":
    main()