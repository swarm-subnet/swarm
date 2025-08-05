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
import resource
from pathlib import Path

# Add swarm to path BEFORE importing swarm modules
swarm_path = str(Path(__file__).resolve().parent.parent.parent)
if swarm_path not in sys.path:
    sys.path.insert(0, swarm_path)

from stable_baselines3 import PPO
from dataclasses import asdict
from swarm.protocol import MapTask, ValidationResult


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
            SUBPROC_MEM_MB = 8192
            rss_bytes = SUBPROC_MEM_MB * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (rss_bytes, rss_bytes))
            resource.setrlimit(resource.RLIMIT_DATA, (rss_bytes, rss_bytes))
        except Exception:
            pass
        
        # Parse task from JSON
        with open(task_json, 'r') as f:
            task_data = json.load(f)
        
        task = MapTask(**task_data)
        
        # First inspect model for fake indicators (safe within container)
        from swarm.validator.forward import _inspect_model_structure, _is_fake_model
        from pathlib import Path
        
        inspection_results = _inspect_model_structure(Path(model_path))
        is_fake, fake_reason = _is_fake_model(inspection_results)
        
        if is_fake:
            # Return fake model detection result
            result = ValidationResult(
                uid=uid,
                success=False,
                time_sec=0.0,
                energy=0.0,
                score=0.0
            )
        else:
            # Model is legitimate, proceed with evaluation
            model = PPO.load(model_path, device="cpu")
            
            # Import and run episode  
            from swarm.validator.forward import _run_episode
            result = _run_episode(task, uid, model)
        
        # Write result to file
        tmp_file = result_file + ".tmp"
        result_dict = asdict(result)
        
        # Add fake model information if detected
        if is_fake:
            result_dict['is_fake_model'] = True
            result_dict['fake_reason'] = fake_reason
            result_dict['inspection_results'] = inspection_results
        
        # Convert numpy types to Python types for JSON serialization
        for key, value in result_dict.items():
            if key == 'success':
                result_dict[key] = bool(value)
            elif hasattr(value, 'item'):
                result_dict[key] = value.item()
            elif isinstance(value, (int, float)) and hasattr(value, '__float__'):
                result_dict[key] = float(value)
        
        with open(tmp_file, 'w') as f:
            json.dump(result_dict, f)
        
        # Atomic rename
        os.rename(tmp_file, result_file)
        
        # Cleanup
        if 'model' in locals():
            del model
        gc.collect()
        
        sys.exit(0)
        
    except Exception as e:
        # Write error result with full traceback
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        error_result = {
            'uid': uid if 'uid' in locals() else 0,
            'success': False,
            'time_sec': 0.0,
            'energy': 0.0,
            'score': 0.0,
            'error': error_msg
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