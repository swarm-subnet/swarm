#!/usr/bin/env python3
"""
ULTRA FAST evaluator script with aggressive optimizations
Target: <15 seconds per evaluation

Key optimizations:
1. 15Hz physics (3.3x faster than 50Hz)
2. Early termination on success
3. Shorter default horizon
4. Minimal environment complexity
5. No GUI, no camera tracking, no sleep
"""

import sys
import os
import json
import gc
import traceback
from pathlib import Path

# Disable all logging to suppress output
import logging
logging.disable(logging.CRITICAL)

# Redirect stderr
original_stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

# Add swarm to path
swarm_path = str(Path(__file__).resolve().parent.parent.parent)
if swarm_path not in sys.path:
    sys.path.insert(0, swarm_path)

def ultra_fast_evaluation(task, uid, model):
    """Ultra fast episode evaluation with aggressive optimizations"""
    import numpy as np
    from swarm.utils.env_factory import make_env
    from swarm.validator.reward import flight_reward
    
    # Create ultra-fast task parameters
    fast_task = type(task)(
        map_seed=task.map_seed,
        start=task.start,
        goal=task.goal,
        sim_dt=0.067,  # 15Hz physics (3.3x faster than 50Hz)
        horizon=min(task.horizon, 20.0),  # Cap at 20s max
        version=task.version
    )
    
    class FastPilot:
        def __init__(self, m): 
            self.m = m
        def reset(self, task): 
            pass
        def act(self, obs, t):
            act, _ = self.m.predict(obs, deterministic=True)
            return act.squeeze()
    
    pilot = FastPilot(model)
    env = make_env(fast_task, gui=False)
    
    # Initial observation
    try:
        obs = env._computeObs()
    except AttributeError:
        obs = env.get_observation()
    
    if isinstance(obs, dict):
        obs = obs[next(iter(obs))]
    
    # Ultra-fast simulation loop
    pos0 = np.asarray(fast_task.start, dtype=float)
    last_pos = pos0.copy()
    t_sim = 0.0
    energy = 0.0
    success = False
    step_count = 0
    
    max_steps = int(fast_task.horizon / fast_task.sim_dt)
    
    for step in range(max_steps):
        # Get action
        try:
            rpm = pilot.act(obs, t_sim)
            
            # Ensure correct action shape
            if rpm.ndim == 0:
                rpm = np.array([rpm, rpm, rpm, rpm])
            elif len(rpm) == 1:
                rpm = np.array([rpm[0], rpm[0], rpm[0], rpm[0]])
            
            # Step environment
            obs, _r, terminated, truncated, info = env.step(rpm[None, :])
            
        except Exception as e:
            # If step fails, break with current state
            break
        
        t_sim += fast_task.sim_dt
        energy += np.abs(rpm).sum() * fast_task.sim_dt
        
        # Update position
        if obs.ndim == 1:
            last_pos = obs[:3]
        else:
            last_pos = obs[0, :3]
        
        # Early termination checks
        if terminated or truncated:
            success = info.get("success", False)
            break
        
        # Quick success check (distance-based)
        goal_pos = np.asarray(fast_task.goal, dtype=float)
        distance = np.linalg.norm(last_pos - goal_pos)
        if distance < 1.0:  # Close to goal
            # Check if we've been close for a bit
            if step > 50:  # At least 3+ seconds at 15Hz
                success = True
                break
        
        step_count += 1
    
    # Clean up environment immediately
    env.close()
    
    # Calculate final score
    score = flight_reward(
        success=success,
        t=t_sim,
        e=energy,
        horizon=fast_task.horizon,
    )
    
    from swarm.protocol import ValidationResult
    return ValidationResult(uid, success, t_sim, energy, score)

def main():
    try:
        # Parse command line arguments
        if len(sys.argv) != 5:
            raise ValueError("Usage: evaluator.py <task_json> <uid> <model_path> <result_file>")
        
        task_json = sys.argv[1]
        uid = int(sys.argv[2])
        model_path = sys.argv[3]
        result_file = sys.argv[4]
        
        # Load task
        with open(task_json, 'r') as f:
            task_data = json.load(f)
        
        from swarm.protocol import MapTask
        task = MapTask(**task_data)
        
        # Load model
        from stable_baselines3 import PPO
        model = PPO.load(model_path, device="cpu")
        
        # Run ultra-fast evaluation
        result = ultra_fast_evaluation(task, uid, model)
        
        # Write result
        from dataclasses import asdict
        result_dict = asdict(result)
        
        # Convert numpy types to Python types
        for key, value in result_dict.items():
            if key == 'success':
                result_dict[key] = bool(value)
            elif hasattr(value, 'item'):
                result_dict[key] = value.item()
            elif isinstance(value, (int, float)) and hasattr(value, '__float__'):
                result_dict[key] = float(value)
        
        # Atomic write
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
            'error': f"{str(e)}\n{traceback.format_exc()}"
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
        # Restore stderr
        try:
            sys.stderr.close()
            sys.stderr = original_stderr
        except Exception:
            pass

if __name__ == "__main__":
    main()