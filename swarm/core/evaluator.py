#!/usr/bin/env python3
"""
Standalone evaluator script for subprocess execution.
Now supports a two-process design (within the same container) to isolate the
model from the environment. In regular evaluation mode, the parent process runs
the environment loop and spawns a child "MODEL_WORKER" process that only loads
the model and returns actions. The two processes exchange strictly numerical
data (observations and actions) via stdio JSON lines.
"""

import sys
import os
import json
import gc
import resource
from pathlib import Path
import subprocess
import json as _json
import select

# Add swarm to path BEFORE importing swarm modules
swarm_path = str(Path(__file__).resolve().parent.parent.parent)
if swarm_path not in sys.path:
    sys.path.insert(0, swarm_path)

from dataclasses import asdict
from swarm.protocol import MapTask, ValidationResult
from swarm.core.secure_loader import secure_load_ppo


def _spawn_model_worker(model_path: str) -> subprocess.Popen:
    """Start a model-only worker process that returns actions for given obs.

    The worker is invoked as: this_script MODEL_WORKER <model_path>
    Communication protocol: parent writes one JSON line {"obs": [...]} per step;
    worker replies with one JSON line {"act": [...]}.
    """
    cmd = [sys.executable, str(Path(__file__).resolve()), "MODEL_WORKER", model_path]
    # Use unbuffered I/O to minimize latency
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    return proc


def _model_predict_via_io(proc: subprocess.Popen, obs_array, timeout_s: float = 1.0):
    """Send observation to the model worker and get an action back as list.

    Returns a Python list representing the action on success; raises RuntimeError
    on timeout or protocol errors.
    """
    if proc.poll() is not None:
        raise RuntimeError("model worker terminated")

    # Ensure 1-D list of floats
    try:
        obs_list = obs_array.tolist() if hasattr(obs_array, "tolist") else list(obs_array)
    except Exception:
        raise RuntimeError("invalid observation format")

    line = _json.dumps({"obs": obs_list}) + "\n"
    try:
        proc.stdin.write(line)
        proc.stdin.flush()
    except Exception as e:
        raise RuntimeError(f"failed to write to model worker: {e}")

    # Wait for one line with timeout
    rlist, _, _ = select.select([proc.stdout], [], [], timeout_s)
    if not rlist:
        raise RuntimeError("model worker timeout")
    reply = proc.stdout.readline()
    if not reply:
        raise RuntimeError("model worker closed pipe")
    try:
        payload = _json.loads(reply)
        act = payload.get("act", None)
        if not isinstance(act, list) or len(act) == 0:
            raise ValueError("bad action payload")
        return act
    except Exception as e:
        raise RuntimeError(f"invalid model worker reply: {e}")


def _wait_model_ready(proc: subprocess.Popen, timeout_s: float = 60.0) -> None:
    """Block until the model worker prints a readiness line {"ready": true}."""
    elapsed = 0.0
    step = 0.1
    while elapsed < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError("model worker terminated prematurely")
        rlist, _, _ = select.select([proc.stdout], [], [], step)
        if rlist:
            line = proc.stdout.readline()
            if not line:
                continue
            try:
                msg = _json.loads(line)
                if msg.get("ready", False) is True:
                    return
            except Exception:
                # ignore non-ready lines
                pass
        elapsed += step
    raise RuntimeError("model worker ready timeout")


def main():
    """Main evaluator entry point"""
    
    # Disable all logging to prevent any logging threads in parent or worker
    import logging
    logging.disable(logging.CRITICAL)
    
    # Redirect stderr to suppress output (but keep for debugging in verify mode)
    original_stderr = sys.stderr
    if len(sys.argv) > 1 and sys.argv[1] != "VERIFY_ONLY":
        sys.stderr = open(os.devnull, 'w')
    
    try:
        # Parse command line arguments - handle both regular evaluation and verification-only
        # Special worker mode: MODEL_WORKER <model_path>
        if len(sys.argv) >= 2 and sys.argv[1] == "MODEL_WORKER":
            # Child process that only loads the model and returns actions via stdio
            # Import here to keep parent process free from model code
            from stable_baselines3 import PPO
            from swarm.core.secure_loader import secure_load_ppo
            from swarm.utils.env_factory import make_env
            from swarm.validator.task_gen import random_task
            from swarm.constants import SIM_DT, HORIZON_SEC
            from pathlib import Path

            model_path = sys.argv[2]
            try:
                # Create minimal environment for policy initialization
                task = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC, seed=1)
                init_env = make_env(task, gui=False)
                try:
                    model = secure_load_ppo(Path(model_path), env=init_env, device="cpu")
                finally:
                    init_env.close()
            except Exception as e:
                sys.stderr.write(f"MODEL_WORKER_LOAD_ERROR: {e}\n")
                sys.stderr.flush()
                sys.exit(1)

            # Signal readiness
            sys.stdout.write(_json.dumps({"ready": True}) + "\n")
            sys.stdout.flush()

            # Serve an infinite loop over stdin
            for line in sys.stdin:
                line = line.strip()
                
                if not line:
                    continue
                try:
                    msg = _json.loads(line)
                    obs = msg.get("obs", None)
                    if not isinstance(obs, list) or len(obs) == 0:
                        raise ValueError("invalid obs")
                    import numpy as _np
                    obs_arr = _np.asarray(obs, dtype=_np.float32)
                    # SB3 can accept 1-D obs; ensure proper shape is handled inside predict
                    act, _ = model.predict(obs_arr, deterministic=True)
                    act_list = act.squeeze().tolist() if hasattr(act, "tolist") else list(act)
                    sys.stdout.write(_json.dumps({"act": act_list}) + "\n")
                    sys.stdout.flush()
                except Exception as e:
                    # Protocol error: respond with an empty action to keep protocol flowing
                    sys.stdout.write(_json.dumps({"act": []}) + "\n")
                    sys.stdout.flush()
            sys.exit(0)

        # Normal modes
        if len(sys.argv) != 5:
            raise ValueError("Usage: evaluator.py <task_json|VERIFY_ONLY> <uid> <model_path> <result_file>")

        first_arg = sys.argv[1]
        uid = int(sys.argv[2])
        model_path = sys.argv[3]
        result_file = sys.argv[4]
        
        # Check if this is verification-only mode
        verify_only_mode = (first_arg == "VERIFY_ONLY")
        
        # Clean verification mode logging
        if verify_only_mode:
            print(f"🔍 Verifying model for UID {uid}")
        
        if not verify_only_mode:
            task_json = first_arg
        
        # Set memory limits
        try:
            SUBPROC_MEM_MB = 8192
            rss_bytes = SUBPROC_MEM_MB * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (rss_bytes, rss_bytes))
            resource.setrlimit(resource.RLIMIT_DATA, (rss_bytes, rss_bytes))
        except Exception:
            pass
        
        # Parse task from JSON (only for regular evaluation)
        if not verify_only_mode:
            with open(task_json, 'r') as f:
                task_data = json.load(f)
            task = MapTask(**task_data)
        
        # First inspect model for fake indicators (safe within container)
        from swarm.core.model_verify import inspect_model_structure, classify_model_validity
        
        inspection_results = inspect_model_structure(Path(model_path))
        model_status, model_reason = classify_model_validity(inspection_results)
        
        # Clean inspection result logging
        if verify_only_mode:
            if model_status == "legitimate":
                status = "LEGITIMATE"
            elif model_status == "missing_metadata":
                status = "MISSING METADATA"
            else:  # fake
                status = "FAKE"
            print(f"📋 Model inspection: {status}" + (f" - {model_reason}" if model_status != "legitimate" else ""))
        
        if model_status == "fake":
            # Return fake model detection result (will be blacklisted)
            result = ValidationResult(
                uid=uid,
                success=False,
                time_sec=0.0,
                energy=0.0,
                score=0.0
            )
        elif model_status == "missing_metadata":
            # Return rejection result (zero score but no blacklist)
            result = ValidationResult(
                uid=uid,
                success=False,
                time_sec=0.0,
                energy=0.0,
                score=0.0
            )
        elif verify_only_mode:
            # VERIFICATION-ONLY MODE: Model is legitimate, return success without evaluation
            result = ValidationResult(
                uid=uid,
                success=True,
                time_sec=0.0,
                energy=0.0,
                score=0.0  # Not applicable for verification-only
            )
        else:
            # REGULAR EVALUATION MODE: Model is legitimate, evaluate via two-process IPC
            # Spawn untrusted model worker
            worker = _spawn_model_worker(model_path)
            # Wait up to 90s for worker to be ready (covers initial model load)
            _wait_model_ready(worker, timeout_s=90.0)

            # Build environment and roll out
            # Import here to avoid bringing SB3 into the parent
            import numpy as _np
            from swarm.utils.env_factory import make_env
            from swarm.constants import SIM_DT

            class _PilotIPC:
                def __init__(self, proc):
                    self.proc = proc
                def reset(self, task):
                    pass
                def act(self, obs, t):
                    act_list = _model_predict_via_io(self.proc, obs, timeout_s=3.0)
                    if not act_list:
                        # fallback to zeros with safe shape
                        return _np.zeros(4, dtype=_np.float32)
                    return _np.asarray(act_list, dtype=_np.float32)

            pilot = _PilotIPC(worker)
            env = make_env(task, gui=False)

            try:
                obs = env._computeObs()
                if isinstance(obs, dict):
                    obs = obs[next(iter(obs))]

                pos0 = _np.asarray(task.start, dtype=float)
                t_sim = 0.0
                energy = 0.0
                success = False
                step_count = 0
                
                lo, hi = env.action_space.low.flatten(), env.action_space.high.flatten()
                last_pos = pos0
                overspeed_streak = 0
                
                while t_sim < task.horizon:
                    act = _np.clip(_np.asarray(pilot.act(obs, t_sim), dtype=_np.float32).reshape(-1), lo, hi)
                    
                    # Apply speed scaling if persistent overspeed in VEL mode
                    if (hasattr(env, 'ACT_TYPE') and hasattr(env, 'SPEED_LIMIT') and overspeed_streak >= 2):
                        from gym_pybullet_drones.utils.enums import ActionType
                        if env.ACT_TYPE == ActionType.VEL and env.SPEED_LIMIT:
                            n = max(_np.linalg.norm(act[:3]), 1e-6)
                            scale = min(1.0, 0.9 / n)
                            act[:3] *= scale
                            act = _np.clip(act, lo, hi)
                    
                    prev = last_pos
                    obs, _r, terminated, truncated, info = env.step(act[None, :])
                    last_pos = env._getDroneStateVector(0)[0:3]
                    
                    # Update overspeed streak for next iteration
                    if hasattr(env, 'SPEED_LIMIT') and env.SPEED_LIMIT:
                        ratio = float(_np.linalg.norm(last_pos - prev) / SIM_DT) / env.SPEED_LIMIT
                        overspeed_streak = (overspeed_streak + 1) if ratio > 1.2 else 0
                    
                    t_sim += SIM_DT
                    energy += _np.abs(act).sum() * SIM_DT
                    if terminated or truncated:
                        success = info.get("success", False)
                        break
                    step_count += 1

                # Compute score using reward function
                from swarm.validator.reward import flight_reward
                score = flight_reward(
                    success=success,
                    t=t_sim,
                    e=energy,
                    horizon=task.horizon,
                )

                result = ValidationResult(uid, success, t_sim, energy, score)
            finally:
                try:
                    env.close()
                except Exception:
                    pass
                try:
                    # Terminate worker process
                    worker.terminate()
                except Exception:
                    pass
        
        # Write result to file
        tmp_file = result_file + ".tmp"
        result_dict = asdict(result)
        
        # Add model status information
        if model_status == "fake":
            result_dict['is_fake_model'] = True
            result_dict['fake_reason'] = model_reason
            result_dict['inspection_results'] = inspection_results
        elif model_status == "missing_metadata":
            result_dict['is_fake_model'] = False
            result_dict['missing_metadata'] = True
            result_dict['rejection_reason'] = model_reason
            result_dict['inspection_results'] = inspection_results
        else:
            # Legitimate model
            result_dict['is_fake_model'] = False
            
        # Clean result logging
        if verify_only_mode:
            print(f"✅ Verification complete")
        
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