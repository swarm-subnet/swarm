#!/usr/bin/env python3
"""
Standalone evaluator script for RPC agent evaluation.
All submissions must be RPC agents with main.py entry point.
"""

import sys
import os
import json
import gc
import resource
from pathlib import Path
import subprocess
import zipfile

swarm_path = str(Path(__file__).resolve().parent.parent.parent)
if swarm_path not in sys.path:
    sys.path.insert(0, swarm_path)

from dataclasses import asdict
from swarm.protocol import MapTask, ValidationResult
from swarm.constants import SPEED_LIMIT
from gym_pybullet_drones.utils.enums import ActionType


def _evaluate_with_rpc(task: MapTask, uid: int, model_path: Path) -> ValidationResult:
    import shutil
    import time
    import asyncio
    
    submission_dir = Path("/tmp") / f"submission_{uid}_{int(time.time())}"
    submission_dir.mkdir(exist_ok=True)
    
    try:
        template_dir = Path(swarm_path) / "swarm" / "submission_template"
        
        with zipfile.ZipFile(model_path, 'r') as zf:
            zf.extractall(submission_dir)
        
        shutil.copy(template_dir / "agent.capnp", submission_dir)
        shutil.copy(template_dir / "agent_server.py", submission_dir)
        shutil.copy(template_dir / "main.py", submission_dir)
        
        requirements_file = submission_dir / "requirements.txt"
        if requirements_file.exists():
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                    capture_output=True,
                    timeout=60,
                    check=False
                )
            except Exception:
                pass
        
        agent_process = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=str(submission_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ.copy()
        )
        
        max_retries = 10
        connected = False
        
        for retry in range(max_retries):
            if agent_process.poll() is not None:
                return ValidationResult(uid, False, 0.0, 0.0)
            
            time.sleep(1)
            
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.5)
                result = sock.connect_ex(('localhost', 8000))
                sock.close()
                if result == 0:
                    connected = True
                    break
            except Exception:
                pass
        
        if not connected:
            agent_process.terminate()
            agent_process.wait(timeout=5)
            return ValidationResult(uid, False, 0.0, 0.0)
        
        try:
            import capnp
            schema_file = Path(__file__).parent.parent / "submission_template" / "agent.capnp"
            agent_capnp = capnp.load(str(schema_file))
            
            async def run_evaluation():
                async with capnp.kj_loop():
                    client = capnp.TwoPartyClient("localhost:8000")
                    agent = client.bootstrap().cast_as(agent_capnp.Agent)
                    
                    ping_result = await agent.ping("test")
                    if ping_result != "pong":
                        raise RuntimeError("RPC ping failed")
                    
                    import numpy as _np
                    from swarm.utils.env_factory import make_env
                    from swarm.constants import SIM_DT
                    from gym_pybullet_drones.utils.enums import ActionType
                    
                    env = make_env(task, gui=False)
                    
                    try:
                        obs, _ = env.reset()
                        
                        pos0 = _np.asarray(task.start, dtype=float)
                        t_sim = 0.0
                        success = False
                        
                        lo, hi = env.action_space.low.flatten(), env.action_space.high.flatten()
                        last_pos = pos0
                        
                        while t_sim < task.horizon:
                            try:
                                obs_tensor = agent_capnp.Tensor.new_message()
                                obs_tensor.data = obs.tobytes()
                                obs_tensor.shape = list(obs.shape)
                                obs_tensor.dtype = str(obs.dtype)
                                
                                observation = agent_capnp.Observation.new_message()
                                entry = observation.init("entries", 1)[0]
                                entry.key = "__value__"
                                entry.tensor = obs_tensor
                                
                                action_tensor = await agent.act(observation)
                                action = _np.frombuffer(
                                    action_tensor.data,
                                    dtype=_np.dtype(action_tensor.dtype)
                                ).reshape(tuple(action_tensor.shape))
                            except Exception:
                                action = _np.zeros(4, dtype=_np.float32)
                            
                            act = _np.clip(_np.asarray(action, dtype=_np.float32).reshape(-1), lo, hi)
                            
                            if hasattr(env, 'ACT_TYPE') and hasattr(env, 'SPEED_LIMIT'):
                                if env.ACT_TYPE == ActionType.VEL and env.SPEED_LIMIT:
                                    n = max(_np.linalg.norm(act[:3]), 1e-6)
                                    scale = min(1.0, SPEED_LIMIT / n)
                                    act[:3] *= scale
                                    act = _np.clip(act, lo, hi)
                            
                            prev = last_pos
                            obs, _r, terminated, truncated, info = env.step(act[None, :])
                            last_pos = env._getDroneStateVector(0)[0:3]
                            
                            t_sim += SIM_DT
                            if terminated or truncated:
                                success = info.get("success", False)
                                break
                        
                        from swarm.validator.reward import flight_reward
                        score = flight_reward(
                            success=success,
                            t=t_sim,
                            horizon=task.horizon,
                            task=task,
                        )
                        
                        return ValidationResult(uid, success, t_sim, score)
                    finally:
                        try:
                            env.close()
                        except Exception:
                            pass
            
            result = asyncio.run(run_evaluation())
        except Exception:
            result = ValidationResult(uid, False, 0.0, 0.0)
        finally:
            try:
                agent_process.terminate()
                agent_process.wait(timeout=5)
            except Exception:
                pass
        
        return result
    finally:
        try:
            shutil.rmtree(submission_dir, ignore_errors=True)
        except Exception:
            pass


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
                score=0.0
            )
        elif model_status == "missing_metadata":
            # Return rejection result (zero score but no blacklist)
            result = ValidationResult(
                uid=uid,
                success=False,
                time_sec=0.0,
                score=0.0
            )
        elif verify_only_mode:
            # VERIFICATION-ONLY MODE: Model is legitimate, return success without evaluation
            result = ValidationResult(
                uid=uid,
                success=True,
                time_sec=0.0,
                score=0.0  # Not applicable for verification-only
            )
        else:
            model_path_obj = Path(model_path)
            result = _evaluate_with_rpc(task, uid, model_path_obj)
        
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