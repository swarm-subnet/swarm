#!/usr/bin/env python3
"""
test_rpc.py

Local test script for RPC agent submissions. Tests your agent exactly 
like the validator does before you submit.

Usage:
    python tests/test_rpc.py swarm/submission_template/ --seed 42 --gui
    python tests/test_rpc.py swarm/submission_template/ --zip
"""

import argparse
import asyncio
import os
import shutil
import socket
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

try:
    import capnp
except ImportError:
    print("ERROR: pycapnp not installed. Install with: pip install pycapnp")
    sys.exit(1)

swarm_path = str(Path(__file__).resolve().parent.parent)
if swarm_path not in sys.path:
    sys.path.insert(0, swarm_path)

from swarm.constants import SIM_DT, HORIZON_SEC, SPEED_LIMIT
from swarm.core.drone import track_drone
from swarm.protocol import ValidationResult, MapTask
from swarm.utils.env_factory import make_env
from swarm.validator.reward import flight_reward
from swarm.validator.task_gen import random_task
from gym_pybullet_drones.utils.enums import ActionType


def _check_folder_structure(folder: Path):
    required_files = ["main.py", "agent_server.py", "drone_agent.py", "agent.capnp"]
    missing = [f for f in required_files if not (folder / f).exists()]
    
    if missing:
        print(f"❌ Missing required files: {', '.join(missing)}")
        return False
    
    print("✅ Folder structure valid")
    return True


def _wait_for_port(port=8000, max_retries=15, retry_delay=1.0):
    for retry in range(max_retries):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            if result == 0:
                return True
        except Exception:
            pass
        time.sleep(retry_delay)
    return False


async def _run_episode(task: MapTask, uid: int, agent, gui=False):
    env = make_env(task, gui=gui)
    
    try:
        obs, _ = env.reset()
        await agent.reset()
        
        pos0 = np.asarray(task.start, dtype=float)
        t_sim = 0.0
        success = False
        speeds = []
        step_count = 0
        
        lo, hi = env.action_space.low.flatten(), env.action_space.high.flatten()
        last_pos = pos0
        
        cli_id = getattr(env, "CLIENT", getattr(env, "_cli", 0))
        frames_per_cam = max(1, int(round(1.0 / (SIM_DT * 60.0))))
        
        schema_file = Path(__file__).parent.parent / "swarm" / "submission_template" / "agent.capnp"
        agent_capnp = capnp.load(str(schema_file))
        
        while t_sim < task.horizon:
            try:
                observation = agent_capnp.Observation.new_message()
                
                if isinstance(obs, dict):
                    entries = observation.init("entries", len(obs))
                    for i, (key, value) in enumerate(obs.items()):
                        arr = np.asarray(value, dtype=np.float32)
                        entries[i].key = key
                        entries[i].tensor.data = arr.tobytes()
                        entries[i].tensor.shape = list(arr.shape)
                        entries[i].tensor.dtype = str(arr.dtype)
                else:
                    arr = np.asarray(obs, dtype=np.float32)
                    entry = observation.init("entries", 1)[0]
                    entry.key = "__value__"
                    entry.tensor.data = arr.tobytes()
                    entry.tensor.shape = list(arr.shape)
                    entry.tensor.dtype = str(arr.dtype)
                
                action_response = await agent.act(observation)
                action = np.frombuffer(
                    action_response.action.data,
                    dtype=np.dtype(action_response.action.dtype)
                ).reshape(tuple(action_response.action.shape))
            except Exception as e:
                print(f"⚠️  Action error at t={t_sim:.2f}s: {e}")
                action = np.zeros(5, dtype=np.float32)
            
            act = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), lo, hi)
            
            if hasattr(env, 'ACT_TYPE') and hasattr(env, 'SPEED_LIMIT'):
                if env.ACT_TYPE == ActionType.VEL and env.SPEED_LIMIT:
                    n = max(np.linalg.norm(act[:3]), 1e-6)
                    scale = min(1.0, SPEED_LIMIT / n)
                    act[:3] *= scale
                    act = np.clip(act, lo, hi)
            
            prev = last_pos
            obs, _r, terminated, truncated, info = env.step(act[None, :])
            last_pos = env._getDroneStateVector(0)[0:3]
            
            speed = np.linalg.norm(last_pos - prev) / SIM_DT
            speeds.append(speed)
            
            t_sim += SIM_DT
            
            if gui and step_count % frames_per_cam == 0:
                try:
                    track_drone(cli=cli_id, drone_id=env.DRONE_IDS[0])
                except Exception:
                    pass
            
            if gui:
                time.sleep(SIM_DT)
            
            if terminated or truncated:
                success = info.get("success", False)
                break
            
            step_count += 1
        
        if not gui:
            env.close()
        
        score = flight_reward(success=success, t=t_sim, horizon=task.horizon, task=task)
        avg_speed = np.mean(speeds) if speeds else 0.0
        
        return ValidationResult(uid, success, t_sim, score), avg_speed
    
    finally:
        try:
            env.close()
        except Exception:
            pass


async def _test_rpc_agent(submission_folder: Path, task: MapTask, gui=False):
    agent_process = subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=str(submission_folder),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ.copy()
    )
    
    try:
        print("🔄 Starting RPC server on port 8000...")
        
        if not _wait_for_port(8000, max_retries=15, retry_delay=1.0):
            print("❌ RPC server failed to start within 15 seconds")
            return None, 0.0
        
        if agent_process.poll() is not None:
            stdout, stderr = agent_process.communicate()
            print(f"❌ Agent process crashed:")
            print(stderr.decode())
            return None, 0.0
        
        print("✅ RPC server started")
        print("✅ Connecting via Cap'n Proto...")
        
        async with capnp.kj_loop():
            stream = await capnp.AsyncIoStream.create_connection(host="localhost", port=8000)
            client = capnp.TwoPartyClient(stream)
            schema_file = Path(__file__).parent.parent / "swarm" / "submission_template" / "agent.capnp"
            agent_capnp = capnp.load(str(schema_file))
            agent = client.bootstrap().cast_as(agent_capnp.Agent)
            
            ping_response = await agent.ping("test")
            if ping_response.response != "pong":
                print(f"❌ RPC ping test failed: got {ping_response.response}")
                return None, 0.0
            
            print("✅ RPC connection established")
            print(f"🚁 Running evaluation (seed: {task.map_seed})...")
            
            result, avg_speed = await _run_episode(task, uid=0, agent=agent, gui=gui)
            
            return result, avg_speed
    
    finally:
        agent_process.terminate()
        try:
            agent_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            agent_process.kill()
            agent_process.wait()


def main():
    parser = argparse.ArgumentParser(
        description="Test RPC agent submission locally (exactly like validator)"
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Path to submission folder (containing main.py, agent_server.py, etc.)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for task generation (default: 42)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Show PyBullet GUI during evaluation",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Create submission.zip after testing",
    )
    args = parser.parse_args()
    
    if not args.folder.exists():
        print(f"❌ Folder not found: {args.folder}")
        sys.exit(1)
    
    if not args.folder.is_dir():
        print(f"❌ Not a directory: {args.folder}")
        sys.exit(1)
    
    print("🔍 Testing RPC Agent Submission\n")
    
    if not _check_folder_structure(args.folder):
        sys.exit(1)
    
    task = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC, seed=args.seed)
    
    print(f"\n📋 Task Details:")
    print(f"   Start: {task.start}")
    print(f"   Goal:  {task.goal}")
    print(f"   Seed:  {task.map_seed}")
    print(f"   Horizon: {task.horizon}s\n")
    
    result, avg_speed = asyncio.run(_test_rpc_agent(args.folder, task, gui=args.gui))
    
    if result is None:
        print("\n❌ Test failed - agent could not be evaluated")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)
    print(f"Success    : {'✅ True' if result.success else '❌ False'}")
    print(f"Time       : {result.time_sec:.2f}s")
    print(f"Score      : {result.score:.3f}")
    print(f"Avg Speed  : {avg_speed:.3f} m/s")
    print("="*60)
    
    if result.success and result.score > 0.7:
        print("\n🎉 Your agent is ready for submission!")
    elif result.success:
        print("\n✅ Mission successful, but try to improve speed for higher scores!")
    else:
        print("\n⚠️  Mission failed. Debug and try again.")
    
    if args.zip:
        _create_submission_zip(args.folder)
    
    sys.exit(0 if result.success else 1)


def _create_submission_zip(folder: Path):
    submission_dir = Path(__file__).parent.parent / "Submission"
    submission_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = submission_dir / "submission.zip"
    
    miner_files = ["drone_agent.py", "requirements.txt", "ppo_policy.zip"]
    model_extensions = [".pt", ".pth", ".onnx", ".pkl", ".h5", ".weights"]
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file in folder.iterdir():
            if file.is_file():
                if file.name in miner_files:
                    zf.write(file, file.name)
                elif any(file.name.endswith(ext) for ext in model_extensions):
                    zf.write(file, file.name)
    
    print(f"\n📦 Created submission: {zip_path}")
    print("   Contents:")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for info in zf.infolist():
            size_kb = info.file_size / 1024
            print(f"   - {info.filename} ({size_kb:.1f} KB)")
    
    print(f"\n✅ Submission ready at: {zip_path}")
    print("   Miner will read from: Submission/submission.zip")


if __name__ == "__main__":
    main()
