import asyncio
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import zipfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import bittensor as bt
import numpy as np

from swarm.protocol import MapTask, ValidationResult
from swarm.constants import EVAL_TIMEOUT_SEC, SIM_DT, SPEED_LIMIT
from swarm.utils.hash import sha256sum
from swarm.core.model_verify import add_to_blacklist
from swarm.validator.reward import flight_reward
from gym_pybullet_drones.utils.enums import ActionType


class DockerSecureEvaluator:
    """Docker-based secure model evaluation"""
    
    _instance = None
    _base_ready = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Only initialize attributes on first instantiation
        if not hasattr(self, 'base_image'):
            self.base_image = "swarm_evaluator_base:latest"
            self.last_fake_model_info = None
        
        if not DockerSecureEvaluator._base_ready:
            self._setup_base_container()
            DockerSecureEvaluator._base_ready = self.base_ready
    
    def _check_docker_available(self):
        """Check if Docker is installed and available"""
        try:
            # Check if Docker command exists
            result = subprocess.run(["docker", "--version"], 
                                  capture_output=True, text=True, check=True)
            bt.logging.info(f"Docker found: {result.stdout.strip()}")
            return True
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            bt.logging.error("🐳 Docker not found! Please install Docker manually.")
            bt.logging.error("📖 See installation instructions in swarm/requirements.txt")
            return False
    
    def _setup_base_container(self):
        """Build base Docker image with all dependencies"""
        try:
            # Check if Docker is installed
            if not self._check_docker_available():
                bt.logging.error("❌ Docker is required but not installed")
                self.base_ready = False
                DockerSecureEvaluator._base_ready = False
                return
            
            # Aggressive cleanup to prevent disk bloat from dangling images/containers
            try:
                # Remove stopped containers and any leftover eval/verify containers
                subprocess.run(["docker", "container", "prune", "-f"], capture_output=True)
                subprocess.run("docker rm -f $(docker ps -aq --filter=name=swarm_eval_)", shell=True, capture_output=True)
                subprocess.run("docker rm -f $(docker ps -aq --filter=name=swarm_verify_)", shell=True, capture_output=True)
            except Exception:
                pass
            try:
                # Remove only dangling images (not all unused images)
                # This prevents accidentally removing the base image
                subprocess.run(["docker", "image", "prune", "-f"], capture_output=True)
                # Remove builder cache to reclaim space
                subprocess.run(["docker", "builder", "prune", "-f"], capture_output=True)
                subprocess.run(["docker", "volume", "prune", "-f"], capture_output=True)
            except Exception:
                pass
            
            dockerfile_path = Path(__file__).parent / "Dockerfile"
            # Build context should be the parent of swarm package
            build_context = Path(__file__).parent.parent.parent.parent
            
            # Build base image (always fresh to get latest swarm scripts)
            cmd = [
                "docker", "build",
                "--no-cache",
                "-t", self.base_image,
                "-f", str(dockerfile_path),
                str(build_context)
            ]
            
            bt.logging.info("Building base Docker image (this may take a few minutes)...")
            bt.logging.debug(f"Docker build command: {' '.join(cmd)}")
            
            # Run with real-time output so user can see progress
            result = subprocess.run(cmd, text=True)
            
            if result.returncode == 0:
                self.base_ready = True
                DockerSecureEvaluator._base_ready = True
                bt.logging.info("✅ Base Docker image ready")
            else:
                bt.logging.error(f"❌ Docker build failed with return code: {result.returncode}")
                self.base_ready = False
                DockerSecureEvaluator._base_ready = False
                
        except Exception as e:
            bt.logging.error(f"Failed to setup Docker environment: {e}")
            self.base_ready = False
            DockerSecureEvaluator._base_ready = False
    
    def _evaluate_with_rpc_sync(
        self,
        task: MapTask,
        uid: int,
        rpc_port: int
    ) -> ValidationResult:
        """Synchronous RPC evaluation - runs in separate thread to avoid kj_loop conflicts"""
        import capnp
        schema_file = Path(__file__).parent.parent.parent / "submission_template" / "agent.capnp"
        agent_capnp = capnp.load(str(schema_file))
        
        async def run_evaluation():
            async with capnp.kj_loop():
                stream = await capnp.AsyncIoStream.create_connection(host="localhost", port=rpc_port)
                client = capnp.TwoPartyClient(stream)
                agent = client.bootstrap().cast_as(agent_capnp.Agent)
                
                ping_response = await agent.ping("test")
                if ping_response.response != "pong":
                    raise RuntimeError("RPC ping failed")
                
                from swarm.utils.env_factory import make_env
                
                env = make_env(task, gui=False)
                
                try:
                    obs, _ = env.reset()
                    
                    pos0 = np.asarray(task.start, dtype=float)
                    t_sim = 0.0
                    success = False
                    
                    lo, hi = env.action_space.low.flatten(), env.action_space.high.flatten()
                    last_pos = pos0
                    
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
                        except Exception:
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
                        
                        t_sim += SIM_DT
                        if terminated or truncated:
                            success = info.get("success", False)
                            break
                    
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
        
        # Run in fresh event loop to avoid kj_loop conflicts
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(run_evaluation())
        finally:
            loop.close()
    
    async def _evaluate_with_rpc_host(
        self,
        task: MapTask,
        uid: int,
        rpc_port: int
    ) -> ValidationResult:
        """Async wrapper that runs RPC evaluation in thread pool"""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._evaluate_with_rpc_sync,
                task,
                uid,
                rpc_port
            )
            return result
        except (BrokenPipeError, ConnectionResetError, ConnectionRefusedError, OSError) as e:
            bt.logging.debug(f"RPC connection error for UID {uid}: {e}")
            return ValidationResult(uid, False, 0.0, 0.0)
        except Exception as e:
            import traceback
            bt.logging.debug(f"RPC evaluation failed for UID {uid}: {e}\n{traceback.format_exc()}")
            return ValidationResult(uid, False, 0.0, 0.0)
    
    def _find_free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    async def evaluate_model(
        self, 
        task: MapTask, 
        uid: int, 
        model_path: Path
    ) -> ValidationResult:
        if not model_path.is_file():
            bt.logging.warning(f"Model path is missing or not a file: {model_path}")
            return ValidationResult(uid, False, 0.0, 0.0)
        
        if not DockerSecureEvaluator._base_ready:
            bt.logging.warning(f"Docker not ready for UID {uid}")
            return ValidationResult(uid, False, 0.0, 0.0)
        
        try:
            check_result = subprocess.run(
                ["docker", "images", "-q", self.base_image],
                capture_output=True,
                text=True
            )
            if not check_result.stdout.strip():
                bt.logging.error(f"Base image {self.base_image} not found - rebuilding...")
                self._setup_base_container()
                if not DockerSecureEvaluator._base_ready:
                    bt.logging.error(f"Failed to rebuild base image for UID {uid}")
                    return ValidationResult(uid, False, 0.0, 0.0)
        except Exception as e:
            bt.logging.warning(f"Failed to check for base image: {e}")
            return ValidationResult(uid, False, 0.0, 0.0)
        
        self.last_fake_model_info = None
        
        try:
            with zipfile.ZipFile(model_path, 'r') as zf:
                if "drone_agent.py" not in zf.namelist():
                    bt.logging.warning(f"Model {uid} missing drone_agent.py - RPC agent required")
                    return ValidationResult(uid, False, 0.0, 0.0)
        except Exception as e:
            bt.logging.warning(f"Failed to validate model {uid}: {e}")
            return ValidationResult(uid, False, 0.0, 0.0)
        
        container_name = f"swarm_eval_{uid}_{int(time.time() * 1000)}"
        host_port = self._find_free_port()
        
        bt.logging.info(f"🐳 Starting Docker container for UID {uid} evaluation...")
        
        tmpdir = None
        try:
            tmpdir = tempfile.mkdtemp()
            os.chown(tmpdir, 1000, 1000)
            os.chmod(tmpdir, 0o755)
            
            submission_dir = Path(tmpdir) / "submission"
            submission_dir.mkdir(exist_ok=True)
            os.chown(submission_dir, 1000, 1000)
            os.chmod(submission_dir, 0o755)
            
            template_dir = Path(__file__).parent.parent.parent / "submission_template"
            
            with zipfile.ZipFile(model_path, 'r') as zf:
                zf.extractall(submission_dir)
            
            shutil.copy(template_dir / "agent.capnp", submission_dir)
            shutil.copy(template_dir / "agent_server.py", submission_dir)
            shutil.copy(template_dir / "main.py", submission_dir)
            
            for f in submission_dir.iterdir():
                os.chown(f, 1000, 1000)
                os.chmod(f, 0o644)
            
            miner_requirements = submission_dir / "requirements.txt"
            has_requirements = miner_requirements.exists()
            
            if has_requirements:
                bt.logging.info(f"📦 Miner has requirements.txt for UID {uid}")
                startup_script = submission_dir / "startup.sh"
                with open(startup_script, 'w') as f:
                    f.write("#!/bin/bash\n")
                    f.write("pip install --no-cache-dir --user -r /workspace/submission/requirements.txt 2>/dev/null\n")
                    f.write("exec python /workspace/submission/main.py\n")
                os.chmod(startup_script, 0o755)
                os.chown(startup_script, 1000, 1000)
                
                cmd = [
                    "docker", "run",
                    "--rm",
                    "-d",
                    "--name", container_name,
                    "--user", "1000:1000",
                    "--memory=6g",
                    "--cpus=2",
                    "--pids-limit=50",
                    "--ulimit", "nofile=256:256",
                    "--ulimit", "fsize=524288000:524288000",
                    "--security-opt", "no-new-privileges",
                    "--network", "bridge",
                    "-p", f"{host_port}:8000",
                    "-v", f"{submission_dir}:/workspace/submission:rw",
                    self.base_image,
                    "bash", "/workspace/submission/startup.sh"
                ]
            else:
                cmd = [
                    "docker", "run",
                    "--rm",
                    "-d",
                    "--name", container_name,
                    "--user", "1000:1000",
                    "--memory=6g",
                    "--cpus=2",
                    "--pids-limit=20",
                    "--ulimit", "nofile=64:64",
                    "--ulimit", "fsize=524288000:524288000",
                    "--security-opt", "no-new-privileges",
                    "--network", "bridge",
                    "-p", f"{host_port}:8000",
                    "-v", f"{submission_dir}:/workspace/submission:ro",
                    self.base_image,
                    "python", "/workspace/submission/main.py"
                ]
            
            bt.logging.debug(f"Docker command for UID {uid}: {' '.join(cmd[:10])}...")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                bt.logging.debug(f"Container start failed for UID {uid}: {result.stderr[:300]}")
                return ValidationResult(uid, False, 0.0, 0.0)
            
            bt.logging.debug(f"Container {container_name} started, waiting for RPC on port {host_port}...")
            
            max_retries = 30 if has_requirements else 15
            connected = False
            
            for retry in range(max_retries):
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(0.5)
                    conn_result = sock.connect_ex(('localhost', host_port))
                    sock.close()
                    if conn_result == 0:
                        connected = True
                        bt.logging.debug(f"Port {host_port} open after {retry + 1} retries")
                        break
                except Exception:
                    pass
                await asyncio.sleep(1)
            
            if connected:
                bt.logging.debug(f"Waiting 10s for RPC server to stabilize...")
                await asyncio.sleep(10)
            
            if not connected:
                try:
                    logs = subprocess.run(
                        ["docker", "logs", container_name],
                        capture_output=True, text=True, timeout=5
                    )
                    bt.logging.debug(f"Container logs for UID {uid}: {logs.stdout[:500] if logs.stdout else 'empty'}")
                    bt.logging.debug(f"Container stderr for UID {uid}: {logs.stderr[:500] if logs.stderr else 'empty'}")
                except Exception:
                    pass
                subprocess.run(["docker", "kill", container_name], capture_output=True)
                subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
                bt.logging.warning(f"RPC server failed to start for UID {uid}")
                return ValidationResult(uid, False, 0.0, 0.0)
            
            try:
                result = await asyncio.wait_for(
                    self._evaluate_with_rpc_host(task, uid, host_port),
                    timeout=EVAL_TIMEOUT_SEC
                )
                
                score = float(result.score)
                if not (0.0 <= score <= 1.0):
                    bt.logging.error(f"🚫 Invalid score {score} for UID {uid} - blacklisting model")
                    model_hash = sha256sum(model_path)
                    add_to_blacklist(model_hash)
                    return ValidationResult(uid, False, 0.0, 0.0)
                
                if result.success and score == 0.0:
                    result.score = 0.01
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"{timestamp} 🔍 DEBUG: UID {uid} result: success={result.success}, time={result.time_sec:.2f}, score={result.score:.4f}")
                
                bt.logging.info(f"🏁 Ending Docker container for UID {uid} - evaluation completed successfully")
                
                return result
                
            except asyncio.TimeoutError:
                bt.logging.warning(f"Evaluation timeout for UID {uid}")
                return ValidationResult(uid, False, 0.0, 0.0)
            finally:
                try:
                    subprocess.run(["docker", "kill", container_name], capture_output=True)
                    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
                except Exception:
                    pass
        
        except Exception as e:
            bt.logging.warning(f"Docker evaluation failed for UID {uid}: {e}")
            try:
                subprocess.run(["docker", "kill", container_name], capture_output=True)
                subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
            except Exception:
                pass
            bt.logging.info(f"🏁 Ending Docker container for UID {uid} - evaluation failed with error")
        finally:
            try:
                subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
            except Exception:
                pass
            if tmpdir:
                try:
                    shutil.rmtree(tmpdir, ignore_errors=True)
                except Exception:
                    pass
            try:
                subprocess.run(["docker", "image", "prune", "-f"], capture_output=True)
                subprocess.run(["docker", "builder", "prune", "-f"], capture_output=True)
            except Exception:
                pass
        
        bt.logging.info(f"🏁 Ending Docker container for UID {uid} - returning default result")
        return ValidationResult(uid, False, 0.0, 0.0)
    
    
    def cleanup(self):
        """Clean up any orphaned containers and prune unused images/cache"""
        try:
            # List all swarm evaluation containers
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", "name=swarm_eval_", "--format", "{{.Names}}"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and result.stdout:
                containers = result.stdout.strip().split('\n')
                for container in containers:
                    if container:
                        subprocess.run(["docker", "rm", "-f", container], capture_output=True)
                        bt.logging.debug(f"Cleaned up orphaned container: {container}")

            # Also clean up verification containers
            result_verify = subprocess.run(
                ["docker", "ps", "-a", "--filter", "name=swarm_verify_", "--format", "{{.Names}}"],
                capture_output=True,
                text=True
            )
            if result_verify.returncode == 0 and result_verify.stdout:
                containers_v = result_verify.stdout.strip().split('\n')
                for container in containers_v:
                    if container:
                        subprocess.run(["docker", "rm", "-f", container], capture_output=True)
                        bt.logging.debug(f"Cleaned up orphaned verify container: {container}")

            # Prune only dangling images and builder cache to reclaim disk space
            # IMPORTANT: Only prune dangling images, NOT all unused images (removed -a flag)
            # This preserves the base image
            subprocess.run(["docker", "image", "prune", "-f"], capture_output=True)
            subprocess.run(["docker", "builder", "prune", "-f"], capture_output=True)
            subprocess.run(["docker", "volume", "prune", "-f"], capture_output=True)
        
        except Exception as e:
            bt.logging.warning(f"Container cleanup failed: {e}")

