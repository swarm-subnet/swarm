import asyncio
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import traceback
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

    def _calculate_docker_hash(self) -> str:
        import hashlib

        dockerfile = Path(__file__).parent / "Dockerfile"
        requirements = Path(__file__).parent / "docker-requirements.txt"
        swarm_pkg = Path(__file__).parent.parent.parent

        hasher = hashlib.sha256()

        if dockerfile.exists():
            hasher.update(dockerfile.read_bytes())
        if requirements.exists():
            hasher.update(requirements.read_bytes())
        if swarm_pkg.exists():
            for f in sorted(swarm_pkg.rglob("*.py")):
                try:
                    hasher.update(str(f.stat().st_mtime).encode())
                except Exception:
                    pass

        return hasher.hexdigest()[:16]

    def _should_rebuild_base_image(self) -> bool:
        current_hash = self._calculate_docker_hash()
        cache_file = Path("/tmp/swarm_docker_hash.txt")

        if cache_file.exists():
            cached_hash = cache_file.read_text().strip()
            if cached_hash == current_hash:
                result = subprocess.run(
                    ["docker", "images", "-q", self.base_image],
                    capture_output=True, text=True
                )
                if result.stdout.strip():
                    bt.logging.info(f"✅ Docker image unchanged (hash: {current_hash})")
                    return False

        cache_file.write_text(current_hash)
        return True

    def _setup_base_container(self):
        try:
            if not self._check_docker_available():
                bt.logging.error("❌ Docker is required but not installed")
                self.base_ready = False
                DockerSecureEvaluator._base_ready = False
                return

            if not self._should_rebuild_base_image():
                self.base_ready = True
                DockerSecureEvaluator._base_ready = True
                return

            bt.logging.info(f"🐳 Building base Docker image {self.base_image}...")

            try:
                subprocess.run(["docker", "container", "prune", "-f"], capture_output=True)
                subprocess.run("docker rm -f $(docker ps -aq --filter=name=swarm_eval_)", shell=True, capture_output=True)
                subprocess.run("docker rm -f $(docker ps -aq --filter=name=swarm_verify_)", shell=True, capture_output=True)
                subprocess.run(["docker", "image", "prune", "-f"], capture_output=True)
                subprocess.run(["docker", "volume", "prune", "-f"], capture_output=True)
            except Exception:
                pass

            dockerfile_path = Path(__file__).parent / "Dockerfile"
            build_context = Path(__file__).parent.parent.parent.parent

            cmd = [
                "docker", "build",
                "-t", self.base_image,
                "-f", str(dockerfile_path),
                str(build_context)
            ]

            bt.logging.info("Building base Docker image (this may take a few minutes)...")
            bt.logging.debug(f"Docker build command: {' '.join(cmd)}")

            env = os.environ.copy()
            env["DOCKER_BUILDKIT"] = "1"
            result = subprocess.run(cmd, text=True, env=env)

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
    
    def _log_container_failure(self, container_name: str, uid: int, failure_type: str):
        """Log detailed container information when evaluation fails"""
        try:
            logs = subprocess.run(
                ["docker", "logs", "--tail", "100", container_name],
                capture_output=True, text=True, timeout=10
            )
            
            bt.logging.warning(f"═══════════════════════════════════════════════════════════")
            bt.logging.warning(f"📋 CONTAINER FAILURE REPORT - UID {uid}")
            bt.logging.warning(f"═══════════════════════════════════════════════════════════")
            bt.logging.warning(f"🔴 Failure Type: {failure_type}")
            bt.logging.warning(f"🐳 Container: {container_name}")
            
            if logs.stdout:
                bt.logging.warning(f"📤 STDOUT (last 100 lines):")
                for line in logs.stdout.strip().split('\n')[-30:]:
                    bt.logging.warning(f"   {line}")
            else:
                bt.logging.warning(f"📤 STDOUT: (empty)")
            
            if logs.stderr:
                bt.logging.warning(f"📥 STDERR (last 100 lines):")
                for line in logs.stderr.strip().split('\n')[-30:]:
                    bt.logging.warning(f"   {line}")
            else:
                bt.logging.warning(f"📥 STDERR: (empty)")
            
            inspect = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Status}} | ExitCode: {{.State.ExitCode}} | Error: {{.State.Error}}", container_name],
                capture_output=True, text=True, timeout=5
            )
            if inspect.stdout.strip():
                bt.logging.warning(f"📊 Container State: {inspect.stdout.strip()}")
            
            bt.logging.warning(f"═══════════════════════════════════════════════════════════")
            
        except Exception as e:
            bt.logging.warning(f"⚠️ Could not retrieve container logs for UID {uid}: {e}")
    
    def _evaluate_with_rpc_sync(
        self,
        task: MapTask,
        uid: int,
        rpc_port: int,
        skip_ping: bool = False
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
                
                if not skip_ping:
                    ping_response = await agent.ping("test")
                    if ping_response.response != "pong":
                        raise RuntimeError("RPC ping failed")
                
                from swarm.utils.env_factory import make_env
                
                env = make_env(task, gui=False)
                
                try:
                    obs, _ = env.reset()
                    await agent.reset()
                    
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
                        legitimate_model=True,
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
        rpc_port: int,
        skip_ping: bool = True
    ) -> ValidationResult:
        """Async wrapper that runs RPC evaluation in thread pool"""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._evaluate_with_rpc_sync(task, uid, rpc_port, skip_ping)
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

    def _check_rpc_ready(self, port: int, timeout: float = 3.0) -> bool:
        """Check if RPC server is ready by testing TCP connection and basic handshake"""
        import socket
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect(("localhost", port))
            sock.send(b'\x00')
            sock.settimeout(0.5)
            try:
                sock.recv(1)
            except socket.timeout:
                pass
            return True
        except Exception:
            return False
        finally:
            if sock:
                try:
                    sock.close()
                except:
                    pass

    def _get_docker_host_ip(self) -> str:
        """Get the Docker bridge gateway IP (host IP as seen from containers)"""
        try:
            result = subprocess.run(
                ["docker", "network", "inspect", "bridge", "-f", "{{range .IPAM.Config}}{{.Gateway}}{{end}}"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass
        return "172.17.0.1"

    def _get_container_pid(self, container_name: str) -> Optional[int]:
        """Get the PID of a running container"""
        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Pid}}", container_name],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                pid = int(result.stdout.strip())
                if pid > 0:
                    return pid
        except Exception:
            pass
        return None

    def _apply_network_lockdown(self, container_pid: int, validator_ip: str) -> bool:
        """Apply iptables rules in container's network namespace from HOST using nsenter"""
        try:
            rules = [
                ["nsenter", "-t", str(container_pid), "-n", "iptables", "-A", "OUTPUT", "-d", validator_ip, "-j", "ACCEPT"],
                ["nsenter", "-t", str(container_pid), "-n", "iptables", "-A", "OUTPUT", "-d", "127.0.0.1", "-j", "ACCEPT"],
                ["nsenter", "-t", str(container_pid), "-n", "iptables", "-A", "OUTPUT", "-m", "state", "--state", "ESTABLISHED,RELATED", "-j", "ACCEPT"],
                ["nsenter", "-t", str(container_pid), "-n", "iptables", "-A", "OUTPUT", "-j", "DROP"],
            ]
            for rule in rules:
                result = subprocess.run(rule, capture_output=True, text=True, timeout=10)
                if result.returncode != 0:
                    bt.logging.warning(f"Failed to apply iptables rule: {' '.join(rule)}")
                    return False
            return True
        except Exception as e:
            bt.logging.warning(f"Network lockdown failed: {e}")
            return False

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
            current_uid = os.getuid()
            current_gid = os.getgid()
            
            tmpdir = tempfile.mkdtemp()
            os.chown(tmpdir, current_uid, current_gid)
            os.chmod(tmpdir, 0o755)
            
            submission_dir = Path(tmpdir) / "submission"
            submission_dir.mkdir(exist_ok=True)
            os.chown(submission_dir, current_uid, current_gid)
            os.chmod(submission_dir, 0o755)
            
            template_dir = Path(__file__).parent.parent.parent / "submission_template"
            
            with zipfile.ZipFile(model_path, 'r') as zf:
                zf.extractall(submission_dir)
            
            shutil.copy(template_dir / "agent.capnp", submission_dir)
            shutil.copy(template_dir / "agent_server.py", submission_dir)
            shutil.copy(template_dir / "main.py", submission_dir)
            
            for f in submission_dir.iterdir():
                os.chown(f, current_uid, current_gid)
                os.chmod(f, 0o644)
            
            miner_requirements = submission_dir / "requirements.txt"
            has_requirements = miner_requirements.exists()
            
            validator_ip = self._get_docker_host_ip()
            
            if has_requirements:
                bt.logging.info(f"📦 Miner has requirements.txt for UID {uid}")
                startup_script = submission_dir / "startup.sh"
                with open(startup_script, 'w') as f:
                    f.write("#!/bin/bash\n")
                    f.write("pip install --no-cache-dir --user -r /workspace/submission/requirements.txt\n")
                    f.write("if [ $? -ne 0 ]; then exit 1; fi\n")
                    f.write("touch /workspace/submission/.pip_done\n")
                    f.write("sleep infinity\n")
                os.chmod(startup_script, 0o755)
                os.chown(startup_script, current_uid, current_gid)
                
                cmd = [
                    "docker", "run",
                    "--rm",
                    "-d",
                    "--name", container_name,
                    "--user", f"{current_uid}:{current_gid}",
                    "--memory=6g",
                    "--cpus=2",
                    "--pids-limit=50",
                    "--ulimit", "nofile=256:256",
                    "--ulimit", "fsize=524288000:524288000",
                    "--security-opt", "no-new-privileges",
                    "--cap-drop", "ALL",
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
                    "--user", f"{current_uid}:{current_gid}",
                    "--memory=6g",
                    "--cpus=2",
                    "--pids-limit=20",
                    "--ulimit", "nofile=256:256",
                    "--ulimit", "fsize=524288000:524288000",
                    "--security-opt", "no-new-privileges",
                    "--cap-drop", "ALL",
                    "--network", "bridge",
                    "-p", f"{host_port}:8000",
                    "-v", f"{submission_dir}:/workspace/submission:ro",
                    self.base_image,
                    "bash", "-c", "sleep infinity"
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
            
            pip_timeout = 120 if has_requirements else 0
            rpc_timeout = 30
            connected = False
            pip_done = False
            
            if has_requirements:
                bt.logging.info(f"⏳ Waiting for pip install to complete (max {pip_timeout}s)...")
                pip_start = time.time()
                pip_done_flag = submission_dir / ".pip_done"
                
                while time.time() - pip_start < pip_timeout:
                    if pip_done_flag.exists():
                        pip_done = True
                        elapsed = time.time() - pip_start
                        bt.logging.info(f"✅ pip install completed in {elapsed:.1f}s")
                        break
                    
                    check = subprocess.run(
                        ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
                        capture_output=True, text=True
                    )
                    if check.returncode != 0 or check.stdout.strip() != "true":
                        bt.logging.warning(f"❌ Container stopped unexpectedly during pip install for UID {uid}")
                        self._log_container_failure(container_name, uid, "container_stopped_during_pip")
                        break
                    
                    await asyncio.sleep(2)
                
                if not pip_done:
                    bt.logging.warning(f"❌ pip install failed for UID {uid}, skipping evaluation")
                    self._log_container_failure(container_name, uid, "pip_install_failed")
                    subprocess.run(["docker", "kill", container_name], capture_output=True)
                    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
                    return ValidationResult(uid, False, 0.0, 0.0)
            else:
                pip_done = True
            
            container_pid = self._get_container_pid(container_name)
            if not container_pid:
                bt.logging.warning(f"❌ Failed to get container PID for UID {uid}")
                subprocess.run(["docker", "kill", container_name], capture_output=True)
                subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
                return ValidationResult(uid, False, 0.0, 0.0)
            
            bt.logging.info(f"🔒 Applying network lockdown from HOST for UID {uid}...")
            if not self._apply_network_lockdown(container_pid, validator_ip):
                bt.logging.warning(f"❌ Failed to apply network lockdown for UID {uid}")
                subprocess.run(["docker", "kill", container_name], capture_output=True)
                subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
                return ValidationResult(uid, False, 0.0, 0.0)
            bt.logging.info(f"✅ Network lockdown applied for UID {uid}")
            
            bt.logging.info(f"Starting untrusted code for UID {uid}...")
            exec_result = subprocess.run(
                ["docker", "exec", "-d", container_name, "python", "/workspace/submission/main.py"],
                capture_output=True, text=True, timeout=10
            )
            if exec_result.returncode != 0:
                bt.logging.warning(f"❌ Failed to start main.py for UID {uid}: {exec_result.stderr[:200]}")
                subprocess.run(["docker", "kill", container_name], capture_output=True)
                subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
                return ValidationResult(uid, False, 0.0, 0.0)
            
            bt.logging.debug(f"Waiting for RPC server on port {host_port} (max {rpc_timeout}s)...")
            rpc_start = time.time()
            
            while time.time() - rpc_start < rpc_timeout:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(0.5)
                    conn_result = sock.connect_ex(('localhost', host_port))
                    sock.close()
                    if conn_result == 0:
                        connected = True
                        elapsed = time.time() - rpc_start
                        bt.logging.debug(f"Port {host_port} open after {elapsed:.1f}s")
                        break
                except Exception:
                    pass
                await asyncio.sleep(1)
            
            if connected:
                rpc_ready = False
                max_rpc_wait = 20
                min_rpc_wait = 4
                rpc_check_interval = 2
                rpc_wait_start = time.time()
                
                await asyncio.sleep(min_rpc_wait)
                
                while time.time() - rpc_wait_start < max_rpc_wait:
                    try:
                        if self._check_rpc_ready(host_port, timeout=3.0):
                            rpc_ready = True
                            elapsed = time.time() - rpc_wait_start
                            bt.logging.debug(f"RPC ready for UID {uid} after {elapsed:.1f}s")
                            break
                    except Exception:
                        pass
                    await asyncio.sleep(rpc_check_interval)
                
                if not rpc_ready:
                    bt.logging.warning(f"RPC server not responding for UID {uid} after {max_rpc_wait}s")
                    self._log_container_failure(container_name, uid, "rpc_not_ready")
                    subprocess.run(["docker", "kill", container_name], capture_output=True)
                    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
                    return ValidationResult(uid, False, 0.0, 0.0)
            
            if not connected:
                bt.logging.warning(f"❌ RPC server failed to start for UID {uid}")
                self._log_container_failure(container_name, uid, "rpc_connection_failed")
                subprocess.run(["docker", "kill", container_name], capture_output=True)
                subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
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
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"{timestamp} 🔍 DEBUG: UID {uid} result: success={result.success}, time={result.time_sec:.2f}, score={result.score:.4f}")
                
                bt.logging.info(f"🏁 Ending Docker container for UID {uid} - evaluation completed successfully")
                
                return result
                
            except asyncio.TimeoutError:
                bt.logging.warning(f"⏱️ Evaluation timeout for UID {uid} (exceeded {EVAL_TIMEOUT_SEC}s)")
                self._log_container_failure(container_name, uid, "evaluation_timeout")
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"{timestamp} 🔍 DEBUG: UID {uid} result: success=False, time=0.00, score=0.0000 (TIMEOUT)")
                return ValidationResult(uid, False, 0.0, 0.0)
            finally:
                try:
                    subprocess.run(["docker", "kill", container_name], capture_output=True)
                    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
                except Exception:
                    pass
        
        except Exception as e:
            bt.logging.warning(f"❌ Docker evaluation failed for UID {uid}: {e}")
            bt.logging.warning(f"📋 Traceback: {traceback.format_exc()}")
            self._log_container_failure(container_name, uid, "exception")
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

            subprocess.run(["docker", "image", "prune", "-f"], capture_output=True)
            subprocess.run(["docker", "volume", "prune", "-f"], capture_output=True)
        
        except Exception as e:
            bt.logging.warning(f"Container cleanup failed: {e}")

