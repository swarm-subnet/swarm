import asyncio
import hashlib
import os
import re
import shutil
import socket
import statistics
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Optional, Callable

import capnp
import bittensor as bt
import numpy as np

from swarm.protocol import ValidationResult
from swarm.constants import (
    SIM_DT, SPEED_LIMIT,
    DOCKER_WORKER_MEMORY, DOCKER_WORKER_CPUS, N_DOCKER_WORKERS,
    DOCKER_PIP_WHITELIST,
    RPC_STEP_TIMEOUT_SEC, RPC_FIRST_STEP_TIMEOUT_SEC,
    RPC_RESET_TIMEOUT_SEC, RPC_PING_TIMEOUT_SEC,
    RPC_MAX_STRIKES_PER_SEED,
    GLOBAL_EVAL_BASE_SEC, GLOBAL_EVAL_PER_SEED_SEC, GLOBAL_EVAL_CAP_SEC,
    MINER_COMPUTE_BUDGET_SEC,
    CALIBRATION_ROUNDS,
    CALIBRATION_OVERHEAD_CAP_SEC,
    CALIBRATION_TIMEOUT_SEC,
    CALIBRATION_BENCHMARK_REF_NS,
    CALIBRATION_CPU_FACTOR_CAP,
    CALIBRATION_MARGIN_SEC,
)
from swarm.utils.hash import sha256sum
from swarm.utils.env_factory import make_env
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
        """Calculate hash of all source files that go into the Docker image."""
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
                    hasher.update(f.read_bytes())
                except Exception:
                    pass

        return hasher.hexdigest()[:16]

    def _get_image_hash_label(self) -> str:
        """Get the code hash label from the existing Docker image."""
        try:
            result = subprocess.run(
                ["docker", "inspect", "--format", "{{index .Config.Labels \"swarm.code_hash\"}}", self.base_image],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return ""

    def _should_rebuild_base_image(self) -> bool:
        """Check if Docker image needs rebuild by comparing code hash with image label."""
        current_hash = self._calculate_docker_hash()

        # Check if image exists
        result = subprocess.run(
            ["docker", "images", "-q", self.base_image],
            capture_output=True, text=True
        )
        if not result.stdout.strip():
            bt.logging.info(f"🐳 Docker image not found, will build (hash: {current_hash})")
            return True

        # Get hash from the image label (not from cache file)
        image_hash = self._get_image_hash_label()

        if image_hash == current_hash:
            bt.logging.info(f"✅ Docker image up-to-date (hash: {current_hash})")
            return False

        if image_hash:
            bt.logging.info(f"🔄 Code changed: image={image_hash}, current={current_hash} - rebuilding")
        else:
            bt.logging.info(f"🔄 Image missing hash label - rebuilding (hash: {current_hash})")

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

            old_image_id = None
            try:
                result = subprocess.run(
                    ["docker", "images", "-q", self.base_image],
                    capture_output=True, text=True
                )
                old_image_id = result.stdout.strip() if result.returncode == 0 else None
            except Exception:
                pass

            try:
                subprocess.run(["docker", "container", "prune", "-f"], capture_output=True)
                subprocess.run("docker rm -f $(docker ps -aq --filter=name=swarm_eval_)", shell=True, capture_output=True)
                subprocess.run("docker rm -f $(docker ps -aq --filter=name=swarm_verify_)", shell=True, capture_output=True)
                subprocess.run(["docker", "image", "prune", "-f"], capture_output=True)
                subprocess.run(["docker", "volume", "prune", "-f"], capture_output=True)
                subprocess.run(["docker", "builder", "prune", "-f", "--keep-storage", "5GB"], capture_output=True)
            except Exception:
                pass

            dockerfile_path = Path(__file__).parent / "Dockerfile"
            build_context = Path(__file__).parent.parent.parent.parent
            current_hash = self._calculate_docker_hash()

            cmd = [
                "docker", "build",
                "--label", f"swarm.code_hash={current_hash}",
                "-t", self.base_image,
                "-f", str(dockerfile_path),
                str(build_context)
            ]

            bt.logging.info(f"Building base Docker image (hash: {current_hash})...")
            bt.logging.debug(f"Docker build command: {' '.join(cmd)}")

            env = os.environ.copy()
            env["DOCKER_BUILDKIT"] = "1"
            result = subprocess.run(cmd, text=True, env=env)

            if result.returncode == 0:
                self.base_ready = True
                DockerSecureEvaluator._base_ready = True
                bt.logging.info("✅ Base Docker image ready")

                if old_image_id:
                    try:
                        new_result = subprocess.run(
                            ["docker", "images", "-q", self.base_image],
                            capture_output=True, text=True
                        )
                        new_image_id = new_result.stdout.strip() if new_result.returncode == 0 else None
                        if new_image_id and old_image_id != new_image_id:
                            subprocess.run(["docker", "rmi", old_image_id], capture_output=True)
                            bt.logging.debug(f"Removed old image: {old_image_id[:12]}")
                    except Exception:
                        pass
            else:
                bt.logging.error(f"❌ Docker build failed with return code: {result.returncode}")
                self.base_ready = False
                DockerSecureEvaluator._base_ready = False

        except Exception as e:
            bt.logging.error(f"Failed to setup Docker environment: {e}")
            self.base_ready = False
            DockerSecureEvaluator._base_ready = False

    def _find_free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    def _check_rpc_ready(self, container_name: str, timeout: float = 5.0) -> bool:
        """Check if the RPC server process is running inside the container."""
        try:
            result = subprocess.run(
                ["docker", "top", container_name],
                capture_output=True, text=True, timeout=timeout
            )
            return result.returncode == 0 and "main.py" in result.stdout
        except Exception:
            return False

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

    @staticmethod
    def _normalize_package_name(name: str) -> str:
        return re.sub(r"[-_.]+", "-", name).lower()

    def _validate_requirements(self, requirements_path: Path, uid: int) -> bool:
        try:
            lines = requirements_path.read_text().splitlines()
        except Exception as e:
            bt.logging.warning(f"UID {uid}: Failed to read requirements.txt: {e}")
            return False

        rejected = []
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("-"):
                bt.logging.warning(f"UID {uid}: Pip option not allowed: {line}")
                return False

            if line.startswith(("git+", "http://", "https://", "file:", "./", "/")):
                bt.logging.warning(f"UID {uid}: Direct URL/path install not allowed: {line}")
                return False

            if " @ " in line:
                bt.logging.warning(f"UID {uid}: PEP 508 direct reference not allowed: {line}")
                return False

            line = line.split("#")[0].strip()
            if not line:
                continue

            line = line.split(";")[0].strip()
            name = re.split(r"[>=<!~\[]", line)[0].strip()
            if not name:
                continue

            normalized = self._normalize_package_name(name)
            if normalized not in DOCKER_PIP_WHITELIST:
                rejected.append(normalized)

        if rejected:
            bt.logging.warning(
                f"UID {uid}: Requirements rejected — packages not whitelisted: {', '.join(rejected)}"
            )
            return False

        return True

    @staticmethod
    def _serialize_observation(agent_capnp, obs):
        """Serialize a numpy observation dict into a Cap'n Proto Observation message."""
        message = agent_capnp.Observation.new_message()
        if isinstance(obs, dict):
            entries = message.init("entries", len(obs))
            for i, (key, value) in enumerate(obs.items()):
                arr = np.asarray(value, dtype=np.float32)
                entries[i].key = key
                entries[i].tensor.data = arr.tobytes()
                entries[i].tensor.shape = list(arr.shape)
                entries[i].tensor.dtype = str(arr.dtype)
        else:
            arr = np.asarray(obs, dtype=np.float32)
            entry = message.init("entries", 1)[0]
            entry.key = "__value__"
            entry.tensor.data = arr.tobytes()
            entry.tensor.shape = list(arr.shape)
            entry.tensor.dtype = str(arr.dtype)
        return message

    def _run_multi_seed_rpc_sync(
        self,
        tasks: list,
        uid: int,
        rpc_port: int,
        on_seed_complete: Optional[Callable[[], None]] = None,
    ) -> list:
        """Run multiple seeds through the same RPC connection.

        This reuses the container for all seeds, calling agent.reset() between each.
        Much faster than creating a new container per seed.
        """
        schema_file = Path(__file__).parent.parent.parent / "submission_template" / "agent.capnp"
        agent_capnp = capnp.load(str(schema_file))

        async def run_all_seeds():
            results = []
            async with capnp.kj_loop():
                stream = None
                agent = None
                max_ping_attempts = 6

                for attempt in range(1, max_ping_attempts + 1):
                    try:
                        stream = await capnp.AsyncIoStream.create_connection(host="localhost", port=rpc_port)
                        client = capnp.TwoPartyClient(stream)
                        agent = client.bootstrap().cast_as(agent_capnp.Agent)

                        ping_response = await asyncio.wait_for(
                            agent.ping("test"), timeout=RPC_PING_TIMEOUT_SEC
                        )
                        if ping_response.response != "pong":
                            raise RuntimeError(f"Unexpected ping response (attempt {attempt}/{max_ping_attempts})")

                        break
                    except asyncio.TimeoutError:
                        if attempt >= max_ping_attempts:
                            bt.logging.warning(
                                f"UID {uid}: RPC ping timeout after {max_ping_attempts} attempts "
                                f"({RPC_PING_TIMEOUT_SEC}s each)"
                            )
                            for _ in tasks:
                                if on_seed_complete:
                                    try:
                                        on_seed_complete()
                                    except Exception:
                                        pass
                            return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]
                        await asyncio.sleep(2)
                    except Exception as e:
                        if attempt >= max_ping_attempts:
                            bt.logging.warning(
                                f"Cap'n Proto connection/ping failed for UID {uid} on port {rpc_port} "
                                f"after {max_ping_attempts} attempts: {e}"
                            )
                            for _ in tasks:
                                if on_seed_complete:
                                    try:
                                        on_seed_complete()
                                    except Exception:
                                        pass
                            return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]
                        await asyncio.sleep(2)

                if agent is None:
                    for _ in tasks:
                        if on_seed_complete:
                            try:
                                on_seed_complete()
                            except Exception:
                                pass
                    return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]

                calibrated_timeout = RPC_STEP_TIMEOUT_SEC
                rpc_overhead_sec = max(RPC_STEP_TIMEOUT_SEC - MINER_COMPUTE_BUDGET_SEC, 0.010)
                cpu_factor = 1.0
                calibrated = False

                for task_idx, task in enumerate(tasks):
                    try:
                        env = make_env(task, gui=False)

                        try:
                            obs, _ = env.reset()
                            await asyncio.wait_for(
                                agent.reset(),
                                timeout=RPC_RESET_TIMEOUT_SEC
                            )

                            if not calibrated:
                                rpc_overhead_sec, cpu_factor = await self._calibrate_rpc_overhead_async(
                                    agent, agent_capnp, obs, uid
                                )
                                calibrated_timeout = (MINER_COMPUTE_BUDGET_SEC * cpu_factor) + rpc_overhead_sec + CALIBRATION_MARGIN_SEC
                                calibrated = True
                                bt.logging.info(
                                    f"UID {uid}: calibrated timeout = {calibrated_timeout*1000:.1f}ms "
                                    f"(budget={MINER_COMPUTE_BUDGET_SEC*1000:.0f}ms x {cpu_factor:.2f} + overhead={rpc_overhead_sec*1000:.1f}ms + margin={CALIBRATION_MARGIN_SEC*1000:.0f}ms)"
                                )

                            pos0 = np.asarray(task.start, dtype=float)
                            t_sim = 0.0
                            success = False
                            info = {}
                            strikes = 0
                            is_first_step = True

                            lo, hi = env.action_space.low.flatten(), env.action_space.high.flatten()
                            last_pos = pos0

                            while t_sim < task.horizon:
                                step_timeout = RPC_FIRST_STEP_TIMEOUT_SEC if is_first_step else calibrated_timeout

                                observation = self._serialize_observation(agent_capnp, obs)

                                try:
                                    t_act_start = time.time()
                                    action_response = await asyncio.wait_for(
                                        agent.act(observation),
                                        timeout=step_timeout
                                    )
                                    action = np.frombuffer(
                                        action_response.action.data,
                                        dtype=np.dtype(action_response.action.dtype)
                                    ).reshape(tuple(action_response.action.shape))
                                except asyncio.TimeoutError:
                                    act_ms = (time.time() - t_act_start) * 1000
                                    strikes += 1
                                    action = np.zeros(5, dtype=np.float32)
                                    if is_first_step:
                                        bt.logging.warning(
                                            f"UID {uid}: first-step act() timeout ({act_ms:.0f}ms > {step_timeout*1000:.0f}ms), "
                                            f"strike {strikes}/{RPC_MAX_STRIKES_PER_SEED}"
                                        )
                                    else:
                                        bt.logging.warning(
                                            f"UID {uid}: act() timeout ({act_ms:.0f}ms > {step_timeout*1000:.0f}ms "
                                            f"[budget={MINER_COMPUTE_BUDGET_SEC*1000:.0f}x{cpu_factor:.2f}+overhead={rpc_overhead_sec*1000:.1f}]), "
                                            f"strike {strikes}/{RPC_MAX_STRIKES_PER_SEED}"
                                        )
                                    if strikes >= RPC_MAX_STRIKES_PER_SEED:
                                        bt.logging.warning(
                                            f"UID {uid} seed {task_idx}: {strikes} RPC timeouts, failing seed"
                                        )
                                        break
                                except Exception:
                                    action = np.zeros(5, dtype=np.float32)

                                is_first_step = False

                                act = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), lo, hi)

                                if hasattr(env, 'ACT_TYPE') and hasattr(env, 'SPEED_LIMIT'):
                                    if env.ACT_TYPE == ActionType.VEL and env.SPEED_LIMIT:
                                        n = max(np.linalg.norm(act[:3]), 1e-6)
                                        scale = min(1.0, SPEED_LIMIT / n)
                                        act[:3] *= scale
                                        act = np.clip(act, lo, hi)

                                obs, _r, terminated, truncated, info = env.step(act[None, :])
                                last_pos = env._getDroneStateVector(0)[0:3]

                                t_sim += SIM_DT
                                if terminated or truncated:
                                    success = info.get("success", False)
                                    break

                            if strikes >= RPC_MAX_STRIKES_PER_SEED:
                                results.append(ValidationResult(uid, False, 0.0, 0.0))
                            else:
                                min_clearance = info.get("min_clearance", None)
                                collision = info.get("collision", False)
                                score = flight_reward(
                                    success=success,
                                    t=t_sim,
                                    horizon=task.horizon,
                                    task=task,
                                    min_clearance=min_clearance,
                                    collision=collision,
                                    legitimate_model=True,
                                )
                                results.append(ValidationResult(uid, success, t_sim, score))

                            if on_seed_complete:
                                try:
                                    on_seed_complete()
                                except Exception:
                                    pass

                        finally:
                            try:
                                env.close()
                            except Exception:
                                pass

                    except Exception as e:
                        bt.logging.warning(f"Seed {task_idx} failed for UID {uid}: {e}")
                        results.append(ValidationResult(uid, False, 0.0, 0.0))
                        if on_seed_complete:
                            try:
                                on_seed_complete()
                            except Exception:
                                pass

            return results

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(run_all_seeds())
        finally:
            loop.close()

    async def _calibrate_rpc_overhead_async(self, agent, agent_capnp, obs, uid: int):
        """Measure RPC pipeline overhead and CPU speed factor."""
        round_trips = []
        benchmark_times_ns = []

        for r in range(CALIBRATION_ROUNDS):
            cal_obs = self._serialize_observation(agent_capnp, obs)

            try:
                t0 = time.time()
                cal_response = await asyncio.wait_for(
                    agent.calibrate(cal_obs),
                    timeout=CALIBRATION_TIMEOUT_SEC
                )
                dt = time.time() - t0
                bench_ns = cal_response.benchmarkNs
                round_trips.append(dt)
                if bench_ns > 0:
                    benchmark_times_ns.append(bench_ns)
            except (asyncio.TimeoutError, Exception) as e:
                bt.logging.warning(
                    f"UID {uid}: calibration round {r+1}/{CALIBRATION_ROUNDS} failed: {e}"
                )

        if len(round_trips) < 3:
            fallback = max(RPC_STEP_TIMEOUT_SEC - MINER_COMPUTE_BUDGET_SEC, 0.010)
            bt.logging.warning(
                f"UID {uid}: calibration mostly failed ({len(round_trips)}/{CALIBRATION_ROUNDS} ok), "
                f"using fallback overhead={fallback*1000:.0f}ms, cpu_factor=1.0"
            )
            return fallback, 1.0

        round_trips.sort()
        trimmed_rt = round_trips[1:-1] if len(round_trips) > 4 else round_trips
        median_overhead = statistics.median(trimmed_rt)

        if median_overhead > CALIBRATION_OVERHEAD_CAP_SEC:
            bt.logging.warning(
                f"UID {uid}: measured RPC overhead {median_overhead*1000:.1f}ms exceeds cap "
                f"{CALIBRATION_OVERHEAD_CAP_SEC*1000:.0f}ms — capping."
            )
            median_overhead = CALIBRATION_OVERHEAD_CAP_SEC

        cpu_factor = 1.0
        if len(benchmark_times_ns) >= 3:
            benchmark_times_ns.sort()
            trimmed_bench = benchmark_times_ns[1:-1] if len(benchmark_times_ns) > 4 else benchmark_times_ns
            median_bench_ns = statistics.median(trimmed_bench)
            cpu_factor = median_bench_ns / CALIBRATION_BENCHMARK_REF_NS
            cpu_factor = max(1.0, min(cpu_factor, CALIBRATION_CPU_FACTOR_CAP))

        bench_median_ms = statistics.median(benchmark_times_ns) / 1e6 if benchmark_times_ns else 0.0
        bt.logging.info(
            f"UID {uid}: calibration results — "
            f"overhead={median_overhead*1000:.1f}ms, "
            f"cpu_factor={cpu_factor:.2f}x, "
            f"benchmark_median={bench_median_ms:.1f}ms, "
            f"rtt=[{', '.join(f'{t*1000:.1f}' for t in round_trips)}]ms"
        )

        return median_overhead, cpu_factor

    async def _run_multi_seed_rpc_host(
        self,
        tasks: list,
        uid: int,
        rpc_port: int,
        on_seed_complete: Optional[Callable[[], None]] = None,
    ) -> list:
        """Async wrapper that runs multi-seed RPC evaluation in thread pool."""
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self._run_multi_seed_rpc_sync(tasks, uid, rpc_port, on_seed_complete)
            )
            return results
        except Exception as e:
            bt.logging.warning(f"Multi-seed RPC evaluation failed for UID {uid}: {e}")
            for _ in tasks:
                if on_seed_complete:
                    try:
                        on_seed_complete()
                    except Exception:
                        pass
            return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]

    async def evaluate_seeds_batch(
        self,
        tasks: list,
        uid: int,
        model_path: Path,
        worker_id: int = 0,
        on_seed_complete: Optional[Callable[[], None]] = None,
    ) -> list:
        """Evaluate multiple seeds in a single container.

        This is the parallel-friendly method for V4 benchmark.
        Instead of creating a new container per seed, this method:
        1. Starts ONE container
        2. Runs all seeds through the same RPC connection
        3. Calls agent.reset() between seeds
        4. Kills container when done

        Args:
            tasks: List of MapTask objects (one per seed)
            uid: Miner UID
            model_path: Path to model zip file
            worker_id: Worker ID for logging (0 to N_DOCKER_WORKERS-1)

        Returns:
            List of ValidationResult objects (one per seed)
        """
        if not tasks:
            return []

        def _notify_all_failed():
            """Call on_seed_complete for all tasks when batch fails early."""
            if on_seed_complete:
                for _ in tasks:
                    try:
                        on_seed_complete()
                    except Exception:
                        pass

        if not model_path.is_file():
            bt.logging.warning(f"[Worker {worker_id}] Model path missing: {model_path}")
            _notify_all_failed()
            return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]

        if not DockerSecureEvaluator._base_ready:
            bt.logging.warning(f"[Worker {worker_id}] Docker not ready for UID {uid}")
            _notify_all_failed()
            return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]

        try:
            with zipfile.ZipFile(model_path, 'r') as zf:
                if "drone_agent.py" not in zf.namelist():
                    bt.logging.warning(f"[Worker {worker_id}] Model {uid} missing drone_agent.py")
                    _notify_all_failed()
                    return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]
        except Exception as e:
            bt.logging.warning(f"[Worker {worker_id}] Failed to validate model {uid}: {e}")
            _notify_all_failed()
            return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]

        container_name = f"swarm_eval_{uid}_w{worker_id}_{int(time.time() * 1000)}"
        host_port = self._find_free_port()

        bt.logging.info(f"[Worker {worker_id}] Starting container for UID {uid} ({len(tasks)} seeds)...")

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

            contents = list(submission_dir.iterdir())
            if len(contents) == 1 and contents[0].is_dir():
                nested_dir = contents[0]
                for item in nested_dir.iterdir():
                    target = submission_dir / item.name
                    if target.exists():
                        if target.is_dir():
                            shutil.rmtree(target)
                        else:
                            target.unlink()
                    shutil.move(str(item), str(target))
                nested_dir.rmdir()

            shutil.copy(template_dir / "agent.capnp", submission_dir)
            shutil.copy(template_dir / "agent_server.py", submission_dir)
            shutil.copy(template_dir / "main.py", submission_dir)

            for f in submission_dir.iterdir():
                os.chown(f, current_uid, current_gid)
                os.chmod(f, 0o644)

            miner_requirements = submission_dir / "requirements.txt"
            has_requirements = miner_requirements.exists()

            if has_requirements and not self._validate_requirements(miner_requirements, uid):
                bt.logging.warning(f"[Worker {worker_id}] UID {uid} requirements.txt rejected")
                _notify_all_failed()
                return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]

            validator_ip = self._get_docker_host_ip()

            if has_requirements:
                bt.logging.info(f"[Worker {worker_id}] Miner has requirements.txt for UID {uid}")
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
                    f"--memory={DOCKER_WORKER_MEMORY}",
                    f"--cpus={DOCKER_WORKER_CPUS}",
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
                    f"--memory={DOCKER_WORKER_MEMORY}",
                    f"--cpus={DOCKER_WORKER_CPUS}",
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

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                bt.logging.warning(f"[Worker {worker_id}] Container start failed: {result.stderr[:300]}")
                _notify_all_failed()
                return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]

            pip_timeout = 120 if has_requirements else 0
            rpc_timeout = 30
            connected = False
            pip_done = False

            if has_requirements:
                bt.logging.info(f"[Worker {worker_id}] Waiting for pip install (max {pip_timeout}s)...")
                pip_start = time.time()
                pip_done_flag = submission_dir / ".pip_done"

                while time.time() - pip_start < pip_timeout:
                    if pip_done_flag.exists():
                        pip_done = True
                        elapsed = time.time() - pip_start
                        bt.logging.info(f"[Worker {worker_id}] pip done in {elapsed:.1f}s")
                        break

                    check = subprocess.run(
                        ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
                        capture_output=True, text=True
                    )
                    if check.returncode != 0 or check.stdout.strip() != "true":
                        bt.logging.warning(f"[Worker {worker_id}] Container stopped during pip")
                        break

                    await asyncio.sleep(2)

                if not pip_done:
                    bt.logging.warning(f"[Worker {worker_id}] pip install failed for UID {uid}")
                    subprocess.run(["docker", "kill", container_name], capture_output=True)
                    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
                    _notify_all_failed()
                    return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]
            else:
                pip_done = True

            container_pid = self._get_container_pid(container_name)
            if not container_pid:
                bt.logging.warning(f"[Worker {worker_id}] Failed to get container PID")
                subprocess.run(["docker", "kill", container_name], capture_output=True)
                subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
                _notify_all_failed()
                return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]

            bt.logging.debug(f"[Worker {worker_id}] Applying network lockdown...")
            if not self._apply_network_lockdown(container_pid, validator_ip):
                bt.logging.warning(f"[Worker {worker_id}] Network lockdown failed")
                subprocess.run(["docker", "kill", container_name], capture_output=True)
                subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
                _notify_all_failed()
                return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]

            exec_result = subprocess.run(
                ["docker", "exec", "-d", container_name, "python", "/workspace/submission/main.py"],
                capture_output=True, text=True, timeout=10
            )
            if exec_result.returncode != 0:
                bt.logging.warning(f"[Worker {worker_id}] Failed to start main.py: {exec_result.stderr[:200]}")
                subprocess.run(["docker", "kill", container_name], capture_output=True)
                subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
                _notify_all_failed()
                return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]

            bt.logging.debug(f"[Worker {worker_id}] Waiting for RPC server for UID {uid} (max {rpc_timeout}s)...")
            rpc_start = time.time()
            max_rpc_wait = 30
            rpc_check_interval = 2
            connected = False

            await asyncio.sleep(4)

            while time.time() - rpc_start < max_rpc_wait:
                try:
                    if self._check_rpc_ready(container_name, timeout=5.0):
                        connected = True
                        elapsed = time.time() - rpc_start
                        bt.logging.debug(f"[Worker {worker_id}] RPC ready for UID {uid} after {elapsed:.1f}s")
                        break
                except Exception:
                    pass
                await asyncio.sleep(rpc_check_interval)

            if not connected:
                bt.logging.warning(f"[Worker {worker_id}] RPC connection failed")
                subprocess.run(["docker", "kill", container_name], capture_output=True)
                subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
                _notify_all_failed()
                return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]

            container_check = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
                capture_output=True, text=True, timeout=5
            )
            if container_check.returncode != 0 or "true" not in container_check.stdout.lower():
                bt.logging.warning(f"[Worker {worker_id}] Container stopped before evaluation")
                subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
                _notify_all_failed()
                return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]

            try:
                batch_timeout = min(
                    GLOBAL_EVAL_BASE_SEC + GLOBAL_EVAL_PER_SEED_SEC * len(tasks),
                    GLOBAL_EVAL_CAP_SEC
                )
                results = await asyncio.wait_for(
                    self._run_multi_seed_rpc_host(tasks, uid, host_port, on_seed_complete),
                    timeout=batch_timeout
                )

                valid_results = []
                for r in results:
                    score = float(r.score)
                    if 0.0 <= score <= 1.0:
                        valid_results.append(r)
                    else:
                        bt.logging.warning(f"[Worker {worker_id}] Invalid score {score}")
                        model_hash = sha256sum(model_path)
                        add_to_blacklist(model_hash)
                        valid_results.append(ValidationResult(uid, False, 0.0, 0.0))

                bt.logging.info(f"[Worker {worker_id}] Completed {len(tasks)} seeds for UID {uid}")
                return valid_results

            except asyncio.TimeoutError:
                bt.logging.warning(f"[Worker {worker_id}] Batch timeout for UID {uid}")
                _notify_all_failed()
                return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]
            finally:
                subprocess.run(["docker", "kill", container_name], capture_output=True)
                subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)

        except Exception as e:
            bt.logging.warning(f"[Worker {worker_id}] Batch evaluation failed: {e}")
            _notify_all_failed()
            try:
                subprocess.run(["docker", "kill", container_name], capture_output=True)
                subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
            except Exception:
                pass
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

        return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]

    async def evaluate_seeds_parallel(
        self,
        tasks: list,
        uid: int,
        model_path: Path,
        num_workers: int = None,
        on_seed_complete: Optional[Callable[[], None]] = None,
    ) -> list:
        """Evaluate seeds in parallel using multiple Docker containers.

        This is the main entry point for V4 benchmark parallelization.
        Splits seeds across N_DOCKER_WORKERS containers running in parallel.

        Args:
            tasks: List of MapTask objects (one per seed)
            uid: Miner UID
            model_path: Path to model zip file
            num_workers: Number of parallel workers (defaults to N_DOCKER_WORKERS)
            on_seed_complete: Optional callback called after each seed completes (thread-safe)

        Returns:
            List of ValidationResult objects (one per seed, in original order)
        """
        if not tasks:
            return []

        if num_workers is None:
            num_workers = N_DOCKER_WORKERS

        num_workers = max(1, min(num_workers, len(tasks)))

        chunks = []
        chunk_size = len(tasks) // num_workers
        remainder = len(tasks) % num_workers

        start = 0
        for i in range(num_workers):
            end = start + chunk_size + (1 if i < remainder else 0)
            chunks.append(tasks[start:end])
            start = end

        bt.logging.info(f"Evaluating {len(tasks)} seeds with {num_workers} parallel workers")

        worker_tasks = [
            self.evaluate_seeds_batch(chunk, uid, model_path, worker_id=i, on_seed_complete=on_seed_complete)
            for i, chunk in enumerate(chunks)
        ]

        chunk_results = await asyncio.gather(*worker_tasks, return_exceptions=True)

        results = []
        for i, chunk_result in enumerate(chunk_results):
            if isinstance(chunk_result, Exception):
                bt.logging.warning(f"Worker {i} failed with exception: {chunk_result}")
                if on_seed_complete:
                    for _ in chunks[i]:
                        try:
                            on_seed_complete()
                        except Exception:
                            pass
                results.extend([ValidationResult(uid, False, 0.0, 0.0) for _ in chunks[i]])
            else:
                results.extend(chunk_result)

        return results

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
            subprocess.run(["docker", "builder", "prune", "-f", "--keep-storage", "5GB"], capture_output=True)

        except Exception as e:
            bt.logging.warning(f"Container cleanup failed: {e}")
