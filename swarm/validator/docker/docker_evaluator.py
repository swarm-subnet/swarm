import asyncio
import json
import os
import secrets
import subprocess
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import bittensor as bt

from swarm.protocol import MapTask, ValidationResult
from swarm.constants import EVAL_TIMEOUT_SEC
from swarm.utils.hash import sha256sum
from swarm.core.model_verify import add_to_blacklist


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
        
        if not hasattr(self, '_isolated_network'):
            self._setup_isolated_network()
    
    def _setup_isolated_network(self):
        pass
    
    def _get_container_ip(self, container_name: str) -> str:
        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", container_name],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except Exception:
            return None
    
    def _apply_container_egress_blocking(self, container_name: str, container_ip: str):
        try:
            gateway_result = subprocess.run(
                ["docker", "network", "inspect", "bridge", "-f", "{{range .IPAM.Config}}{{.Gateway}}{{end}}"],
                capture_output=True,
                text=True,
                check=True
            )
            gateway_ip = gateway_result.stdout.strip()
            
            if not gateway_ip:
                bt.logging.warning(f"Could not get bridge gateway IP for egress blocking")
                return
            
            iptables_result = subprocess.run([
                "iptables", "-I", "DOCKER-USER", "1",
                "-s", container_ip,
                "!", "-d", gateway_ip,
                "-j", "DROP"
            ], capture_output=True)
            
            if iptables_result.returncode == 0:
                bt.logging.info(f"✅ Egress blocking applied for container {container_name}")
            else:
                bt.logging.error(f"❌ SECURITY WARNING: Failed to apply egress blocking for {container_name}")
                bt.logging.error(f"Container may have outbound internet access")
                bt.logging.error(f"Error: {iptables_result.stderr.decode()}")
        except Exception as e:
            bt.logging.error(f"❌ SECURITY WARNING: Exception applying egress blocking: {e}")

    async def evaluate_model(
        self, 
        task: MapTask, 
        uid: int, 
        model_path: Path
    ) -> ValidationResult:
        """Evaluate model in isolated Docker container with proper lifecycle management"""
        
        if not model_path.is_file():
            bt.logging.warning(f"Model path is missing or not a file: {model_path}")
            return ValidationResult(uid, False, 0.0, 0.0)
        
        if not DockerSecureEvaluator._base_ready:
            bt.logging.warning(f"Docker not ready for UID {uid}")
            return ValidationResult(uid, False, 0.0, 0.0)
        
        # Double-check that base image exists before proceeding
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
        
        from swarm.core.model_verify import inspect_model_structure, classify_model_validity
        
        inspection_results = inspect_model_structure(model_path)
        model_status, model_reason = classify_model_validity(inspection_results)
        
        if model_status == "fake":
            bt.logging.warning(f"🚫 FAKE MODEL DETECTED for UID {uid}: {model_reason}")
            bt.logging.debug(f"Inspection results: {inspection_results}")
            
            self.last_fake_model_info = {
                'uid': uid,
                'reason': model_reason,
                'inspection_results': inspection_results
            }
            
            model_hash = sha256sum(model_path)
            add_to_blacklist(model_hash)
            
            bt.logging.info(f"🏁 Ending evaluation for UID {uid} - fake model detected")
            return ValidationResult(uid, False, 0.0, 0.0)
        
        if model_status == "missing_metadata":
            bt.logging.warning(f"Model {uid} missing metadata: {model_reason}")
            bt.logging.info(f"🏁 Ending evaluation for UID {uid} - missing metadata")
            return ValidationResult(uid, False, 0.0, 0.0)
        
        container_name = f"swarm_eval_{uid}_{int(time.time() * 1000)}"
        
        bt.logging.info(f"🐳 Starting Docker container for UID {uid} evaluation...")
        
        auth_token = secrets.token_hex(32)
        
        try:
            cmd = [
                "docker", "run",
                "--rm",
                "--name", container_name,
                "--user", "1000:1000",
                "--memory=6g",
                "--cpus=2",
                "--pids-limit=20",
                "--ulimit", "nofile=64:64",
                "--ulimit", "fsize=524288000:524288000",
                "--security-opt", "no-new-privileges",
                "--network", "bridge",
                "-e", f"AUTH_TOKEN={auth_token}",
                "-e", f"RPC_PORT=9000",
                "-v", f"{model_path.absolute()}:/workspace/model.zip:ro",
                self.base_image,
                "container_launcher",
                "/workspace/model.zip"
            ]
            
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            bt.logging.info(f"⏳ Waiting 3 minutes for pip install to complete (miner's requirements.txt)...")
            await asyncio.sleep(180)
            
            container_ip = self._get_container_ip(container_name)
            if container_ip:
                self._apply_container_egress_blocking(container_name, container_ip)
            if not container_ip:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5)
                except Exception:
                    pass
                subprocess.run(["docker", "kill", container_name], capture_output=True)
                return ValidationResult(uid, False, 0.0, 0.0)
            
            from swarm.core.evaluator_host import evaluate_with_rpc_client
            
            try:
                result = await asyncio.wait_for(
                    evaluate_with_rpc_client(task, uid, container_ip, auth_token),
                    timeout=EVAL_TIMEOUT_SEC
                )
                
                if not (0.0 <= result.score <= 1.0):
                    bt.logging.error(f"🚫 Invalid score {result.score} for UID {uid} - blacklisting model")
                    model_hash = sha256sum(model_path)
                    add_to_blacklist(model_hash)
                    result = ValidationResult(uid, False, 0.0, 0.0)
                
                if result.score == 0.0:
                    result = ValidationResult(uid, result.success, result.time_sec, 0.01)
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"{timestamp} 🔍 DEBUG: UID {uid} result: uid={result.uid}, success={result.success}, time={result.time_sec}, score={result.score}")
                
                bt.logging.info(f"🏁 Ending Docker container for UID {uid} - evaluation completed successfully")
                
                return result
                
            except asyncio.TimeoutError:
                bt.logging.warning(f"Container timeout for UID {uid}")
                subprocess.run(["docker", "kill", container_name], capture_output=True)
                return ValidationResult(uid, False, 0.0, 0.0)
            
            finally:
                try:
                    proc.terminate()
                    await asyncio.wait_for(proc.wait(), timeout=5)
                except Exception:
                    subprocess.run(["docker", "kill", container_name], capture_output=True)
        
        except Exception as e:
            bt.logging.warning(f"Docker evaluation failed for UID {uid}: {e}")
            subprocess.run(["docker", "kill", container_name], capture_output=True)
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
        finally:
            # Best-effort ensure container is removed even if --rm didn't trigger
            try:
                subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
            except Exception:
                pass
            # Periodically prune dangling images/caches to keep disk usage low
            # IMPORTANT: Only prune dangling images, NOT all unused images (removed -a flag)
            # This preserves the base image between evaluations
            try:
                subprocess.run(["docker", "image", "prune", "-f"], capture_output=True)
                subprocess.run(["docker", "builder", "prune", "-f"], capture_output=True)
            except Exception:
                pass
        
        # ENHANCED LOGGING: Final fallback ending message
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

