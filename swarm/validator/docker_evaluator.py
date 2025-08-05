import asyncio
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import bittensor as bt

from swarm.protocol import MapTask, ValidationResult
from swarm.constants import EVAL_TIMEOUT_SEC


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
            self.base_image = "swarm_evaluator_base"
            self.last_fake_model_info = None
        
        if not DockerSecureEvaluator._base_ready:
            self._setup_base_container()
            DockerSecureEvaluator._base_ready = self.base_ready
    
    def _ensure_docker_installed(self):
        """Check if Docker is installed, install automatically if missing"""
        try:
            # Check if Docker command exists
            result = subprocess.run(["docker", "--version"], 
                                  capture_output=True, text=True, check=True)
            bt.logging.info(f"Docker found: {result.stdout.strip()}")
            return True
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            bt.logging.warning("🐳 Docker not found - installing automatically...")
            
            try:
                # Install Docker using official script
                bt.logging.info("Downloading Docker installation script...")
                
                # Download Docker install script
                download_cmd = ["curl", "-fsSL", "https://get.docker.com", "-o", "/tmp/get-docker.sh"]
                subprocess.run(download_cmd, check=True, capture_output=True)
                
                # Make script executable
                subprocess.run(["chmod", "+x", "/tmp/get-docker.sh"], check=True)
                
                # Run Docker installation
                bt.logging.info("Installing Docker (this may take a few minutes)...")
                install_result = subprocess.run(["sudo", "/tmp/get-docker.sh"], 
                                              capture_output=True, text=True, timeout=300)
                
                if install_result.returncode == 0:
                    bt.logging.info("✅ Docker installed successfully!")
                    
                    # Add current user to docker group
                    import os
                    username = os.getenv("USER", "root")
                    subprocess.run(["sudo", "usermod", "-aG", "docker", username], 
                                 capture_output=True)
                    
                    # Start Docker service
                    subprocess.run(["sudo", "systemctl", "start", "docker"], 
                                 capture_output=True)
                    subprocess.run(["sudo", "systemctl", "enable", "docker"], 
                                 capture_output=True)
                    
                    bt.logging.info("🔄 Docker service started")
                    
                    # Cleanup
                    subprocess.run(["rm", "-f", "/tmp/get-docker.sh"], capture_output=True)
                    
                    # Wait a moment for Docker to be ready
                    import time
                    time.sleep(3)
                    
                    # Verify installation
                    verify_result = subprocess.run(["docker", "--version"], 
                                                 capture_output=True, text=True)
                    if verify_result.returncode == 0:
                        bt.logging.info(f"✅ Docker ready: {verify_result.stdout.strip()}")
                        return True
                    else:
                        bt.logging.error("❌ Docker installation verification failed")
                        return False
                        
                else:
                    bt.logging.error(f"❌ Docker installation failed: {install_result.stderr}")
                    return False
                    
            except subprocess.TimeoutExpired:
                bt.logging.error("❌ Docker installation timed out")
                return False
            except Exception as e:
                bt.logging.error(f"❌ Docker installation error: {e}")
                return False
    
    def _setup_base_container(self):
        """Build base Docker image with all dependencies"""
        try:
            # Check if Docker is installed, install if missing
            self._ensure_docker_installed()
            
            dockerfile_path = Path(__file__).parent / "Dockerfile"
            # Build context should be the parent of swarm package
            build_context = Path(__file__).parent.parent.parent
            
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
    

    async def evaluate_model(
        self, 
        task: MapTask, 
        uid: int, 
        model_path: Path
    ) -> ValidationResult:
        """Evaluate model in isolated Docker container"""
        
        if not DockerSecureEvaluator._base_ready:
            bt.logging.warning(f"Docker not ready for UID {uid}")
            return ValidationResult(uid, False, 0.0, 0.0, 0.0)
        
        # Track fake models detected for this evaluation
        self.last_fake_model_info = None
        
        container_name = f"swarm_eval_{uid}_{int(time.time() * 1000)}"
        
        try:
            # Create temp directory for task/result files
            with tempfile.TemporaryDirectory() as tmpdir:
                task_file = Path(tmpdir) / "task.json"
                result_file = Path(tmpdir) / "result.json"
                
                # Write task data
                from dataclasses import asdict
                with open(task_file, 'w') as f:
                    json.dump(asdict(task), f)
                
                # Docker run command with security limits - mount entire temp dir
                cmd = [
                    "docker", "run",
                    "--rm",
                    "--name", container_name,
                    "--user", "root",  # Run as root to allow writing to mounted directories
                    "--memory=6g",
                    "--cpus=2",
                    "--pids-limit=20",
                    "--ulimit", "nofile=64:64",
                    "--ulimit", "fsize=524288000:524288000",
                    "--security-opt", "no-new-privileges",
                    "--network", "none",
                    "-v", f"{tmpdir}:/workspace/shared",
                    "-v", f"{model_path.absolute()}:/workspace/model.zip:ro",
                    self.base_image,
                    "/workspace/shared/task.json",
                    str(uid),
                    "/workspace/model.zip",
                    "/workspace/shared/result.json"
                ]
                
                # Execute with timeout
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(),
                        timeout=EVAL_TIMEOUT_SEC + 10
                    )
                    
                    if proc.returncode != 0:
                        stderr_str = stderr.decode() if stderr else ""
                        bt.logging.debug(f"Container failed for UID {uid}: {stderr_str[:300]}")
                    
                except asyncio.TimeoutError:
                    # Kill container if timeout
                    subprocess.run(["docker", "kill", container_name], capture_output=True)
                    bt.logging.warning(f"Container timeout for UID {uid}")
                
                # Read results
                if result_file.exists():
                    try:
                        with open(result_file, 'r') as f:
                            data = json.load(f)
                        
                        # Check for fake model detection
                        if data.get('is_fake_model', False):
                            bt.logging.warning(f"🚫 FAKE MODEL DETECTED for UID {uid}: {data.get('fake_reason', 'Unknown')}")
                            bt.logging.debug(f"Inspection results: {data.get('inspection_results', {})}")
                            
                            # Store fake model info for blacklisting
                            self.last_fake_model_info = {
                                'uid': uid,
                                'reason': data.get('fake_reason', 'Unknown'),
                                'inspection_results': data.get('inspection_results', {})
                            }
                            
                            # Return zero score for fake models
                            return ValidationResult(uid, False, 0.0, 0.0, 0.0)
                        
                        had_error = "error" in data
                        if had_error:
                            bt.logging.debug(f"Evaluation error for UID {uid}: {data['error']}")
                        
                        result_data = {k: v for k, v in data.items() if k not in ["error", "is_fake_model", "fake_reason", "inspection_results"]}
                        
                        # Clear fake model info since this was a legitimate evaluation
                        self.last_fake_model_info = None
                        
                        # Apply reward floor logic
                        if not had_error and float(result_data.get("score", 0.0)) == 0.0:
                            result_data["score"] = 0.01
                        
                        # Log result data exactly as requested - custom format with emoji
                        from datetime import datetime
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"{timestamp} 🔍 DEBUG: UID {uid} result_data: {result_data}")
                        
                        return ValidationResult(**result_data)
                        
                    except Exception as e:
                        bt.logging.warning(f"Failed to parse result for UID {uid}: {e}")
                
        except Exception as e:
            bt.logging.warning(f"Docker evaluation failed for UID {uid}: {e}")
            # Ensure container is killed
            subprocess.run(["docker", "kill", container_name], capture_output=True)
        
        return ValidationResult(uid, False, 0.0, 0.0, 0.0)
    
    
    def cleanup(self):
        """Clean up any orphaned containers"""
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
        
        except Exception as e:
            bt.logging.warning(f"Container cleanup failed: {e}")