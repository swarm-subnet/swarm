import asyncio
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import bittensor as bt

from swarm.constants import TRAINING_TIMEOUT_SEC


class DockerTrainer:

    _trainer_base_ready = False
    _trainer_base_image = "swarm_trainer_base:latest"

    @classmethod
    def ensure_base_image(cls):
        """Ensure training base image exists"""
        if cls._trainer_base_ready:
            return True

        try:
            result = subprocess.run(
                ["docker", "images", "-q", cls._trainer_base_image],
                capture_output=True,
                text=True
            )
            if result.stdout.strip():
                cls._trainer_base_ready = True
                return True

            bt.logging.info("Building Docker training base image...")
            dockerfile_path = Path(__file__).parent / "Dockerfile.training"
            build_context = Path(__file__).parent.parent.parent.parent

            cmd = [
                "docker", "build",
                "--no-cache",
                "-t", cls._trainer_base_image,
                "-f", str(dockerfile_path),
                str(build_context)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                cls._trainer_base_ready = True
                bt.logging.info("Training base image built successfully")
                return True
            else:
                bt.logging.error(f"Failed to build training image: {result.stderr}")
                return False

        except Exception as e:
            bt.logging.error(f"Docker training setup failed: {e}")
            return False

    @classmethod
    async def run_training(
        cls,
        training_code_path: Path,
        output_dir: Path,
        seed: int,
        timesteps: int,
        uid: int,
    ) -> Optional[Path]:
        """
        Run training in Docker container.
        Returns path to trained model if successful, None otherwise.
        """
        if not cls.ensure_base_image():
            bt.logging.error("Training base image not available")
            return None

        container_name = f"swarm_train_{uid}_{int(time.time() * 1000)}"
        output_model_path = output_dir / "trained_model.zip"

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                os.chown(tmpdir, 1000, 1000)
                os.chmod(tmpdir, 0o755)

                code_extract_dir = Path(tmpdir) / "code"
                code_extract_dir.mkdir(parents=True, exist_ok=True)

                import zipfile
                with zipfile.ZipFile(training_code_path, 'r') as zf:
                    zf.extractall(code_extract_dir)

                for root, dirs, files in os.walk(code_extract_dir):
                    os.chown(root, 1000, 1000)
                    for d in dirs:
                        os.chown(os.path.join(root, d), 1000, 1000)
                    for f in files:
                        os.chown(os.path.join(root, f), 1000, 1000)

                output_dir.mkdir(parents=True, exist_ok=True)
                os.chown(output_dir, 1000, 1000)
                os.chmod(output_dir, 0o755)

                cmd = [
                    "docker", "run",
                    "--rm",
                    "--name", container_name,
                    "--user", "1000:1000",
                    "--memory=8g",
                    "--cpus=4",
                    "--pids-limit=20",
                    "--ulimit", "nofile=64:64",
                    "--ulimit", "fsize=524288000:524288000",
                    "--security-opt", "no-new-privileges",
                    "--network", "none",
                    "-v", f"{code_extract_dir}:/workspace/code:ro",
                    "-v", f"{output_dir}:/workspace/output",
                    cls._trainer_base_image,
                    "--seed", str(seed),
                    "--timesteps", str(timesteps),
                    "--output", "/workspace/output/trained_model.zip"
                ]

                bt.logging.info(f"Starting training for UID {uid} (seed={seed}, timesteps={timesteps})")

                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(),
                        timeout=TRAINING_TIMEOUT_SEC
                    )

                    if proc.returncode == 0:
                        if output_model_path.exists():
                            bt.logging.info(f"Training completed successfully for UID {uid}")
                            return output_model_path
                        else:
                            bt.logging.warning(f"Training completed but no output model for UID {uid}")
                            return None
                    else:
                        stderr_str = stderr.decode() if stderr else ""
                        bt.logging.warning(f"Training failed for UID {uid}: {stderr_str[:300]}")
                        return None

                except asyncio.TimeoutError:
                    subprocess.run(["docker", "kill", container_name], capture_output=True)
                    bt.logging.warning(f"Training timeout for UID {uid}")
                    return None

        except Exception as e:
            bt.logging.error(f"Training execution failed for UID {uid}: {e}")
            subprocess.run(["docker", "kill", container_name], capture_output=True)
            return None
        finally:
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)

    @classmethod
    def cleanup(cls):
        """Clean up training containers"""
        try:
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", "name=swarm_train_", "--format", "{{.Names}}"],
                capture_output=True,
                text=True
            )

            if result.returncode == 0 and result.stdout:
                containers = result.stdout.strip().split('\n')
                for container in containers:
                    if container:
                        subprocess.run(["docker", "rm", "-f", container], capture_output=True)

        except Exception as e:
            bt.logging.warning(f"Training container cleanup failed: {e}")
