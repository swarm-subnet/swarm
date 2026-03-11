import hashlib
import os
import subprocess
from pathlib import Path
from typing import Optional

import bittensor as bt

from swarm.config import DockerRuntimeSettings, env_bool
from swarm.constants import DOCKER_WORKER_CPUS, DOCKER_WORKER_MEMORY

from ._shared import _THREAD_CAP_ENV_VARS


def __new__(cls):
    if cls._instance is None:
        cls._instance = super().__new__(cls)
    return cls._instance


def _docker_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def _swarm_package_dir() -> Path:
    return _docker_dir().parent.parent


def _repo_root() -> Path:
    return _swarm_package_dir().parent

def __init__(self):
    evaluator_cls = self.__class__
    # Only initialize attributes on first instantiation
    if not hasattr(self, "base_image"):
        self.base_image = "swarm_evaluator_base:latest"
        self.last_fake_model_info = None

    if not evaluator_cls._base_ready:
        self._setup_base_container()
        evaluator_cls._base_ready = self.base_ready

def _check_docker_available(self):
    """Check if Docker is installed and available"""
    try:
        # Check if Docker command exists
        result = subprocess.run(
            ["docker", "--version"], capture_output=True, text=True, check=True
        )
        bt.logging.info(f"Docker found: {result.stdout.strip()}")
        return True

    except (subprocess.CalledProcessError, FileNotFoundError):
        bt.logging.error("🐳 Docker not found! Please install Docker manually.")
        bt.logging.error(
            "📖 See installation instructions in swarm/requirements.txt"
        )
        return False

@staticmethod
def _env_truthy(name: str) -> bool:
    return env_bool(name, False)

@classmethod
def _docker_env_overrides(cls) -> dict[str, str]:
    settings = DockerRuntimeSettings.from_env()
    return settings.docker_env_overrides(_THREAD_CAP_ENV_VARS)

@staticmethod
def _split_worker_cpuset_map(raw: str) -> list[str]:
    return DockerRuntimeSettings.split_cpuset_map(raw)

@classmethod
def _resolve_worker_limits(cls, worker_id: int) -> dict[str, Optional[str]]:
    limits = DockerRuntimeSettings.from_env().resolve_worker_limits(worker_id)
    return {
        "cpus": limits.cpus if limits.cpus not in (None, "") else DOCKER_WORKER_CPUS,
        "memory": limits.memory if limits.memory not in (None, "") else DOCKER_WORKER_MEMORY,
        "cpuset_cpus": limits.cpuset_cpus if limits.cpuset_cpus not in (None, "") else None,
    }

def _calculate_docker_hash(self) -> str:
    """Calculate hash of all source files that go into the Docker image."""
    dockerfile = _docker_dir() / "Dockerfile"
    requirements = _docker_dir() / "docker-requirements.txt"
    swarm_pkg = _swarm_package_dir()

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
            [
                "docker",
                "inspect",
                "--format",
                '{{index .Config.Labels "swarm.code_hash"}}',
                self.base_image,
            ],
            capture_output=True,
            text=True,
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
        ["docker", "images", "-q", self.base_image], capture_output=True, text=True
    )
    if not result.stdout.strip():
        bt.logging.info(
            f"🐳 Docker image not found, will build (hash: {current_hash})"
        )
        return True

    # Get hash from the image label (not from cache file)
    image_hash = self._get_image_hash_label()

    if image_hash == current_hash:
        bt.logging.info(f"✅ Docker image up-to-date (hash: {current_hash})")
        return False

    if image_hash:
        bt.logging.info(
            f"🔄 Code changed: image={image_hash}, current={current_hash} - rebuilding"
        )
    else:
        bt.logging.info(
            f"🔄 Image missing hash label - rebuilding (hash: {current_hash})"
        )

    return True

def _setup_base_container(self):
    evaluator_cls = self.__class__
    try:
        if not self._check_docker_available():
            bt.logging.error("❌ Docker is required but not installed")
            self.base_ready = False
            evaluator_cls._base_ready = False
            return

        if not self._should_rebuild_base_image():
            self.base_ready = True
            evaluator_cls._base_ready = True
            return

        bt.logging.info(f"🐳 Building base Docker image {self.base_image}...")

        old_image_id = None
        try:
            result = subprocess.run(
                ["docker", "images", "-q", self.base_image],
                capture_output=True,
                text=True,
            )
            old_image_id = result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            pass

        try:
            subprocess.run(
                ["docker", "container", "prune", "-f"], capture_output=True
            )
            subprocess.run(
                "docker rm -f $(docker ps -aq --filter=name=swarm_eval_)",
                shell=True,
                capture_output=True,
            )
            subprocess.run(
                "docker rm -f $(docker ps -aq --filter=name=swarm_verify_)",
                shell=True,
                capture_output=True,
            )
            subprocess.run(["docker", "image", "prune", "-f"], capture_output=True)
            subprocess.run(["docker", "volume", "prune", "-f"], capture_output=True)
            subprocess.run(
                ["docker", "builder", "prune", "-f", "--keep-storage", "5GB"],
                capture_output=True,
            )
        except Exception:
            pass

        dockerfile_path = _docker_dir() / "Dockerfile"
        build_context = _repo_root()
        current_hash = self._calculate_docker_hash()

        cmd = [
            "docker",
            "build",
            "--label",
            f"swarm.code_hash={current_hash}",
            "-t",
            self.base_image,
            "-f",
            str(dockerfile_path),
            str(build_context),
        ]

        bt.logging.info(f"Building base Docker image (hash: {current_hash})...")
        bt.logging.debug(f"Docker build command: {' '.join(cmd)}")

        env = os.environ.copy()
        env["DOCKER_BUILDKIT"] = "1"
        result = subprocess.run(cmd, text=True, env=env)

        if result.returncode == 0:
            self.base_ready = True
            evaluator_cls._base_ready = True
            bt.logging.info("✅ Base Docker image ready")

            if old_image_id:
                try:
                    new_result = subprocess.run(
                        ["docker", "images", "-q", self.base_image],
                        capture_output=True,
                        text=True,
                    )
                    new_image_id = (
                        new_result.stdout.strip()
                        if new_result.returncode == 0
                        else None
                    )
                    if new_image_id and old_image_id != new_image_id:
                        subprocess.run(
                            ["docker", "rmi", old_image_id], capture_output=True
                        )
                        bt.logging.debug(f"Removed old image: {old_image_id[:12]}")
                except Exception:
                    pass
        else:
            bt.logging.error(
                f"❌ Docker build failed with return code: {result.returncode}"
            )
            self.base_ready = False
            evaluator_cls._base_ready = False

    except Exception as e:
        bt.logging.error(f"Failed to setup Docker environment: {e}")
        self.base_ready = False
        evaluator_cls._base_ready = False
