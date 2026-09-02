# The MIT License (MIT)
# Copyright © 2026 Swarm

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import hashlib
import os
import stat
import subprocess
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Optional

import bittensor as bt

from swarm.challenge_families.base import ChallengeFamilyRuntimeProfile
from swarm.config import DockerRuntimeSettings, env_bool
from swarm.constants import DOCKER_WORKER_CPUS, DOCKER_WORKER_MEMORY

from ._shared import _THREAD_CAP_ENV_VARS
from .batch import remove_all_model_images
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
    if not hasattr(self, "last_fake_model_info"):
        self.last_fake_model_info = None
    if not hasattr(self, "base_images"):
        self.base_images = {"base": self.base_image}
    else:
        self.base_images.setdefault("base", self.base_image)
        self.base_image = self.base_images["base"]
    if not hasattr(self, "family_runtime_profiles"):
        self.family_runtime_profiles = {}
    if not hasattr(self, "last_runtime_profile_info"):
        self.last_runtime_profile_info = None
    if not hasattr(self, "last_selected_run_image"):
        self.last_selected_run_image = None
    if not hasattr(self, "last_selected_worker_limits"):
        self.last_selected_worker_limits = None
    if not hasattr(self, "last_selected_runtime_profile"):
        self.last_selected_runtime_profile = None
    if not hasattr(self, "last_selected_runtime_env"):
        self.last_selected_runtime_env = None

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
            "Docker is required for graph evaluation."
        )
        return False

@classmethod
def _docker_env_overrides(cls, thread_cap: Optional[int] = None) -> dict[str, str]:
    settings = DockerRuntimeSettings.from_env()
    return settings.docker_env_overrides(
        _THREAD_CAP_ENV_VARS,
        thread_cap=thread_cap,
    )


@staticmethod
def _coerce_runtime_profile(profile: Optional[Any]) -> Optional[ChallengeFamilyRuntimeProfile]:
    if profile is None:
        return None
    if isinstance(profile, ChallengeFamilyRuntimeProfile):
        return profile
    if isinstance(profile, dict):
        return ChallengeFamilyRuntimeProfile.from_mapping(profile)
    raise TypeError(f"Unsupported runtime profile payload: {type(profile)!r}")

@staticmethod
def _split_worker_cpuset_map(raw: str) -> list[str]:
    return DockerRuntimeSettings.split_cpuset_map(raw)

@classmethod
def _resolve_worker_limits(
    cls,
    worker_id: int,
    runtime_profile: Optional[Any] = None,
) -> dict[str, Optional[str]]:
    profile = cls._coerce_runtime_profile(runtime_profile)
    settings = DockerRuntimeSettings.from_env()
    limits = settings.resolve_worker_limits(worker_id)
    cpus_requested = settings.cpus_override not in (None, "")
    resolved = {
        "cpus": limits.cpus if limits.cpus not in (None, "") else DOCKER_WORKER_CPUS,
        "memory": limits.memory if limits.memory not in (None, "") else DOCKER_WORKER_MEMORY,
        "cpuset_cpus": limits.cpuset_cpus if limits.cpuset_cpus not in (None, "") else None,
    }
    if profile is not None:
        if profile.docker_worker_cpus not in (None, ""):
            resolved["cpus"] = str(profile.docker_worker_cpus)
            cpus_requested = True
        if profile.docker_worker_memory not in (None, ""):
            resolved["memory"] = str(profile.docker_worker_memory)
    drop_default_quota = (
        resolved["cpuset_cpus"]
        and not cpus_requested
        and not env_bool("SWARM_DOCKER_KEEP_CPU_QUOTA", False)
    )
    if drop_default_quota:
        # Pinned cores already bound the container; a quota on top only adds
        # mid-step throttling stalls.
        resolved["cpus"] = None
    return resolved


def _resolve_base_image_for_key(self, image_key: str) -> str:
    normalized = str(image_key or "base").strip() or "base"
    env_name = f"SWARM_DOCKER_BASE_IMAGE_{normalized.upper()}"
    env_override = os.getenv(env_name)
    if env_override:
        return env_override
    return str(getattr(self, "base_images", {}).get(normalized, self.base_image))

def _is_excluded_from_build(rel_posix: str, patterns: list[str]) -> bool:
    """Whether .dockerignore keeps this path out of the build context."""
    for pattern in patterns:
        if pattern.startswith("**/"):
            tail = pattern[3:]
            if any(fnmatch(part, tail) for part in rel_posix.split("/")):
                return True
        elif rel_posix == pattern or rel_posix.startswith(f"{pattern}/"):
            return True
    return False


def _build_context_files(root: Path, swarm_pkg: Path) -> list[Path]:
    """Every path `COPY swarm` puts in the image, after .dockerignore.

    Directories are included: COPY carries their permissions across, so a mode
    change on one alters the image with no file having changed."""
    try:
        lines = (root / ".dockerignore").read_text().splitlines()
    except OSError:
        lines = []
    patterns = [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]
    # A negation puts files back that an earlier line excluded; rather than work out
    # which, drop the patterns and hash the whole tree. Over-hashing only costs a build.
    if any(p.startswith("!") for p in patterns):
        patterns = []
    return sorted(
        p
        for p in swarm_pkg.rglob("*")
        if not _is_excluded_from_build(p.relative_to(root).as_posix(), patterns)
    )


def _calculate_docker_hash(self) -> str:
    """Cache key over every input that shapes the runner image.

    Entries are framed as path + type + mode + length + payload so file
    boundaries are unambiguous and a permission change is not mistaken for no
    change; a missing or unreadable input hashes as a distinct marker instead of
    silently keeping a stale key."""
    root = _repo_root()
    inputs = [
        root / ".dockerignore",
        _docker_dir() / "Dockerfile",
        _docker_dir() / "docker-requirements.txt",
    ]
    inputs.extend(_build_context_files(root, _swarm_package_dir()))

    hasher = hashlib.sha256()
    for f in inputs:
        try:
            if f.is_symlink():
                kind, mode, payload = b"l", 0, os.readlink(f).encode()
            elif f.is_dir():
                kind, mode, payload = b"d", stat.S_IMODE(f.stat().st_mode), b""
            else:
                kind, mode = b"f", stat.S_IMODE(f.stat().st_mode)
                payload = f.read_bytes()
        except OSError:
            kind, mode, payload = b"x", 0, b"__missing_input__"
        rel = f.relative_to(root).as_posix().encode()
        hasher.update(len(rel).to_bytes(4, "big"))
        hasher.update(rel)
        hasher.update(kind)
        hasher.update(mode.to_bytes(4, "big"))
        hasher.update(len(payload).to_bytes(8, "big"))
        hasher.update(payload)

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
            remove_all_model_images()
            subprocess.run(["docker", "image", "prune", "-f"], capture_output=True)
            subprocess.run(["docker", "volume", "prune", "-f"], capture_output=True)
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
