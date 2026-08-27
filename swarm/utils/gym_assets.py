"""Stage family drone URDFs in a writable dir and point BaseAviary at them.

BaseAviary hardcodes its URDF lookup to
``pkg_resources.resource_filename('gym_pybullet_drones', 'assets/' + self.URDF)``
(both ``_parseURDFParameters`` and ``_housekeeping``), so historically the family
URDFs were copied into the installed package's assets dir at runtime. That write
fails whenever site-packages is not writable — non-root containers (the CI test
image runs as the host uid), read-only mounts, shared installs.

Instead, the URDFs are staged under a per-uid temp dir and a pass-through
resolver is installed over BaseAviary's module-level ``pkg_resources``: it
returns the staged absolute path for registered basenames and delegates every
other lookup (plane.urdf, cf2x.urdf, ...) to the real pkg_resources. Only
BaseAviary's global is replaced — BaseControl/CTBRControl keep their own
imports and resolve stock assets untouched.
"""
from __future__ import annotations

import os
import tempfile

import gym_pybullet_drones.envs.BaseAviary as _base_aviary_mod

# basename -> absolute staged path, consulted by the resolver below
_STAGED: dict[str, str] = {}

_REAL_PKG_RESOURCES = _base_aviary_mod.pkg_resources


class _StagedAssetResolver:
    """Drop-in for BaseAviary's ``pkg_resources`` global: staged basenames
    resolve to their writable-dir copies, everything else passes through."""

    def resource_filename(self, package: str, resource: str) -> str:
        if package == "gym_pybullet_drones" and resource.startswith("assets/"):
            staged = _STAGED.get(resource[len("assets/"):])
            if staged is not None:
                return staged
        return _REAL_PKG_RESOURCES.resource_filename(package, resource)

    def __getattr__(self, name):
        return getattr(_REAL_PKG_RESOURCES, name)


# Installed at import; the family modules import this one before any env is
# built, and Python's import lock makes the swap race-free.
if not isinstance(_base_aviary_mod.pkg_resources, _StagedAssetResolver):
    _base_aviary_mod.pkg_resources = _StagedAssetResolver()


def stage_dir() -> str:
    """Writable staging dir, per-uid so shared hosts don't collide."""
    uid = os.getuid() if hasattr(os, "getuid") else "na"
    path = os.path.join(tempfile.gettempdir(), f"swarm_gym_assets_v1_{uid}")
    os.makedirs(path, exist_ok=True)
    return path


def register(basename: str, abs_path: str) -> str:
    """Route ``assets/<basename>`` lookups to ``abs_path``; returns the basename
    for assignment to ``self.URDF``."""
    _STAGED[basename] = abs_path
    return basename


def staged_path(basename: str) -> str:
    """Absolute path of a registered staged asset, for direct p.loadURDF calls."""
    return _STAGED[basename]


def copy_verified(src: str, dst: str) -> None:
    """Copy src to dst unless already byte-identical. Atomic and race-safe
    across workers; re-verified after the write so every validator parses
    byte-identical physical constants."""
    with open(src, "rb") as f:
        src_bytes = f.read()
    if os.path.exists(dst):
        with open(dst, "rb") as f:
            if f.read() == src_bytes:
                return
    tmp = f"{dst}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(src_bytes)
    os.replace(tmp, dst)  # atomic, race-safe across workers
    with open(dst, "rb") as f:
        if f.read() != src_bytes:
            raise RuntimeError(f"{os.path.basename(dst)} content mismatch in staged assets")
