"""
swarm.validator.loader
~~~~~~~~~~~~~~~~~~~~~~

Light‑weight utilities to import miner wheels in sandboxed virtual
environments.  Each wheel is installed **once** into its own venv that
lives alongside the cached .whl file; subsequent loads are instant.
"""
from __future__ import annotations

import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from venv import EnvBuilder


@contextmanager
def temp_venv(wheel_path: Path) -> Iterator[str]:
    """
    Context manager that guarantees the wheel at *wheel_path* is installed
    in an isolated virtual‑env and temporarily exposes its site‑packages
    directory on `sys.path`.

    Parameters
    ----------
    wheel_path : Path
        Absolute path to the miner's wheel file.

    Yields
    ------
    site_packages : str
        Path string of the virtual‑env's site‑packages (first on sys.path
        during the context).
    """
    if not wheel_path.is_file():
        raise FileNotFoundError(wheel_path)

    venv_dir = wheel_path.with_suffix(".venv")
    site_packages: Path

    # Initialise venv + install wheel (idempotent)
    if not venv_dir.exists():
        EnvBuilder(with_pip=True, clear=False).create(venv_dir)
        pip_exe = venv_dir / "bin" / "pip"
        subprocess.check_call([pip_exe, "install", wheel_path])

    # Locate `…/python3.X/site-packages`
    candidates = list((venv_dir / "lib").glob("python*/site-packages"))
    if not candidates:
        raise RuntimeError(f"Cannot locate site‑packages under {venv_dir}")
    site_packages = candidates[0]

    # Prepend to import path
    sys.path.insert(0, str(site_packages))
    try:
        yield str(site_packages)
    finally:
        # Remove the path while preserving wheel/venv on disk for re‑use
        with suppress(ValueError):
            sys.path.remove(str(site_packages))
