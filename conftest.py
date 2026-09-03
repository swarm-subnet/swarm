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

"""Pytest configuration for ensuring local package imports."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest


def _ensure_repo_on_syspath() -> None:
    repo_root = Path(__file__).resolve().parent
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _configure_xdist_worker() -> None:
    worker = os.getenv("PYTEST_XDIST_WORKER")
    if not worker:
        return
    if not os.getenv("SWARM_TERRAIN_CACHE_DIR"):
        cache_dir = Path(tempfile.gettempdir()) / "swarm_terrain_cache" / worker
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["SWARM_TERRAIN_CACHE_DIR"] = str(cache_dir)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")


_ensure_repo_on_syspath()
_configure_xdist_worker()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-e2e",
        action="store_true",
        default=False,
        help="Run e2e/runtime tests that are skipped by default.",
    )
    parser.addoption(
        "--run-full",
        action="store_true",
        default=False,
        help="Run full-suite heavy tests that are skipped by default.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    run_e2e = config.getoption("--run-e2e") or os.getenv("SWARM_RUN_E2E") == "1"
    run_full = config.getoption("--run-full") or os.getenv("SWARM_RUN_FULL") == "1"

    skip_e2e = pytest.mark.skip(
        reason="e2e/runtime tests are opt-in; use --run-e2e or SWARM_RUN_E2E=1"
    )
    skip_full = pytest.mark.skip(
        reason="full-suite heavy tests are opt-in; use --run-full or SWARM_RUN_FULL=1"
    )
    for item in items:
        if not run_e2e and item.get_closest_marker("e2e") is not None:
            item.add_marker(skip_e2e)
        if not run_full and item.get_closest_marker("full") is not None:
            item.add_marker(skip_full)


# ── packaging ────────────────────────────────────────────────────────────────
# Shared because both test roots ask questions about the same distribution, and
# building a wheel twice to answer them would double the cost.

@pytest.fixture(scope="session")
def selected_files() -> set[str]:
    """What MANIFEST.in selects, resolved by the code that builds the package."""
    from setuptools._distutils.filelist import FileList

    repo_root = Path(__file__).resolve().parent
    cwd = os.getcwd()
    os.chdir(repo_root)
    try:
        file_list = FileList()
        file_list.findall()
        for raw in Path("MANIFEST.in").read_text().splitlines():
            line = raw.strip()
            if line and not line.startswith("#"):
                file_list.process_template_line(line)
        return {name.replace(os.sep, "/") for name in file_list.files}
    finally:
        os.chdir(cwd)


@pytest.fixture(scope="session")
def setuptools_config() -> dict:
    import tomllib

    repo_root = Path(__file__).resolve().parent
    with (repo_root / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh).get("tool", {}).get("setuptools", {})


_WHEEL_SOURCE_JUNK = shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "*.egg-info", "build")


def wheel_source_ignore():
    """What never belongs in the copy the wheel is built from.

    Virtualenvs are recognised by the `pyvenv.cfg` they contain rather than by
    name: the setup scripts make `validator_env` and `miner_env`, the install
    steps make `.venv`, and .gitignore lists several more. Their `bin/python`
    points at the interpreter that created them, so copying one from a mounted
    repository into a container follows a symlink to a path that is not there."""
    def ignore(directory, names):
        ignored = set(_WHEEL_SOURCE_JUNK(directory, names))
        for name in names:
            if (Path(directory) / name / "pyvenv.cfg").is_file():
                ignored.add(name)
        return ignored

    return ignore


@pytest.fixture(scope="session")
def wheel_contents(tmp_path_factory) -> set[str]:
    """What a built wheel actually holds.

    The packaging rules say what should ship; only the artifact says what does.
    Built from a copy so a concurrent test cannot see a half-written build tree."""
    import shutil
    import subprocess
    import zipfile

    repo_root = Path(__file__).resolve().parent
    source = tmp_path_factory.mktemp("wheel_src") / "repo"
    # A link pointing outside the repository cannot be part of a wheel, and letting
    # copytree fail on one turns an unrelated stray file into an error here.
    shutil.copytree(
        repo_root, source,
        ignore=wheel_source_ignore(), ignore_dangling_symlinks=True,
    )
    out = tmp_path_factory.mktemp("wheel_out")
    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(out), str(source)],
        cwd=str(source), capture_output=True, text=True, timeout=900,
    )
    built = sorted(out.glob("*.whl"))
    if result.returncode != 0 or not built:
        pytest.fail(f"building the wheel failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
    with zipfile.ZipFile(built[0]) as wheel:
        return set(wheel.namelist())
