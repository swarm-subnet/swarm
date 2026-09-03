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

"""Data files an installed copy needs.

The package is code plus a handful of data files it loads by name at runtime.
Whether those reach an install depends on MANIFEST.in selecting them and on the
package-data settings not throwing them back out, and nothing else in the suite
would notice them going missing: every other test reads the source tree, where
they are there either way.

The miner's side of the same question lives in miner/tests/test_packaging.py.
"""
from __future__ import annotations

import shutil

import pytest
from setuptools._distutils.filelist import translate_pattern

from conftest import wheel_source_ignore

REQUIRED_DATA_FILES = [
    "swarm/model_graph/model_graph.schema.json",
    "swarm/model_graph/execution_profile.v1.json",
    "swarm/validator/calibration/baseline_manifest.json",
    "swarm/validator/calibration/baseline_model.zip",
]


@pytest.mark.parametrize("relative_path", REQUIRED_DATA_FILES)
def test_manifest_selects_the_data_files_the_package_loads(relative_path, selected_files):
    assert relative_path in selected_files, (
        f"{relative_path} is loaded at runtime but MANIFEST.in does not select it, "
        "so an installed copy would not have it"
    )


def test_package_data_is_included(setuptools_config):
    """Turning this off drops every data file MANIFEST.in selected."""
    assert setuptools_config.get("include-package-data", True) is True


@pytest.mark.parametrize("relative_path", REQUIRED_DATA_FILES)
def test_nothing_excludes_the_data_files_again(relative_path, setuptools_config):
    """exclude-package-data runs after selection, so it can take them back out."""
    package, _, within = relative_path.partition("/")
    for scope, patterns in setuptools_config.get("exclude-package-data", {}).items():
        if scope not in ("*", package):
            continue
        for pattern in patterns:
            assert not translate_pattern(pattern).match(within), (
                f"{relative_path} is excluded again by '{pattern}' under '{scope}'"
            )


def test_the_wheel_carries_what_an_install_needs(wheel_contents):
    """The rules above say what should ship; the artifact says what does."""
    missing = [p for p in REQUIRED_DATA_FILES if p not in wheel_contents]
    assert missing == [], f"the wheel is missing {missing}"


def _repo_with_virtualenv(root, name):
    """A repository holding a virtualenv made by a different interpreter."""
    venv = root / name
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /nonexistent/host/bin\n")
    for executable in ("python", "python3", "python3.11"):
        (venv / "bin" / executable).symlink_to("/nonexistent/host/bin/python3.11")
    (root / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    return root


# validator_env and miner_env come from the setup scripts, .venv from the install
# steps, and .gitignore lists several more. The last name is arbitrary on purpose:
# the copy has to recognise a virtualenv by what it holds, not what it is called.
@pytest.mark.parametrize(
    "name", ["validator_env", "miner_env", ".venv", "venv", "swarm_env", "whatever_env"]
)
def test_the_wheel_source_copy_skips_a_virtualenv(tmp_path, name):
    """A virtualenv's `bin/python` points at the interpreter that made it. Copying
    one from a mounted repository into a container follows that symlink to a path
    that is not there, so every test needing a wheel errors before it runs."""
    repo = _repo_with_virtualenv(tmp_path / "repo", name)

    shutil.copytree(
        repo, tmp_path / "copy",
        ignore=wheel_source_ignore(), ignore_dangling_symlinks=True,
    )

    assert not (tmp_path / "copy" / name).exists()


def test_the_wheel_source_copy_survives_a_stray_dangling_symlink(tmp_path):
    """Second layer: a link pointing outside the repository is skipped rather than
    raising, whatever it is and wherever it sits."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    (repo / "stray").symlink_to("/nonexistent/target")

    shutil.copytree(
        repo, tmp_path / "copy",
        ignore=wheel_source_ignore(), ignore_dangling_symlinks=True,
    )

    assert (tmp_path / "copy" / "pyproject.toml").is_file()
