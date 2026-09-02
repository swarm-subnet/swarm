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


def test_the_wheel_source_copy_skips_a_virtualenv(tmp_path):
    """The install instructions create `.venv` in the repository root. Its
    `bin/python` points at the host interpreter, which does not exist inside the
    test container, so copying it raises shutil.Error and every test that needs a
    wheel errors before it runs."""
    repo = tmp_path / "repo"
    (repo / ".venv" / "bin").mkdir(parents=True)
    (repo / ".venv" / "bin" / "python").symlink_to("/nonexistent/host/python")
    (repo / "pyproject.toml").write_text("[project]\nname = 'x'\n")

    shutil.copytree(repo, tmp_path / "copy", ignore=wheel_source_ignore())

    assert not (tmp_path / "copy" / ".venv").exists()
