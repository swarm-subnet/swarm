"""Data files an installed copy needs.

The package is code plus a handful of data files it loads by name at runtime.
Whether those reach an install depends on MANIFEST.in selecting them and on the
package-data settings not throwing them back out, and nothing else in the suite
would notice them going missing: every other test reads the source tree, where
they are there either way.
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

import pytest
from setuptools._distutils.filelist import FileList
from setuptools._distutils.filelist import translate_pattern

REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DATA_FILES = [
    "swarm/model_graph/model_graph.schema.json",
    "swarm/model_graph/execution_profile.v1.json",
    "swarm/validator/calibration/baseline_manifest.json",
    "swarm/validator/calibration/baseline_model.zip",
]


@pytest.fixture(scope="module")
def selected_files() -> set[str]:
    """What MANIFEST.in selects, resolved by the code that builds the package."""
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
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


@pytest.fixture(scope="module")
def setuptools_config() -> dict:
    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh).get("tool", {}).get("setuptools", {})


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
