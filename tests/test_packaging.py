"""Data files an installed copy needs.

The package is code plus a handful of data files it loads by name at runtime.
Those only reach an install if MANIFEST.in names them, and nothing else in the
suite would notice their absence: every other test runs against the source tree,
where the files are there either way.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from setuptools._distutils.filelist import FileList

REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DATA_FILES = [
    "swarm/model_graph/model_graph.schema.json",
    "swarm/model_graph/execution_profile.v1.json",
    "swarm/validator/calibration/baseline_manifest.json",
    "swarm/validator/calibration/baseline_model.zip",
]


@pytest.fixture(scope="module")
def packaged_files() -> set[str]:
    """What MANIFEST.in selects, resolved by the code that builds the sdist."""
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


@pytest.mark.parametrize("relative_path", REQUIRED_DATA_FILES)
def test_package_ships_the_data_files_it_loads(relative_path, packaged_files):
    assert relative_path in packaged_files, (
        f"{relative_path} is loaded at runtime but MANIFEST.in does not select it, "
        "so an installed copy would not have it"
    )
