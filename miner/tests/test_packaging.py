"""What the miner's side of the package ships.

The miner's code ships so an installed copy can run it; the miner's tests do not.
Python files reach a distribution through package discovery and everything else
only if MANIFEST.in names it, so the shell scripts are the ones listed here.
"""
from __future__ import annotations

import pytest

MINER_DATA_FILES = [
    "miner/src/scripts/setup.sh",
    "miner/src/scripts/install_dependencies.sh",
]

MINER_IN_THE_WHEEL = [
    *MINER_DATA_FILES,
    "miner/src/miner.py",
    "miner/src/submission_template/drone_agent.py",
]


@pytest.mark.parametrize("relative_path", MINER_DATA_FILES)
def test_manifest_selects_the_miner_scripts(relative_path, selected_files):
    assert relative_path in selected_files, f"{relative_path} is not selected for the package"


def test_the_miner_package_is_discovered(setuptools_config):
    include = setuptools_config.get("packages", {}).get("find", {}).get("include", [])
    assert "miner*" in include, "the miner package would not be discovered"


def test_the_miner_tests_do_not_ship(selected_files, setuptools_config):
    exclude = setuptools_config.get("packages", {}).get("find", {}).get("exclude", [])
    assert "miner.tests*" in exclude, "miner tests would be discovered as a package"
    shipped = [p for p in selected_files if p.startswith("miner/tests/")]
    assert shipped == [], f"MANIFEST.in still selects miner tests: {shipped}"


def test_the_wheel_carries_the_miner_side(wheel_contents):
    missing = [p for p in MINER_IN_THE_WHEEL if p not in wheel_contents]
    assert missing == [], f"the wheel is missing {missing}"

    shipped_tests = sorted(n for n in wheel_contents if n.startswith("miner/tests"))
    assert shipped_tests == [], f"the wheel carries miner tests: {shipped_tests}"
