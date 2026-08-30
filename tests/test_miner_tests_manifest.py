"""The miner manifest has to stay honest about what it covers.

A miner-only pull request runs the tests this manifest names and nothing else, so
a row whose tests do not exist would report a pass over an unchecked change. These
check the manifest's shape; whether the tests themselves are good is a separate
question they cannot answer.
"""
from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "ci" / "miner_tests.toml"


@pytest.fixture(scope="module")
def manifest() -> dict:
    with MANIFEST_PATH.open("rb") as fh:
        return tomllib.load(fh)


@pytest.fixture(scope="module")
def files(manifest) -> dict[str, list[str]]:
    return {k: v for k, v in manifest["files"].items() if k != "extra_tests"}


def _selectors(manifest, files) -> list[str]:
    named = [s for group in files.values() for s in group]
    return named + list(manifest["files"].get("extra_tests", []))


def test_every_listed_file_names_its_tests(files):
    empty = [path for path, tests in files.items() if not tests]
    assert empty == [], f"listed with nothing covering them: {empty}"


def test_every_listed_file_exists(files):
    missing = [path for path in files if not (REPO_ROOT / path).exists()]
    assert missing == [], f"listed but not in the tree: {missing}"


def test_no_path_is_a_pattern(files):
    globs = [p for p in files if any(c in p for c in "*?[")]
    assert globs == [], f"paths are exact, these are not: {globs}"


def test_every_selector_collects(manifest, files):
    """A selector that names nothing would silently shrink the miner job."""
    selectors = _selectors(manifest, files)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", *selectors],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"a selector collects nothing:\n{result.stdout[-2000:]}"


def test_listed_files_stay_inside_the_declared_bounds(manifest, files):
    bounds = manifest["bounds"]
    exact, prefixes = set(bounds["exact"]), tuple(bounds["prefixes"])
    outside = [p for p in files if p not in exact and not p.startswith(prefixes)]
    assert outside == [], f"outside the declared miner side: {outside}"


def test_routing_files_are_never_listed(manifest, files):
    """Routing decides what runs, so a change to it is not a miner-only change."""
    protected = tuple(manifest["bounds"]["protected"])
    listed = [p for p in files if p.startswith(protected)]
    assert listed == [], f"routing files cannot be miner-only: {listed}"


def test_the_workflow_reads_the_rules_from_the_base_branch():
    """These rules only mean anything if a branch cannot supply its own copy."""
    workflow = (REPO_ROOT / ".github" / "workflows" / "tests.yml").read_text()
    for path in ["ci/classify_changes.py", "ci/miner_tests.toml"]:
        assert f'git show "$GITHUB_SHA^1:{path}"' in workflow, (
            f"{path} must come from the first parent, not the branch under test"
        )
