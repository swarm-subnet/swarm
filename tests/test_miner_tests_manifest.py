"""The miner manifest has to stay honest about what it covers.

A miner-only pull request runs the tests this manifest names and nothing else, so
a row whose tests do not exist would report a pass over an unchecked change. These
check the manifest's shape; whether the tests themselves are good is a separate
question they cannot answer.
"""
from __future__ import annotations

import os
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


def _lane_script() -> str:
    """The workflow's own lane-picking shell, so this tests what CI will run."""
    import yaml

    workflow = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "tests.yml").read_text())
    steps = workflow["jobs"]["tests"]["steps"]
    picks = [s["run"] for s in steps if s.get("id") == "pick"]
    assert len(picks) == 1, "expected exactly one lane-picking step"
    return picks[0]


@pytest.mark.parametrize(
    "change, expected",
    [
        ('printf "\\n" >> miner/src/miner.py', "miner"),
        ('printf "\\n" >> swarm/validator/reward.py', "full"),
        # the branch rewrites the rules it is about to be judged by
        ('printf "\\n" >> swarm/validator/reward.py; '
         'printf \'[files]\\n"swarm/validator/reward.py" = ["tests/test_github.py"]\\n'
         '[bounds]\\nexact=[]\\nprefixes=[]\\nprotected=[]\\n\' > ci/miner_tests.toml', "full"),
        ('printf "\\n" >> swarm/validator/reward.py; '
         'printf \'print("miner")\\n\' > ci/classify_changes.py', "full"),
    ],
)
def test_the_lane_comes_from_the_base_branch(change, expected, tmp_path):
    """A branch must not be able to widen the rules that decide how much of the
    suite it faces, so the lane is picked using the first parent's copy."""
    repo = tmp_path / "repo"
    run = lambda *a, **k: subprocess.run(*a, cwd=repo, check=True, capture_output=True, **k)
    # Only the routing files and the two paths the cases touch: cloning the real
    # repository for this costs hundreds of megabytes and proves nothing extra.
    (repo / "ci").mkdir(parents=True)
    (repo / "miner" / "src").mkdir(parents=True)
    (repo / "swarm" / "validator").mkdir(parents=True)
    for name in ("classify_changes.py", "miner_tests.toml"):
        (repo / "ci" / name).write_bytes((REPO_ROOT / "ci" / name).read_bytes())
    (repo / "miner" / "src" / "miner.py").write_text("x = 1\n")
    (repo / "swarm" / "validator" / "reward.py").write_text("x = 1\n")
    run(["git", "init", "-q", "."])
    run(["git", "config", "user.email", "t@t"])
    run(["git", "config", "user.name", "t"])
    run(["git", "add", "-A"])
    run(["git", "commit", "-q", "-m", "base"])
    base = run(["git", "rev-parse", "HEAD"], text=True).stdout.strip()

    subprocess.run(change, cwd=repo, shell=True, check=True)
    run(["git", "add", "-A"])
    run(["git", "commit", "-q", "-m", "pr"])
    head = run(["git", "rev-parse", "HEAD"], text=True).stdout.strip()
    run(["git", "checkout", "-q", base])
    merge = run(["git", "commit-tree", f"{head}^{{tree}}", "-p", base, "-p", head,
                 "-m", "merge"], text=True).stdout.strip()

    result = subprocess.run(
        _lane_script(), cwd=repo, shell=True, capture_output=True, text=True,
        env={"PATH": os.environ["PATH"], "HOME": str(tmp_path),
             "GITHUB_EVENT_NAME": "pull_request", "GITHUB_SHA": merge,
             "GITHUB_OUTPUT": str(tmp_path / "out")},
    )
    assert result.returncode == 0, result.stderr[-1500:]
    assert f"lane={expected}" in (tmp_path / "out").read_text(), result.stdout
