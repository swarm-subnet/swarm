"""Paths under `validator/` that are written relative to the file holding them.

Moving a file changes what `parents[n]` means, and the ones that walk up to the
repository root go wrong quietly: a script inserts the wrong directory on
`sys.path`, or a test reads a schema that is not there and treats the absence as
an empty result.

Each entry names its intended target, because they do not share one: most want
the repository root, one wants a sibling checkout beside it, and a few
deliberately want their own directory.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = REPO_ROOT / "validator"

_PATHLIB = re.compile(r"Path\(__file__\)(?:\.resolve\(\))?((?:\.parents\[\d\]|\.parent)+)")
_OSPATH = re.compile(r"os\.path\.dirname\(os\.path\.abspath\(__file__\)\)((?:,\s*os\.pardir)+)")
# A root derived from an alias rather than from __file__ directly. This is how the
# leak got through: _SCRIPT_DIR was correct, and the root built from it was not.
_ALIAS_ROOT = re.compile(r"^_REPO_ROOT\s*=\s*_SCRIPT_DIR((?:\.parents\[\d\]|\.parent)+)", re.M)

# Files whose walk is meant to reach the repository root.
REACH_THE_REPO_ROOT = [
    "scripts/bench_full_eval.py",
    "scripts/dump_depth_frame.py",
    "scripts/gen_family_io_tables.py",
    "scripts/prebake_mannequin_parts.py",
    "scripts/profile_walltime.py",
    "scripts/sync_family_registry.py",
    "scripts/test_timings.py",
    "scripts/verify_render_identity.py",
    "scripts/health/check_current_epoch_weights.py",
    "scripts/health/check_validator_health.py",
    "tests/sar/test_mannequin.py",
    "tests/sar/test_no_coord_leak.py",
    "tests/test_benchmark_default_model_fixed_seeds.py",
    "tests/test_challenge_family_boundaries.py",
    "tests/test_cli.py",
    "tests/test_docker_evaluator.py",
    "tests/test_domain_model_naming.py",
    "tests/test_github.py",
    "tests/test_scripts_shell.py",
    "tests/test_submission_manifest.py",
]

# These want their own directory, not the root.
STAY_LOCAL = {
    "scripts/generate_video.py",
    "scripts/stress_benchmark_compare.py",
    "scripts/visualize_map.py",
    "tests/test_swarm_autopilot_regression.py",
}


def _steps(expr: str) -> int:
    """`parents[k]` is k+1 levels up; `.parent` is one."""
    total = 0
    for m in re.finditer(r"\.parents\[(\d)\]|\.parent\b", expr):
        total += int(m.group(1)) + 1 if m.group(1) else 1
    return total


def _resolved(source: Path) -> list[Path]:
    """Where every file-relative walk in this file actually lands."""
    text = source.read_text(errors="ignore")
    out = []
    for m in _PATHLIB.finditer(text):
        landed = source.resolve()
        for _ in range(_steps(m.group(1))):
            landed = landed.parent
        out.append(landed)
    for m in _OSPATH.finditer(text):
        landed = source.resolve().parent
        for _ in range(m.group(1).count("os.pardir")):
            landed = landed.parent
        out.append(landed)
    return out


def _alias_roots(source: Path) -> list[Path]:
    """Roots built from `_SCRIPT_DIR` rather than from `__file__` directly."""
    out = []
    for m in _ALIAS_ROOT.finditer(source.read_text(errors="ignore")):
        landed = source.resolve().parent          # _SCRIPT_DIR
        for _ in range(_steps(m.group(1))):
            landed = landed.parent
        out.append(landed)
    return out


@pytest.mark.parametrize(
    "relative",
    ["scripts/visualize_map.py", "scripts/generate_video.py", "scripts/stress_benchmark_compare.py"],
    ids=lambda p: p,
)
def test_a_root_built_from_the_script_dir_is_still_the_repo_root(relative):
    """These insert their root on sys.path. Pointing it at `validator/` would make
    the old top-level `scripts` importable again from an installed copy."""
    roots = _alias_roots(VALIDATOR / relative)
    assert roots, f"{relative} no longer derives a root from _SCRIPT_DIR"
    assert set(roots) == {REPO_ROOT}, (
        f"{relative} derives {sorted(str(r) for r in roots)}, not the repository root"
    )


@pytest.mark.parametrize("relative", REACH_THE_REPO_ROOT, ids=lambda p: p)
def test_the_walk_reaches_the_repo_root(relative):
    source = VALIDATOR / relative
    landed = _resolved(source)
    assert landed, f"{relative} no longer computes a path from its own location"
    assert REPO_ROOT in landed, (
        f"{relative} walks up to {sorted({str(p) for p in landed})}, "
        f"none of which is the repository root"
    )


def test_the_sibling_backend_path_stays_beside_the_repo():
    """Compared lexically: the checkout is usually absent, and a guard that
    turned into a skip when it is would protect nothing."""
    source = VALIDATOR / "tests" / "test_submission_manifest.py"
    expected = REPO_ROOT.parent / "swarm-backend" / "app"
    text = source.read_text()
    assert 'parents[3] / "swarm-backend" / "app"' in text, (
        f"the sibling checkout path in {source.name} no longer reaches {expected}"
    )


def test_every_resolver_is_accounted_for():
    """A file that grows a new walk must be listed, or it goes unchecked."""
    found = set()
    for source in VALIDATOR.rglob("*.py"):
        if "__pycache__" in str(source) or source.name == Path(__file__).name:
            continue
        if _resolved(source):
            found.add(str(source.relative_to(VALIDATOR)))
    listed = set(REACH_THE_REPO_ROOT) | STAY_LOCAL
    assert found == listed, (
        f"unlisted resolvers: {sorted(found - listed)}; "
        f"listed but no longer present: {sorted(listed - found)}"
    )
