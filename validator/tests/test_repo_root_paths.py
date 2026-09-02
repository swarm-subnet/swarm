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

"""Paths under `validator/` that are written relative to the file holding them.

Moving a file changes what `parents[n]` means, and these go wrong quietly: a
script inserts the wrong directory on `sys.path`, or a test reads a schema that
is not there and treats the absence as an empty result.

Every expression is pinned, not every file. A file with five resolvers can have
four right and one wrong, and a check that only asked whether *any* of them
reached the repository root would pass. The expected values also include the
ones that deliberately stay local, so a walk cannot quietly grow a level.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = REPO_ROOT / "validator"

_PATHLIB = re.compile(r"Path\(__file__\)(?:\.resolve\(\))?((?:\.parents\[\d\]|\.parent)+)")
_OSPATH = re.compile(r"os\.path\.dirname\(os\.path\.abspath\(__file__\)\)((?:,\s*os\.pardir)+)")
# A root derived from an alias rather than from __file__ directly. This is how a leak
# got through once: _SCRIPT_DIR was right, and the root built from it was not.
_ALIAS_ROOT = re.compile(r"^_REPO_ROOT\s*=\s*_SCRIPT_DIR((?:\.parents\[\d\]|\.parent)+)", re.M)

# path relative to validator/ -> the target of every resolver in it, sorted.
# "SIBLING" is the swarm-backend checkout beside the repository, not inside it.
EXPECTED: dict[str, list[str]] = {
    "scripts/bench_full_eval.py": ["REPO_ROOT"],
    "scripts/dump_depth_frame.py": ["REPO_ROOT"],
    "scripts/gen_family_io_tables.py": ["REPO_ROOT"],
    "scripts/generate_video.py": ["REPO_ROOT", "validator/scripts"],
    "scripts/health/check_current_epoch_weights.py": ["REPO_ROOT"],
    "scripts/health/check_validator_health.py": ["REPO_ROOT"],
    "scripts/prebake_mannequin_parts.py": ["REPO_ROOT"],
    "scripts/profile_walltime.py": ["REPO_ROOT"],
    "scripts/stress_benchmark_compare.py": ["REPO_ROOT", "validator/scripts"],
    "scripts/sync_family_registry.py": ["REPO_ROOT"],
    "scripts/test_timings.py": ["REPO_ROOT"],
    "scripts/verify_render_identity.py": ["REPO_ROOT"],
    "scripts/visualize_map.py": ["REPO_ROOT", "validator/scripts"],
    "tests/sar/test_mannequin.py": ["REPO_ROOT"],
    "tests/sar/test_no_coord_leak.py": ["REPO_ROOT"],
    "tests/test_benchmark_default_model_fixed_seeds.py": ["REPO_ROOT"],
    "tests/test_challenge_family_boundaries.py": ["REPO_ROOT"],
    "tests/test_cli.py": ["REPO_ROOT"],
    "tests/test_docker_evaluator.py": ["REPO_ROOT", "REPO_ROOT"],
    "tests/test_domain_model_naming.py": ["REPO_ROOT"],
    "tests/test_github.py": ["REPO_ROOT"] * 5,
    "tests/test_scripts_shell.py": ["REPO_ROOT"],
    "tests/test_submission_manifest.py": ["REPO_ROOT", "SIBLING"],
    "tests/test_swarm_autopilot_regression.py": ["validator/tests"],
}


def _steps(expr: str) -> int:
    """`parents[k]` is k+1 levels up; `.parent` is one."""
    total = 0
    for m in re.finditer(r"\.parents\[(\d)\]|\.parent\b", expr):
        total += int(m.group(1)) + 1 if m.group(1) else 1
    return total


def _label(path: Path) -> str:
    if path == REPO_ROOT:
        return "REPO_ROOT"
    if path == REPO_ROOT.parent:
        return "SIBLING"
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _targets(source: Path) -> list[str]:
    """Where every file-relative walk in this file actually lands."""
    text = source.read_text(errors="ignore")
    found = []
    for m in _PATHLIB.finditer(text):
        p = source.resolve()
        for _ in range(_steps(m.group(1))):
            p = p.parent
        found.append(_label(p))
    for m in _OSPATH.finditer(text):
        p = source.resolve().parent
        for _ in range(m.group(1).count("os.pardir")):
            p = p.parent
        found.append(_label(p))
    for m in _ALIAS_ROOT.finditer(text):
        p = source.resolve().parent          # _SCRIPT_DIR
        for _ in range(_steps(m.group(1))):
            p = p.parent
        found.append(_label(p))
    return sorted(found)


@pytest.mark.parametrize("relative", sorted(EXPECTED), ids=lambda p: p)
def test_every_resolver_lands_where_it_should(relative):
    source = VALIDATOR / relative
    assert source.is_file(), f"{relative} is listed here but not in the tree"
    assert _targets(source) == sorted(EXPECTED[relative]), (
        f"{relative} resolves to {_targets(source)}, expected {sorted(EXPECTED[relative])}"
    )


def test_the_sibling_backend_path_is_complete():
    """The generic check above only sees how far the walk goes, not what is
    appended to it. Renaming the directory would leave the walk at the same
    depth while the production test quietly skipped instead of comparing."""
    source = VALIDATOR / "tests" / "test_submission_manifest.py"
    expected = REPO_ROOT.parent / "swarm-backend" / "app"
    text = source.read_text()
    assert 'parents[3] / "swarm-backend" / "app"' in text, (
        f"the sibling checkout path no longer reads {expected}; a change here turns "
        "the manifest comparison into a permanent skip"
    )


def test_no_file_resolves_paths_without_being_listed():
    """A file that grows a recognised resolver must be listed, or it goes unchecked."""
    found = {
        str(p.relative_to(VALIDATOR))
        for p in VALIDATOR.rglob("*.py")
        if "__pycache__" not in str(p) and p.name != Path(__file__).name and _targets(p)
    }
    assert found == set(EXPECTED), (
        f"unlisted: {sorted(found - set(EXPECTED))}; "
        f"listed but gone: {sorted(set(EXPECTED) - found)}"
    )
