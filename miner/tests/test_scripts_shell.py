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

"""The miner's setup scripts.

They live outside `scripts/`, so the checks that walk that directory do not see
them. Losing them once already went unnoticed: the suite stayed green while the
scripts went unchecked.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

MINER_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "src" / "scripts"
EXPECTED = {"setup.sh", "install_dependencies.sh"}


def _miner_shell_scripts() -> list[Path]:
    return sorted(MINER_SCRIPTS_DIR.glob("*.sh"))


def test_the_expected_scripts_are_there():
    """Named rather than counted, so a rename cannot quietly reduce the set."""
    assert {p.name for p in _miner_shell_scripts()} == EXPECTED


@pytest.mark.parametrize("script", _miner_shell_scripts(), ids=lambda p: p.name)
def test_shell_script_has_shebang(script: Path):
    first = script.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    assert first.startswith("#!"), f"missing shebang in {script}"


@pytest.mark.parametrize("script", _miner_shell_scripts(), ids=lambda p: p.name)
def test_shell_script_parses_with_bash_n(script: Path):
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, f"{script} does not parse:\n{result.stderr}"


@pytest.mark.parametrize("script", _miner_shell_scripts(), ids=lambda p: p.name)
def test_shell_script_is_not_world_writable(script: Path):
    assert not (script.stat().st_mode & 0o002), f"{script} is world-writable"


def test_setup_script_stays_executable():
    """A permission change arrives as an ordinary edit, so pin the bit itself."""
    setup = MINER_SCRIPTS_DIR / "setup.sh"
    assert setup.stat().st_mode & 0o111, f"{setup} is not executable"
