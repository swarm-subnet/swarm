from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _all_shell_scripts() -> list[Path]:
    return sorted(SCRIPTS_DIR.rglob("*.sh"))


def test_all_shell_scripts_discovered():
    scripts = _all_shell_scripts()
    assert scripts, "No shell scripts found under scripts/"


@pytest.mark.parametrize("script_path", _all_shell_scripts(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_shell_script_has_shebang(script_path: Path):
    first_line = script_path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    assert first_line.startswith("#!"), f"Missing shebang in {script_path}"


@pytest.mark.parametrize("script_path", _all_shell_scripts(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_shell_script_parses_with_bash_n(script_path: Path):
    result = subprocess.run(
        ["bash", "-n", str(script_path)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, f"{script_path} failed bash -n: {result.stderr}"


def test_shell_scripts_use_strict_mode_for_deploy_scripts():
    deploy_scripts = [
        REPO_ROOT / "scripts" / "validator" / "update" / "update_deploy.sh",
        REPO_ROOT / "scripts" / "validator" / "update" / "auto_update_deploy.sh",
    ]
    for script in deploy_scripts:
        content = script.read_text(encoding="utf-8", errors="ignore")
        assert "set -euo pipefail" in content


def test_setup_scripts_define_main_entrypoint():
    setup_scripts = [
        REPO_ROOT / "scripts" / "miner" / "setup.sh",
        REPO_ROOT / "scripts" / "validator" / "main" / "setup.sh",
    ]
    for script in setup_scripts:
        content = script.read_text(encoding="utf-8", errors="ignore")
        assert "main()" in content
        assert 'main "$@"' in content


def test_scripts_are_not_world_writable():
    for script in _all_shell_scripts():
        mode = script.stat().st_mode
        assert not (mode & 0o002), f"{script} is world-writable"
