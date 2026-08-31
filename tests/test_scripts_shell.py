from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
MINER_SCRIPTS_DIR = REPO_ROOT / "miner" / "src" / "scripts"


def _all_shell_scripts() -> list[Path]:
    return sorted(
        p for d in (SCRIPTS_DIR, MINER_SCRIPTS_DIR) if d.is_dir() for p in d.rglob("*.sh")
    )


def test_all_shell_scripts_discovered():
    scripts = _all_shell_scripts()
    assert scripts, "No shell scripts found under scripts/"
    # the miner's scripts live outside scripts/, so a glob of it alone misses them
    found = {p.name for p in scripts if p.parent == MINER_SCRIPTS_DIR}
    assert found == {"setup.sh", "install_dependencies.sh"}, (
        f"expected the miner's two scripts under {MINER_SCRIPTS_DIR}, found {sorted(found)}"
    )


@pytest.mark.parametrize(
    "script_path", _all_shell_scripts(), ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_shell_script_has_shebang(script_path: Path):
    first_line = script_path.read_text(encoding="utf-8", errors="ignore").splitlines()[
        0
    ]
    assert first_line.startswith("#!"), f"Missing shebang in {script_path}"


@pytest.mark.parametrize(
    "script_path", _all_shell_scripts(), ids=lambda p: str(p.relative_to(REPO_ROOT))
)
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
        REPO_ROOT / "miner" / "src" / "scripts" / "setup.sh",
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


def test_setup_scripts_stay_executable():
    """A permission change shows up as an ordinary edit, so pin the bit itself."""
    for script in [
        REPO_ROOT / "miner" / "src" / "scripts" / "setup.sh",
        REPO_ROOT / "scripts" / "validator" / "main" / "setup.sh",
    ]:
        assert script.stat().st_mode & 0o111, f"{script} is not executable"
