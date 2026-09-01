from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "validator" / "scripts"


def _all_shell_scripts() -> list[Path]:
    return sorted(SCRIPTS_DIR.rglob("*.sh"))


# Named, not counted: losing one of the four would drop its parametrised checks
# while a "there is at least one" assertion stayed green.
EXPECTED_SCRIPTS = {
    "main/install_dependencies.sh",
    "main/setup.sh",
    "update/auto_update_deploy.sh",
    "update/update_deploy.sh",
}


def test_all_shell_scripts_discovered():
    found = {str(p.relative_to(SCRIPTS_DIR)) for p in _all_shell_scripts()}
    assert found == EXPECTED_SCRIPTS, (
        f"expected {sorted(EXPECTED_SCRIPTS)} under {SCRIPTS_DIR}, found {sorted(found)}"
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
        SCRIPTS_DIR / "update" / "update_deploy.sh",
        SCRIPTS_DIR / "update" / "auto_update_deploy.sh",
    ]
    for script in deploy_scripts:
        content = script.read_text(encoding="utf-8", errors="ignore")
        assert "set -euo pipefail" in content


def test_setup_scripts_define_main_entrypoint():
    setup_scripts = [
        REPO_ROOT / "miner" / "src" / "scripts" / "setup.sh",
        SCRIPTS_DIR / "main" / "setup.sh",
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
        SCRIPTS_DIR / "main" / "setup.sh",
    ]:
        assert script.stat().st_mode & 0o111, f"{script} is not executable"


def test_the_miner_tests_are_actually_collected():
    """A default run has to reach miner/tests. Reading the setting is not enough:
    an ignore in addopts leaves testpaths looking right and collects nothing."""
    result = subprocess.run(
        # -n 0 overrides the `-n auto` in addopts: this only lists tests, and
        # starting a worker per core to do it changes the output as well as the cost.
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         "-p", "no:cacheprovider", "-n", "0"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout[-2000:]}\nstderr:\n{result.stderr[-2000:]}"
    )
    assert "miner/tests/test_miner.py::" in result.stdout, (
        "a default run does not collect the miner's tests"
    )


# The auto-updater is registered with PM2 by path on operator machines, and PM2 keeps
# that path across restarts. A forwarder stays at the old location so the pull that
# delivers this move does not break the updater that performed it.
LEGACY_UPDATER = REPO_ROOT / "scripts" / "validator" / "update" / "auto_update_deploy.sh"


def test_the_legacy_updater_path_still_answers():
    assert LEGACY_UPDATER.is_file(), (
        f"{LEGACY_UPDATER} is what operators registered with PM2; removing it breaks "
        "their updater on the next pull"
    )
    assert LEGACY_UPDATER.stat().st_mode & 0o111, f"{LEGACY_UPDATER} is not executable"


def test_the_forwarder_points_at_the_real_updater():
    target = SCRIPTS_DIR / "update" / "auto_update_deploy.sh"
    assert target.is_file(), f"{target} is missing, so the forwarder leads nowhere"
    assert "validator/scripts/update/auto_update_deploy.sh" in LEGACY_UPDATER.read_text(), (
        "the forwarder no longer names the script it forwards to"
    )
