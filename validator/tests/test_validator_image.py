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

"""The validator container: its image, its compose service and the scripts that deploy it."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER_DIR = REPO_ROOT / ".docker"
UPDATE_DIR = REPO_ROOT / "validator" / "scripts" / "update"
UPDATE_SCRIPT = UPDATE_DIR / "update_deploy.sh"
AUTO_UPDATE_SCRIPT = UPDATE_DIR / "auto_update_deploy.sh"
VALIDATOR_DOCKERFILE = DOCKER_DIR / "validator.Dockerfile"
COMPOSE_FILE = DOCKER_DIR / "docker-compose.yml"
ENTRYPOINT = DOCKER_DIR / "validator-entrypoint.sh"
VALIDATOR_REQUIREMENTS = REPO_ROOT / "validator" / "requirements.txt"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "validator-image.yml"


@pytest.mark.parametrize("script", [UPDATE_SCRIPT, AUTO_UPDATE_SCRIPT], ids=lambda p: p.name)
def test_the_deploy_scripts_are_valid_bash(script: Path) -> None:
    """Proves neither script has a syntax error, which would strand a host mid-update."""
    assert script.is_file(), f"missing {script}"
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_the_entrypoint_runs_a_plain_command_without_a_wallet(tmp_path: Path) -> None:
    """Proves a test job can use the image with no validator environment at all.

    A command replaces the validator; a flag is still appended to it, and then the
    wallet variables are required as before.
    """
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    ran = subprocess.run(
        ["bash", str(ENTRYPOINT), "echo", "ran as", "given"],
        capture_output=True, text=True, env=env,
    )
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.strip() == "ran as given"

    flagged = subprocess.run(
        ["bash", str(ENTRYPOINT), "--logging.debug"], capture_output=True, text=True, env=env,
    )
    assert flagged.returncode == 2
    assert "SWARM_WALLET_NAME" in flagged.stderr


def test_the_version_compare_orders_releases_properly() -> None:
    """Proves the watcher redeploys on a newer version and never on an equal or older one.

    It is the whole trigger: wrong here and every host either stops updating or
    redeploys on a loop.
    """
    probe = f"""
        source <(sed -n '/^is_remote_newer()/,/^}}/p' {AUTO_UPDATE_SCRIPT})
        for pair in "5.1.5.6 5.1.5.7" "5.1.5.7 5.1.6.0" "5.1.9.0 5.1.10.0"; do
            set -- $pair
            is_remote_newer "$1" "$2" && echo "newer" || echo "not"
        done
        for pair in "5.1.5.7 5.1.5.7" "5.1.6.0 5.1.5.7" "5.1.10.0 5.1.9.0"; do
            set -- $pair
            is_remote_newer "$1" "$2" && echo "newer" || echo "not"
        done
    """
    result = subprocess.run(["bash", "-c", probe], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["newer", "newer", "newer", "not", "not", "not"]


def test_the_validator_image_builds_on_the_published_base() -> None:
    """Proves the validator ships the same package set the flights are scored on."""
    text = VALIDATOR_DOCKERFILE.read_text()
    assert 'ARG BASE_IMAGE=ghcr.io/swarm-subnet/swarm:base' in text
    assert "FROM ${BASE_IMAGE}" in text
    assert 'LABEL swarm.__version__=' in text, "the updater compares this label"


def test_the_runner_image_inputs_are_untouched() -> None:
    """Proves the root dockerignore is not what the validator image reads.

    The runner image's cache key hashes that file, so editing it would rebuild the
    runner on every validator for no reason.
    """
    assert (DOCKER_DIR / "validator.Dockerfile.dockerignore").is_file()
    root_ignore = (REPO_ROOT / ".dockerignore").read_text()
    assert "validator" in root_ignore.split(), "root ignore should still exclude validator/"


def test_the_cross_machine_check_pins_the_published_image() -> None:
    """Proves the check flies inside the shared image unless a host is opted out.

    Defaulting to whatever a host already built would make the check compare the
    per-host builds it exists to rule out, and it would still print a verdict.
    """
    sys.path.insert(0, str(REPO_ROOT / "validator" / "scripts"))
    try:
        import cross_machine_check as check
    finally:
        sys.path.pop(0)

    parser = check._build_parser()
    assert parser.parse_args(["run"]).image == check.SHARED_IMAGE
    assert parser.parse_args(["run", "--host-image"]).image is None


def _pins(path: Path) -> dict[str, str]:
    """Map each requirement name in a file to its full line, comments and blanks dropped."""
    lines = [line.strip() for line in path.read_text().splitlines()]
    return {
        re.split(r"[=@ ]", line, maxsplit=1)[0]: line
        for line in lines if line and not line.startswith("#")
    }


def test_the_validator_pins_match_the_root_pins() -> None:
    """Proves the validator's own list can narrow the root one but never disagree with it.

    A version chosen here would put the validator on different physics from the
    flights it scores, which is the drift the shared base image exists to rule out.
    """
    root = _pins(REPO_ROOT / "requirements.txt")
    for name, line in _pins(VALIDATOR_REQUIREMENTS).items():
        assert root.get(name) == line, f"{name} differs from the root requirements.txt"


def test_the_image_installs_the_validator_requirements() -> None:
    """Proves the image is built from the validator's list rather than the root one."""
    assert "validator/requirements.txt" in VALIDATOR_DOCKERFILE.read_text()


def test_the_update_script_finds_the_checkout_from_its_copy(tmp_path: Path) -> None:
    """Proves the script still resolves the repository once it re-runs itself from a copy.

    The copy lives in the temp directory, so a path derived from the running file
    points outside the checkout and the update dies before it has done anything.
    """
    checkout = tmp_path / "checkout"
    script_dir = checkout / "validator" / "scripts" / "update"
    script_dir.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    head = UPDATE_SCRIPT.read_text().split("\nSTEP=0\n", 1)[0]
    probe = script_dir / "update_deploy.sh"
    probe.write_text(head + '\necho "$REPO_ROOT"\n')

    result = subprocess.run(
        ["bash", str(probe)], capture_output=True, text=True, cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).resolve() == checkout.resolve()


def test_the_publish_workflow_watches_files_that_exist() -> None:
    """Proves a renamed image input cannot leave the rebuild check watching nothing."""
    text = WORKFLOW.read_text()
    watched = re.findall(r"^\s+(\.docker/\S+|validator/\S+?);?\s*(?:\\|then)?$", text, re.M)
    assert "validator/requirements.txt" in watched
    assert ".docker/validator.Dockerfile" in watched
    for path in watched:
        assert (REPO_ROOT / path).is_file(), f"the workflow watches a missing file: {path}"


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker is not installed")
def test_the_validator_service_can_be_built_locally() -> None:
    """Proves compose can build the image from the validator Dockerfile while still naming the published one."""
    result = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "--profile", "validator",
         "config", "--format", "json"],
        capture_output=True, text=True, cwd=str(DOCKER_DIR),
    )
    if result.returncode != 0:
        pytest.skip(f"docker compose could not run here: {result.stderr.strip()[:200]}")

    service = json.loads(result.stdout)["services"]["validator"]
    assert service["build"]["dockerfile"] == ".docker/validator.Dockerfile"
    assert Path(service["build"]["context"]).resolve() == REPO_ROOT
    assert service["pull_policy"] == "always", "`up` must keep pulling the published image"


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker is not installed")
def test_everything_the_validator_writes_lands_in_the_state_directory() -> None:
    """Proves the code's own state folders, the log and the home all map to the state dir.

    Inside the image those folders belong to root and vanish on the next image, so a
    validator without these mounts loses its model cache, its queues and its scores.
    """
    result = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "--profile", "validator",
         "config", "--format", "json"],
        capture_output=True, text=True, cwd=str(DOCKER_DIR),
        env={**os.environ, "SWARM_STATE_DIR": "/srv/state"},
    )
    if result.returncode != 0:
        pytest.skip(f"docker compose could not run here: {result.stderr.strip()[:200]}")

    service = json.loads(result.stdout)["services"]["validator"]
    mounts = {v["target"]: v["source"] for v in service["volumes"]}
    assert mounts["/opt/swarm-validator/state"] == "/srv/state/state"
    assert mounts["/opt/swarm-validator/swarm/state"] == "/srv/state/swarm-state"
    assert mounts["/srv/state"] == "/srv/state", "the path the daemon is handed must match"
    assert service["working_dir"] == "/srv/state", "logfile.log is written to the cwd"
    assert service["environment"]["HOME"] == "/srv/state"
    assert mounts["/wallets"].endswith("/.bittensor/wallets")
    assert service["environment"]["BT_WALLET_PATH"] == "/wallets"


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker is not installed")
def test_the_compose_file_describes_the_validator_service() -> None:
    """Proves the service parses, pulls rather than builds, and is behind its profile."""
    result = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "--profile", "validator", "config"],
        capture_output=True, text=True, cwd=str(DOCKER_DIR),
    )
    if result.returncode != 0:
        pytest.skip(f"docker compose could not run here: {result.stderr.strip()[:200]}")

    rendered = result.stdout
    assert "swarm-validator" in rendered, "the service should run the published image"
    assert "/var/run/docker.sock" in rendered, "it drives the host daemon"
    for mount in ("/tmp", "/dev/shm"):
        assert mount in rendered, f"{mount} must be mounted at the same path on both sides"


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker is not installed")
def test_the_default_profile_does_not_start_the_validator() -> None:
    """Proves `docker compose up` on a dev box still only brings up the test services."""
    result = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "config", "--services"],
        capture_output=True, text=True, cwd=str(DOCKER_DIR),
    )
    if result.returncode != 0:
        pytest.skip(f"docker compose could not run here: {result.stderr.strip()[:200]}")
    assert "validator" not in result.stdout.split()
