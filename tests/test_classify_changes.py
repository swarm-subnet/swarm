"""Which lane a change set lands in.

Every case that is not plainly a listed miner file goes to the whole suite. That
is the safe direction: running everything costs minutes, while running the miner
tests alone on something they do not cover reports a pass nobody checked.
"""
from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ci"))

import classify_changes as cc  # noqa: E402

MANIFEST = cc.load_manifest(REPO_ROOT / "ci" / "miner_tests.toml")

MINER_FILE = "swarm/submission_manifest/__init__.py"
OTHER_MINER_FILE = "neurons/miner.py"


def route(*records: tuple[str, ...]) -> str:
    """Records as git writes them: a status then one path, or two for a rename."""
    payload = b"\0".join(field.encode() for r in records for field in r) + b"\0"
    return cc.classify(cc.paths_from_name_status(payload), MANIFEST)


# ── the miner lane ────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "path",
    [
        "swarm/submission_manifest/__init__.py",
        "swarm/submission_manifest/submission_manifest.schema.json",
        "swarm/templates/README.md",
        "neurons/miner.py",
        "scripts/miner/setup.sh",
        "scripts/miner/install_dependencies.sh",
        "swarm/submission_template/drone_agent.py",
        "swarm/submission_template/office_drone_agent.py",
    ],
)
def test_a_listed_file_on_its_own_runs_the_miner_tests(path):
    assert route(("M", path)) == "miner"


def test_several_listed_files_together_run_the_miner_tests():
    assert route(("M", MINER_FILE), ("M", OTHER_MINER_FILE)) == "miner"


def test_a_rename_between_two_listed_files_runs_the_miner_tests():
    assert route(("R100", MINER_FILE, OTHER_MINER_FILE)) == "miner"


# ── everything else ───────────────────────────────────────────────────────────

def test_a_validator_change_runs_the_whole_suite():
    assert route(("M", "swarm/validator/reward.py")) == "full"


def test_a_miner_change_beside_a_validator_change_runs_the_whole_suite():
    assert route(("M", MINER_FILE), ("M", "swarm/validator/reward.py")) == "full"


@pytest.mark.parametrize(
    "path",
    [
        "swarm/submission_template/main.py",
        "swarm/submission_template/agent_server.py",
        "swarm/submission_template/runtime_caps.py",
        "swarm/submission_template/agent.capnp",
    ],
)
def test_files_the_validator_supplies_run_the_whole_suite(path):
    """These are replaced in every submission, so they are not the miner's."""
    assert route(("M", path)) == "full"


@pytest.mark.parametrize(
    "path",
    ["swarm/policy_interface.py", "swarm/protocol.py", "swarm/constants.py"],
)
def test_shared_contracts_run_the_whole_suite(path):
    assert route(("M", path)) == "full"


def test_the_trusted_runner_runs_the_whole_suite():
    assert route(("M", "swarm/model_graph/runner.py")) == "full"


def test_an_untested_miner_area_runs_the_whole_suite():
    """RL/ is the miner's, but nothing covers it, so it cannot be listed."""
    assert route(("M", "RL/common.py")) == "full"


def test_a_new_file_beside_listed_ones_runs_the_whole_suite():
    assert route(("A", "swarm/submission_manifest/helper.py")) == "full"


def test_a_new_file_of_another_kind_runs_the_whole_suite():
    """A README next to the shell scripts is not covered by the shell test."""
    assert route(("A", "scripts/miner/README.txt")) == "full"


@pytest.mark.parametrize(
    "path",
    [
        ".github/workflows/tests.yml",
        "ci/miner_tests.toml",
        "ci/classify_changes.py",
        "tests/test_miner_tests_manifest.py",
    ],
)
def test_changing_the_routing_runs_the_whole_suite(path):
    """Routing decides what runs, so the run that changes it is the full one."""
    assert route(("M", path)) == "full"


def test_a_deletion_runs_the_whole_suite():
    """Nothing selected asserts a deleted file ought to have existed."""
    assert route(("D", MINER_FILE)) == "full"


@pytest.mark.parametrize(
    "records",
    [
        (("R100", MINER_FILE, "swarm/core/moved.py"),),
        (("R100", "swarm/core/moved.py", MINER_FILE),),
    ],
)
def test_a_rename_across_the_boundary_runs_the_whole_suite(records):
    assert route(*records) == "full"


@pytest.mark.parametrize("path", ["scripts/minerish/x.sh", "RL_backup/x.py"])
def test_a_similar_looking_path_runs_the_whole_suite(path):
    assert route(("M", path)) == "full"


# ── anything the classifier cannot read ───────────────────────────────────────

def test_an_empty_change_set_runs_the_whole_suite():
    assert cc.classify(cc.paths_from_name_status(b""), MANIFEST) == "full"


def test_an_unreadable_record_runs_the_whole_suite():
    assert cc.classify(cc.paths_from_name_status(b"nonsense\0"), MANIFEST) == "full"


def test_a_truncated_rename_record_runs_the_whole_suite():
    assert cc.classify(cc.paths_from_name_status(b"R100\0only-one-side\0"), MANIFEST) == "full"


def test_a_path_containing_a_space_is_read_correctly():
    """The diff is NUL-delimited, so an awkward filename is still one path."""
    assert route(("A", "scripts/miner/a file.sh")) == "full"
    assert cc.paths_from_name_status(b"M\0swarm/a b.py\0") == ["swarm/a b.py"]


def test_a_missing_manifest_runs_the_whole_suite(tmp_path, capsys):
    diff = tmp_path / "diff"
    diff.write_bytes(b"M\0" + MINER_FILE.encode() + b"\0")
    cc.main(["--manifest", str(tmp_path / "absent.toml"), "--diff", str(diff)])
    assert capsys.readouterr().out.strip() == "full"


def test_the_command_line_prints_the_lane(tmp_path, capsys):
    diff = tmp_path / "diff"
    diff.write_bytes(b"M\0" + MINER_FILE.encode() + b"\0")
    cc.main(["--manifest", str(REPO_ROOT / "ci" / "miner_tests.toml"), "--diff", str(diff)])
    assert capsys.readouterr().out.strip() == "miner"
