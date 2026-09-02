from __future__ import annotations

import ast
import re
import zipfile
from pathlib import Path

from swarm.validator.docker.docker_evaluator_parts import batch as batch_mod

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATOR_DOCKERFILE = REPO_ROOT / "swarm" / "validator" / "docker" / "Dockerfile"

# The interpreter the per-submission install must target, and the environment the
# evaluation container resolves a bare `python` to.
BASE_ENV = "/opt/env"
BASE_PYTHON = f"{BASE_ENV}/bin/python"


def _startup_script_source() -> str:
    """The startup script literal written into each submission directory."""
    source = Path(batch_mod.__file__).read_text(encoding="utf-8")
    match = re.search(r'startup_script\.write_text\(\s*(.*?)\n\s*\)\n', source, re.S)
    assert match, "could not locate the generated startup script in batch.py"
    return match.group(1)


def test_startup_script_installs_into_the_base_environment():
    """A fresh closure would ignore the CPU torch already in the image and refetch it."""
    script = _startup_script_source()
    assert "uv pip install" in script
    assert BASE_PYTHON in script
    assert "pip install --user" not in script


def test_install_target_survives_the_commit_and_the_runtime_tmpfs():
    """The submission dir is a bind mount and /tmp is remounted for evaluation, so
    neither can hold dependencies the committed image needs."""
    script = _startup_script_source()
    assert "--target" not in script
    assert "/workspace/submission/.deps" not in script
    assert not re.search(r"--(target|prefix)[= ]/tmp", script)


def test_startup_script_marks_completion_only_after_a_clean_install():
    """The marker is the only signal gating `docker commit`, so a non-zero install
    must exit before it is written."""
    body = _startup_script_source().replace('"', "").replace("\\n", "\n")
    install_at = body.index("uv pip install")
    guard_at = body.index("if [ $? -ne 0 ]; then exit 1; fi")
    marker_at = body.index("touch /workspace/submission/.pip_done")
    assert install_at < guard_at < marker_at


def test_an_extracted_marker_would_end_the_wait_immediately(tmp_path: Path):
    """Guards the fixture the next test relies on: extraction really does write a
    submission-supplied marker into the directory the poll loop watches."""
    archive = tmp_path / "model.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("drone_agent.py", "class Agent: pass\n")
        zf.writestr(".pip_done", "")

    submission_dir = tmp_path / "submission"
    submission_dir.mkdir()
    batch_mod._extract_submission(archive, submission_dir)
    assert (submission_dir / ".pip_done").exists()


def test_prepare_model_image_removes_the_marker_before_launching():
    """Deleting the unlink from prepare_model_image must fail here: the poll at
    `.pip_done` runs before the running-state check, so a stale marker commits the
    image mid-install."""
    source = Path(batch_mod.__file__).read_text(encoding="utf-8")
    start = source.index("def prepare_model_image(")
    body = source[start : source.index("\ndef ", start + 1)]

    extract_at = body.index("_extract_submission(model_path, submission_dir)")
    unlink_at = body.index('(submission_dir / ".pip_done").unlink(')
    launch_at = body.index("startup_script = submission_dir")
    assert extract_at < unlink_at < launch_at


def test_evaluator_image_builds_the_base_environment_and_puts_it_first_on_path():
    dockerfile = EVALUATOR_DOCKERFILE.read_text(encoding="utf-8")
    assert f"uv venv {BASE_ENV}" in dockerfile
    assert f"ENV PATH={BASE_ENV}/bin:$PATH" in dockerfile
    assert "pip install" not in dockerfile.replace("uv pip install", "")


def test_no_runtime_profile_overrides_the_interpreter_search_path():
    """`batch.py` runs a bare `python`, so a profile defining PATH would silently
    bypass the base environment. The profile env is merged over the image's own."""
    families_dir = REPO_ROOT / "swarm" / "challenge_families"
    literals = 0
    for source_file in sorted(families_dir.glob("*.py")):
        tree = ast.parse(source_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.keyword) or node.arg != "docker_env":
                continue
            # base.py forwards a caller-supplied mapping through a comprehension;
            # what it forwards is declared as a literal in the family modules below.
            if not isinstance(node.value, ast.Dict):
                continue
            literals += 1
            keys = {
                key.value
                for key in node.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
            offenders = keys & {"PATH", "PYTHONPATH"}
            assert not offenders, f"{source_file.name} sets {offenders} in docker_env"
    assert literals >= 6, f"expected a docker_env literal per family, found {literals}"
