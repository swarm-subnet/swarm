import stat
import zipfile
from pathlib import Path

from swarm.core.submission_policy import (
    FORBIDDEN_SUFFIXES,
    MAX_UNCOMPRESSED_BYTES,
    REQUIRED_ROOT_FILES,
    check_safety,
    check_structure,
    validate_submission_zip,
)


def _make_zip(path: Path, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)


def _make_symlink_zip(path: Path) -> None:
    info = zipfile.ZipInfo("link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, "drone_agent.py")


def test_check_safety_accepts_clean_zip(tmp_path):
    z = tmp_path / "ok.zip"
    _make_zip(z, {"drone_agent.py": b"class A: pass"})
    assert check_safety(z) == (True, "ok")


def test_check_safety_rejects_path_traversal(tmp_path):
    z = tmp_path / "bad.zip"
    _make_zip(z, {"../escape.py": b"x"})
    ok, reason = check_safety(z)
    assert not ok
    assert "Path traversal" in reason


def test_check_safety_rejects_absolute_path(tmp_path):
    z = tmp_path / "bad.zip"
    _make_zip(z, {"/abs/escape.py": b"x"})
    ok, reason = check_safety(z)
    assert not ok
    assert "Path traversal" in reason


def test_check_safety_rejects_symlink(tmp_path):
    z = tmp_path / "bad.zip"
    _make_symlink_zip(z)
    ok, reason = check_safety(z)
    assert not ok
    assert "Symlink" in reason


def test_check_safety_rejects_corrupt_archive(tmp_path):
    z = tmp_path / "corrupt.zip"
    z.write_bytes(b"not a zip")
    ok, reason = check_safety(z)
    assert not ok
    assert "Corrupted" in reason


def test_check_safety_honors_custom_max_uncompressed(tmp_path):
    z = tmp_path / "ok.zip"
    _make_zip(z, {"drone_agent.py": b"a" * 1024})
    ok, reason = check_safety(z, max_uncompressed=512)
    assert not ok
    assert "too large" in reason


def test_check_safety_rejects_oversized_uncompressed(tmp_path):
    z = tmp_path / "huge.zip"
    payload = b"a" * (MAX_UNCOMPRESSED_BYTES + 1)
    _make_zip(z, {"drone_agent.py": payload})
    ok, reason = check_safety(z)
    assert not ok
    assert "too large" in reason


def test_check_structure_accepts_valid_layout(tmp_path):
    z = tmp_path / "ok.zip"
    _make_zip(z, {"drone_agent.py": b"x", "weights.pt": b"bin"})
    assert check_structure(z) == (True, "ok")


def test_check_structure_rejects_missing_required(tmp_path):
    z = tmp_path / "bad.zip"
    _make_zip(z, {"main.py": b"x"})
    ok, reason = check_structure(z)
    assert not ok
    assert reason.startswith("missing_required_file:")


def test_check_structure_rejects_forbidden_suffixes(tmp_path):
    z = tmp_path / "bad.zip"
    _make_zip(z, {"drone_agent.py": b"x", "payload.sh": b"x"})
    ok, reason = check_structure(z)
    assert not ok
    assert reason.startswith("forbidden_suffix:")
    assert "payload.sh" in reason


def test_check_structure_rejects_nested_drone_agent(tmp_path):
    z = tmp_path / "bad.zip"
    _make_zip(z, {"agent/drone_agent.py": b"x"})
    ok, reason = check_structure(z)
    assert not ok
    assert reason.startswith("missing_required_file:")


def test_validate_submission_zip_accepts_valid(tmp_path):
    z = tmp_path / "ok.zip"
    _make_zip(z, {"drone_agent.py": b"x"})
    assert validate_submission_zip(z) == (True, "ok")


def test_forbidden_suffixes_match_backend():
    assert set(FORBIDDEN_SUFFIXES) == {".exe", ".so", ".dll", ".sh", ".bat", ".pyc"}


def test_required_root_files_match_backend():
    assert REQUIRED_ROOT_FILES == ("drone_agent.py",)
