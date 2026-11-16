"""Compiled native code is refused at intake by its header, whatever the file is called or wherever it is nested."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
from onnx import TensorProto, helper

from swarm.core import submission_policy
from swarm.core.model_verify import classify_model_validity, inspect_model_structure
from swarm.core.submission_policy import (
    NESTED_ARCHIVE_MAX_DEPTH,
    check_native_code,
    is_native_executable,
    profile_digest,
    validate_submission_zip,
)

# Validators send this with every score; the backend refuses scores whose digest differs from the model's.
_PINNED_PROFILE_DIGEST = "5f1f6061517938492774e6d22f2f9e0df651aa9c76d84095d291e81f70b8c586"

_AGENT = b"class DroneFlightController: pass\n"
_ELF = b"\x7fELF\x02\x01\x01\x00" + bytes(56)


def _pe_image() -> bytes:
    """A minimal PE image: the DOS stub's offset at 0x3c points at the PE signature."""
    stub = bytearray(0x80)
    stub[0:2] = b"MZ"
    stub[0x3C:0x40] = (0x80).to_bytes(4, "little")
    return bytes(stub) + b"PE\x00\x00" + bytes(20)


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    """An in-memory zip holding ``files``."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buffer.getvalue()


def _submission(tmp_path: Path, files: dict[str, bytes]) -> Path:
    """A submission zip with the agent entry point plus ``files``."""
    path = tmp_path / "submission.zip"
    path.write_bytes(_zip_bytes({"drone_agent.py": _AGENT, **files}))
    return path


def _onnx_model() -> bytes:
    """A real one-node ONNX model, serialized."""
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "policy",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
    )
    return helper.make_model(graph).SerializeToString()


def _npz() -> bytes:
    """A real NumPy weights archive."""
    buffer = io.BytesIO()
    np.savez(buffer, w=np.arange(16, dtype=np.float32), b=np.zeros(4))
    return buffer.getvalue()


def _nested(depth: int, payload: bytes) -> bytes:
    """``payload`` wrapped in ``depth`` levels of zip, each named ``inner.zip`` in its parent."""
    data = payload
    name = "lib.bin"
    for _ in range(depth):
        data = _zip_bytes({name: data})
        name = "inner.zip"
    return data


def test_profile_digest_is_unchanged():
    """The execution profile digest stays at the value every stored model was admitted under."""
    assert profile_digest() == _PINNED_PROFILE_DIGEST


def test_an_elf_named_like_weights_is_refused(tmp_path):
    """A shared library renamed to weights.bin no longer passes on its suffix."""
    assert validate_submission_zip(_submission(tmp_path, {"weights.bin": _ELF})) == (False, "native_code:weights.bin")


def test_an_elf_without_an_extension_is_refused(tmp_path):
    """An executable with no extension at all is refused by its content."""
    assert validate_submission_zip(_submission(tmp_path, {"tools/helper": _ELF})) == (False, "native_code:tools/helper")


def test_an_elf_inside_a_nested_zip_is_refused(tmp_path):
    """An executable packed inside a zip inside the submission is found and named through its archive."""
    path = _submission(tmp_path, {"assets/pack.dat": _zip_bytes({"lib/core.bin": _ELF, "readme.txt": b"hi"})})
    assert validate_submission_zip(path) == (False, "native_code:assets/pack.dat!lib/core.bin")


@pytest.mark.parametrize(
    "header",
    [
        b"\xfe\xed\xfa\xce", b"\xce\xfa\xed\xfe", b"\xfe\xed\xfa\xcf", b"\xcf\xfa\xed\xfe",
        b"\xca\xfe\xba\xbe\x00\x00\x00\x02", b"\xca\xfe\xba\xbf\x00\x00\x00\x02",
    ],
    ids=["macho32-be", "macho32-le", "macho64-be", "macho64-le", "fat", "fat64"],
)
def test_mach_o_binaries_are_refused(tmp_path, header):
    """Thin Mach-O in both byte orders and widths, and fat binaries, are refused."""
    ok, reason = validate_submission_zip(_submission(tmp_path, {"model.bin": header + bytes(64)}))
    assert (ok, reason) == (False, "native_code:model.bin")


def test_a_pe_image_is_refused(tmp_path):
    """A Windows PE image is refused once its header offset leads to the PE signature."""
    assert validate_submission_zip(_submission(tmp_path, {"model.dat": _pe_image()})) == (False, "native_code:model.dat")


def test_real_model_files_pass(tmp_path):
    """ONNX and NumPy weight files, including the zip inside an .npz, are accepted."""
    path = _submission(tmp_path, {"model.onnx": _onnx_model(), "weights.npz": _npz(), "config.json": b"{}"})
    assert validate_submission_zip(path) == (True, "ok")


def test_text_that_starts_with_mz_passes(tmp_path):
    """A file whose first bytes are MZ but which carries no PE signature is not mistaken for an executable."""
    text = b"MZ readings per waypoint\n" + b"0.1,0.2,0.3\n" * 20
    assert validate_submission_zip(_submission(tmp_path, {"notes.txt": text})) == (True, "ok")


def test_a_java_class_file_is_not_taken_for_a_fat_binary():
    """The fat Mach-O magic followed by a class-file major version is left alone."""
    assert not is_native_executable(b"\xca\xfe\xba\xbe\x00\x00\x00\x34" + bytes(16))


def test_nesting_is_followed_to_the_depth_limit_and_no_further(tmp_path):
    """An executable at the deepest inspected level is found; one level deeper is never opened."""
    at_limit = _submission(tmp_path, {"pack.zip": _nested(NESTED_ARCHIVE_MAX_DEPTH, _ELF)})
    assert check_native_code(at_limit)[0] is False
    beyond = _submission(tmp_path, {"pack.zip": _nested(NESTED_ARCHIVE_MAX_DEPTH + 1, _ELF)})
    assert check_native_code(beyond) == (True, "ok")


def test_nested_archives_past_the_byte_budget_are_not_read(tmp_path, monkeypatch):
    """A nested archive larger than what is left of the byte budget is skipped rather than decompressed."""
    monkeypatch.setattr(submission_policy, "NESTED_ARCHIVE_MAX_BYTES", 16)
    path = _submission(tmp_path, {"pack.zip": _zip_bytes({"lib.bin": _ELF})})
    assert check_native_code(path) == (True, "ok")


def test_the_miner_is_told_what_was_refused_and_why(tmp_path):
    """The verifier names the files, says they are compiled code, and treats the model as dangerous."""
    inspection = inspect_model_structure(_submission(tmp_path, {"weights.bin": _ELF}))
    assert "weights.bin" in inspection["error"]
    assert "compiled native code" in inspection["error"]
    assert classify_model_validity(inspection)[0] == "fake"


def test_backend_mirror_matches():
    """The backend's copy of the policy carries the same code below its module docstring."""
    ours = Path(submission_policy.__file__)
    theirs = Path(__file__).resolve().parents[3] / "swarm-backend" / "app" / "submission_policy.py"
    if not theirs.is_file():
        pytest.skip("swarm-backend repository is not checked out next to swarm")

    def _code(path: Path) -> str:
        """The module source from its first import on."""
        source = path.read_text()
        return source[source.index("\nimport "):]

    assert _code(ours) == _code(theirs)
