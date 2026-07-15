from pathlib import Path

import pytest


MIRRORED_FILES = (
    "admission.py",
    "archive.py",
    "constants.py",
    "errors.py",
    "execution_profile.v1.json",
    "manifest.py",
    "model_graph.schema.json",
    "onnx_profile.py",
    "resource_accounting.py",
)


def test_mirror_is_byte_identical_to_backend():
    swarm = Path(__file__).resolve().parents[2] / "swarm" / "model_graph"
    backend = Path(__file__).resolve().parents[3] / "swarm-backend" / "app" / "model_graph"
    if not backend.is_dir():
        pytest.skip("swarm-backend repository is not checked out next to swarm")
    for name in MIRRORED_FILES:
        assert (swarm / name).read_bytes() == (backend / name).read_bytes(), name
