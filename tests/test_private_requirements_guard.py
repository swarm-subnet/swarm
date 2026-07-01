import zipfile
from unittest.mock import MagicMock


def test_set_private_marker(tmp_path):
    from swarm.validator.utils_parts.model_fetch import _set_private_marker

    model_fp = tmp_path / "UID_1.zip"
    model_fp.write_bytes(b"x")
    _set_private_marker(model_fp, True)
    assert model_fp.with_suffix(".private").exists()
    _set_private_marker(model_fp, False)
    assert not model_fp.with_suffix(".private").exists()


def test_prepare_model_image_rejects_private_with_requirements(tmp_path):
    from swarm.validator.docker.docker_evaluator_parts.batch import prepare_model_image

    model_path = tmp_path / "UID_99.zip"
    with zipfile.ZipFile(model_path, "w") as archive:
        archive.writestr("requirements.txt", "numpy\n")
        archive.writestr("policy.py", "x = 1\n")
    model_path.with_suffix(".private").touch()

    # Private model + requirements must be rejected before any networked pip build.
    result = prepare_model_image(MagicMock(), 99, model_path)
    assert result is None
