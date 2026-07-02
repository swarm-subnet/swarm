def test_prepare_model_image_reuses_existing_image(tmp_path, monkeypatch):
    import zipfile
    from unittest.mock import MagicMock

    from swarm.utils.hash import sha256sum
    from swarm.validator.docker.docker_evaluator_parts import batch

    model_path = tmp_path / "UID_7.zip"
    with zipfile.ZipFile(model_path, "w") as archive:
        archive.writestr("requirements.txt", "numpy\n")
        archive.writestr("policy.py", "x = 1\n")

    monkeypatch.setattr(batch, "_image_exists", lambda tag: True)

    result = batch.prepare_model_image(MagicMock(), 7, model_path)
    assert result == batch.model_image_tag(sha256sum(model_path))


def test_prepare_model_image_private_ignores_cached_image(tmp_path, monkeypatch):
    import zipfile
    from unittest.mock import MagicMock

    from swarm.validator.docker.docker_evaluator_parts import batch

    model_path = tmp_path / "UID_7.zip"
    with zipfile.ZipFile(model_path, "w") as archive:
        archive.writestr("requirements.txt", "numpy\n")
        archive.writestr("policy.py", "x = 1\n")
    model_path.with_suffix(".private").touch()

    monkeypatch.setattr(batch, "_image_exists", lambda tag: True)

    assert batch.prepare_model_image(MagicMock(), 7, model_path) is None


def test_cleanup_removes_only_orphan_images(tmp_path, monkeypatch):
    import zipfile
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from swarm.utils.hash import sha256sum
    from swarm.validator.docker.docker_evaluator_parts import batch

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_path = model_dir / "UID_7.zip"
    with zipfile.ZipFile(model_path, "w") as archive:
        archive.writestr("requirements.txt", "numpy\n")
        archive.writestr("policy.py", "x = 1\n")

    live = batch.model_image_tag(sha256sum(model_path))
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ["docker", "images"]:
            return SimpleNamespace(
                returncode=0,
                stdout=f"{live}\nswarm_eval_model_deadbeef0000:latest\n",
            )
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(batch, "MODEL_DIR", model_dir)
    monkeypatch.setattr(batch.subprocess, "run", fake_run)

    batch.cleanup(MagicMock())

    assert ["docker", "rmi", "swarm_eval_model_deadbeef0000:latest"] in calls
    assert ["docker", "rmi", live] not in calls


def test_remove_all_model_images(monkeypatch):
    from types import SimpleNamespace

    from swarm.validator.docker.docker_evaluator_parts import batch

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ["docker", "images"]:
            return SimpleNamespace(
                returncode=0,
                stdout="swarm_eval_model_aaa111:latest\nswarm_eval_model_bbb222:latest\n",
            )
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(batch.subprocess, "run", fake_run)

    batch.remove_all_model_images()

    assert ["docker", "rmi", "swarm_eval_model_aaa111:latest"] in calls
    assert ["docker", "rmi", "swarm_eval_model_bbb222:latest"] in calls
