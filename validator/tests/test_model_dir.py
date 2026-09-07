from swarm.validator.utils_parts import model_fetch


def test_legacy_model_folder_is_adopted_on_first_start(monkeypatch, tmp_path):
    legacy = tmp_path / "miner_models_v2"
    legacy.mkdir()
    (legacy / "UID_3.zip").write_bytes(b"zip")
    new = tmp_path / "swarm" / "state" / "miner_models"
    monkeypatch.setattr(model_fetch, "LEGACY_MODEL_DIR", legacy)
    monkeypatch.setattr(model_fetch, "MODEL_DIR", new)

    assert model_fetch.ensure_model_dir() == new
    assert (new / "UID_3.zip").read_bytes() == b"zip"
    assert not legacy.exists()


def test_an_empty_model_dir_still_adopts_the_legacy_folder(monkeypatch, tmp_path):
    legacy = tmp_path / "miner_models_v2"
    legacy.mkdir()
    (legacy / "UID_3.zip").write_bytes(b"zip")
    new = tmp_path / "miner_models"
    new.mkdir()
    monkeypatch.setattr(model_fetch, "LEGACY_MODEL_DIR", legacy)
    monkeypatch.setattr(model_fetch, "MODEL_DIR", new)

    model_fetch.ensure_model_dir()
    assert (new / "UID_3.zip").read_bytes() == b"zip"
    assert not legacy.exists()


def test_a_symlinked_legacy_folder_keeps_pointing_at_its_disk(monkeypatch, tmp_path):
    disk = tmp_path / "cache_disk" / "models"
    disk.mkdir(parents=True)
    (disk / "UID_3.zip").write_bytes(b"zip")
    legacy = tmp_path / "miner_models_v2"
    legacy.symlink_to("cache_disk/models")
    new = tmp_path / "swarm" / "state" / "miner_models"
    monkeypatch.setattr(model_fetch, "LEGACY_MODEL_DIR", legacy)
    monkeypatch.setattr(model_fetch, "MODEL_DIR", new)

    model_fetch.ensure_model_dir()
    assert new.is_symlink() and new.resolve() == disk.resolve()
    assert (new / "UID_3.zip").read_bytes() == b"zip"
    assert not legacy.exists()


def test_existing_model_dir_is_left_alone(monkeypatch, tmp_path):
    legacy = tmp_path / "miner_models_v2"
    legacy.mkdir()
    (legacy / "UID_3.zip").write_bytes(b"old")
    new = tmp_path / "miner_models"
    new.mkdir()
    (new / "UID_3.zip").write_bytes(b"new")
    monkeypatch.setattr(model_fetch, "LEGACY_MODEL_DIR", legacy)
    monkeypatch.setattr(model_fetch, "MODEL_DIR", new)

    model_fetch.ensure_model_dir()
    assert (new / "UID_3.zip").read_bytes() == b"new"
    assert legacy.exists()


def test_model_dir_lives_under_the_package_state_dir():
    from swarm.constants import MODEL_DIR
    assert MODEL_DIR.is_absolute()
    assert MODEL_DIR.parts[-3:] == ("swarm", "state", "miner_models")
