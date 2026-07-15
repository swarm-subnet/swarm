from __future__ import annotations

import json
from types import SimpleNamespace
import zipfile

from neurons import miner
from tests.model_graph.fixtures import base_manifest, build_artifact, dense_model


def _make_bad_zip(tmp_path, names=("payload.bin",)):
    path = tmp_path / "submission.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for name in names:
            zf.writestr(name, "data")
    return str(path)


def _make_graph(tmp_path, family_id="cf_search_and_rescue"):
    state_width, action_width = {
        "cf_autopilot": (141, 5),
        "cf_search_and_rescue": (165, 6),
    }[family_id]
    model = dense_model({"state": (state_width,)}, "action", (action_width,))
    manifest = base_manifest(
        family_id,
        models=[{"id": "p", "file": "models/p.onnx", "sha256": "0" * 64}],
        nodes=[{"id": "p", "model": "p", "inputs": {"state": "obs.state"}}],
        action="p.action",
    )
    return str(build_artifact(tmp_path, manifest, {"models/p.onnx": model}))


def test_validate_github_url_strips_git_suffix():
    assert (
        miner._validate_github_url("https://github.com/example/project.git/")
        == "https://github.com/example/project"
    )


def test_main_uses_set_commitment(monkeypatch):
    hotkey = "5ExampleHotkey"
    commit_calls = []

    class FakeWallet:
        def __init__(self, name, hotkey):
            self.hotkey = SimpleNamespace(ss58_address="5ExampleHotkey")

    class FakeSubtensor:
        def __init__(self, network):
            self.network = network

        def metagraph(self, netuid):
            return SimpleNamespace(hotkeys=[hotkey])

        def set_commitment(self, wallet, netuid, data, *, mev_protection=False):
            commit_calls.append((wallet.hotkey.ss58_address, netuid, data, mev_protection))
            return SimpleNamespace(success=True, message="ok")

        def set_reveal_commitment(self, **_kwargs):
            raise AssertionError("set_reveal_commitment should not be used")

    fake_logging = SimpleNamespace(
        set_debug=lambda *_args, **_kwargs: None,
        info=lambda *_args, **_kwargs: None,
        warning=lambda *_args, **_kwargs: None,
        error=lambda *_args, **_kwargs: None,
    )

    if hasattr(miner.bt, "Wallet"):
        monkeypatch.setattr(miner.bt, "Wallet", FakeWallet)
    else:
        monkeypatch.setattr(miner.bt, "wallet", FakeWallet)

    if hasattr(miner.bt, "Subtensor"):
        monkeypatch.setattr(miner.bt, "Subtensor", FakeSubtensor)
    else:
        monkeypatch.setattr(miner.bt, "subtensor", FakeSubtensor)
    monkeypatch.setattr(miner.bt, "logging", fake_logging)

    exit_code = miner.main(
        [
            "--github_url",
            "https://github.com/example/project.git",
            "--wallet.name",
            "miner",
            "--wallet.hotkey",
            "default",
            "--netuid",
            "124",
            "--subtensor.network",
            "finney",
        ]
    )

    assert exit_code == 0
    assert commit_calls == [
        ("5ExampleHotkey", 124, "https://github.com/example/project", False)
    ]


def test_private_unknown_family_blocks_before_wallet(monkeypatch, tmp_path):
    class FakeWallet:
        def __init__(self, name, hotkey):
            raise AssertionError("wallet should not be constructed")

    fake_logging = SimpleNamespace(
        set_debug=lambda *_args, **_kwargs: None,
        info=lambda *_args, **_kwargs: None,
        error=lambda *_args, **_kwargs: None,
        warning=lambda *_args, **_kwargs: None,
    )

    monkeypatch.setattr(miner, "_fetch_backend_families", lambda url: None)
    if hasattr(miner.bt, "Wallet"):
        monkeypatch.setattr(miner.bt, "Wallet", FakeWallet)
    else:
        monkeypatch.setattr(miner.bt, "wallet", FakeWallet)
    monkeypatch.setattr(miner.bt, "logging", fake_logging)

    exit_code = miner.main(
        [
            "--family_id",
            "cf_definitely_not_real",
            "--artifact",
            _make_bad_zip(tmp_path),
            "--backend_url",
            "http://backend.test",
        ]
    )

    assert exit_code == 1


def test_private_public_family_blocks_with_guidance(monkeypatch, tmp_path):
    error_lines = []
    fake_logging = SimpleNamespace(
        set_debug=lambda *_args, **_kwargs: None,
        info=lambda *_args, **_kwargs: None,
        error=lambda message, *_args, **_kwargs: error_lines.append(message),
        warning=lambda *_args, **_kwargs: None,
    )

    monkeypatch.setattr(
        miner, "_fetch_backend_families", lambda url: {"cf_search_and_rescue": "public"}
    )
    monkeypatch.setattr(miner.bt, "logging", fake_logging)

    exit_code = miner.main(
        [
            "--family_id",
            "cf_search_and_rescue",
            "--artifact",
            _make_bad_zip(tmp_path),
            "--backend_url",
            "http://backend.test",
        ]
    )

    assert exit_code == 1
    assert any("github_url" in line for line in error_lines)


def test_private_bad_zip_blocks_before_commit(monkeypatch, tmp_path):
    class FakeWallet:
        def __init__(self, name, hotkey):
            raise AssertionError("wallet should not be constructed")

    fake_logging = SimpleNamespace(
        set_debug=lambda *_args, **_kwargs: None,
        info=lambda *_args, **_kwargs: None,
        error=lambda *_args, **_kwargs: None,
        warning=lambda *_args, **_kwargs: None,
    )

    monkeypatch.setattr(
        miner, "_fetch_backend_families", lambda url: {"cf_search_and_rescue": "private"}
    )
    if hasattr(miner.bt, "Wallet"):
        monkeypatch.setattr(miner.bt, "Wallet", FakeWallet)
    else:
        monkeypatch.setattr(miner.bt, "wallet", FakeWallet)
    monkeypatch.setattr(miner.bt, "logging", fake_logging)

    exit_code = miner.main(
        [
            "--family_id",
            "cf_search_and_rescue",
            "--artifact",
            _make_bad_zip(tmp_path),
            "--backend_url",
            "http://backend.test",
        ]
    )

    assert exit_code == 1


def test_private_valid_submission_commits(monkeypatch, tmp_path):
    hotkey = "5ExampleHotkey"
    commit_calls = []

    class FakeWallet:
        def __init__(self, name, hotkey):
            self.hotkey = SimpleNamespace(
                ss58_address="5ExampleHotkey",
                sign=lambda message: b"signature",
            )

    class FakeSubtensor:
        def __init__(self, network):
            self.network = network

        def metagraph(self, netuid):
            return SimpleNamespace(hotkeys=[hotkey])

        def set_commitment(self, wallet, netuid, data, *, mev_protection=False):
            commit_calls.append((wallet.hotkey.ss58_address, netuid, data, mev_protection))
            return SimpleNamespace(success=True, message="ok")

    fake_logging = SimpleNamespace(
        set_debug=lambda *_args, **_kwargs: None,
        info=lambda *_args, **_kwargs: None,
        error=lambda *_args, **_kwargs: None,
        warning=lambda *_args, **_kwargs: None,
    )

    monkeypatch.setattr(
        miner, "_fetch_backend_families", lambda url: {"cf_search_and_rescue": "private"}
    )
    monkeypatch.setattr(miner, "_upload_private_artifact", lambda *args, **kwargs: True)
    if hasattr(miner.bt, "Wallet"):
        monkeypatch.setattr(miner.bt, "Wallet", FakeWallet)
    else:
        monkeypatch.setattr(miner.bt, "wallet", FakeWallet)
    if hasattr(miner.bt, "Subtensor"):
        monkeypatch.setattr(miner.bt, "Subtensor", FakeSubtensor)
    else:
        monkeypatch.setattr(miner.bt, "subtensor", FakeSubtensor)
    monkeypatch.setattr(miner.bt, "logging", fake_logging)

    exit_code = miner.main(
        [
            "--family_id",
            "cf_search_and_rescue",
            "--artifact",
            _make_graph(tmp_path),
            "--backend_url",
            "http://backend.test",
            "--wallet.name",
            "miner",
            "--wallet.hotkey",
            "default",
            "--netuid",
            "124",
            "--subtensor.network",
            "finney",
        ]
    )

    assert exit_code == 0
    assert len(commit_calls) == 1
    _, _, data, _ = commit_calls[0]
    commit = json.loads(data)
    assert commit["visibility"] == "private"
    assert commit["family_id"] == "cf_search_and_rescue"


def test_validate_artifact_rules(tmp_path):
    artifact = _make_graph(tmp_path)
    assert miner._validate_artifact(artifact, family_id="cf_search_and_rescue") is None
    assert "does not match" in miner._validate_artifact(
        artifact, family_id="cf_autopilot"
    )
    not_zip = tmp_path / "not_zip.txt"
    not_zip.write_text("data")
    assert "MG_ARCHIVE_REJECTED" in miner._validate_artifact(str(not_zip))


def test_load_local_families_reads_schema():
    families = miner._load_local_families()

    assert isinstance(families, dict)
    assert "cf_autopilot" in families
    assert "cf_search_and_rescue" in families
