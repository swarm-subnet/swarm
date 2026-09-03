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

from __future__ import annotations

import json
from types import SimpleNamespace
import zipfile

from miner.src import miner


def _make_bad_zip(tmp_path, names=("payload.bin",)):
    path = tmp_path / "submission.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for name in names:
            zf.writestr(name, "data")
    return str(path)


def _make_submission(tmp_path, family_id="cf_search_and_rescue"):
    action_width = {"cf_autopilot": 5, "cf_search_and_rescue": 6}[family_id]
    path = tmp_path / "submission.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            "drone_agent.py",
            "import numpy as np\n\n\n"
            "class DroneFlightController:\n"
            "    def act(self, observation):\n"
            f"        return np.zeros({action_width}, dtype=np.float32)\n\n"
            "    def reset(self):\n"
            "        pass\n",
        )
    return str(path)


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
    monkeypatch.setattr(miner, "_backend_reachable", lambda url: True)
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
            _make_submission(tmp_path),
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
    assert len(data.encode()) <= 128  # chain commitment cap (Raw128)
    commit = json.loads(data)
    assert commit["v"] == 1
    assert commit["f"] == "cf_search_and_rescue"
    assert len(commit["s"]) == 64


def test_validate_artifact_rules(tmp_path):
    artifact = _make_submission(tmp_path)
    assert miner._validate_artifact(artifact, family_id="cf_search_and_rescue") is None
    bad = _make_bad_zip(tmp_path)
    assert "missing_required_file" in miner._validate_artifact(bad)
    not_zip = tmp_path / "not_zip.txt"
    not_zip.write_text("data")
    assert miner._validate_artifact(str(not_zip)) is not None


def test_validate_artifact_rejects_a_family_it_was_not_packaged_for(tmp_path):
    """A commitment must not claim a family the packaged contract disagrees with."""
    from swarm.cli import _package_model_artifact

    source = tmp_path / "src"
    source.mkdir()
    (source / "drone_agent.py").write_text(
        "import numpy as np\n\n\n"
        "class DroneFlightController:\n"
        "    def act(self, observation):\n"
        "        return np.zeros(6, dtype=np.float32)\n\n"
        "    def reset(self):\n"
        "        pass\n"
    )
    packaged = _package_model_artifact(
        source_dir=source,
        output_zip=tmp_path / "sar.zip",
        family_id="cf_search_and_rescue",
        interface_version=None,
        overwrite=True,
    )
    path = str(packaged.output_zip)

    assert miner._validate_artifact(path, family_id="cf_search_and_rescue") is None
    assert "does not match" in miner._validate_artifact(path, family_id="cf_autopilot")


def test_load_local_families_reads_schema():
    families = miner._load_local_families()

    assert isinstance(families, dict)
    assert "cf_autopilot" in families
    assert "cf_search_and_rescue" in families


def test_private_unreachable_backend_blocks_before_commit(monkeypatch, tmp_path):
    class FakeWallet:
        def __init__(self, name, hotkey):
            raise AssertionError("wallet should not be constructed")

    fake_logging = SimpleNamespace(
        set_debug=lambda *_args, **_kwargs: None,
        info=lambda *_args, **_kwargs: None,
        warning=lambda *_args, **_kwargs: None,
        error=lambda *_args, **_kwargs: None,
    )

    monkeypatch.setattr(
        miner, "_fetch_backend_families", lambda url: {"cf_search_and_rescue": "private"}
    )
    monkeypatch.setattr(miner, "_validate_artifact", lambda path, family_id: None)
    monkeypatch.setattr(miner, "_backend_reachable", lambda url: False)
    if hasattr(miner.bt, "Wallet"):
        monkeypatch.setattr(miner.bt, "Wallet", FakeWallet)
    else:
        monkeypatch.setattr(miner.bt, "wallet", FakeWallet)
    monkeypatch.setattr(miner.bt, "logging", fake_logging)

    artifact = tmp_path / "submission.zip"
    artifact.write_bytes(b"zip-bytes")
    exit_code = miner.main(
        [
            "--family_id",
            "cf_search_and_rescue",
            "--artifact",
            str(artifact),
            "--backend_url",
            "http://backend.test",
        ]
    )

    assert exit_code == 1
