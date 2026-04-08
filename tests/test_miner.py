from __future__ import annotations

from types import SimpleNamespace

from neurons import miner


def test_validate_github_url_strips_git_suffix():
    assert (
        miner._validate_github_url("https://github.com/example/project.git/")
        == "https://github.com/example/project"
    )


def test_main_uses_plain_commit(monkeypatch):
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

        def commit(self, *, wallet, netuid, data):
            commit_calls.append((wallet.hotkey.ss58_address, netuid, data))
            return True

        def set_reveal_commitment(self, **_kwargs):
            raise AssertionError("set_reveal_commitment should not be used")

    fake_logging = SimpleNamespace(
        set_debug=lambda *_args, **_kwargs: None,
        info=lambda *_args, **_kwargs: None,
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
        ("5ExampleHotkey", 124, "https://github.com/example/project")
    ]
