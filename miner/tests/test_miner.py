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

"""What the miner CLI refuses before it ever commits, and how the private digest-then-upload track behaves once it does."""

from __future__ import annotations

import hashlib
import io
import json
import shlex
import zipfile
from types import SimpleNamespace

from miner.src import miner


def _make_bad_zip(tmp_path, names=("payload.bin",)):
    """Write a zip holding only the named junk entries, with no drone_agent.py in it."""
    path = tmp_path / "submission.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for name in names:
            zf.writestr(name, "data")
    return str(path)


def _make_submission(tmp_path, family_id="cf_search_and_rescue"):
    """Write a passing zip whose act() returns the action width that family expects."""
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
    """A trailing `.git/` is trimmed back to the bare owner/repo address."""
    assert (
        miner._validate_github_url("https://github.com/example/project.git/")
        == "https://github.com/example/project"
    )


def test_main_uses_set_commitment(monkeypatch):
    """The public track commits the normalized URL through `set_commitment`, never the reveal variant."""
    hotkey = "5ExampleHotkey"
    commit_calls = []

    class FakeWallet:
        """A wallet carrying one fixed ss58 address and no signing key at all."""

        def __init__(self, name, hotkey):
            """Keep the coldkey name and hotkey string, and expose the fixed ss58 address."""
            self.name, self.hotkey_str = name, hotkey
            self.hotkey = SimpleNamespace(ss58_address="5ExampleHotkey")

    class FakeSubtensor:
        """Chain stand-in that records every commitment and reports it landed."""

        def __init__(self, network):
            """Keep the network name; nothing else about the chain is needed here."""
            self.network = network

        def metagraph(self, netuid):
            """Return a metagraph holding only the registered hotkey."""
            return SimpleNamespace(hotkeys=[hotkey])

        def set_commitment(self, wallet, netuid, data, *, mev_protection=False):
            """Record the hotkey, netuid, payload and MEV flag, then report success."""
            commit_calls.append((wallet.hotkey.ss58_address, netuid, data, mev_protection))
            return SimpleNamespace(success=True, message="ok")

        def set_reveal_commitment(self, **_kwargs):
            """Fail the run: the miner must use the plain commitment call, not the timed reveal."""
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
    monkeypatch.setattr(miner, "_fetch_backend_families", lambda url: {"cf_autopilot": "public"})
    monkeypatch.setattr(miner, "_check_public_manifest", lambda url: (None, "cf_autopilot"))

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
    """An unrecognised family_id is refused: the run exits 1."""
    class FakeWallet:
        """A wallet that fails the run the moment anything tries to open it."""

        def __init__(self, name, hotkey):
            """Fail on construction: reaching the wallet means a check let the run through."""
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
    """A public-track family refuses the private submission and points the miner at `--github_url`."""
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
    """An archive with no agent module is refused: the run exits 1."""
    class FakeWallet:
        """A wallet that fails the run the moment anything tries to open it."""

        def __init__(self, name, hotkey):
            """Fail on construction: reaching the wallet means a check let the run through."""
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
    """The chain payload stays inside the 128-byte cap and carries v1, the family and a 64-character digest."""
    hotkey = "5ExampleHotkey"
    commit_calls = []

    class FakeWallet:
        """A wallet whose hotkey signs everything with the same fixed bytes."""

        def __init__(self, name, hotkey):
            """Keep the coldkey name and hotkey string, and expose a hotkey that always signs."""
            self.name, self.hotkey_str = name, hotkey
            self.hotkey = SimpleNamespace(
                ss58_address="5ExampleHotkey",
                sign=lambda message: b"signature",
            )

    class FakeSubtensor:
        """Chain stand-in that records every commitment and reports it landed."""

        def __init__(self, network):
            """Keep the network name; nothing else about the chain is needed here."""
            self.network = network

        def metagraph(self, netuid):
            """Return a metagraph holding only the registered hotkey."""
            return SimpleNamespace(hotkeys=[hotkey])

        def set_commitment(self, wallet, netuid, data, *, mev_protection=False):
            """Record the hotkey, netuid, payload and MEV flag, then report success."""
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
    monkeypatch.setattr(miner, "_submission_window", lambda url: {"open": True})
    monkeypatch.setattr(miner, "_submission_status", lambda url, digest: None)
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
    """A packaged zip passes; one with no agent module reports `missing_required_file`, and a text file is refused."""
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
    """The repo's own domain schema loads as a dict carrying cf_autopilot and cf_search_and_rescue."""
    families = miner._load_local_families()

    assert isinstance(families, dict)
    assert "cf_autopilot" in families
    assert "cf_search_and_rescue" in families


def test_private_unreachable_backend_blocks_before_commit(monkeypatch, tmp_path):
    """A backend that fails its health check is refused: the run exits 1."""
    class FakeWallet:
        """A wallet that fails the run the moment anything tries to open it."""

        def __init__(self, name, hotkey):
            """Fail on construction: reaching the wallet means a check let the run through."""
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


# ── local checks that run before the chain commit ─────────────────────────────

def _make_submission_with_requirements(tmp_path, requirements: str) -> str:
    """Write a passing zip carrying the given requirements.txt text alongside the agent."""
    path = tmp_path / "submission.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            "drone_agent.py",
            "import numpy as np\n\n\n"
            "class DroneFlightController:\n"
            "    def act(self, observation):\n"
            "        return np.zeros(6, dtype=np.float32)\n\n"
            "    def reset(self):\n"
            "        pass\n",
        )
        zf.writestr("requirements.txt", requirements)
    return str(path)


def test_validate_artifact_accepts_whitelisted_requirements(tmp_path):
    """Pinned numpy and torch lines, trailing comment and all, pass the whitelist untouched."""
    artifact = _make_submission_with_requirements(tmp_path, "numpy>=1.26\ntorch==2.3.0  # pinned\n")
    assert miner._validate_artifact(artifact) is None


def test_validate_artifact_rejects_requirements_off_the_whitelist(tmp_path):
    """An off-list package is named in the reason, and a git+https line is refused as a URL install."""
    artifact = _make_submission_with_requirements(tmp_path, "numpy\nrequests\n")
    reason = miner._validate_artifact(artifact)
    assert reason is not None and reason.startswith("requirements_rejected:")
    assert "requests" in reason

    url_install = _make_submission_with_requirements(tmp_path, "git+https://example.com/x.git\n")
    assert "URL and path installs" in miner._validate_artifact(url_install)


def test_validate_artifact_rejects_an_oversized_archive(tmp_path, monkeypatch):
    """An archive past MAX_MODEL_BYTES reports `artifact_too_large` before anything is committed."""
    artifact = _make_submission(tmp_path)
    monkeypatch.setattr(miner, "MAX_MODEL_BYTES", 10)
    assert miner._validate_artifact(artifact).startswith("artifact_too_large:")


def test_public_commit_is_refused_for_a_private_family(monkeypatch):
    """A family that moved to the private track exits 1 and points the miner at `swarm model submit`."""
    error_lines = []
    fake_logging = SimpleNamespace(
        set_debug=lambda *_a, **_k: None, info=lambda *_a, **_k: None,
        warning=lambda *_a, **_k: None, error=lambda m, *_a, **_k: error_lines.append(m),
    )

    class FakeWallet:
        """A wallet that fails the run the moment anything tries to open it."""

        def __init__(self, name, hotkey):
            """Fail on construction: reaching the wallet means a check let the run through."""
            raise AssertionError("wallet should not be constructed")

    monkeypatch.setattr(miner.bt, "logging", fake_logging)
    monkeypatch.setattr(miner, "_fetch_backend_families", lambda url: {"cf_autopilot": "private"})
    monkeypatch.setattr(miner, "_check_public_manifest", lambda url: (None, "cf_autopilot"))
    if hasattr(miner.bt, "Wallet"):
        monkeypatch.setattr(miner.bt, "Wallet", FakeWallet)
    else:
        monkeypatch.setattr(miner.bt, "wallet", FakeWallet)

    exit_code = miner.main(["--github_url", "https://github.com/example/project"])

    assert exit_code == 1
    assert any("swarm model submit" in line for line in error_lines)


def test_private_submit_refuses_during_the_freeze(monkeypatch, tmp_path):
    """A closed submission window exits 1 and reports the reopen time in minutes."""
    error_lines = []
    fake_logging = SimpleNamespace(
        set_debug=lambda *_a, **_k: None, info=lambda *_a, **_k: None,
        warning=lambda *_a, **_k: None, error=lambda m, *_a, **_k: error_lines.append(m),
    )

    class FakeWallet:
        """A wallet that fails the run the moment anything tries to open it."""

        def __init__(self, name, hotkey):
            """Fail on construction: reaching the wallet means a check let the run through."""
            raise AssertionError("wallet should not be constructed")

    monkeypatch.setattr(miner.bt, "logging", fake_logging)
    monkeypatch.setattr(miner, "_fetch_backend_families", lambda url: {"cf_search_and_rescue": "private"})
    monkeypatch.setattr(miner, "_backend_reachable", lambda url: True)
    monkeypatch.setattr(miner, "_submission_window", lambda url: {"open": False, "reopens_in_seconds": 3600})
    if hasattr(miner.bt, "Wallet"):
        monkeypatch.setattr(miner.bt, "Wallet", FakeWallet)
    else:
        monkeypatch.setattr(miner.bt, "wallet", FakeWallet)

    exit_code = miner.main([
        "--family_id", "cf_search_and_rescue", "--artifact", _make_submission(tmp_path),
        "--backend_url", "http://backend.test",
    ])

    assert exit_code == 1
    assert any("freeze" in line and "60 minutes" in line for line in error_lines)


# ── the upload and how it talks to the miner ──────────────────────────────────

class _Response:
    """An httpx-shaped reply carrying a status code and a JSON body."""

    def __init__(self, status_code, payload=None):
        """Keep the status code and payload, and render the body as JSON text."""
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = json.dumps(self._payload)

    def json(self):
        """Return the payload the reply was built with."""
        return self._payload


def _wallet(name="my cold", hotkey_str="my_hot"):
    """A signing wallet whose coldkey name contains a space, so quoting can be checked."""
    return SimpleNamespace(
        name=name, hotkey_str=hotkey_str,
        hotkey=SimpleNamespace(ss58_address="5ExampleHotkey", sign=lambda message: b"signature"),
    )


def _upload_env(monkeypatch, responses, *, status=None):
    """Serve the queued replies in order, drop every sleep, and capture the logged lines."""
    calls = []
    logs = {"info": [], "error": []}

    def fake_post(url, **kwargs):
        """Record the URL and pop the next queued reply."""
        calls.append(url)
        return responses.pop(0)

    monkeypatch.setattr(miner.httpx, "post", fake_post)
    monkeypatch.setattr(miner.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(miner, "_submission_status", lambda url, digest: status)
    monkeypatch.setattr(miner.bt, "logging", SimpleNamespace(
        info=lambda m, *_a, **_k: logs["info"].append(m),
        warning=lambda *_a, **_k: None,
        error=lambda m, *_a, **_k: logs["error"].append(m),
        set_debug=lambda *_a, **_k: None,
    ))
    return calls, logs


def test_upload_retries_until_the_backend_has_scanned_the_commit(monkeypatch, tmp_path):
    """A 404 and then a 503 are both retried, and the third attempt stores the archive."""
    calls, logs = _upload_env(
        monkeypatch,
        [_Response(404), _Response(503), _Response(200, {"stored": True})],
        status={"registered": True, "awaiting_upload": True},
    )
    ok = miner._upload_private_artifact("http://backend.test", _make_submission(tmp_path), "a" * 64, _wallet())
    assert ok is True
    assert len(calls) == 3
    assert any("has not scanned" in line for line in logs["info"])


def test_upload_stops_with_the_reason_when_the_commit_was_rejected(monkeypatch, tmp_path):
    """A 404 the backend explains as a hash collision stops after one attempt, with that reason."""
    calls, logs = _upload_env(
        monkeypatch, [_Response(404)],
        status={"registered": False, "reason_code": "HASH_COLLISION",
                "detail": "digest already registered to another model"},
    )
    ok = miner._upload_private_artifact("http://backend.test", _make_submission(tmp_path), "a" * 64, _wallet())
    assert ok is False
    assert len(calls) == 1
    assert any("earliest on-chain commit wins" in line for line in logs["error"])


def test_upload_shows_the_backend_message_on_a_rejection(monkeypatch, tmp_path):
    """A 409 passes the backend's own duplicate-submission text straight through to the miner."""
    detail = "Duplicate submission: same model as an existing submission (UID 18)"
    calls, logs = _upload_env(monkeypatch, [_Response(409, {"detail": detail})])
    ok = miner._upload_private_artifact("http://backend.test", _make_submission(tmp_path), "a" * 64, _wallet())
    assert ok is False
    assert len(calls) == 1
    assert any(detail in line for line in logs["error"])


def test_upload_gives_up_with_the_resume_command(monkeypatch, tmp_path):
    """The command must carry the wallet that signed the commitment, or the retry signs as someone else."""
    monkeypatch.setattr(miner, "UPLOAD_RETRY_BUDGET_SEC", 0)
    calls, logs = _upload_env(monkeypatch, [_Response(404)], status=None)
    artifact = _make_submission(tmp_path)
    ok = miner._upload_private_artifact("http://backend.test", artifact, "a" * 64, _wallet())
    assert ok is False
    resume = next(line for line in logs["error"] if "--upload-only" in line)
    assert "--wallet.name 'my cold'" in resume
    assert "--wallet.hotkey my_hot" in resume
    assert "--backend-url http://backend.test" in resume
    assert shlex.quote(artifact) in resume


def test_the_resume_command_leaves_out_the_default_backend(tmp_path):
    """The default backend URL is omitted from the resume line, the wallet never is."""
    resume = miner._resume_hint(
        "cf_autopilot", str(tmp_path / "submission.zip"), _wallet(), miner.DEFAULT_BACKEND_URL)
    assert "--backend-url" not in resume
    assert "--wallet.name 'my cold'" in resume


# ── the whole submission against a fake backend ───────────────────────────────

class _FakeBackend:
    """Just enough of the backend for a miner to submit against: it registers a digest
    when the chain commit lands, needs one scan before the upload is accepted, and
    refuses an upload that repeats a model it already holds."""

    def __init__(self):
        """Start with nothing registered, one scan still owed, and no uploads or fingerprints yet."""
        self.registered = {}
        self.scans_needed = 1
        self.uploads = []
        self.fingerprints = set()

    def _json(self, status, payload):
        """Wrap a status code and payload as a reply object."""
        return _Response(status, payload)

    def get(self, url, **kwargs):
        """Answer health, family metadata, the submission window and digest status; anything else fails the run."""
        if url.endswith("/health"):
            return self._json(200, {"status": "ok"})
        if url.endswith("/families/metadata"):
            return self._json(200, {"challenge_families": {"cf_search_and_rescue": {"visibility": "private"}}})
        if url.endswith("/miners/submission-window"):
            return self._json(200, {"open": True, "reopens_in_seconds": 0})
        if "/miners/models/" in url and url.endswith("/status"):
            digest = url.rsplit("/", 2)[1]
            if digest in self.registered:
                return self._json(200, {"registered": True, "uid": 7, "awaiting_upload": True})
            return self._json(200, {"registered": False, "reason_code": None, "detail": None})
        raise AssertionError(f"unexpected GET {url}")

    def post(self, url, *, headers, files, timeout):
        """Accept the upload once the commit is scanned and the bytes hash to the digest; 409 on a repeated model."""
        assert url.endswith("/private-upload")
        digest = url.rsplit("/", 2)[1]
        assert headers["X-Miner-Hotkey"] == "5ExampleHotkey"
        assert headers["X-Miner-Signature"] == b"signature".hex()
        if digest not in self.registered or self.scans_needed > 0:
            self.scans_needed -= 1
            return self._json(404, {"detail": "No private model for this digest"})
        body = files["file"][1].read()
        assert hashlib.sha256(body).hexdigest() == digest
        fingerprint = _content_key(body)
        if fingerprint in self.fingerprints:
            return self._json(409, {"detail": "Duplicate submission: same model as an existing submission (UID 7)"})
        self.fingerprints.add(fingerprint)
        self.uploads.append(digest)
        return self._json(200, {"stored": True, "status": "PENDING_BENCHMARK"})

    def commit(self, data):
        """Register the digest against its family, the way the chain scanner would."""
        payload = json.loads(data)
        self.registered[payload["s"]] = payload["f"]


def _content_key(archive: bytes) -> str:
    """What the backend fingerprints: the python source with comment lines dropped."""
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        source = zf.read("drone_agent.py").decode()
    return "\n".join(line for line in source.splitlines() if not line.strip().startswith("#"))


def _chain(monkeypatch, backend: _FakeBackend):
    """Point the bittensor wallet and subtensor classes at fakes that commit into the given backend."""
    class FakeWallet:
        """A wallet with the fixed test hotkey and a signature that is always the same bytes."""

        def __init__(self, name, hotkey):
            """Keep the coldkey name and hotkey string, and expose a hotkey that signs."""
            self.name, self.hotkey_str = name, hotkey
            self.hotkey = SimpleNamespace(ss58_address="5ExampleHotkey", sign=lambda m: b"signature")

    class FakeSubtensor:
        """Chain stand-in that forwards every commitment straight into the fake backend."""

        def __init__(self, network):
            """Ignore the network name; the fake chain has nothing to connect to."""
            pass

        def metagraph(self, netuid):
            """Return a metagraph where the fixed test hotkey is the only registrant."""
            return SimpleNamespace(hotkeys=["5ExampleHotkey"])

        def set_commitment(self, wallet, netuid, data, *, mev_protection=False):
            """Hand the payload to the fake backend and report the extrinsic succeeded."""
            backend.commit(data)
            return SimpleNamespace(success=True)

    monkeypatch.setattr(miner.bt, "Wallet" if hasattr(miner.bt, "Wallet") else "wallet", FakeWallet)
    monkeypatch.setattr(miner.bt, "Subtensor" if hasattr(miner.bt, "Subtensor") else "subtensor", FakeSubtensor)


def test_submit_private_end_to_end_against_a_fake_backend(monkeypatch, tmp_path):
    """The full private flow commits the digest, waits out one unscanned poll and lands the upload.
    A repack of the same model with one comment added is still refused as a duplicate."""
    backend = _FakeBackend()
    logs = {"info": [], "error": []}
    monkeypatch.setattr(miner, "httpx", backend)
    monkeypatch.setattr(miner.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(miner.bt, "logging", SimpleNamespace(
        info=lambda m, *_a, **_k: logs["info"].append(m), warning=lambda *_a, **_k: None,
        error=lambda m, *_a, **_k: logs["error"].append(m), set_debug=lambda *_a, **_k: None,
    ))
    _chain(monkeypatch, backend)
    artifact = _make_submission(tmp_path)
    digest = miner._sha256_file(artifact)

    code = miner.submit_private(
        family_id="cf_search_and_rescue", artifact=artifact, backend_url="http://backend.test",
        wallet_name="w", wallet_hotkey="h",
    )

    assert code == 0
    assert backend.registered == {digest: "cf_search_and_rescue"}
    assert backend.uploads == [digest]
    assert any("PRIVATE MODEL SUBMITTED SUCCESSFULLY" in line for line in logs["info"])
    assert any("has not scanned" in line for line in logs["info"])

    # a copycat repacks the same model with one comment line and a fresh hotkey
    copy_dir = tmp_path / "copy"
    copy_dir.mkdir()
    copy = copy_dir / "submission.zip"
    with zipfile.ZipFile(artifact) as original, zipfile.ZipFile(copy, "w") as target:
        for info in original.infolist():
            payload = original.read(info)
            if info.filename.endswith(".py"):
                payload = b"# new version\n" + payload
            target.writestr(info.filename, payload)
    copy_digest = miner._sha256_file(str(copy))
    assert copy_digest != digest
    backend.scans_needed = 0

    code = miner.submit_private(
        family_id="cf_search_and_rescue", artifact=str(copy), backend_url="http://backend.test",
        wallet_name="w", wallet_hotkey="h",
    )

    assert code == 1
    assert backend.uploads == [digest]
    assert any("same model as an existing submission (UID 7)" in line for line in logs["error"])
    assert any("--upload-only" in line for line in logs["error"])


def test_explain_rejection_reads_like_a_sentence():
    """A known reason code becomes advice the miner can act on; an unknown one falls back to the raw detail."""
    text = miner.explain_rejection("ONE_TASK_PER_HOTKEY", "hotkey already used in cf_autopilot")
    assert "register a new hotkey" in text and "cf_autopilot" in text
    assert miner.explain_rejection("something_new", "raw detail") == "raw detail"
