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

"""How a validator handles private-track bytes: fetched from the backend only,
owner-only on disk, never kept for forensics, gone once the task is done."""
from __future__ import annotations

import asyncio
import hashlib
import stat
import zipfile
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from swarm.core import model_verify
from swarm.core.submission_policy import SUBMISSION_INTERFACE_VERSION
from swarm.validator.backend_api import BackendApiClient
from swarm.validator.utils_parts import model_fetch
from swarm.validator.utils_parts import run_task as run_task_mod

AGENT = (
    "import numpy as np\n\n\n"
    "class DroneFlightController:\n"
    "    def act(self, observation):\n"
    "        return np.zeros(6, dtype=np.float32)\n\n"
    "    def reset(self):\n"
    "        pass\n"
)


def _submission_bytes() -> bytes:
    """Return a zip holding one drone_agent.py, the smallest thing that reads as a submission."""
    import io

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("drone_agent.py", AGENT)
    return buffer.getvalue()


# ── the fetch itself ─────────────────────────────────────────────────────────

def _client_stub(handler):
    """Return a BackendApiClient stand-in whose requests are answered by handler, signing stubbed out."""
    async def fence(_resp):
        """Let every response through: the duplicate-instance guard has nothing to judge here."""
        return None

    return SimpleNamespace(
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        base_url="https://backend.test",
        _sign_request=lambda method, endpoint, body: {"X-Validator-Hotkey": "hk"},
        _fence_duplicate_instance=fence,
    )


def test_private_fetch_writes_the_bytes_owner_only(tmp_path):
    """The artifact lands byte for byte at mode 0600, so no other account on the box can read it."""
    body = _submission_bytes()
    stub = _client_stub(lambda request: httpx.Response(200, content=body))
    dest = tmp_path / "UID_7.zip"

    ok = asyncio.run(BackendApiClient.fetch_private_artifact(stub, "a" * 64, dest))

    assert ok is True
    assert dest.read_bytes() == body
    assert stat.S_IMODE(dest.stat().st_mode) == 0o600


def test_private_fetch_refusal_leaves_nothing_on_disk(tmp_path):
    """A 403 from the vault reports failure and leaves no partial file behind."""
    stub = _client_stub(lambda request: httpx.Response(403, json={"detail": "not trusted"}))
    dest = tmp_path / "UID_7.zip"
    assert asyncio.run(BackendApiClient.fetch_private_artifact(stub, "a" * 64, dest)) is False
    assert not dest.exists()


def test_private_fetch_drops_an_oversized_artifact(tmp_path, monkeypatch):
    """A stream that runs past the byte cap is cut off and the part already written is removed."""
    from swarm.validator import backend_api

    monkeypatch.setattr(backend_api, "MAX_MODEL_BYTES", 16)
    stub = _client_stub(lambda request: httpx.Response(200, content=b"x" * 64))
    dest = tmp_path / "UID_7.zip"
    assert asyncio.run(BackendApiClient.fetch_private_artifact(stub, "a" * 64, dest)) is False
    assert not dest.exists()


# ── discovery: a private entry never touches GitHub ──────────────────────────

def _discovery_env(monkeypatch, tmp_path, fetched: bytes | None):
    """Return a validator stand-in serving `fetched` as the private artifact, cache redirected to tmp_path."""
    async def fetch(model_hash, dest):
        """Write the prepared bytes to dest, or report failure when there are none."""
        if fetched is None:
            return False
        dest.write_bytes(fetched)
        return True

    async def no_docker(*_args, **_kwargs):
        """Stand in for the container verification step and do nothing at all."""
        return None

    monkeypatch.setattr(model_fetch, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(model_fetch, "load_blacklist", lambda: set())
    monkeypatch.setattr(model_fetch, "verify_new_model_with_docker", no_docker)
    return SimpleNamespace(backend_api=SimpleNamespace(fetch_private_artifact=fetch))


def _entry(model_hash: str, *, is_private: bool, github_url: str = "") -> dict:
    """Return one backend discovery record for UID 7, shaped the way the fetcher reads them."""
    return {
        "uid": 7, "model_hash": model_hash, "family_id": "cf_search_and_rescue",
        "interface_version": SUBMISSION_INTERFACE_VERSION, "github_url": github_url,
        "artifact_path": "artifacts/x.zip" if github_url else "", "is_private": is_private,
    }


def test_private_model_is_fetched_from_the_backend_and_marked(monkeypatch, tmp_path):
    """A private entry comes from the vault, lands under its UID, and gets a .private marker beside it."""
    body = _submission_bytes()
    digest = hashlib.sha256(body).hexdigest()
    self = _discovery_env(monkeypatch, tmp_path, body)

    paths = asyncio.run(model_fetch._ensure_models_from_backend(self, [_entry(digest, is_private=True)]))

    model_fp = tmp_path / "UID_7.zip"
    assert paths == {7: (model_fp, "")}
    assert model_fp.read_bytes() == body
    assert model_fp.with_suffix(".private").exists()


def test_private_model_without_github_url_is_not_skipped(monkeypatch, tmp_path):
    """The public path needs a repo; the private path must not be dropped for lacking one."""
    body = _submission_bytes()
    digest = hashlib.sha256(body).hexdigest()
    self = _discovery_env(monkeypatch, tmp_path, body)
    assert asyncio.run(model_fetch._ensure_models_from_backend(self, [_entry(digest, is_private=False)])) == {}
    assert asyncio.run(model_fetch._ensure_models_from_backend(self, [_entry(digest, is_private=True)])) != {}


def test_failed_private_fetch_clears_the_marker(monkeypatch, tmp_path):
    """A download that fails leaves neither bytes nor marker, so nothing half-done is read as a cached model."""
    self = _discovery_env(monkeypatch, tmp_path, None)
    paths = asyncio.run(model_fetch._ensure_models_from_backend(self, [_entry("b" * 64, is_private=True)]))
    assert paths == {}
    assert not (tmp_path / "UID_7.zip").exists()
    assert not (tmp_path / "UID_7.private").exists()


def test_hash_mismatch_on_private_bytes_is_rejected(monkeypatch, tmp_path):
    """Private bytes whose digest is not the one the backend announced are refused and wiped."""
    self = _discovery_env(monkeypatch, tmp_path, _submission_bytes())
    paths = asyncio.run(model_fetch._ensure_models_from_backend(self, [_entry("c" * 64, is_private=True)]))
    assert paths == {}
    assert not (tmp_path / "UID_7.zip").exists()


# ── after the task ───────────────────────────────────────────────────────────

def _run(monkeypatch, tmp_path, *, is_private: bool) -> Path:
    """Drive one BENCHMARK task over a pre-staged model file, fetch and phase stubbed, and return that path."""
    model_fp = tmp_path / "UID_7.zip"
    model_fp.write_bytes(_submission_bytes())
    model_fetch._set_private_marker(model_fp, is_private)
    digest = hashlib.sha256(model_fp.read_bytes()).hexdigest()

    async def ensure(_self, _entries):
        """Hand back the pre-staged file as UID 7's model instead of downloading one."""
        return {7: (model_fp, "")}

    async def phase(*_args, **_kwargs):
        """Check that the model bytes are still on disk while the evaluation phase runs."""
        assert model_fp.exists(), "the bytes must be present while the phase runs"

    monkeypatch.setattr(run_task_mod, "_ensure_models_from_backend", ensure)
    monkeypatch.setattr(run_task_mod, "_run_phase", phase)
    task = {"uid": 7, "phase": "BENCHMARK", "task_id": 1, "model_hash": digest, "is_private": is_private}
    asyncio.run(run_task_mod.run_task(
        SimpleNamespace(), task, cancel_flag=asyncio.Event(), wake_flag=asyncio.Event(),
    ))
    return model_fp


def test_private_bytes_are_deleted_once_the_task_is_done(monkeypatch, tmp_path):
    """Both the archive and its marker are gone the moment the task returns: nothing private outlives it."""
    model_fp = _run(monkeypatch, tmp_path, is_private=True)
    assert not model_fp.exists()
    assert not model_fp.with_suffix(".private").exists()


def test_public_bytes_stay_cached_between_tasks(monkeypatch, tmp_path):
    """A public archive survives its task, so the next one reuses it rather than downloading again."""
    model_fp = _run(monkeypatch, tmp_path, is_private=False)
    assert model_fp.exists()


def test_private_bytes_are_dropped_when_the_fetch_itself_is_cancelled(monkeypatch, tmp_path):
    """Cancellation during the download must not leave the bytes on disk with nothing to sweep them."""
    monkeypatch.setattr(model_fetch, "MODEL_DIR", tmp_path)
    model_fp = tmp_path / "UID_7.zip"

    async def ensure(_self, _entries):
        """Stage the private bytes and their marker, then raise as if the download were cancelled."""
        model_fp.write_bytes(_submission_bytes())
        model_fetch._set_private_marker(model_fp, True)
        raise asyncio.CancelledError

    monkeypatch.setattr(run_task_mod, "_ensure_models_from_backend", ensure)
    task = {"uid": 7, "phase": "BENCHMARK", "task_id": 1, "model_hash": "d" * 64, "is_private": True}
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run_task_mod.run_task(
            SimpleNamespace(), task, cancel_flag=asyncio.Event(), wake_flag=asyncio.Event(),
        ))

    assert not model_fp.exists()
    assert not model_fp.with_suffix(".private").exists()


# ── forensics ────────────────────────────────────────────────────────────────

def test_flagged_private_model_is_not_kept_for_analysis(monkeypatch, tmp_path):
    """Saving a flagged submission for forensics copies public bytes and silently skips marked ones."""
    monkeypatch.setattr(model_verify, "MODEL_DIR", tmp_path)
    model_fp = tmp_path / "UID_9.zip"
    model_fp.write_bytes(_submission_bytes())

    model_fetch._set_private_marker(model_fp, True)
    model_verify.save_fake_model_for_analysis(model_fp, 9, "d" * 64, "test", {})
    assert not (tmp_path / "UID_9_fake").exists()

    model_fetch._set_private_marker(model_fp, False)
    model_verify.save_fake_model_for_analysis(model_fp, 9, "d" * 64, "test", {})
    assert (tmp_path / "UID_9_fake" / "1" / "model.zip").exists()
