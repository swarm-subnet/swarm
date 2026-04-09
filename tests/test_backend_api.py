from __future__ import annotations

import asyncio
import hashlib
import json

import httpx

from swarm.validator import backend_api


def _run(coro):
    return asyncio.run(coro)


class _DummyHotkey:
    ss58_address = "validator_hotkey"

    def sign(self, message: bytes) -> bytes:
        _ = message
        return b"\x01\x02"


class _DummyWallet:
    hotkey = _DummyHotkey()


class _FakeResponse:
    def __init__(
        self,
        payload: dict,
        status_code: int = 200,
        raise_error: Exception | None = None,
    ):
        self._payload = payload
        self.status_code = status_code
        self._raise_error = raise_error

    def raise_for_status(self):
        if self._raise_error:
            raise self._raise_error

    def json(self):
        return self._payload


class _FakeAsyncClient:
    def __init__(self):
        self.post_response = _FakeResponse({})
        self.get_response = _FakeResponse({})
        self.post_error = None
        self.get_error = None
        self.last_post = None
        self.last_get = None

    async def post(self, url, **kwargs):
        self.last_post = (url, kwargs)
        if self.post_error:
            raise self.post_error
        return self.post_response

    async def get(self, url, **kwargs):
        self.last_get = (url, kwargs)
        if self.get_error:
            raise self.get_error
        return self.get_response

    async def aclose(self):
        return None


def _build_client(monkeypatch, tmp_path, wallet=None):
    monkeypatch.setattr(backend_api, "STATE_DIR", tmp_path)
    monkeypatch.setattr(
        backend_api, "RUNTIME_STATE_FILE", tmp_path / "runtime_state.json"
    )
    client = backend_api.BackendApiClient(
        wallet=wallet, base_url="http://backend.local"
    )
    return client


def test_load_runtime_state_missing_file_returns_defaults(monkeypatch, tmp_path):
    monkeypatch.setattr(backend_api, "RUNTIME_STATE_FILE", tmp_path / "missing.json")
    state = backend_api._load_runtime_state()
    assert state == {"last_weights": {}, "reeval_queue": [], "last_sync": 0, "benchmark_epoch": 0}


def test_save_runtime_state_writes_file(monkeypatch, tmp_path):
    monkeypatch.setattr(backend_api, "STATE_DIR", tmp_path)
    runtime_file = tmp_path / "runtime_state.json"
    monkeypatch.setattr(backend_api, "RUNTIME_STATE_FILE", runtime_file)

    backend_api._save_runtime_state({"last_weights": {"1": 1.0}, "benchmark_epoch": 7})
    assert runtime_file.exists()
    assert json.loads(runtime_file.read_text())["last_weights"] == {"1": 1.0}


def test_sign_request_without_wallet_returns_empty_headers(monkeypatch, tmp_path):
    client = _build_client(monkeypatch, tmp_path, wallet=None)
    try:
        assert client._sign_request("POST", "/x", b"{}") == {}
    finally:
        _run(client.close())


def test_sign_request_with_wallet_contains_expected_fields(monkeypatch, tmp_path):
    client = _build_client(monkeypatch, tmp_path, wallet=_DummyWallet())
    try:
        headers = client._sign_request("POST", "/validators/sync", b"{}")
        assert headers["X-Validator-Hotkey"] == "validator_hotkey"
        assert headers["X-Validator-Signature"] == "0102"
        assert "X-Validator-Nonce" in headers
        assert "X-Validator-Timestamp" in headers
    finally:
        _run(client.close())


def test_post_signed_success(monkeypatch, tmp_path):
    client = _build_client(monkeypatch, tmp_path, wallet=_DummyWallet())
    fake_http = _FakeAsyncClient()
    fake_http.post_response = _FakeResponse({"ok": True})
    client.client = fake_http

    try:
        data = _run(client._post_signed("/validators/heartbeat", {"status": "idle"}))
        assert data == {"ok": True}
        assert fake_http.last_post[0] == "http://backend.local/validators/heartbeat"
    finally:
        _run(client.close())


def test_post_signed_http_error_returns_json_body(monkeypatch, tmp_path):
    client = _build_client(monkeypatch, tmp_path, wallet=_DummyWallet())
    fake_http = _FakeAsyncClient()
    response = httpx.Response(
        status_code=400,
        json={"detail": "bad"},
        request=httpx.Request("POST", "http://backend.local/x"),
    )
    fake_http.post_error = httpx.HTTPStatusError(
        "boom", request=response.request, response=response
    )
    client.client = fake_http

    try:
        data = _run(client._post_signed("/x", {"a": 1}))
        assert data == {"detail": "bad"}
    finally:
        _run(client.close())


def test_post_new_model_maps_backend_response(monkeypatch, tmp_path):
    client = _build_client(monkeypatch, tmp_path, wallet=_DummyWallet())
    try:
        monkeypatch.setattr(client, "_get_miner_hotkey", lambda uid: "miner_hotkey")

        async def _fake_post(endpoint, data):
            assert endpoint == "/validators/models/new"
            assert data["hotkey"] == "miner_hotkey"
            return {"model_id": 55}

        monkeypatch.setattr(client, "_post_signed", _fake_post)
        result = _run(client.post_new_model(1, "hash", "coldkey", "validator"))
        assert result == {"accepted": True, "model_id": 55}
    finally:
        _run(client.close())


def test_post_new_model_forwards_github_url(monkeypatch, tmp_path):
    client = _build_client(monkeypatch, tmp_path, wallet=_DummyWallet())
    try:
        monkeypatch.setattr(client, "_get_miner_hotkey", lambda uid: "miner_hotkey")

        async def _fake_post(endpoint, data):
            assert endpoint == "/validators/models/new"
            assert data["github_url"] == "https://github.com/example/model"
            return {"model_id": 56}

        monkeypatch.setattr(client, "_post_signed", _fake_post)
        result = _run(
            client.post_new_model(
                1,
                "hash",
                "coldkey",
                "validator",
                github_url="https://github.com/example/model",
            )
        )
        assert result == {"accepted": True, "model_id": 56}
    finally:
        _run(client.close())


def test_post_new_model_handles_missing_hotkey(monkeypatch, tmp_path):
    client = _build_client(monkeypatch, tmp_path, wallet=_DummyWallet())
    try:
        monkeypatch.setattr(client, "_get_miner_hotkey", lambda uid: "")
        result = _run(client.post_new_model(1, "hash", "coldkey", "validator"))
        assert result["accepted"] is False
    finally:
        _run(client.close())


def test_sync_success_updates_runtime_state(monkeypatch, tmp_path):
    client = _build_client(monkeypatch, tmp_path, wallet=_DummyWallet())
    try:

        async def _fake_get(endpoint, **kwargs):
            assert endpoint == "/validators/sync"
            assert kwargs["extra_headers"] == {
                "X-Benchmark-Version": backend_api.BENCHMARK_VERSION
            }
            return {
                "current_champion": {
                    "uid": 2,
                    "benchmark_score": 0.91,
                    "model_hash": "abc",
                },
                "weights": {"2": 1.0},
                "reeval_queue": [{"uid": 2, "reason": "reeval"}],
                "leaderboard_version": 9,
                "benchmark_epoch": 7,
                "pending_models": [
                    {
                        "uid": 3,
                        "model_hash": "pending-hash",
                        "github_url": "https://github.com/example/pending-model",
                    }
                ],
            }

        monkeypatch.setattr(client, "_get_signed", _fake_get)
        result = _run(client.sync())
        assert result["current_top"]["uid"] == 2
        assert result["weights"] == {"2": 1.0}
        assert result["leaderboard_version"] == 9
        assert result["benchmark_epoch"] == 7
        assert result["pending_models"] == [
            {
                "uid": 3,
                "model_hash": "pending-hash",
                "github_url": "https://github.com/example/pending-model",
            }
        ]
    finally:
        _run(client.close())


def test_sync_fallback_returns_cached_runtime_state(monkeypatch, tmp_path):
    client = _build_client(monkeypatch, tmp_path, wallet=_DummyWallet())
    client._runtime_state = {
        "current_top": {"uid": 7},
        "last_weights": {"7": 1.0},
        "reeval_queue": [{"uid": 7, "reason": "cached"}],
        "last_sync": 1,
        "benchmark_epoch": 8,
    }
    try:

        async def _fake_get(endpoint, **kwargs):
            _ = endpoint
            _ = kwargs
            return {"error": "backend down"}

        monkeypatch.setattr(client, "_get_signed", _fake_get)
        result = _run(client.sync())
        assert result["fallback"] is True
        assert result["weights"] == {"7": 1.0}
        assert result["current_top"] == {"uid": 7}
        assert result["benchmark_epoch"] == 8
    finally:
        _run(client.close())


def test_upload_model_file_requires_wallet(monkeypatch, tmp_path):
    client = _build_client(monkeypatch, tmp_path, wallet=None)
    model = tmp_path / "model.zip"
    model.write_bytes(b"zip")
    try:
        result = _run(client.upload_model_file(1, model))
        assert result == {"error": "no wallet"}
    finally:
        _run(client.close())


def test_upload_model_file_posts_signed_request(monkeypatch, tmp_path):
    client = _build_client(monkeypatch, tmp_path, wallet=_DummyWallet())
    fake_http = _FakeAsyncClient()
    fake_http.post_response = _FakeResponse({"stored": True})
    client.client = fake_http

    model = tmp_path / "model.zip"
    payload = b"zip-contents"
    model.write_bytes(payload)
    expected_hash = hashlib.sha256(payload).hexdigest()
    try:
        result = _run(client.upload_model_file(4, model))
        assert result == {"stored": True}
        _, kwargs = fake_http.last_post
        assert kwargs["headers"]["X-Model-Hash"] == expected_hash
        assert kwargs["files"]["file"][0] == "model.zip"
    finally:
        _run(client.close())


def test_publish_epoch_seeds_posts_expected_payload(monkeypatch, tmp_path):
    client = _build_client(monkeypatch, tmp_path, wallet=_DummyWallet())
    try:

        async def _fake_post(endpoint, data):
            assert endpoint == "/validators/epoch/publish"
            assert data == {
                "epoch_number": 7,
                "seeds": [11, 12, 13],
                "started_at": "2026-03-25T10:00:00Z",
                "ended_at": "2026-03-25T10:10:00Z",
                "benchmark_version": "v4.0.0",
            }
            return {"accepted": True}

        monkeypatch.setattr(client, "_post_signed", _fake_post)
        result = _run(
            client.publish_epoch_seeds(
                epoch_number=7,
                seeds=[11, 12, 13],
                started_at="2026-03-25T10:00:00Z",
                ended_at="2026-03-25T10:10:00Z",
                benchmark_version="v4.0.0",
            )
        )
        assert result == {"accepted": True}
    finally:
        _run(client.close())


def test_classify_backend_failure_marks_epoch_mismatch_and_missing_github_as_terminal():
    is_terminal, reason = backend_api.classify_backend_failure(
        {"detail": "Benchmark epoch mismatch. Current benchmark epoch is 7, got 6."},
        "score",
    )
    assert is_terminal is True
    assert "epoch mismatch" in reason.lower()

    is_terminal, reason = backend_api.classify_backend_failure(
        {"detail": "github_url is required and must be https://github.com/{owner}/{repo}"},
        "new_model",
    )
    assert is_terminal is True
    assert "github_url" in reason
