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

import asyncio
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
    assert state == {
        "last_weights": {},
        "last_kings": [],
        "reeval_queue": [],
        "assigned_tasks": [],
        "rollout": {},
        "validator_compatibility": {},
        "leaderboard_version": 0,
        "last_sync": 0,
        "benchmark_epoch": 0,
    }


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
        assert headers["X-Validator-Session"] == client.session_id
        assert headers["X-Swarm-Validator-Contract"] == "agent_rpc.v1"
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


def test_post_heartbeat_forwards_queue_and_active_task(monkeypatch, tmp_path):
    client = _build_client(monkeypatch, tmp_path, wallet=_DummyWallet())
    try:
        captured = {}

        async def _fake_post(endpoint, data):
            captured["endpoint"] = endpoint
            captured["data"] = data
            return {"recorded": True}

        monkeypatch.setattr(client, "_post_signed", _fake_post)
        result = _run(
            client.post_heartbeat(
                status="evaluating_benchmark",
                current_uid=114,
                progress=10,
                total_seeds=800,
                queue=[{"uid": 114, "phase": "benchmark", "status": "benchmark", "enqueue_time": "2026-04-15T17:00:00Z"}],
                active_task={"uid": 114, "phase": "BENCHMARK", "assignment_id": 9, "family_id": "cf_autopilot"},
                backend_decision_version=12,
            )
        )
        assert result == {"recorded": True}
        assert captured["endpoint"] == "/validators/heartbeat"
        assert captured["data"]["active_task"]["assignment_id"] == 9
        assert captured["data"]["active_task"]["family_id"] == "cf_autopilot"
        assert captured["data"]["backend_decision_version"] == 12
        assert captured["data"]["session_id"] == client.session_id
    finally:
        _run(client.close())


def test_duplicate_session_response_exits_process(monkeypatch, tmp_path):
    client = _build_client(monkeypatch, tmp_path, wallet=_DummyWallet())
    fake_http = _FakeAsyncClient()
    response = httpx.Response(
        status_code=409,
        json={"detail": "DUPLICATE_VALIDATOR_INSTANCE"},
        request=httpx.Request("POST", "http://backend.local/validators/heartbeat"),
    )
    fake_http.post_response = response
    client.client = fake_http

    try:
        try:
            _run(client._post_signed("/validators/heartbeat", {"status": "idle"}))
            assert False, "duplicate session must terminate the validator"
        except SystemExit as exc:
            assert exc.code == 1
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
        assert data["detail"] == "bad"
        assert data["error"] == "HTTP 400"
        assert data["status_code"] == 400
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
                    "family_id": "cf_search_and_rescue",
                    "benchmark_score": 0.91,
                    "model_hash": "abc",
                },
                "weights": {"2": 1.0},
                "reeval_queue": [{"uid": 2, "family_id": "cf_search_and_rescue", "reason": "reeval"}],
                "leaderboard_version": 9,
                "rollout": {
                    "mode": "staged",
                    "default_family_id": "cf_search_and_rescue",
                    "enabled_family_ids": ["cf_search_and_rescue"],
                    "required_validator_contract": "family_sync.v1",
                },
                "validator_compatibility": {
                    "validator_contract": "family_sync.v1",
                    "compatible": True,
                    "upgrade_required": False,
                    "safe_family_ids": ["cf_search_and_rescue"],
                    "reason": "validator_contract_accepted",
                },
                "benchmark_epoch": 7,
                "assigned_tasks": [
                    {
                        "task_id": 5,
                        "uid": 3,
                        "phase": "SCREENING",
                        "family_id": "cf_autopilot",
                        "status": "QUEUED",
                        "priority": 20,
                        "assigned_at": "2026-04-15T18:00:00Z",
                    }
                ],
                "pending_models": [
                    {
                        "uid": 3,
                        "family_id": "cf_autopilot",
                        "model_hash": "pending-hash",
                        "github_url": "https://github.com/example/pending-model",
                    }
                ],
            }

        monkeypatch.setattr(client, "_get_signed", _fake_get)
        result = _run(client.sync())
        assert result["current_top"]["uid"] == 2
        assert result["current_top"]["family_id"] == "cf_search_and_rescue"
        assert result["weights"] == {"2": 1.0}
        assert result["reeval_queue"] == [
            {"uid": 2, "family_id": "cf_search_and_rescue", "reason": "reeval"}
        ]
        assert result["leaderboard_version"] == 9
        assert result["benchmark_epoch"] == 7
        assert result["assigned_tasks"][0]["task_id"] == 5
        assert result["assigned_tasks"][0]["family_id"] == "cf_autopilot"
        assert result["rollout"]["mode"] == "staged"
        assert result["validator_compatibility"]["compatible"] is True
        assert result["pending_models"] == [
            {
                "uid": 3,
                "family_id": "cf_autopilot",
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
        "assigned_tasks": [{"task_id": 11, "uid": 7, "phase": "SCREENING", "status": "QUEUED"}],
        "rollout": {"mode": "single_family"},
        "validator_compatibility": {"compatible": True},
        "leaderboard_version": 3,
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
        assert result["assigned_tasks"] == [{"task_id": 11, "uid": 7, "phase": "SCREENING", "status": "QUEUED"}]
        assert result["rollout"] == {"mode": "single_family"}
        assert result["validator_compatibility"] == {"compatible": True}
        assert result["benchmark_epoch"] == 8
    finally:
        _run(client.close())


def test_publish_epoch_seeds_posts_expected_payload(monkeypatch, tmp_path):
    client = _build_client(monkeypatch, tmp_path, wallet=_DummyWallet())
    try:

        async def _fake_post(endpoint, data):
            assert endpoint == "/validators/epoch/publish"
            assert data == {
                "epoch_number": 7,
                "family_id": "cf_autopilot",
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
                family_id="cf_autopilot",
                seeds=[11, 12, 13],
                started_at="2026-03-25T10:00:00Z",
                ended_at="2026-03-25T10:10:00Z",
                benchmark_version="v4.0.0",
            )
        )
        assert result == {"accepted": True}
    finally:
        _run(client.close())


def test_post_signed_5xx_tags_transport_failure(monkeypatch, tmp_path):
    client = _build_client(monkeypatch, tmp_path, wallet=_DummyWallet())
    fake_http = _FakeAsyncClient()
    response = httpx.Response(
        status_code=502,
        json={"detail": "Bad Gateway"},
        request=httpx.Request("POST", "http://backend.local/x"),
    )
    fake_http.post_error = httpx.HTTPStatusError(
        "boom", request=response.request, response=response
    )
    client.client = fake_http
    try:
        data = _run(client._post_signed("/x", {"a": 1}))
        assert data.get("transport_failure") is True
        assert data.get("status_code") == 502
    finally:
        _run(client.close())


def test_post_signed_4xx_does_not_tag_transport_failure(monkeypatch, tmp_path):
    client = _build_client(monkeypatch, tmp_path, wallet=_DummyWallet())
    fake_http = _FakeAsyncClient()
    response = httpx.Response(
        status_code=409,
        json={"detail": "already submitted"},
        request=httpx.Request("POST", "http://backend.local/x"),
    )
    fake_http.post_error = httpx.HTTPStatusError(
        "conflict", request=response.request, response=response
    )
    client.client = fake_http
    try:
        data = _run(client._post_signed("/x", {"a": 1}))
        assert "transport_failure" not in data
        assert data["detail"] == "already submitted"
        assert data["status_code"] == 409
    finally:
        _run(client.close())


def test_post_signed_connection_error_tags_transport_failure(monkeypatch, tmp_path):
    client = _build_client(monkeypatch, tmp_path, wallet=_DummyWallet())
    fake_http = _FakeAsyncClient()
    fake_http.post_error = httpx.ConnectError("cannot connect")
    client.client = fake_http
    try:
        data = _run(client._post_signed("/x", {"a": 1}))
        assert data.get("transport_failure") is True
        assert "error" in data
    finally:
        _run(client.close())


def test_post_signed_timeout_tags_transport_failure(monkeypatch, tmp_path):
    client = _build_client(monkeypatch, tmp_path, wallet=_DummyWallet())
    fake_http = _FakeAsyncClient()
    fake_http.post_error = httpx.ReadTimeout("read timed out")
    client.client = fake_http
    try:
        data = _run(client._post_signed("/x", {"a": 1}))
        assert data.get("transport_failure") is True
    finally:
        _run(client.close())


def test_get_signed_5xx_tags_transport_failure(monkeypatch, tmp_path):
    client = _build_client(monkeypatch, tmp_path, wallet=_DummyWallet())
    fake_http = _FakeAsyncClient()
    response = httpx.Response(
        status_code=503,
        json={"detail": "Service Unavailable"},
        request=httpx.Request("GET", "http://backend.local/x"),
    )
    fake_http.get_error = httpx.HTTPStatusError(
        "boom", request=response.request, response=response
    )
    client.client = fake_http
    try:
        data = _run(client._get_signed("/x"))
        assert data.get("transport_failure") is True
        assert data.get("status_code") == 503
    finally:
        _run(client.close())


def test_authorize_with_retry_passes_through_success():
    async def _auth():
        return {"authorized": True, "task_id": 42}

    result = _run(
        backend_api.authorize_with_retry(_auth, attempts=3, base_delay=0.0)
    )
    assert result == {"authorized": True, "task_id": 42}


def test_authorize_with_retry_does_not_retry_real_denial():
    calls = {"n": 0}

    async def _auth():
        calls["n"] += 1
        return {"authorized": False, "reason": "epoch rotated"}

    result = _run(
        backend_api.authorize_with_retry(_auth, attempts=3, base_delay=0.0)
    )
    assert result == {"authorized": False, "reason": "epoch rotated"}
    assert calls["n"] == 1


def test_authorize_with_retry_recovers_after_transient_failures():
    calls = {"n": 0}

    async def _auth():
        calls["n"] += 1
        if calls["n"] < 3:
            return {"error": "502 Bad Gateway", "transport_failure": True}
        return {"authorized": True, "task_id": 7}

    result = _run(
        backend_api.authorize_with_retry(_auth, attempts=5, base_delay=0.0)
    )
    assert result == {"authorized": True, "task_id": 7}
    assert calls["n"] == 3


def test_authorize_with_retry_raises_after_exhausting_transients():
    calls = {"n": 0}

    async def _auth():
        calls["n"] += 1
        return {"error": "read timed out", "transport_failure": True}

    try:
        _run(
            backend_api.authorize_with_retry(_auth, attempts=3, base_delay=0.0)
        )
    except backend_api.BackendTransportError as exc:
        assert "read timed out" in str(exc)
    else:
        raise AssertionError("expected BackendTransportError")
    assert calls["n"] == 3
