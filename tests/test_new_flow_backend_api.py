from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from swarm.validator import backend_api


class _DummyHotkey:
    ss58_address = "validator_hotkey"

    def sign(self, message: bytes) -> bytes:
        _ = message
        return b"\x01\x02"


class _DummyWallet:
    hotkey = _DummyHotkey()


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


class _StubGetClient:
    def __init__(self, response: _FakeResponse, exc: Exception | None = None):
        self._response = response
        self._exc = exc
        self.last_url: str | None = None
        self.last_headers: dict | None = None

    async def get(self, url: str, headers=None, **_kw):
        self.last_url = url
        self.last_headers = headers or {}
        if self._exc is not None:
            raise self._exc
        return self._response

    def stream(self, *_a, **_kw):
        raise NotImplementedError


def _client(stub_get: _StubGetClient | None = None) -> backend_api.BackendApiClient:
    client = backend_api.BackendApiClient(
        wallet=_DummyWallet(), base_url="https://example/test"
    )
    if stub_get is not None:
        client.client = stub_get  # type: ignore[assignment]
    return client


def test_next_task_returns_task_dict():
    stub = _StubGetClient(_FakeResponse({"task": {"task_id": 42, "uid": 7}}))
    client = _client(stub)

    result = asyncio.run(client.next_task())

    assert result == {"task_id": 42, "uid": 7}
    assert stub.last_url.endswith("/validators/next-task")
    assert "X-Benchmark-Version" in stub.last_headers


def test_next_task_returns_none_when_payload_has_null_task():
    stub = _StubGetClient(_FakeResponse({"task": None}))
    client = _client(stub)

    assert asyncio.run(client.next_task()) is None


def test_next_task_raises_protocol_mismatch_on_404():
    stub = _StubGetClient(_FakeResponse({}, status_code=404))
    client = _client(stub)

    with pytest.raises(backend_api.BackendProtocolMismatchError):
        asyncio.run(client.next_task())


def test_next_task_raises_protocol_mismatch_on_405():
    stub = _StubGetClient(_FakeResponse({}, status_code=405))
    client = _client(stub)

    with pytest.raises(backend_api.BackendProtocolMismatchError):
        asyncio.run(client.next_task())


def test_next_task_raises_transport_on_500():
    stub = _StubGetClient(_FakeResponse({}, status_code=503))
    client = _client(stub)

    with pytest.raises(backend_api.BackendTransportError):
        asyncio.run(client.next_task())


def test_next_task_raises_transport_on_network_error():
    stub = _StubGetClient(
        _FakeResponse({}),
        exc=httpx.ConnectError("connection refused"),
    )
    client = _client(stub)

    with pytest.raises(backend_api.BackendTransportError):
        asyncio.run(client.next_task())


def test_submit_task_result_posts_signed_payload(monkeypatch):
    captured: dict = {}

    async def _fake_post_signed(self, endpoint: str, data: dict):
        captured["endpoint"] = endpoint
        captured["data"] = data
        return {"recorded": True, "task_status": "SUBMITTED"}

    monkeypatch.setattr(
        backend_api.BackendApiClient, "_post_signed", _fake_post_signed,
    )
    client = _client()

    result = asyncio.run(client.submit_task_result(
        task_id=33,
        score=0.42,
        per_type_scores={"city": 0.4, "open": 0.5},
        seeds_evaluated=200,
        early_failed=False,
        epoch_number=5,
    ))

    assert result["recorded"] is True
    assert captured["endpoint"] == "/validators/tasks/33/result"
    assert captured["data"]["score"] == 0.42
    assert captured["data"]["seeds_evaluated"] == 200
    assert captured["data"]["early_failed"] is False
    assert captured["data"]["epoch_number"] == 5
    assert captured["data"]["per_type_scores"] == {"city": 0.4, "open": 0.5}


def test_parse_sse_block_returns_payload_with_event_id():
    block = ["event: state", "id: 7", 'data: {"type":"wake"}']
    payload = backend_api._parse_sse_block(block)
    assert payload == {"type": "wake", "event_id": 7}


def test_parse_sse_block_preserves_existing_event_id_field():
    block = ["id: 9", 'data: {"type":"wake","event_id":12}']
    payload = backend_api._parse_sse_block(block)
    assert payload == {"type": "wake", "event_id": 12}


def test_parse_sse_block_returns_none_on_invalid_json():
    block = ["data: not-json-at-all"]
    payload = backend_api._parse_sse_block(block)
    assert payload is None


def test_parse_sse_block_ignores_comment_lines():
    block = [": keepalive comment", 'data: {"type":"wake"}']
    payload = backend_api._parse_sse_block(block)
    assert payload == {"type": "wake"}


class _StreamCM:
    def __init__(self, status_code: int, lines: list[str]):
        self.status_code = status_code
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _StubStreamClient:
    def __init__(self, status_code: int, lines: list[str]):
        self._status = status_code
        self._lines = lines
        self.last_headers: dict | None = None

    def stream(self, _method, _url, headers=None, **_kw):
        self.last_headers = headers or {}
        return _StreamCM(self._status, self._lines)


def test_events_yields_parsed_frames():
    stub = _StubStreamClient(
        200,
        [
            "event: state",
            "id: 1",
            'data: {"type":"wake"}',
            "",
            "event: state",
            "id: 2",
            'data: {"type":"screening_failed","uid":7}',
            "",
        ],
    )
    client = _client()
    client.client = stub  # type: ignore[assignment]

    async def _collect():
        events = []
        async for event in client.events(last_event_id=0):
            events.append(event)
        return events

    events = asyncio.run(_collect())
    assert events[0]["type"] == "wake"
    assert events[0]["event_id"] == 1
    assert events[1]["type"] == "screening_failed"
    assert events[1]["event_id"] == 2
    assert stub.last_headers.get("Last-Event-ID") == "0"
    assert stub.last_headers.get("Accept") == "text/event-stream"


def test_events_raises_protocol_mismatch_on_404():
    stub = _StubStreamClient(404, [])
    client = _client()
    client.client = stub  # type: ignore[assignment]

    async def _collect():
        events = []
        async for event in client.events():
            events.append(event)
        return events

    with pytest.raises(backend_api.BackendProtocolMismatchError):
        asyncio.run(_collect())
