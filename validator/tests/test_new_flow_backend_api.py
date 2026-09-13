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

"""Tests for the task-driven backend flow: next-task polling, result submission and the SSE stream."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from swarm.validator import backend_api


class _DummyHotkey:
    """A hotkey that signs anything as the two bytes 0102."""
    ss58_address = "validator_hotkey"

    def sign(self, message: bytes) -> bytes:
        """Return a fixed two-byte signature, ignoring the message."""
        _ = message
        return b"\x01\x02"


class _DummyWallet:
    """A wallet carrying only the stub hotkey the client signs with."""
    hotkey = _DummyHotkey()


class _FakeResponse:
    """A minimal response exposing only a status code and a JSON payload."""
    def __init__(self, payload, status_code=200):
        """Store the payload and the status code the client will branch on."""
        self._payload = payload
        self.status_code = status_code

    def json(self):
        """Return the canned payload the response was built with."""
        return self._payload


class _StubGetClient:
    """A client whose get() records the URL and headers, then returns or raises what it was given."""
    def __init__(self, response: _FakeResponse, exc: Exception | None = None):
        """Hold the response to return, the optional exception to raise, and blank call records."""
        self._response = response
        self._exc = exc
        self.last_url: str | None = None
        self.last_headers: dict | None = None

    async def get(self, url: str, headers=None, **_kw):
        """Record the URL and headers, then raise the stored exception or return the response."""
        self.last_url = url
        self.last_headers = headers or {}
        if self._exc is not None:
            raise self._exc
        return self._response

    def stream(self, *_a, **_kw):
        """Raise NotImplementedError; the stub answers plain GETs only."""
        raise NotImplementedError


def _client(stub_get: _StubGetClient | None = None) -> backend_api.BackendApiClient:
    """Return a BackendApiClient signed by the stub wallet, with its HTTP layer optionally replaced."""
    client = backend_api.BackendApiClient(
        wallet=_DummyWallet(), base_url="https://example/test"
    )
    if stub_get is not None:
        client.client = stub_get  # type: ignore[assignment]
    return client


def test_next_task_returns_task_dict():
    """The task is unwrapped from its envelope, and the benchmark version header goes out."""
    stub = _StubGetClient(_FakeResponse({"task": {"task_id": 42, "uid": 7}}))
    client = _client(stub)

    result = asyncio.run(client.next_task())

    assert result == {"task_id": 42, "uid": 7}
    assert stub.last_url.endswith("/validators/next-task")
    assert "X-Benchmark-Version" in stub.last_headers


def test_next_task_returns_none_when_payload_has_null_task():
    """A null task in the envelope reads as no work, not as a missing key."""
    stub = _StubGetClient(_FakeResponse({"task": None}))
    client = _client(stub)

    assert asyncio.run(client.next_task()) is None


def test_next_task_raises_protocol_mismatch_on_404():
    """A 404 means the backend predates the endpoint, and that is raised, never swallowed."""
    stub = _StubGetClient(_FakeResponse({}, status_code=404))
    client = _client(stub)

    with pytest.raises(backend_api.BackendProtocolMismatchError):
        asyncio.run(client.next_task())


def test_next_task_raises_protocol_mismatch_on_405():
    """A 405 is treated the same way as a 404: an old backend, not a transient fault."""
    stub = _StubGetClient(_FakeResponse({}, status_code=405))
    client = _client(stub)

    with pytest.raises(backend_api.BackendProtocolMismatchError):
        asyncio.run(client.next_task())


def test_next_task_raises_transport_on_500():
    """A 5xx is a retryable fault: BackendTransportError, not a protocol complaint."""
    stub = _StubGetClient(_FakeResponse({}, status_code=503))
    client = _client(stub)

    with pytest.raises(backend_api.BackendTransportError):
        asyncio.run(client.next_task())


def test_next_task_raises_transport_on_network_error():
    """A refused connection surfaces as BackendTransportError, never as a raw httpx error."""
    stub = _StubGetClient(
        _FakeResponse({}),
        exc=httpx.ConnectError("connection refused"),
    )
    client = _client(stub)

    with pytest.raises(backend_api.BackendTransportError):
        asyncio.run(client.next_task())


def test_next_task_warns_once_on_426_upgrade_required():
    """A 426 yields no work and latches the warning flag, which the next good poll clears."""
    stub = _StubGetClient(_FakeResponse({}, status_code=426))
    client = _client(stub)

    assert asyncio.run(client.next_task()) is None
    assert client._upgrade_warned is True
    assert asyncio.run(client.next_task()) is None

    stub._response = _FakeResponse({"task": {"task_id": 5, "uid": 1}})
    assert asyncio.run(client.next_task()) == {"task_id": 5, "uid": 1}
    assert client._upgrade_warned is False


def test_submit_task_result_posts_signed_payload(monkeypatch):
    """A result goes to the per-task result endpoint with the per-type scores under both breakdown keys."""
    captured: dict = {}

    async def _fake_post_signed(self, endpoint: str, data: dict):
        """Record the endpoint and body, and answer as if the backend stored them."""
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
    assert captured["data"]["metric_breakdown"] == {"city": 0.4, "open": 0.5}
    assert captured["data"]["per_type_scores"] == {"city": 0.4, "open": 0.5}


def test_parse_sse_block_returns_payload_with_event_id():
    """The SSE id line is folded into the decoded data as event_id."""
    block = ["event: state", "id: 7", 'data: {"type":"wake"}']
    payload = backend_api._parse_sse_block(block)
    assert payload == {"type": "wake", "event_id": 7}


def test_parse_sse_block_preserves_existing_event_id_field():
    """An event_id already inside the data wins over the SSE id line."""
    block = ["id: 9", 'data: {"type":"wake","event_id":12}']
    payload = backend_api._parse_sse_block(block)
    assert payload == {"type": "wake", "event_id": 12}


def test_parse_sse_block_returns_none_on_invalid_json():
    """A frame whose data will not decode is dropped rather than raising."""
    block = ["data: not-json-at-all"]
    payload = backend_api._parse_sse_block(block)
    assert payload is None


def test_parse_sse_block_ignores_comment_lines():
    """A leading colon line is skipped, so keepalives do not corrupt the frame."""
    block = [": keepalive comment", 'data: {"type":"wake"}']
    payload = backend_api._parse_sse_block(block)
    assert payload == {"type": "wake"}


class _StreamCM:
    """An async context manager that replays a fixed list of SSE lines."""
    def __init__(self, status_code: int, lines: list[str]):
        """Hold the status code and the lines the stream will yield."""
        self.status_code = status_code
        self._lines = lines

    async def __aenter__(self):
        """Return the stream itself as the context value."""
        return self

    async def __aexit__(self, *_):
        """Return False so an exception raised inside the block propagates."""
        return False

    async def aiter_lines(self):
        """Yield each stored line in order, as httpx would over the wire."""
        for line in self._lines:
            yield line


class _StubStreamClient:
    """A client whose stream() records the request headers and replays canned SSE lines."""
    def __init__(self, status_code: int, lines: list[str]):
        """Hold the status code and the SSE lines every stream will replay."""
        self._status = status_code
        self._lines = lines
        self.last_headers: dict | None = None

    def stream(self, _method, _url, headers=None, **_kw):
        """Record the headers and return a context manager over the canned lines."""
        self.last_headers = headers or {}
        return _StreamCM(self._status, self._lines)


def test_events_yields_parsed_frames():
    """Frames arrive as dicts carrying their event id, and Last-Event-ID and Accept headers go out."""
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
        """Return every event the stream yields, drained into a list."""
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
    """A 404 on the stream endpoint raises BackendProtocolMismatchError before any frame."""
    stub = _StubStreamClient(404, [])
    client = _client()
    client.client = stub  # type: ignore[assignment]

    async def _collect():
        """Return every event the stream yields, drained into a list."""
        events = []
        async for event in client.events():
            events.append(event)
        return events

    with pytest.raises(backend_api.BackendProtocolMismatchError):
        asyncio.run(_collect())
