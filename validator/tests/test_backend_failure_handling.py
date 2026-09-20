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

"""Tests that a backend failure is never read as a backend answer, call by call."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from swarm.validator import backend_api
from swarm.validator.backend_api import (
    BackendApiClient,
    BackendRejectedError,
    BackendTransportError,
)


class _Hotkey:
    """A hotkey that signs anything with two fixed bytes."""
    ss58_address = "validator_hotkey"

    def sign(self, message: bytes) -> bytes:
        """Return a fixed signature, ignoring the message."""
        return b"\x01\x02"


class _Wallet:
    """A wallet carrying only the stub hotkey the client signs with."""
    hotkey = _Hotkey()


def _reply(status_code: int, body) -> httpx.Response:
    """Build a real httpx response with a JSON body, or raw text when body is a string."""
    request = httpx.Request("POST", "http://backend.local/x")
    if isinstance(body, str):
        return httpx.Response(status_code=status_code, text=body, request=request)
    return httpx.Response(status_code=status_code, json=body, request=request)


class _ScriptedHttp:
    """An httpx client double that plays a script of replies and exceptions, repeating the last."""

    def __init__(self, script):
        """Queue the replies or exceptions to play and start with an empty call log."""
        self._script = list(script)
        self.calls: list[str] = []
        self.bodies: list[dict] = []

    def _next(self):
        """Pop the next scripted step, keeping the last one for every call after it."""
        step = self._script.pop(0) if len(self._script) > 1 else self._script[0]
        if isinstance(step, Exception):
            raise step
        return step

    async def post(self, url, **kwargs):
        """Log the POST and play the next scripted step."""
        self.calls.append(url)
        self.bodies.append(json.loads(kwargs["content"]))
        return self._next()

    async def get(self, url, **kwargs):
        """Log the GET and play the next scripted step."""
        self.calls.append(url)
        return self._next()

    async def aclose(self):
        """Return None; the double owns no socket."""
        return None


@pytest.fixture
def make_client(monkeypatch, tmp_path):
    """Build a client on a scripted transport, with state in tmp_path and every sleep skipped."""
    monkeypatch.setattr(backend_api, "STATE_DIR", tmp_path)
    monkeypatch.setattr(backend_api, "RUNTIME_STATE_FILE", tmp_path / "runtime_state.json")

    async def _no_sleep(_seconds):
        """Return at once so retry backoff costs the test no time."""
        return None

    monkeypatch.setattr(backend_api.asyncio, "sleep", _no_sleep)

    def _make(script):
        """Return the client and the scripted transport it talks through."""
        client = BackendApiClient(wallet=_Wallet(), base_url="http://backend.local")
        http = _ScriptedHttp(script)
        client.client = http
        return client, http

    return _make


_UNAVAILABLE = [
    _reply(500, {"detail": "boom"}),
    _reply(502, "<html>bad gateway</html>"),
    _reply(429, {"detail": "Rate limit exceeded"}),
    httpx.ReadTimeout("slow"),
    httpx.ConnectError("refused"),
    _reply(200, "<html>proxy login page</html>"),
    _reply(200, ["not", "an", "object"]),
]


@pytest.mark.parametrize("step", _UNAVAILABLE)
def test_post_outage_raises_transport_error(make_client, step):
    """A server error, a rate limit, a timeout or an unreadable body raises instead of returning a dict."""
    client, _http = make_client([step])
    with pytest.raises(BackendTransportError):
        asyncio.run(client._post_signed("/x", {"a": 1}))


@pytest.mark.parametrize("step", _UNAVAILABLE)
def test_get_outage_raises_transport_error(make_client, step):
    """The GET sender treats an outage exactly as the POST sender does."""
    client, _http = make_client([step])
    with pytest.raises(BackendTransportError):
        asyncio.run(client._get_signed("/x"))


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 410, 426])
def test_refusal_raises_rejected_error_with_status_and_detail(make_client, status):
    """A 4xx raises a rejection that carries the status code and the backend's own reason."""
    client, _http = make_client([_reply(status, {"detail": "task_not_running"})])
    with pytest.raises(BackendRejectedError) as caught:
        asyncio.run(client._post_signed("/x", {"a": 1}))
    assert caught.value.status_code == status
    assert caught.value.detail == "task_not_running"
    assert not isinstance(caught.value, BackendTransportError)


def test_rejected_error_never_leaks_the_backend_url(make_client):
    """A refusal whose body quotes a URL is scrubbed before it can reach a log."""
    client, _http = make_client([_reply(400, {"detail": "see http://backend.local/secret"})])
    with pytest.raises(BackendRejectedError) as caught:
        asyncio.run(client._post_signed("/x", {}))
    assert "backend.local" not in str(caught.value)


def test_claim_seeds_outage_is_not_an_empty_pool(make_client):
    """A failed claim raises, so it can never be read as a grant of nothing from a drained pool."""
    client, _http = make_client([_reply(500, {"detail": "boom"})])
    with pytest.raises(BackendTransportError):
        asyncio.run(client.claim_seeds(7, count=2))


def test_claim_seeds_refusal_raises_rejected(make_client):
    """A claim on a task the backend already closed raises a rejection naming the reason."""
    client, _http = make_client([_reply(409, {"detail": "task_not_running"})])
    with pytest.raises(BackendRejectedError) as caught:
        asyncio.run(client.claim_seeds(7, count=2))
    assert caught.value.detail == "task_not_running"


def test_claim_seeds_returns_the_real_grant(make_client):
    """A real reply still comes back as the grant the backend sent."""
    grant = {"granted": [4, 9], "pending": 3, "leased_other": 1, "done": 0}
    client, _http = make_client([_reply(200, grant)])
    assert asyncio.run(client.claim_seeds(7, count=2)) == grant


def test_submit_task_result_retries_through_an_outage(make_client):
    """A result that meets two failures is sent again until the backend records it."""
    recorded = {"recorded": True, "task_status": "SUBMITTED"}
    client, http = make_client(
        [_reply(503, {"detail": "down"}), httpx.ReadTimeout("slow"), _reply(200, recorded)]
    )
    result = asyncio.run(client.submit_task_result(
        5, score=0.5, per_type_scores={}, seeds_evaluated=10,
        early_failed=False, epoch_number=3,
    ))
    assert result == recorded
    assert len(http.calls) == 3


def test_submit_task_result_gives_up_after_its_attempt_budget(make_client, monkeypatch):
    """A backend that never comes back ends in a transport error after the fixed number of attempts."""
    monkeypatch.setattr(backend_api, "RESULT_SUBMIT_ATTEMPTS", 4)
    client, http = make_client([_reply(503, {"detail": "down"})])
    with pytest.raises(BackendTransportError):
        asyncio.run(client.submit_task_result(
            5, score=0.5, per_type_scores={}, seeds_evaluated=10,
            early_failed=False, epoch_number=3,
        ))
    assert len(http.calls) == 4


def test_submit_task_result_does_not_retry_a_refusal(make_client):
    """A finalized task refuses the result once, and the client does not ask again."""
    client, http = make_client([_reply(410, {"recorded": False, "reason": "task_finalized"})])
    with pytest.raises(BackendRejectedError) as caught:
        asyncio.run(client.submit_task_result(
            5, score=0.5, per_type_scores={}, seeds_evaluated=10,
            early_failed=False, epoch_number=3,
        ))
    assert caught.value.status_code == 410
    assert len(http.calls) == 1


def test_seed_scores_retry_an_outage_then_raise(make_client):
    """Score uploads retry an outage for their attempt budget, then raise instead of returning an error dict."""
    client, http = make_client([_reply(502, "bad gateway")])
    with pytest.raises(BackendTransportError):
        asyncio.run(client.post_seed_scores_batch(
            model_uid=1, epoch_number=1,
            scores=[{"seed_index": 0, "score": 0.5, "map_type": "city"}],
            retries=3,
        ))
    assert len(http.calls) == 3


def test_seed_scores_do_not_retry_a_refusal(make_client):
    """A refused score batch is sent once: asking again cannot change the backend's answer."""
    client, http = make_client([_reply(409, {"detail": "Submission provenance mismatch"})])
    with pytest.raises(BackendRejectedError):
        asyncio.run(client.post_seed_scores_batch(
            model_uid=1, epoch_number=1,
            scores=[{"seed_index": 0, "score": 0.5, "map_type": "city"}],
        ))
    assert len(http.calls) == 1


def test_seed_scores_recover_after_a_short_outage(make_client):
    """One failed upload followed by a confirmation returns the confirmation."""
    client, http = make_client(
        [httpx.ConnectError("refused"), _reply(200, {"recorded": 1, "message": "ok"})]
    )
    result = asyncio.run(client.post_seed_scores_batch(
        model_uid=1, epoch_number=1,
        scores=[{"seed_index": 0, "score": 0.5, "map_type": "city"}],
    ))
    assert result["recorded"] == 1
    assert len(http.calls) == 2


def test_announce_startup_keeps_knocking_through_an_outage(make_client, monkeypatch):
    """A failed startup heartbeat is not read as settled: startup retries until the backend accepts."""
    monkeypatch.setattr(backend_api, "DUPLICATE_SESSION_RETRY_SEC", 0)
    client, http = make_client(
        [_reply(503, {"detail": "down"}), httpx.ConnectError("refused"), _reply(200, {"recorded": True})]
    )
    asyncio.run(client.announce_startup())
    assert len(http.calls) == 3
    assert client._announcing is False


def test_announce_startup_stops_on_a_refusal_that_is_not_a_duplicate(make_client, monkeypatch):
    """A refusal other than the duplicate-session one ends the announce without looping."""
    monkeypatch.setattr(backend_api, "DUPLICATE_SESSION_RETRY_SEC", 0)
    client, http = make_client([_reply(401, {"detail": "Invalid signature"})])
    asyncio.run(client.announce_startup())
    assert len(http.calls) == 1


def test_stand_down_raises_when_the_hand_back_does_not_land(make_client):
    """A stand-down the backend never received raises, so shutdown cannot log it as delivered."""
    client, _http = make_client([httpx.ConnectError("refused")])
    with pytest.raises(BackendTransportError):
        asyncio.run(client.stand_down())


def test_sync_falls_back_to_the_cache_on_an_outage(make_client):
    """Sync still answers from the saved state, flagged as fallback, when the backend is down."""
    client, _http = make_client([_reply(500, {"detail": "boom"})])
    client._runtime_state["last_kings"] = [{"uid": 3}]
    data = asyncio.run(client.sync())
    assert data["fallback"] is True
    assert data["kings"] == [{"uid": 3}]


def test_sync_falls_back_on_a_refusal_too(make_client):
    """A refused sync is never parsed as a leaderboard: the cached one is served instead."""
    client, _http = make_client([_reply(403, {"detail": "not allowed"})])
    data = asyncio.run(client.sync())
    assert data["fallback"] is True
