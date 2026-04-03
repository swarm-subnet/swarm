from __future__ import annotations

import asyncio
import json
import time
import threading

from swarm.validator import backend_api
from swarm.constants import BACKEND_GRACE_PERIOD_SEC


def _run(coro):
    return asyncio.run(coro)


class _DummyHotkey:
    ss58_address = "validator_hotkey"
    def sign(self, message: bytes) -> bytes:
        return b"\x01\x02"


class _DummyWallet:
    hotkey = _DummyHotkey()


def _build_client(monkeypatch, tmp_path, wallet=None):
    monkeypatch.setattr(backend_api, "STATE_DIR", tmp_path)
    monkeypatch.setattr(backend_api, "RUNTIME_STATE_FILE", tmp_path / "runtime_state.json")
    return backend_api.BackendApiClient(wallet=wallet, base_url="http://backend.local")


def test_cached_weights_returned_within_grace_period(monkeypatch, tmp_path):
    state_file = tmp_path / "runtime_state.json"
    state_file.write_text(json.dumps({
        "last_weights": {"42": 1.0},
        "reeval_queue": [],
        "last_sync": time.time() - 60,
        "benchmark_epoch": 1,
    }))
    client = _build_client(monkeypatch, tmp_path, wallet=_DummyWallet())
    try:
        assert client.last_sync_ts > 0
        assert client.get_cached_weights() == {"42": 1.0}
        assert (time.time() - client.last_sync_ts) < BACKEND_GRACE_PERIOD_SEC
    finally:
        _run(client.close())


def test_cached_weights_empty_after_grace_period(monkeypatch, tmp_path):
    state_file = tmp_path / "runtime_state.json"
    state_file.write_text(json.dumps({
        "last_weights": {"42": 1.0},
        "reeval_queue": [],
        "last_sync": time.time() - BACKEND_GRACE_PERIOD_SEC - 100,
        "benchmark_epoch": 1,
    }))
    client = _build_client(monkeypatch, tmp_path, wallet=_DummyWallet())
    try:
        assert (time.time() - client.last_sync_ts) > BACKEND_GRACE_PERIOD_SEC
    finally:
        _run(client.close())


def test_heartbeat_remove_uid_from_queue():
    from swarm.validator.utils_parts.heartbeat import HeartbeatManager

    class FakeApi:
        async def post_heartbeat(self, **kw):
            pass

    loop = asyncio.new_event_loop()
    hb = HeartbeatManager(FakeApi(), loop)
    queue = [{"uid": 10, "phase": "screening"}, {"uid": 20, "phase": "benchmark"}]
    hb.set_queue(queue)

    hb.remove_uid_from_queue(10)
    assert len(hb._queue) == 1
    assert hb._queue[0]["uid"] == 20
    assert hb._queue is queue

    loop.close()


def test_heartbeat_finish_removes_uid_and_sends_idle():
    from swarm.validator.utils_parts.heartbeat import HeartbeatManager

    calls = []

    class FakeApi:
        async def post_heartbeat(self, **kw):
            calls.append(kw)

    loop = asyncio.new_event_loop()
    hb = HeartbeatManager(FakeApi(), loop)
    queue = [{"uid": 10, "phase": "screening"}, {"uid": 20, "phase": "benchmark"}]

    async def run_test():
        hb._queue = queue
        hb._progress = 5
        hb._total = 200
        hb._status = "evaluating_screening"
        hb._uid = 10
        hb._active = True
        hb._session_id = 1

        await hb._finish_async(5, 1, 10)

        assert all(q["uid"] != 10 for q in hb._queue)
        idle_call = calls[-1]
        assert idle_call["status"] == "idle"
        assert isinstance(idle_call["queue"], list)

    loop.run_until_complete(run_test())
    loop.close()
