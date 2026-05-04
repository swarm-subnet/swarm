from __future__ import annotations

import asyncio

import pytest

from swarm.validator import sse_listener as sse_module
from swarm.validator.backend_api import (
    BackendProtocolMismatchError,
    BackendTransportError,
)


class _StubBackendApi:
    def __init__(self, sequences: list):
        self._sequences = sequences
        self._call_count = 0
        self.last_event_ids: list = []

    def events(self, last_event_id=None):
        self.last_event_ids.append(last_event_id)
        sequence = self._sequences[self._call_count]
        self._call_count += 1

        async def _gen():
            if isinstance(sequence, Exception):
                raise sequence
            for event in sequence:
                yield event

        return _gen()


def _flags():
    return asyncio.Event(), asyncio.Event()


@pytest.mark.asyncio
async def test_handle_cancel_event_sets_both_flags():
    cancel, wake = _flags()
    listener = sse_module.SseListener(_StubBackendApi([]), cancel, wake)
    listener._handle({"type": "screening_failed", "event_id": 5})
    assert cancel.is_set()
    assert wake.is_set()
    assert listener._last_event_id == 5


@pytest.mark.asyncio
async def test_handle_wake_only_sets_wake():
    cancel, wake = _flags()
    listener = sse_module.SseListener(_StubBackendApi([]), cancel, wake)
    listener._handle({"type": "wake", "event_id": 3})
    assert not cancel.is_set()
    assert wake.is_set()


@pytest.mark.asyncio
async def test_handle_resync_keeps_last_event_id():
    """resync_required must preserve the replay anchor: a same-stream
    reconnect should resume from there. If the backend ring buffer can't
    satisfy it, the server emits another resync_required."""
    cancel, wake = _flags()
    listener = sse_module.SseListener(_StubBackendApi([]), cancel, wake)
    listener._last_event_id = 42
    listener._handle({"type": "resync_required", "event_id": 50})
    assert cancel.is_set()
    assert wake.is_set()
    assert listener._last_event_id == 50


@pytest.mark.asyncio
async def test_handle_epoch_transition_cancels_and_wakes():
    cancel, wake = _flags()
    listener = sse_module.SseListener(_StubBackendApi([]), cancel, wake)
    listener._handle({"type": "epoch_transition", "epoch_number": 8, "event_id": 7})
    assert cancel.is_set()
    assert wake.is_set()


@pytest.mark.asyncio
async def test_run_forever_consumes_events_then_reconnects(monkeypatch):
    cancel, wake = _flags()
    sleeps: list[float] = []

    async def _fake_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) >= 2:
            raise RuntimeError("stop test loop")

    monkeypatch.setattr(sse_module.asyncio, "sleep", _fake_sleep)

    backend = _StubBackendApi([
        [{"type": "wake", "event_id": 1}],
        BackendTransportError("network blip"),
        [{"type": "wake", "event_id": 2}],
    ])
    listener = sse_module.SseListener(backend, cancel, wake, initial_backoff=0.1)

    with pytest.raises(RuntimeError, match="stop test loop"):
        await listener.run_forever()

    assert wake.is_set()
    # First connect with no last_event_id, second after the wake event
    # at id=1, third reconnect with id=1.
    assert backend.last_event_ids[0] is None
    assert backend.last_event_ids[1] == 1
    # First sleep is the clean-close floor after a finished iterator;
    # second is the transport-error backoff.
    assert sleeps[0] == pytest.approx(0.5)
    assert sleeps[1] == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_run_forever_propagates_protocol_mismatch():
    cancel, wake = _flags()
    backend = _StubBackendApi([
        BackendProtocolMismatchError("backend too old"),
    ])
    listener = sse_module.SseListener(backend, cancel, wake)

    with pytest.raises(BackendProtocolMismatchError):
        await listener.run_forever()
