"""SSE listener that turns backend state-change events into local flags.

Runs as a supervised background task off ``forward.py``. The forward
loop reads ``cancel_flag`` between chunks to abort in-flight work and
``wake_flag`` to short-circuit its idle sleep so it polls /next-task
again as soon as a state change happens.

Reconnect is automatic with exponential backoff. ``Last-Event-ID`` is
re-sent on each reconnect so the backend can replay missed events
within its ring buffer (or emit ``resync_required`` if it cannot).
``BackendProtocolMismatchError`` propagates so the supervisor can fail
the validator loudly instead of silently downgrading.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

import bittensor as bt

from .backend_api import (
    BackendApiClient,
    BackendProtocolMismatchError,
    BackendTransportError,
)


_CANCEL_TYPES = {"screening_failed", "evaluation_failed", "benchmark_done", "stop"}
_WAKE_ONLY_TYPES = {"screening_passed", "wake"}
_RESYNC_TYPES = {"epoch_transition", "resync_required"}


class SseListener:
    def __init__(
        self,
        backend_api: BackendApiClient,
        cancel_flag: asyncio.Event,
        wake_flag: asyncio.Event,
        *,
        initial_backoff: float = 1.0,
        max_backoff: float = 30.0,
    ) -> None:
        self.backend_api = backend_api
        self.cancel_flag = cancel_flag
        self.wake_flag = wake_flag
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff
        self._last_event_id: Optional[int] = None

    async def run_forever(self) -> None:
        backoff = self._initial_backoff
        # Minimum wait between reconnects so a misbehaving proxy that
        # closes the stream immediately can't pin the loop at 100% CPU.
        clean_close_floor = 0.5
        # A stream that lived this long counts as healthy: routine proxy
        # idle-timeouts shouldn't escalate the reconnect backoff.
        healthy_threshold = 10.0
        loop = asyncio.get_event_loop()
        while True:
            stream_opened_at = loop.time()
            try:
                async for event in self.backend_api.events(
                    last_event_id=self._last_event_id,
                ):
                    self._handle(event)
                backoff = self._initial_backoff
                await asyncio.sleep(clean_close_floor)
            except BackendProtocolMismatchError:
                bt.logging.error(
                    "SSE listener: backend version mismatch; aborting validator"
                )
                raise
            except BackendTransportError as exc:
                elapsed = loop.time() - stream_opened_at
                bt.logging.warning(
                    f"SSE stream dropped after {elapsed:.1f}s: {exc}; "
                    f"reconnecting in {backoff:.1f}s"
                )
                await asyncio.sleep(backoff)
                if elapsed >= healthy_threshold:
                    backoff = self._initial_backoff
                else:
                    backoff = min(backoff * 2, self._max_backoff)
            except Exception as exc:
                elapsed = loop.time() - stream_opened_at
                bt.logging.warning(
                    f"SSE listener unexpected error after {elapsed:.1f}s: {exc}; "
                    f"reconnecting in {backoff:.1f}s"
                )
                await asyncio.sleep(backoff)
                if elapsed >= healthy_threshold:
                    backoff = self._initial_backoff
                else:
                    backoff = min(backoff * 2, self._max_backoff)

    def _handle(self, event: Dict[str, Any]) -> None:
        event_id = event.get("event_id")
        if isinstance(event_id, int):
            self._last_event_id = event_id

        event_type = event.get("type")
        if event_type in _CANCEL_TYPES:
            self.cancel_flag.set()
            self.wake_flag.set()
        elif event_type in _RESYNC_TYPES:
            # Drop in-flight state and refresh from /next-task. Keep
            # _last_event_id intact so a same-stream restart can replay
            # from the anchor; if the backend's ring buffer can't satisfy
            # it, the server will emit another resync_required.
            self.cancel_flag.set()
            self.wake_flag.set()
        elif event_type in _WAKE_ONLY_TYPES:
            self.wake_flag.set()
