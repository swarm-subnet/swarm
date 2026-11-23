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

"""Notice a validator whose worker has stopped turning, and say where every thread stands before it exits.

The worker thread runs the event loop for each forward pass and the chain sync between them. A healthy loop
answers a ping within seconds, even deep in a long evaluation. A worker stuck in a chain call, or a coroutine that
blocks the loop, stops answering: it then neither takes work nor sets weights, while the process looks alive. After
SWARM_VALIDATOR_STALL_SEC without an answer (default 1800 s, never under 300 s, 0 turns it off) the validator logs every
thread's stack and exits, so its supervisor (Docker, pm2) starts it fresh and the log names the call it was stuck in.
"""

from __future__ import annotations

import asyncio
import math
import os
import sys
import threading
import time
import traceback
from typing import Callable, Optional, Tuple

STALL_LIMIT_ENV = "SWARM_VALIDATOR_STALL_SEC"
DEFAULT_STALL_LIMIT_SEC = 1800.0     # longer than any legitimate pause of the loop; 0 turns the watchdog off
MIN_STALL_LIMIT_SEC = 300.0          # the sync between forwards pauses the loop, retries included, for far less
STALL_EXIT_CODE = 75                 # a supervisor restarts on it; the stacks above it say why
REPORT_DEADLINE_SEC = 10.0           # a stuck log sink must not keep a frozen validator from exiting


def stall_limit_sec() -> float:
    """The configured stall limit in seconds, never under the minimum; the default when unset, invalid or not finite.

    0 or less turns the watchdog off."""
    raw = os.environ.get(STALL_LIMIT_ENV, "").strip()
    try:
        limit = float(raw) if raw else DEFAULT_STALL_LIMIT_SEC
    except ValueError:
        return DEFAULT_STALL_LIMIT_SEC
    if not math.isfinite(limit):
        return DEFAULT_STALL_LIMIT_SEC
    return limit if limit <= 0 else max(limit, MIN_STALL_LIMIT_SEC)


class LoopLiveness:
    """Pings an event loop from another thread and times how long the oldest unanswered ping has waited."""

    def __init__(self, loop: asyncio.AbstractEventLoop, *, clock: Callable[[], float] = time.monotonic):
        """Start with no ping outstanding, so nothing counts as stalled before the first one."""
        self._loop = loop
        self._clock = clock
        self._lock = threading.Lock()
        self._pending_since: Optional[float] = None

    def ping(self) -> None:
        """Ask the loop to answer; a ping still waiting is not doubled and keeps its start time."""
        with self._lock:
            if self._pending_since is not None:
                return
            self._pending_since = self._clock()
        try:
            self._loop.call_soon_threadsafe(self._answer)
        except RuntimeError:
            # A closed loop never answers; the refused ping stays outstanding so the stall keeps counting.
            pass

    def _answer(self) -> None:
        """Run on the loop: clear the outstanding ping."""
        with self._lock:
            self._pending_since = None

    def stalled_for(self) -> float:
        """Seconds the outstanding ping has waited, or 0 when the loop has answered every ping."""
        with self._lock:
            if self._pending_since is None:
                return 0.0
            return self._clock() - self._pending_since


def thread_stacks() -> str:
    """Every live thread's name and current stack, the form a stall report needs."""
    names = {thread.ident: thread.name for thread in threading.enumerate()}
    parts = []
    for ident, frame in sys._current_frames().items():
        parts.append(f"--- thread {names.get(ident, '?')} ({ident}) ---\n{''.join(traceback.format_stack(frame))}")
    return "\n".join(parts)


def write_stderr(message: str) -> None:
    """Print the message to stderr, which a supervisor keeps even when the log handlers are gone."""
    print(message, file=sys.stderr, flush=True)


def report_stall(build: Callable[[], str], writers: Tuple[Callable[[str], None], ...], *,
                 deadline_sec: Optional[float] = None) -> None:
    """Build the report and hand it to every writer at once, all off this thread, waiting no longer than the deadline.

    Building and each writer run on daemon threads; one that raises or blocks is given up on, so the report can never
    keep the process from exiting. The deadline defaults to REPORT_DEADLINE_SEC."""
    deadline = time.monotonic() + (REPORT_DEADLINE_SEC if deadline_sec is None else deadline_sec)

    def write(writer: Callable[[str], None], message: str) -> None:
        """Run one writer, swallowing its failure so the others still run."""
        try:
            writer(message)
        except Exception:
            pass

    def build_and_write() -> None:
        """Build the message, then run every writer on its own thread until the shared deadline."""
        try:
            message = build()
        except Exception as exc:
            message = f"Stall report could not be built: {exc!r}"
        threads = [threading.Thread(target=write, args=(writer, message), name="stall_report", daemon=True)
                   for writer in writers]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))

    reporter = threading.Thread(target=build_and_write, name="stall_report", daemon=True)
    reporter.start()
    reporter.join(max(0.0, deadline - time.monotonic()))
