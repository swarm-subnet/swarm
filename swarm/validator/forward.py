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

"""Validator forward loop.

Single-task model: pull one task from /next-task, run it, repeat. The
SSE listener runs as a supervised background task and uses two
``asyncio.Event`` flags to talk to the foreground loop:

  * ``cancel_flag`` — set on STOP / screening_failed / benchmark_done /
    epoch_transition / resync_required. The streaming evaluator checks
    it at chunk boundaries and aborts.
  * ``wake_flag`` — set on any state change. Cuts short the idle sleep
    so the validator polls /next-task again right away.
"""
from __future__ import annotations

import asyncio
import traceback

import bittensor as bt

from swarm.constants import FORWARD_SLEEP_SEC, N_DOCKER_WORKERS
from swarm.validator.calibration import baseline_model_available

from .backend_api import (
    BackendApiClient,
    BackendProtocolMismatchError,
    BackendTransportError,
)
from .docker.docker_evaluator import DockerSecureEvaluator
from .docker.docker_evaluator_parts.batch import (
    _ensure_host_speed_factor,
    host_speed_factor_is_fresh,
)
from .runtime_telemetry import tracker_call
from .seed_manager import BenchmarkSeedManager
from .sse_listener import SseListener
from .utils import (
    _apply_backend_weights_to_scores,
    _publish_pending_epoch_seeds,
    accept_sync_version,
    compute_koth_weights_from_sync,
    stamp_local_weights_on_kings,
)
from .utils_parts.run_task import run_task


async def forward(self) -> None:
    if not hasattr(self, "_forward_lock"):
        self._forward_lock = asyncio.Lock()
    async with self._forward_lock:
        await _forward_iteration(self)


async def _forward_iteration(self) -> None:
    try:
        self.forward_count = getattr(self, "forward_count", 0) + 1
        tracker_call(self, "mark_forward_started", forward_count=self.forward_count)
        bt.logging.info(f"[Forward #{self.forward_count}] start")

        await _ensure_components(self)

        if getattr(self, "_sse_fatal", None) is not None:
            tracker_call(
                self, "mark_forward_failed", error=str(self._sse_fatal),
            )
            raise self._sse_fatal

        tracker_call(self, "mark_backend_sync_started")
        sync_data = await self.backend_api.sync()
        tracker_call(
            self,
            "mark_backend_sync_completed",
            fallback=bool(sync_data.get("fallback", False)),
            leaderboard_version=sync_data.get("leaderboard_version"),
            error=str(sync_data.get("error", "")),
        )

        backend_epoch = int(
            sync_data.get("benchmark_epoch")
            or sync_data.get("current_epoch")
            or 0
        )
        if backend_epoch > 0 and backend_epoch != self.seed_manager.epoch_number:
            old_epoch = self.seed_manager.align_to_epoch(backend_epoch)
            if old_epoch is not None:
                self.docker_evaluator.cleanup()
                bt.logging.info(
                    f"Aligned validator seed epoch: {old_epoch} -> "
                    f"{self.seed_manager.epoch_number}"
                )

        local_koth_weights = compute_koth_weights_from_sync(
            sync_data, metagraph=self.metagraph
        )
        if local_koth_weights is not None and accept_sync_version(self, sync_data):
            _apply_backend_weights_to_scores(self, local_koth_weights)

        kings_payload = sync_data.get("kings") or []
        stamped = stamp_local_weights_on_kings(kings_payload)
        tracker_call(self, "update_koth_window", stamped)

        await _publish_pending_epoch_seeds(self)

        if not await _host_may_score(self):
            await _idle_until_wake(self, FORWARD_SLEEP_SEC)
            tracker_call(self, "mark_forward_completed", forward_count=self.forward_count)
            return

        # Reset flags before each long-poll so SSE events that arrive
        # during the next /next-task wait can wake us promptly.
        self._cancel_flag.clear()
        self._wake_flag.clear()

        try:
            task = await self.backend_api.next_task()
        except BackendProtocolMismatchError as exc:
            self._sse_fatal = exc
            tracker_call(self, "mark_forward_failed", error=str(exc))
            raise
        except BackendTransportError as exc:
            bt.logging.warning(f"next_task transport error: {exc}")
            await asyncio.sleep(FORWARD_SLEEP_SEC)
            return

        if task is None:
            await _idle_until_wake(self, FORWARD_SLEEP_SEC)
            tracker_call(self, "mark_forward_completed", forward_count=self.forward_count)
            return

        bt.logging.info(
            f"[Forward #{self.forward_count}] task uid={task.get('uid')} "
            f"phase={task.get('phase')}"
        )
        await run_task(
            self, task,
            cancel_flag=self._cancel_flag, wake_flag=self._wake_flag,
        )

        self.docker_evaluator.cleanup()
        tracker_call(self, "mark_forward_completed", forward_count=self.forward_count)

    except BackendProtocolMismatchError:
        raise
    except Exception as exc:
        tracker_call(self, "mark_forward_failed", error=str(exc))
        bt.logging.error(f"Validator forward error: {exc}")
        bt.logging.error(traceback.format_exc())
        await asyncio.sleep(FORWARD_SLEEP_SEC)


async def _ensure_components(self) -> None:
    if not hasattr(self, "seed_manager"):
        self.seed_manager = BenchmarkSeedManager()
        _invalidate_local_state_for_regenerated_seeds(self)
    if not hasattr(self, "backend_api"):
        try:
            self.backend_api = BackendApiClient(wallet=self.wallet)
        except ValueError as exc:
            tracker_call(self, "mark_forward_failed", error=str(exc))
            bt.logging.error(f"Backend API init failed: {exc}")
            bt.logging.error("Set SWARM_BACKEND_API_URL")
            raise
    if not hasattr(self, "docker_evaluator") or not DockerSecureEvaluator._base_ready:
        tracker_call(self, "mark_forward_failed", error="Docker evaluator not ready")
        raise RuntimeError("Docker evaluator not ready")
    if hasattr(self, "docker_evaluator"):
        setattr(
            self.docker_evaluator,
            "runtime_tracker",
            getattr(self, "runtime_tracker", None),
        )

    if not hasattr(self, "_cancel_flag"):
        self._cancel_flag = asyncio.Event()
    if not hasattr(self, "_wake_flag"):
        self._wake_flag = asyncio.Event()
    if not hasattr(self, "_sse_fatal"):
        self._sse_fatal = None
    if not hasattr(self, "_sse_task") or self._sse_task is None or self._sse_task.done():
        listener = SseListener(
            self.backend_api,
            self._cancel_flag,
            self._wake_flag,
        )
        self._sse_listener = listener
        self._sse_task = asyncio.create_task(listener.run_forever())
        self._sse_task.add_done_callback(
            lambda task: _record_sse_listener_exit(self, task)
        )


def _record_sse_listener_exit(self, task: asyncio.Task) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is None:
        return
    if isinstance(exc, BackendProtocolMismatchError):
        self._sse_fatal = exc
        cancel_flag = getattr(self, "_cancel_flag", None)
        wake_flag = getattr(self, "_wake_flag", None)
        if cancel_flag is not None:
            cancel_flag.set()
        if wake_flag is not None:
            wake_flag.set()
    bt.logging.error(f"SSE listener exited: {exc}")


def _image_provenance_ok(self) -> bool:
    evaluator = self.docker_evaluator
    try:
        label = str(evaluator._get_image_hash_label() or "")
        expected = str(evaluator._calculate_docker_hash())
    except Exception as exc:
        bt.logging.warning(f"Runner image provenance unavailable: {exc}")
        return False
    if label != expected:
        bt.logging.warning(
            f"Runner image label {label!r} does not match the expected build hash "
            f"{expected!r}; not polling for scoring tasks until the image is rebuilt"
        )
        return False
    return True


async def _host_may_score(self) -> bool:
    """Every scoring worker needs a fresh, eligible speed factor and a
    provenance-verified runner image before this host leases work.

    A worker without a valid calibration fails closed: an unmeasurable host
    must not score miners with unnormalized timing."""
    if not _image_provenance_ok(self):
        return False
    if baseline_model_available() and not host_speed_factor_is_fresh(N_DOCKER_WORKERS):
        await _announce_calibrating(self)
    speed = await _ensure_host_speed_factor(self.docker_evaluator, N_DOCKER_WORKERS)
    if speed is None:
        bt.logging.error(
            "HOST EXCLUDED FROM SCORING: no valid reference calibration; "
            "not polling for scoring tasks until calibration succeeds"
        )
        tracker_call(
            self, "mark_forward_failed", error="host_excluded_no_calibration",
        )
        return False
    if not speed.eligible:
        bt.logging.error(
            f"HOST EXCLUDED FROM SCORING: speed factor {speed.factor:.2f}x exceeds "
            "the eligibility limit; not polling for scoring tasks until recalibration passes"
        )
        tracker_call(
            self, "mark_forward_failed", error=f"host_excluded_speed_{speed.factor:.2f}x",
        )
        return False
    return True


async def _announce_calibrating(self) -> None:
    """Tell the backend this host is about to measure itself, so its seeds go
    back to the pool instead of waiting out a calibration it cannot interrupt."""
    bt.logging.info("Host calibration due; standing down from the pool while it runs")
    try:
        await self.backend_api.post_heartbeat(status="calibrating", in_flight_seeds=[])
    except Exception as exc:
        bt.logging.warning(f"Could not announce calibration: {exc}")


async def _idle_until_wake(self, timeout: float) -> None:
    try:
        await asyncio.wait_for(self._wake_flag.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        return
    finally:
        self._wake_flag.clear()


def _invalidate_local_state_for_regenerated_seeds(self) -> None:
    seed_manager = getattr(self, "seed_manager", None)
    if seed_manager is None:
        return
    if not getattr(seed_manager, "current_epoch_requires_state_invalidation", False):
        return
    seed_manager.current_epoch_requires_state_invalidation = False
    bt.logging.warning(
        "Current epoch seeds were regenerated locally; cleared local state"
    )
