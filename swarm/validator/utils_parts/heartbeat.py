from ._shared import *


class HeartbeatManager:
    """Thread-safe heartbeat progress manager for evaluation tracking.

    Sends throttled heartbeat updates to the backend during seed evaluation.
    Designed to be called from worker threads while safely dispatching async
    heartbeat calls to the main event loop.
    """

    def __init__(self, backend_api: BackendApiClient, main_loop: asyncio.AbstractEventLoop):
        self.backend_api = backend_api
        self.main_loop = main_loop
        self._progress = 0
        self._total = 0
        self._last_sent = 0
        self._lock = threading.Lock()
        self._status = "idle"
        self._uid: Optional[int] = None
        self._session_id = 0
        self._active = False
        self._queue: Optional[list] = None
        self._active_task: Optional[dict] = None
        self._backend_decision_version: Optional[int] = None

    def set_queue(self, queue: list) -> None:
        with self._lock:
            self._queue = queue

    def start(
        self,
        status: str,
        uid: int,
        total: int,
        queue: Optional[list] = None,
        active_task: Optional[dict] = None,
        backend_decision_version: Optional[int] = None,
    ) -> None:
        with self._lock:
            self._session_id += 1
            self._status = status
            self._uid = uid
            self._total = total
            self._progress = 0
            self._last_sent = 0
            self._active = True
            if queue is not None:
                self._queue = queue
            if active_task is not None:
                self._active_task = active_task
            elif queue is not None:
                matched = next((item for item in queue if int(item.get("uid", -1)) == uid), None)
                if matched is not None:
                    self._active_task = {
                        "uid": uid,
                        "phase": matched.get("phase"),
                        "assignment_id": matched.get("assignment_id"),
                        "epoch_number": matched.get("epoch_number"),
                        "benchmark_version": matched.get("benchmark_version"),
                    }
            self._backend_decision_version = backend_decision_version

        asyncio.run_coroutine_threadsafe(
            self._safe_heartbeat(0, self._session_id),
            self.main_loop
        )

    def on_seed_complete(self) -> None:
        """Called from worker thread after each seed completes (throttled)."""
        with self._lock:
            if not self._active:
                return
            self._progress += 1
            progress = self._progress
            session_id = self._session_id
            if progress - self._last_sent < 10:
                return
            self._last_sent = progress

        self.main_loop.call_soon_threadsafe(
            lambda p=progress, s=session_id: asyncio.create_task(self._safe_heartbeat(p, s))
        )

    def remove_uid_from_queue(self, uid: int) -> None:
        with self._lock:
            if self._queue is not None:
                self._queue[:] = [q for q in self._queue if q.get("uid") != uid]

    def finish(self) -> None:
        with self._lock:
            final_progress = self._progress
            session_id = self._session_id
            uid = self._uid
            self._active = False

        asyncio.run_coroutine_threadsafe(
            self._finish_async(final_progress, session_id, uid),
            self.main_loop
        )

    async def _finish_async(self, final_progress: int, session_id: int, uid: Optional[int]) -> None:
        if final_progress > 0:
            await self._safe_heartbeat(final_progress, session_id, allow_inactive=True)
        if uid is not None:
            self.remove_uid_from_queue(uid)
        await self._send_idle()

    async def _safe_heartbeat(
        self, progress: int, session_id: int, allow_inactive: bool = False
    ) -> None:
        with self._lock:
            if session_id != self._session_id:
                return
            if not allow_inactive and not self._active:
                return
            status = self._status
            uid = self._uid
            total = self._total
            queue = list(self._queue) if self._queue is not None else None
            active_task = dict(self._active_task) if self._active_task is not None else None
            decision_version = self._backend_decision_version

        try:
            await asyncio.wait_for(
                self.backend_api.post_heartbeat(
                    status=status,
                    current_uid=uid,
                    progress=progress,
                    total_seeds=total,
                    queue=queue,
                    active_task=active_task,
                    backend_decision_version=decision_version,
                ),
                timeout=2.0
            )
        except Exception:
            pass

    async def _send_idle(self) -> None:
        with self._lock:
            queue = list(self._queue) if self._queue is not None else []
            decision_version = self._backend_decision_version
        try:
            await asyncio.wait_for(
                self.backend_api.post_heartbeat(
                    status="idle",
                    queue=queue,
                    backend_decision_version=decision_version,
                ),
                timeout=2.0
            )
        except Exception:
            pass



# ──────────────────────────────────────────────────────────────────────────
# Model hash tracker (UID → hash persistence)
