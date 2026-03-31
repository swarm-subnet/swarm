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

    def set_queue(self, queue: list) -> None:
        with self._lock:
            self._queue = queue

    def start(self, status: str, uid: int, total: int, queue: Optional[list] = None) -> None:
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

    def finish(self) -> None:
        with self._lock:
            final_progress = self._progress
            session_id = self._session_id
            self._active = False

        asyncio.run_coroutine_threadsafe(
            self._finish_async(final_progress, session_id),
            self.main_loop
        )

    async def _finish_async(self, final_progress: int, session_id: int) -> None:
        if final_progress > 0:
            await self._safe_heartbeat(final_progress, session_id, allow_inactive=True)
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

        try:
            await asyncio.wait_for(
                self.backend_api.post_heartbeat(
                    status=status,
                    current_uid=uid,
                    progress=progress,
                    total_seeds=total,
                    queue=self._queue,
                ),
                timeout=2.0
            )
        except Exception:
            pass

    async def _send_idle(self) -> None:
        try:
            await asyncio.wait_for(
                self.backend_api.post_heartbeat(status="idle"),
                timeout=2.0
            )
        except Exception:
            pass



# ──────────────────────────────────────────────────────────────────────────
# Model hash tracker (UID → hash persistence)
