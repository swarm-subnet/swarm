from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional


STATE_DIR = Path(__file__).resolve().parent.parent / "state"
RUNTIME_SNAPSHOT_FILE = STATE_DIR / "validator_runtime.json"
RUNTIME_EVENTS_FILE = STATE_DIR / "validator_events.jsonl"
_TRACKER_SCHEMA_VERSION = 1


def tracker_call(target: Any, method: str, *args: Any, **kwargs: Any) -> Any:
    tracker = getattr(target, "runtime_tracker", None)
    if tracker is None and callable(getattr(target, method, None)):
        tracker = target
    if tracker is None:
        return None
    func = getattr(tracker, method, None)
    if not callable(func):
        return None
    try:
        return func(*args, **kwargs)
    except Exception:
        return None


def _new_counts() -> dict[str, int]:
    return {
        "pending": 0,
        "processing": 0,
        "retry": 0,
        "completed": 0,
        "terminal_rejected": 0,
    }


def _new_stage_state() -> dict[str, Any]:
    return {
        "active": False,
        "uid": None,
        "started_at": None,
        "progress": 0,
        "total": 0,
        "last_completed_at": None,
        "last_uid": None,
        "last_duration_sec": None,
        "last_result": None,
        "last_note": "",
    }


class ValidatorRuntimeTracker:
    def __init__(
        self,
        *,
        state_dir: Path | None = None,
        snapshot_file: Path | None = None,
        events_file: Path | None = None,
        process_label: str = "validator",
    ) -> None:
        self.state_dir = Path(state_dir) if state_dir is not None else STATE_DIR
        self.snapshot_file = (
            Path(snapshot_file) if snapshot_file is not None else self.state_dir / RUNTIME_SNAPSHOT_FILE.name
        )
        self.events_file = (
            Path(events_file) if events_file is not None else self.state_dir / RUNTIME_EVENTS_FILE.name
        )
        self._lock = threading.Lock()
        self._queue_item_stages: dict[str, str] = {}
        self._queue_item_progress: dict[str, dict[str, Any]] = {}
        self._recent_events: deque[dict[str, Any]] = deque(maxlen=32)
        self._last_snapshot_write_monotonic = 0.0
        self._snapshot_dirty = False
        now = time.time()
        self.snapshot: dict[str, Any] = {
            "version": _TRACKER_SCHEMA_VERSION,
            "updated_at": now,
            "process": {
                "label": process_label,
                "pid": os.getpid(),
                "started_at": now,
                "worker_thread_alive": False,
            },
            "forward": {
                "count": 0,
                "in_progress": False,
                "last_started_at": None,
                "last_completed_at": None,
                "last_duration_sec": None,
                "last_error": "",
                "last_completed_forward_count": 0,
            },
            "backend": {
                "sync_in_progress": False,
                "last_sync_started_at": None,
                "last_sync_completed_at": None,
                "last_sync_success_at": None,
                "last_sync_duration_sec": None,
                "fallback": False,
                "consecutive_failures": 0,
                "pending_models_count": 0,
                "reeval_queue_count": 0,
                "leaderboard_version": None,
                "last_error": "",
            },
            "epoch": {
                "epoch_number": None,
                "seconds_until_end": None,
                "freeze_active": False,
                "last_transition_at": None,
                "last_transition_old": None,
                "last_transition_new": None,
            },
            "queue": {
                "counts": _new_counts(),
                "processable_count": 0,
                "oldest_age_sec": 0.0,
                "max_retry_attempts": 0,
                "active_items": [],
            },
            "evaluation": {
                "screening": _new_stage_state(),
                "benchmark": _new_stage_state(),
            },
            "reeval": {
                "queue_length": 0,
                "active": False,
                "active_uid": None,
                "active_reason": "",
                "active_model_hash": "",
                "active_started_at": None,
                "last_started_key": None,
                "repeat_count": 0,
                "last_completed_uid": None,
                "last_completed_reason": "",
                "last_completed_at": None,
                "last_error": "",
            },
            "docker": {
                "requested_workers": None,
                "effective_workers": None,
                "active_worker_cap": None,
                "dispatch_count": 0,
                "worker_restarts": 0,
                "worker_stalls": 0,
                "worker_crashes": 0,
                "last_batch_group": "",
                "last_seed": None,
                "last_backoff_note": "",
                "last_backoff_at": None,
                "cleanup_count": 0,
                "last_cleanup_duration_sec": None,
                "last_cleanup_reason": "",
            },
            "weights": {
                "last_attempt_at": None,
                "last_success_at": None,
                "last_error": "",
                "last_nonzero_uids": 0,
            },
            "chain_sync": {
                "in_progress": False,
                "context": "",
                "last_started_at": None,
                "last_completed_at": None,
                "last_success_at": None,
                "last_duration_sec": None,
                "last_error": "",
            },
            "counters": {
                "models_processed_total": 0,
                "screening_started_total": 0,
                "screening_completed_total": 0,
                "benchmark_started_total": 0,
                "benchmark_completed_total": 0,
                "screening_submit_started_total": 0,
                "screening_submit_success_total": 0,
                "screening_submit_failure_total": 0,
                "score_submit_started_total": 0,
                "score_submit_success_total": 0,
                "score_submit_failure_total": 0,
                "reeval_started_total": 0,
                "reeval_completed_total": 0,
            },
            "alerts": [],
        }
        self.record_event("tracker_started", force_snapshot=True, process_label=process_label)

    def flush(self) -> None:
        with self._lock:
            self._persist_snapshot_locked(force=True)

    def increment_counter(self, counter: str, amount: int = 1) -> None:
        with self._lock:
            self._record_counter(counter, amount=amount)
            self._persist_snapshot_locked(force=False)

    def _record_counter(self, counter: str, amount: int = 1) -> None:
        counters = self.snapshot["counters"]
        counters[counter] = int(counters.get(counter, 0)) + amount

    def _record_event_locked(
        self,
        event: str,
        *,
        severity: str = "info",
        force_snapshot: bool = False,
        **fields: Any,
    ) -> None:
        payload = {
            "ts": time.time(),
            "severity": severity,
            "event": event,
            "fields": fields,
        }
        self._recent_events.append(payload)
        self.snapshot["updated_at"] = payload["ts"]
        self.snapshot["alerts"] = compute_alerts(self.snapshot, now=payload["ts"])
        self._append_event_jsonl(payload)
        self._persist_snapshot_locked(force=force_snapshot)

    def record_event(
        self,
        event: str,
        *,
        severity: str = "info",
        force_snapshot: bool = False,
        **fields: Any,
    ) -> None:
        with self._lock:
            self._record_event_locked(
                event,
                severity=severity,
                force_snapshot=force_snapshot,
                **fields,
            )

    def mark_worker_thread_alive(self, alive: bool) -> None:
        with self._lock:
            self.snapshot["process"]["worker_thread_alive"] = bool(alive)
            self._record_event_locked(
                "worker_thread_alive" if alive else "worker_thread_stopped",
                alive=bool(alive),
            )

    def mark_forward_started(self, forward_count: int) -> None:
        now = time.time()
        with self._lock:
            forward = self.snapshot["forward"]
            forward["count"] = int(forward_count)
            forward["in_progress"] = True
            forward["last_started_at"] = now
            self._record_event_locked("forward_started", forward_count=int(forward_count))

    def mark_forward_completed(self, forward_count: int) -> None:
        now = time.time()
        with self._lock:
            forward = self.snapshot["forward"]
            started_at = forward.get("last_started_at")
            forward["in_progress"] = False
            forward["last_completed_at"] = now
            forward["last_completed_forward_count"] = int(forward_count)
            forward["last_duration_sec"] = (
                max(0.0, now - float(started_at)) if started_at is not None else None
            )
            forward["last_error"] = ""
            self._record_event_locked(
                "forward_completed",
                forward_count=int(forward_count),
                duration_sec=forward["last_duration_sec"],
                force_snapshot=True,
            )

    def mark_forward_failed(self, error: str) -> None:
        now = time.time()
        with self._lock:
            forward = self.snapshot["forward"]
            started_at = forward.get("last_started_at")
            forward["in_progress"] = False
            forward["last_completed_at"] = now
            forward["last_duration_sec"] = (
                max(0.0, now - float(started_at)) if started_at is not None else None
            )
            forward["last_error"] = str(error)
            self._record_event_locked(
                "forward_failed",
                severity="error",
                error=str(error),
                duration_sec=forward["last_duration_sec"],
                force_snapshot=True,
            )

    def mark_epoch_state(
        self,
        *,
        epoch_number: int,
        seconds_until_end: float,
        freeze_active: bool,
    ) -> None:
        with self._lock:
            epoch = self.snapshot["epoch"]
            epoch["epoch_number"] = int(epoch_number)
            epoch["seconds_until_end"] = float(seconds_until_end)
            epoch["freeze_active"] = bool(freeze_active)
            self._persist_snapshot_locked(force=False)

    def mark_epoch_transition(self, *, old_epoch: int, new_epoch: int) -> None:
        now = time.time()
        with self._lock:
            epoch = self.snapshot["epoch"]
            epoch["last_transition_at"] = now
            epoch["last_transition_old"] = int(old_epoch)
            epoch["last_transition_new"] = int(new_epoch)
            epoch["epoch_number"] = int(new_epoch)
            self._record_event_locked(
                "epoch_transition",
                old_epoch=int(old_epoch),
                new_epoch=int(new_epoch),
                force_snapshot=True,
            )

    def mark_backend_sync_started(self) -> None:
        now = time.time()
        with self._lock:
            backend = self.snapshot["backend"]
            backend["sync_in_progress"] = True
            backend["last_sync_started_at"] = now
            self._record_event_locked("backend_sync_started")

    def mark_backend_sync_completed(
        self,
        *,
        fallback: bool,
        pending_models_count: int,
        reeval_queue_count: int,
        leaderboard_version: int | None,
        error: str = "",
    ) -> None:
        now = time.time()
        with self._lock:
            backend = self.snapshot["backend"]
            started_at = backend.get("last_sync_started_at")
            backend["sync_in_progress"] = False
            backend["last_sync_completed_at"] = now
            backend["last_sync_duration_sec"] = (
                max(0.0, now - float(started_at)) if started_at is not None else None
            )
            backend["fallback"] = bool(fallback)
            backend["pending_models_count"] = int(pending_models_count)
            backend["reeval_queue_count"] = int(reeval_queue_count)
            backend["leaderboard_version"] = leaderboard_version
            backend["last_error"] = str(error)
            if fallback:
                backend["consecutive_failures"] = int(backend.get("consecutive_failures", 0)) + 1
            else:
                backend["consecutive_failures"] = 0
                backend["last_sync_success_at"] = now
            self.snapshot["reeval"]["queue_length"] = int(reeval_queue_count)
            self._record_event_locked(
                "backend_sync_completed",
                severity="warning" if fallback else "info",
                fallback=bool(fallback),
                pending_models_count=int(pending_models_count),
                reeval_queue_count=int(reeval_queue_count),
                leaderboard_version=leaderboard_version,
                duration_sec=backend["last_sync_duration_sec"],
                error=str(error),
                force_snapshot=True,
            )

    def mark_reeval_started(
        self,
        *,
        uid: int,
        reason: str,
        model_hash: str = "",
    ) -> None:
        now = time.time()
        with self._lock:
            reeval = self.snapshot["reeval"]
            started_key = f"{uid}:{model_hash}:{reason}"
            if reeval.get("last_started_key") == started_key:
                reeval["repeat_count"] = int(reeval.get("repeat_count", 0)) + 1
            else:
                reeval["repeat_count"] = 1
                reeval["last_started_key"] = started_key
            reeval["active"] = True
            reeval["active_uid"] = int(uid)
            reeval["active_reason"] = str(reason)
            reeval["active_model_hash"] = str(model_hash)
            reeval["active_started_at"] = now
            reeval["last_error"] = ""
            self._record_counter("reeval_started_total")
            self._record_event_locked(
                "reeval_started",
                uid=int(uid),
                reason=str(reason),
                model_hash=str(model_hash),
            )

    def mark_reeval_completed(
        self,
        *,
        uid: int,
        reason: str,
        success: bool,
        total_score: float | None = None,
        error: str = "",
    ) -> None:
        now = time.time()
        with self._lock:
            reeval = self.snapshot["reeval"]
            reeval["active"] = False
            reeval["last_completed_uid"] = int(uid)
            reeval["last_completed_reason"] = str(reason)
            reeval["last_completed_at"] = now
            reeval["last_error"] = str(error)
            if success:
                self._record_counter("reeval_completed_total")
            self._record_event_locked(
                "reeval_completed" if success else "reeval_failed",
                severity="warning" if not success else "info",
                uid=int(uid),
                reason=str(reason),
                total_score=total_score,
                error=str(error),
                force_snapshot=True,
            )

    def mark_reeval_missing_model(self, *, uid: int, reason: str) -> None:
        with self._lock:
            self._record_event_locked(
                "reeval_missing_model",
                severity="warning",
                uid=int(uid),
                reason=str(reason),
            )

    def mark_chain_sync_started(self, *, context: str) -> None:
        now = time.time()
        with self._lock:
            chain_sync = self.snapshot["chain_sync"]
            chain_sync["in_progress"] = True
            chain_sync["context"] = str(context)
            chain_sync["last_started_at"] = now
            self._record_event_locked("chain_sync_started", context=str(context))

    def mark_chain_sync_completed(
        self,
        *,
        context: str,
        success: bool,
        error: str = "",
    ) -> None:
        now = time.time()
        with self._lock:
            chain_sync = self.snapshot["chain_sync"]
            started_at = chain_sync.get("last_started_at")
            chain_sync["in_progress"] = False
            chain_sync["context"] = str(context)
            chain_sync["last_completed_at"] = now
            chain_sync["last_duration_sec"] = (
                max(0.0, now - float(started_at)) if started_at is not None else None
            )
            chain_sync["last_error"] = str(error)
            if success:
                chain_sync["last_success_at"] = now
            self._record_event_locked(
                "chain_sync_completed" if success else "chain_sync_failed",
                severity="error" if not success else "info",
                context=str(context),
                duration_sec=chain_sync["last_duration_sec"],
                error=str(error),
                force_snapshot=True,
            )

    def mark_weights_attempt(self) -> None:
        with self._lock:
            self.snapshot["weights"]["last_attempt_at"] = time.time()
            self._record_event_locked("weights_attempt")

    def mark_weights_result(
        self,
        *,
        success: bool,
        error: str = "",
        nonzero_uids: int = 0,
    ) -> None:
        now = time.time()
        with self._lock:
            weights = self.snapshot["weights"]
            weights["last_nonzero_uids"] = int(nonzero_uids)
            weights["last_error"] = str(error)
            if success:
                weights["last_success_at"] = now
            self._record_event_locked(
                "weights_success" if success else "weights_failed",
                severity="error" if not success else "info",
                nonzero_uids=int(nonzero_uids),
                error=str(error),
                force_snapshot=True,
            )

    def mark_screening_started(
        self,
        *,
        uid: int,
        total_seeds: int,
        threshold: float,
    ) -> None:
        now = time.time()
        with self._lock:
            screening = self.snapshot["evaluation"]["screening"]
            screening["active"] = True
            screening["uid"] = int(uid)
            screening["started_at"] = now
            screening["progress"] = 0
            screening["total"] = int(total_seeds)
            screening["last_note"] = f"threshold={float(threshold):.4f}"
            self._record_counter("screening_started_total")
            self._record_event_locked(
                "screening_started",
                uid=int(uid),
                total_seeds=int(total_seeds),
                threshold=float(threshold),
            )

    def mark_screening_progress(
        self,
        *,
        uid: int,
        progress: int,
        total_seeds: int,
        running_median: float,
        note: str = "",
    ) -> None:
        with self._lock:
            screening = self.snapshot["evaluation"]["screening"]
            screening["uid"] = int(uid)
            screening["progress"] = int(progress)
            screening["total"] = int(total_seeds)
            screening["last_result"] = float(running_median)
            if note:
                screening["last_note"] = str(note)
            self._record_event_locked(
                "screening_progress",
                uid=int(uid),
                progress=int(progress),
                total_seeds=int(total_seeds),
                running_median=float(running_median),
                note=str(note),
            )

    def mark_screening_completed(
        self,
        *,
        uid: int,
        evaluated: int,
        total_seeds: int,
        median_score: float,
        note: str = "",
    ) -> None:
        now = time.time()
        with self._lock:
            screening = self.snapshot["evaluation"]["screening"]
            started_at = screening.get("started_at")
            screening["active"] = False
            screening["uid"] = int(uid)
            screening["progress"] = int(evaluated)
            screening["total"] = int(total_seeds)
            screening["last_completed_at"] = now
            screening["last_uid"] = int(uid)
            screening["last_duration_sec"] = (
                max(0.0, now - float(started_at)) if started_at is not None else None
            )
            screening["last_result"] = float(median_score)
            screening["last_note"] = str(note)
            self._record_counter("screening_completed_total")
            self._record_event_locked(
                "screening_completed",
                uid=int(uid),
                evaluated=int(evaluated),
                total_seeds=int(total_seeds),
                median_score=float(median_score),
                note=str(note),
                duration_sec=screening["last_duration_sec"],
                force_snapshot=True,
            )

    def mark_benchmark_started(self, *, uid: int, total_seeds: int, note: str = "") -> None:
        now = time.time()
        with self._lock:
            benchmark = self.snapshot["evaluation"]["benchmark"]
            benchmark["active"] = True
            benchmark["uid"] = int(uid)
            benchmark["started_at"] = now
            benchmark["progress"] = 0
            benchmark["total"] = int(total_seeds)
            benchmark["last_note"] = str(note)
            self._record_counter("benchmark_started_total")
            self._record_event_locked(
                "benchmark_started",
                uid=int(uid),
                total_seeds=int(total_seeds),
                note=str(note),
            )

    def mark_benchmark_progress(
        self,
        *,
        uid: int,
        progress: int,
        total_seeds: int,
        note: str = "",
    ) -> None:
        with self._lock:
            benchmark = self.snapshot["evaluation"]["benchmark"]
            benchmark["uid"] = int(uid)
            benchmark["progress"] = int(progress)
            benchmark["total"] = int(total_seeds)
            if note:
                benchmark["last_note"] = str(note)
            self._record_event_locked(
                "benchmark_progress",
                uid=int(uid),
                progress=int(progress),
                total_seeds=int(total_seeds),
                note=str(note),
            )

    def mark_benchmark_completed(
        self,
        *,
        uid: int,
        evaluated: int,
        total_seeds: int,
        median_score: float,
        note: str = "",
    ) -> None:
        now = time.time()
        with self._lock:
            benchmark = self.snapshot["evaluation"]["benchmark"]
            started_at = benchmark.get("started_at")
            benchmark["active"] = False
            benchmark["uid"] = int(uid)
            benchmark["progress"] = int(evaluated)
            benchmark["total"] = int(total_seeds)
            benchmark["last_completed_at"] = now
            benchmark["last_uid"] = int(uid)
            benchmark["last_duration_sec"] = (
                max(0.0, now - float(started_at)) if started_at is not None else None
            )
            benchmark["last_result"] = float(median_score)
            benchmark["last_note"] = str(note)
            self._record_counter("benchmark_completed_total")
            self._record_event_locked(
                "benchmark_completed",
                uid=int(uid),
                evaluated=int(evaluated),
                total_seeds=int(total_seeds),
                median_score=float(median_score),
                note=str(note),
                duration_sec=benchmark["last_duration_sec"],
                force_snapshot=True,
            )

    def mark_submission_started(
        self,
        *,
        stage: str,
        uid: int,
        model_hash: str = "",
    ) -> None:
        with self._lock:
            counter = "score_submit_started_total" if stage == "score" else "screening_submit_started_total"
            self._record_counter(counter)
            self._record_event_locked(
                f"{stage}_submit_started",
                uid=int(uid),
                model_hash=str(model_hash),
            )

    def mark_submission_result(
        self,
        *,
        stage: str,
        uid: int,
        success: bool,
        terminal: bool,
        reason: str = "",
        model_hash: str = "",
    ) -> None:
        with self._lock:
            if stage == "score":
                counter = "score_submit_success_total" if success else "score_submit_failure_total"
            else:
                counter = (
                    "screening_submit_success_total" if success else "screening_submit_failure_total"
                )
            self._record_counter(counter)
            severity = "info" if success else ("error" if terminal else "warning")
            self._record_event_locked(
                f"{stage}_submit_success" if success else f"{stage}_submit_failed",
                severity=severity,
                uid=int(uid),
                terminal=bool(terminal),
                reason=str(reason),
                model_hash=str(model_hash),
                force_snapshot=not success,
            )

    def mark_queue_item_stage(
        self,
        *,
        queue: dict | None,
        key: str,
        item: dict[str, Any],
        stage: str,
        progress_done: int | None = None,
        progress_total: int | None = None,
        severity: str = "info",
        note: str = "",
    ) -> None:
        with self._lock:
            self._queue_item_stages[str(key)] = str(stage)
            if progress_done is not None or progress_total is not None or note:
                payload = self._queue_item_progress.setdefault(str(key), {})
                if progress_done is not None:
                    payload["done"] = int(progress_done)
                if progress_total is not None:
                    payload["total"] = int(progress_total)
                if note:
                    payload["note"] = str(note)
            if queue is not None:
                self._update_queue_state_locked(queue)
            self._record_event_locked(
                "queue_item_stage",
                severity=severity,
                key=str(key),
                uid=int(item.get("uid", -1)),
                status=str(item.get("status", "")),
                stage=str(stage),
                model_hash=str(item.get("model_hash", ""))[:16],
                retry_attempts=int(item.get("retry_attempts", 0) or 0),
                note=str(note),
            )

    def update_queue_state(self, queue: dict) -> None:
        with self._lock:
            self._update_queue_state_locked(queue)
            self._persist_snapshot_locked(force=False)

    def _update_queue_state_locked(self, queue: dict) -> None:
        items = dict(queue.get("items", {}))
        now = time.time()
        counts = _new_counts()
        processable_count = 0
        oldest_age_sec = 0.0
        max_retry_attempts = 0
        summaries: list[dict[str, Any]] = []
        live_keys = set(items.keys())
        self._queue_item_stages = {
            key: stage for key, stage in self._queue_item_stages.items() if key in live_keys
        }
        self._queue_item_progress = {
            key: progress for key, progress in self._queue_item_progress.items() if key in live_keys
        }
        for key, item in items.items():
            status = str(item.get("status", "pending"))
            counts.setdefault(status, 0)
            counts[status] = int(counts.get(status, 0)) + 1
            retry_attempts = int(item.get("retry_attempts", 0) or 0)
            max_retry_attempts = max(max_retry_attempts, retry_attempts)
            created_at = float(item.get("created_at", now) or now)
            updated_at = float(item.get("updated_at", created_at) or created_at)
            age_sec = max(0.0, now - created_at)
            updated_age_sec = max(0.0, now - updated_at)
            if status not in ("completed", "terminal_rejected"):
                oldest_age_sec = max(oldest_age_sec, age_sec)
                next_retry_at = float(item.get("next_retry_at", 0) or 0)
                if next_retry_at <= now:
                    processable_count += 1
            progress = self._queue_item_progress.get(key, {})
            summaries.append(
                {
                    "key": str(key),
                    "uid": int(item.get("uid", -1)),
                    "status": status,
                    "stage": self._queue_item_stages.get(key, status),
                    "model_hash": str(item.get("model_hash", ""))[:12],
                    "retry_attempts": retry_attempts,
                    "age_sec": age_sec,
                    "updated_age_sec": updated_age_sec,
                    "last_error": str(item.get("last_error", "")),
                    "screening_score": item.get("screening_score"),
                    "seeds_evaluated": int(item.get("seeds_evaluated", 0) or 0),
                    "progress_done": progress.get("done"),
                    "progress_total": progress.get("total"),
                    "note": progress.get("note", ""),
                }
            )
        summaries.sort(key=lambda entry: (-float(entry["age_sec"]), entry["uid"], entry["key"]))
        queue_snapshot = self.snapshot["queue"]
        queue_snapshot["counts"] = counts
        queue_snapshot["processable_count"] = int(processable_count)
        queue_snapshot["oldest_age_sec"] = float(oldest_age_sec)
        queue_snapshot["max_retry_attempts"] = int(max_retry_attempts)
        queue_snapshot["active_items"] = summaries[:8]

    def mark_docker_run_started(
        self,
        *,
        requested_workers: int,
        effective_workers: int,
        total_tasks: int,
    ) -> None:
        with self._lock:
            docker = self.snapshot["docker"]
            docker["requested_workers"] = int(requested_workers)
            docker["effective_workers"] = int(effective_workers)
            if docker.get("active_worker_cap") is None:
                docker["active_worker_cap"] = int(effective_workers)
            self._record_event_locked(
                "docker_run_started",
                requested_workers=int(requested_workers),
                effective_workers=int(effective_workers),
                total_tasks=int(total_tasks),
            )

    def mark_docker_dispatch(
        self,
        *,
        worker_slot: int,
        batch_index: int,
        group: str,
        seed: int,
        active_worker_cap: int,
    ) -> None:
        with self._lock:
            docker = self.snapshot["docker"]
            docker["dispatch_count"] = int(docker.get("dispatch_count", 0)) + 1
            docker["last_batch_group"] = str(group)
            docker["last_seed"] = int(seed)
            docker["active_worker_cap"] = int(active_worker_cap)
            self._record_event_locked(
                "docker_batch_dispatched",
                worker_slot=int(worker_slot),
                batch_index=int(batch_index),
                group=str(group),
                seed=int(seed),
                active_worker_cap=int(active_worker_cap),
            )

    def mark_docker_backoff(self, *, active_worker_cap: int, note: str) -> None:
        now = time.time()
        with self._lock:
            docker = self.snapshot["docker"]
            docker["active_worker_cap"] = int(active_worker_cap)
            docker["last_backoff_note"] = str(note)
            docker["last_backoff_at"] = now
            self._record_event_locked(
                "docker_backoff",
                severity="warning",
                active_worker_cap=int(active_worker_cap),
                note=str(note),
            )

    def mark_docker_worker_failure(
        self,
        *,
        status: str,
        worker_slot: int,
        error: str,
    ) -> None:
        with self._lock:
            docker = self.snapshot["docker"]
            if status == "worker_stall_timeout":
                docker["worker_stalls"] = int(docker.get("worker_stalls", 0)) + 1
            else:
                docker["worker_crashes"] = int(docker.get("worker_crashes", 0)) + 1
            self._record_event_locked(
                "docker_worker_failure",
                severity="warning",
                status=str(status),
                worker_slot=int(worker_slot),
                error=str(error),
            )

    def mark_docker_worker_restart(self, *, worker_slot: int) -> None:
        with self._lock:
            docker = self.snapshot["docker"]
            docker["worker_restarts"] = int(docker.get("worker_restarts", 0)) + 1
            self._record_event_locked(
                "docker_worker_restart",
                worker_slot=int(worker_slot),
            )

    def mark_docker_cleanup(self, *, duration_sec: float, reason: str) -> None:
        with self._lock:
            docker = self.snapshot["docker"]
            docker["cleanup_count"] = int(docker.get("cleanup_count", 0)) + 1
            docker["last_cleanup_duration_sec"] = float(duration_sec)
            docker["last_cleanup_reason"] = str(reason)
            self._record_event_locked(
                "docker_cleanup",
                duration_sec=float(duration_sec),
                reason=str(reason),
                force_snapshot=True,
            )

    def snapshot_copy(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self.snapshot))

    def _append_event_jsonl(self, payload: dict[str, Any]) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with self.events_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def _persist_snapshot_locked(self, *, force: bool) -> None:
        now_monotonic = time.monotonic()
        if not force and (now_monotonic - self._last_snapshot_write_monotonic) < 0.25:
            self._snapshot_dirty = True
            return
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot["updated_at"] = time.time()
        self.snapshot["alerts"] = compute_alerts(self.snapshot, now=self.snapshot["updated_at"])
        temp_path = self.snapshot_file.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(self.snapshot, handle, indent=2, sort_keys=True)
        temp_path.replace(self.snapshot_file)
        self._last_snapshot_write_monotonic = now_monotonic
        self._snapshot_dirty = False


def load_runtime_snapshot(path: Path | None = None) -> dict[str, Any]:
    target = Path(path) if path is not None else RUNTIME_SNAPSHOT_FILE
    return json.loads(target.read_text(encoding="utf-8"))


def load_recent_events(path: Path | None = None, limit: int = 8) -> list[dict[str, Any]]:
    target = Path(path) if path is not None else RUNTIME_EVENTS_FILE
    if not target.exists():
        return []
    recent: deque[dict[str, Any]] = deque(maxlen=max(0, int(limit)))
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                recent.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return list(recent)


def compute_alerts(snapshot: dict[str, Any], *, now: float | None = None) -> list[dict[str, str]]:
    current_time = time.time() if now is None else float(now)
    alerts: list[dict[str, str]] = []

    def _add(code: str, severity: str, message: str) -> None:
        alerts.append({"code": code, "severity": severity, "message": message})

    process = snapshot.get("process", {})
    forward = snapshot.get("forward", {})
    backend = snapshot.get("backend", {})
    epoch = snapshot.get("epoch", {})
    queue = snapshot.get("queue", {})
    docker = snapshot.get("docker", {})
    chain_sync = snapshot.get("chain_sync", {})
    reeval = snapshot.get("reeval", {})

    if process.get("worker_thread_alive") is False and forward.get("count", 0):
        _add("worker_thread_dead", "critical", "validator worker thread is not alive")

    if forward.get("in_progress") and forward.get("last_started_at") is not None:
        age = current_time - float(forward["last_started_at"])
        if age > 300:
            _add("forward_stalled", "critical", f"forward has been running for {age:.0f}s")
        elif age > 120:
            _add("forward_slow", "warning", f"forward has been running for {age:.0f}s")

    if chain_sync.get("in_progress") and chain_sync.get("last_started_at") is not None:
        age = current_time - float(chain_sync["last_started_at"])
        if age > 180:
            _add("chain_sync_stalled", "critical", f"chain sync has been running for {age:.0f}s")
        elif age > 60:
            _add("chain_sync_slow", "warning", f"chain sync has been running for {age:.0f}s")

    if backend.get("fallback"):
        severity = "critical" if (current_time - float(backend.get("last_sync_success_at") or 0)) > 300 else "warning"
        _add(
            "backend_fallback",
            severity,
            "backend sync is in fallback mode; new pending models are not being discovered",
        )

    oldest_age = float(queue.get("oldest_age_sec", 0.0) or 0.0)
    if oldest_age > 1800:
        _add("queue_oldest_critical", "critical", f"oldest queue item age is {oldest_age:.0f}s")
    elif oldest_age > 900:
        _add("queue_oldest_warning", "warning", f"oldest queue item age is {oldest_age:.0f}s")

    max_retry_attempts = int(queue.get("max_retry_attempts", 0) or 0)
    if max_retry_attempts >= 10:
        _add("queue_retry_critical", "critical", f"queue item retry count reached {max_retry_attempts}")
    elif max_retry_attempts >= 5:
        _add("queue_retry_warning", "warning", f"queue item retry count reached {max_retry_attempts}")

    if epoch.get("freeze_active") and int(queue.get("processable_count", 0) or 0) > 0:
        _add("freeze_backlog", "warning", "epoch freeze is blocking queue items that are otherwise ready")

    requested_workers = docker.get("requested_workers")
    active_worker_cap = docker.get("active_worker_cap")
    if requested_workers and active_worker_cap and int(active_worker_cap) < int(requested_workers):
        _add(
            "docker_backoff_active",
            "warning",
            f"docker adaptive backoff is limiting workers to {active_worker_cap}/{requested_workers}",
        )

    if reeval.get("repeat_count", 0) and int(reeval["repeat_count"]) > 1:
        _add(
            "reeval_repeat",
            "warning",
            f"same re-evaluation has started {int(reeval['repeat_count'])} times in a row",
        )

    severity_order = {"critical": 0, "warning": 1, "info": 2}
    alerts.sort(key=lambda alert: (severity_order.get(alert["severity"], 9), alert["code"]))
    return alerts


__all__ = [
    "STATE_DIR",
    "RUNTIME_EVENTS_FILE",
    "RUNTIME_SNAPSHOT_FILE",
    "ValidatorRuntimeTracker",
    "compute_alerts",
    "load_recent_events",
    "load_runtime_snapshot",
    "tracker_call",
]
