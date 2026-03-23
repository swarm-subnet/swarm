from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Iterable

from .runtime_telemetry import (
    RUNTIME_EVENTS_FILE,
    RUNTIME_SNAPSHOT_FILE,
    load_recent_events,
    load_runtime_snapshot,
)


def _fmt_age(ts: float | None, now: float) -> str:
    if ts in (None, 0):
        return "never"
    delta = max(0.0, now - float(ts))
    if delta < 1:
        return "just now"
    if delta < 60:
        return f"{delta:.0f}s ago"
    if delta < 3600:
        return f"{delta / 60:.1f}m ago"
    return f"{delta / 3600:.1f}h ago"


def _fmt_duration(value: Any) -> str:
    if value in (None, "", 0):
        return "0s" if value == 0 else "-"
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return str(value)
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds / 60:.1f}m"


def _health_label(alerts: list[dict[str, str]]) -> str:
    if any(alert.get("severity") == "critical" for alert in alerts):
        return "CRITICAL"
    if any(alert.get("severity") == "warning" for alert in alerts):
        return "WARNING"
    return "HEALTHY"


def _render_section(title: str, rows: Iterable[tuple[str, str]]) -> list[str]:
    lines = [f"{title}"]
    for key, value in rows:
        lines.append(f"  {key:<22} {value}")
    return lines


def render_runtime_dashboard(
    snapshot: dict[str, Any],
    *,
    events: list[dict[str, Any]] | None = None,
    now: float | None = None,
) -> str:
    current_time = time.time() if now is None else float(now)
    alerts = list(snapshot.get("alerts", []))
    process = snapshot.get("process", {})
    forward = snapshot.get("forward", {})
    backend = snapshot.get("backend", {})
    epoch = snapshot.get("epoch", {})
    queue = snapshot.get("queue", {})
    evaluation = snapshot.get("evaluation", {})
    screening = evaluation.get("screening", {})
    benchmark = evaluation.get("benchmark", {})
    reeval = snapshot.get("reeval", {})
    docker = snapshot.get("docker", {})
    weights = snapshot.get("weights", {})
    chain_sync = snapshot.get("chain_sync", {})
    counters = snapshot.get("counters", {})

    output: list[str] = []
    output.append("Swarm Validator Monitor")
    output.append(
        "  "
        f"health={_health_label(alerts)}  "
        f"pid={process.get('pid', '?')}  "
        f"updated={_fmt_age(snapshot.get('updated_at'), current_time)}  "
        f"worker_alive={process.get('worker_thread_alive', False)}"
    )
    output.append("")

    if alerts:
        output.append("Alerts")
        for alert in alerts[:8]:
            output.append(
                f"  [{alert.get('severity', 'info').upper():<8}] {alert.get('message', '')}"
            )
        output.append("")

    output.extend(
        _render_section(
            "Forward",
            [
                ("count", str(forward.get("count", 0))),
                ("in progress", str(bool(forward.get("in_progress", False)))),
                ("last started", _fmt_age(forward.get("last_started_at"), current_time)),
                ("last completed", _fmt_age(forward.get("last_completed_at"), current_time)),
                ("last duration", _fmt_duration(forward.get("last_duration_sec"))),
                ("last error", str(forward.get("last_error", "") or "-")),
            ],
        )
    )
    output.append("")
    output.extend(
        _render_section(
            "Backend",
            [
                ("sync in progress", str(bool(backend.get("sync_in_progress", False)))),
                ("fallback", str(bool(backend.get("fallback", False)))),
                ("last sync", _fmt_age(backend.get("last_sync_completed_at"), current_time)),
                ("sync duration", _fmt_duration(backend.get("last_sync_duration_sec"))),
                ("pending models", str(backend.get("pending_models_count", 0))),
                ("re-eval queue", str(backend.get("reeval_queue_count", 0))),
                ("consecutive failures", str(backend.get("consecutive_failures", 0))),
            ],
        )
    )
    output.append("")
    output.extend(
        _render_section(
            "Epoch",
            [
                ("epoch", str(epoch.get("epoch_number", "-"))),
                ("freeze active", str(bool(epoch.get("freeze_active", False)))),
                (
                    "seconds until end",
                    _fmt_duration(epoch.get("seconds_until_end")),
                ),
                (
                    "last transition",
                    _fmt_age(epoch.get("last_transition_at"), current_time),
                ),
            ],
        )
    )
    output.append("")
    output.extend(
        _render_section(
            "Queue",
            [
                ("pending", str(queue.get("counts", {}).get("pending", 0))),
                ("processing", str(queue.get("counts", {}).get("processing", 0))),
                ("retry", str(queue.get("counts", {}).get("retry", 0))),
                ("processable", str(queue.get("processable_count", 0))),
                ("oldest item", _fmt_duration(queue.get("oldest_age_sec"))),
                ("max retries", str(queue.get("max_retry_attempts", 0))),
                ("models processed", str(counters.get("models_processed_total", 0))),
            ],
        )
    )
    output.append("")
    output.extend(
        _render_section(
            "Evaluation",
            [
                (
                    "screening",
                    (
                        f"uid={screening.get('uid', '-')} "
                        f"active={bool(screening.get('active', False))} "
                        f"progress={screening.get('progress', 0)}/{screening.get('total', 0)} "
                        f"last={screening.get('last_result', '-')}"
                    ),
                ),
                (
                    "benchmark",
                    (
                        f"uid={benchmark.get('uid', '-')} "
                        f"active={bool(benchmark.get('active', False))} "
                        f"progress={benchmark.get('progress', 0)}/{benchmark.get('total', 0)} "
                        f"last={benchmark.get('last_result', '-')}"
                    ),
                ),
                (
                    "re-eval",
                    (
                        f"active={bool(reeval.get('active', False))} "
                        f"uid={reeval.get('active_uid', '-')} "
                        f"reason={reeval.get('active_reason', '-') or '-'} "
                        f"repeat={reeval.get('repeat_count', 0)}"
                    ),
                ),
            ],
        )
    )
    output.append("")
    output.extend(
        _render_section(
            "Docker",
            [
                (
                    "workers",
                    f"{docker.get('active_worker_cap', '-')} / "
                    f"{docker.get('effective_workers', '-')} effective "
                    f"(requested {docker.get('requested_workers', '-')})",
                ),
                ("dispatch count", str(docker.get("dispatch_count", 0))),
                ("worker restarts", str(docker.get("worker_restarts", 0))),
                ("worker stalls", str(docker.get("worker_stalls", 0))),
                ("worker crashes", str(docker.get("worker_crashes", 0))),
                (
                    "last cleanup",
                    f"{_fmt_duration(docker.get('last_cleanup_duration_sec'))} "
                    f"({docker.get('last_cleanup_reason', '-') or '-'})",
                ),
                ("backoff note", str(docker.get("last_backoff_note", "") or "-")),
            ],
        )
    )
    output.append("")
    output.extend(
        _render_section(
            "Chain / Weights",
            [
                (
                    "chain sync",
                    (
                        f"active={bool(chain_sync.get('in_progress', False))} "
                        f"context={chain_sync.get('context', '-') or '-'} "
                        f"last={_fmt_age(chain_sync.get('last_completed_at'), current_time)}"
                    ),
                ),
                ("weight attempt", _fmt_age(weights.get("last_attempt_at"), current_time)),
                ("weight success", _fmt_age(weights.get("last_success_at"), current_time)),
                ("weight nonzero uids", str(weights.get("last_nonzero_uids", 0))),
                ("weight error", str(weights.get("last_error", "") or "-")),
            ],
        )
    )
    output.append("")
    output.append("Queue Items")
    output.append(
        "  "
        f"{'uid':>4} {'status':<18} {'stage':<18} {'age':>8} {'retry':>5} {'progress':>11}  {'error'}"
    )
    for item in queue.get("active_items", []):
        progress = "-"
        if item.get("progress_total"):
            progress = f"{item.get('progress_done', 0)}/{item.get('progress_total', 0)}"
        output.append(
            "  "
            f"{int(item.get('uid', -1)):>4} "
            f"{str(item.get('status', '-')):<18.18} "
            f"{str(item.get('stage', '-')):<18.18} "
            f"{_fmt_duration(item.get('age_sec')):>8} "
            f"{int(item.get('retry_attempts', 0)):>5} "
            f"{progress:>11}  "
            f"{str(item.get('last_error', '') or item.get('note', '') or '-')[:70]}"
        )
    if not queue.get("active_items"):
        output.append("  -")
    output.append("")

    if events:
        output.append("Recent Events")
        for event in events[-8:]:
            output.append(
                "  "
                f"{_fmt_age(event.get('ts'), current_time):>10} "
                f"[{str(event.get('severity', 'info')).upper():<7}] "
                f"{event.get('event', '-')}"
            )
        output.append("")

    return "\n".join(output).rstrip() + "\n"


def run_runtime_dashboard(
    *,
    snapshot_path: Path | None = None,
    events_path: Path | None = None,
    refresh_sec: float = 1.0,
    once: bool = False,
    no_clear: bool = False,
    max_events: int = 8,
    stream: Any = None,
) -> int:
    target_snapshot = Path(snapshot_path) if snapshot_path is not None else RUNTIME_SNAPSHOT_FILE
    target_events = Path(events_path) if events_path is not None else RUNTIME_EVENTS_FILE
    stream = stream if stream is not None else sys.stdout
    if not target_snapshot.exists():
        stream.write(f"Telemetry snapshot not found: {target_snapshot}\n")
        stream.flush()
        return 1

    while True:
        snapshot = load_runtime_snapshot(target_snapshot)
        events = load_recent_events(target_events, limit=max_events)
        frame = render_runtime_dashboard(snapshot, events=events)
        if not no_clear:
            stream.write("\x1b[2J\x1b[H")
        stream.write(frame)
        stream.flush()
        if once:
            return 0
        time.sleep(max(0.1, float(refresh_sec)))


__all__ = [
    "render_runtime_dashboard",
    "run_runtime_dashboard",
]
