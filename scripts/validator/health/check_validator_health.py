#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.validator.weight_history import (  # noqa: E402
    build_subtensor,
    collect_history,
    get_last_update_vector,
    get_subnet_epoch_state,
    get_uid_for_hotkey,
)

DEFAULT_NETUID = 124
DEFAULT_HOTKEY = "5FF6pxRem43f7wCisfXevqYVURZtxxnC4kYTx4dnNAWqi9vg"
DEFAULT_NETWORK = "archive"
DEFAULT_REPORT_HOURS = 10
DEFAULT_BLOCK_TIME_SECONDS = 12


@dataclass
class HealthCheckResult:
    ok: bool
    status: str
    message: str
    netuid: int
    hotkey: str
    validator_uid: int | None
    checked_at_block: int
    epoch_start: int
    epoch_end: int
    last_update_block: int | None


@dataclass
class HourlyHealthCheckResult:
    hours_ago: int
    target_time_utc: datetime
    checked_at_block: int
    checked_at_time_utc: datetime
    status: str
    epoch_start: int
    epoch_end: int
    last_update_block: int | None
    last_update_time_utc: datetime | None


def is_validator_healthy(
    netuid: int,
    hotkey: str,
    network: str = DEFAULT_NETWORK,
    chain_endpoint: str = "",
    block: int | None = None,
    current_epoch: bool = False,
) -> bool:
    result = check_validator_health(
        netuid=netuid,
        hotkey=hotkey,
        network=network,
        chain_endpoint=chain_endpoint,
        block=block,
        current_epoch=current_epoch,
    )
    return result.ok


def were_last_epochs_healthy(
    epochs: int = 10,
    netuid: int = DEFAULT_NETUID,
    hotkey: str = DEFAULT_HOTKEY,
    network: str = DEFAULT_NETWORK,
    chain_endpoint: str = "",
) -> bool:
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    target = chain_endpoint or network
    subtensor = build_subtensor(target)

    try:
        records = collect_history(
            subtensor=subtensor,
            netuid=netuid,
            hotkey=hotkey,
            epochs=epochs,
            mechid=0,
            top_k=0,
            include_current_epoch=False,
        )
        return len(records) == epochs and all(
            record.registered and record.last_update_in_epoch for record in records
        )
    finally:
        close_fn = getattr(subtensor, "close", None)
        if callable(close_fn):
            close_fn()


def check_validator_health(
    netuid: int,
    hotkey: str,
    network: str = DEFAULT_NETWORK,
    chain_endpoint: str = "",
    block: int | None = None,
    current_epoch: bool = False,
) -> HealthCheckResult:
    target = chain_endpoint or network
    subtensor = build_subtensor(target)

    try:
        checked_at_block = subtensor.get_current_block() if block is None else int(block)
        return _check_validator_health_with_subtensor(
            subtensor=subtensor,
            netuid=netuid,
            hotkey=hotkey,
            checked_at_block=checked_at_block,
            current_epoch=current_epoch,
        )
    finally:
        close_fn = getattr(subtensor, "close", None)
        if callable(close_fn):
            close_fn()


def _check_validator_health_with_subtensor(
    subtensor,
    netuid: int,
    hotkey: str,
    checked_at_block: int,
    current_epoch: bool,
) -> HealthCheckResult:
    tempo, blocks_since = get_subnet_epoch_state(
        subtensor, netuid=netuid, block=checked_at_block
    )
    current_epoch_start = checked_at_block - blocks_since

    if current_epoch:
        epoch_start = current_epoch_start
        epoch_end = checked_at_block
        epoch_label = "current epoch"
        query_block = checked_at_block
    else:
        epoch_end = current_epoch_start - 1
        if epoch_end < 0:
            return HealthCheckResult(
                ok=False,
                status="ERROR",
                message="ERROR: No completed epoch is available for health evaluation yet.",
                netuid=netuid,
                hotkey=hotkey,
                validator_uid=None,
                checked_at_block=checked_at_block,
                epoch_start=-1,
                epoch_end=-1,
                last_update_block=None,
            )
        epoch_start = epoch_end - tempo
        epoch_label = "latest completed epoch"
        query_block = epoch_end

    validator_uid = get_uid_for_hotkey(
        subtensor, netuid=netuid, hotkey=hotkey, block=query_block
    )
    if validator_uid is None:
        return HealthCheckResult(
            ok=False,
            status="ERROR",
            message=(
                f"ERROR: Validator hotkey is not registered on netuid {netuid} "
                f"at block {query_block}."
            ),
            netuid=netuid,
            hotkey=hotkey,
            validator_uid=None,
            checked_at_block=checked_at_block,
            epoch_start=epoch_start,
            epoch_end=epoch_end,
            last_update_block=None,
        )

    last_updates = get_last_update_vector(subtensor, netuid=netuid, block=query_block)
    if validator_uid >= len(last_updates):
        return HealthCheckResult(
            ok=False,
            status="ERROR",
            message=(
                f"ERROR: LastUpdate data is unavailable for validator uid {validator_uid} "
                f"on netuid {netuid}."
            ),
            netuid=netuid,
            hotkey=hotkey,
            validator_uid=validator_uid,
            checked_at_block=checked_at_block,
            epoch_start=epoch_start,
            epoch_end=epoch_end,
            last_update_block=None,
        )

    last_update_block = int(last_updates[validator_uid])
    ok = epoch_start <= last_update_block <= epoch_end
    if ok:
        message = (
            f"OK: Validator is healthy. Weights were set in the {epoch_label} "
            f"(uid={validator_uid}, epoch={epoch_start}-{epoch_end}, "
            f"last_update_block={last_update_block})."
        )
        status = "OK"
    else:
        message = (
            f"ERROR: No weights set in the {epoch_label} "
            f"(uid={validator_uid}, epoch={epoch_start}-{epoch_end}, "
            f"last_update_block={last_update_block})."
        )
        status = "ERROR"

    return HealthCheckResult(
        ok=ok,
        status=status,
        message=message,
        netuid=netuid,
        hotkey=hotkey,
        validator_uid=validator_uid,
        checked_at_block=checked_at_block,
        epoch_start=epoch_start,
        epoch_end=epoch_end,
        last_update_block=last_update_block,
    )


def get_block_time_utc(subtensor, block: int) -> datetime:
    block_hash = subtensor.substrate.get_block_hash(block)
    result = subtensor.substrate.query(
        module="Timestamp",
        storage_function="Now",
        block_hash=block_hash,
    )
    timestamp_ms = float(getattr(result, "value", result))
    return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)


def find_block_for_target_time(
    subtensor,
    target_time_utc: datetime,
    current_block: int,
) -> int:
    current_time_utc = get_block_time_utc(subtensor, current_block)
    delta_seconds = (current_time_utc - target_time_utc).total_seconds()
    block = current_block - int(round(delta_seconds / DEFAULT_BLOCK_TIME_SECONDS))
    block = max(0, min(current_block, block))

    for _ in range(8):
        block_time_utc = get_block_time_utc(subtensor, block)
        block_delta_seconds = (block_time_utc - target_time_utc).total_seconds()
        if abs(block_delta_seconds) <= DEFAULT_BLOCK_TIME_SECONDS:
            break
        adjustment = int(round(block_delta_seconds / DEFAULT_BLOCK_TIME_SECONDS))
        if adjustment == 0:
            break
        block = max(0, min(current_block, block - adjustment))

    return block


def collect_recent_hourly_health_checks(
    netuid: int = DEFAULT_NETUID,
    hotkey: str = DEFAULT_HOTKEY,
    network: str = DEFAULT_NETWORK,
    chain_endpoint: str = "",
    hours: int = DEFAULT_REPORT_HOURS,
    now_utc: datetime | None = None,
) -> list[HourlyHealthCheckResult]:
    target = chain_endpoint or network
    subtensor = build_subtensor(target)
    block_time_cache: dict[int, datetime] = {}

    def cached_block_time(block: int | None) -> datetime | None:
        if block is None:
            return None
        if block not in block_time_cache:
            block_time_cache[block] = get_block_time_utc(subtensor, block)
        return block_time_cache[block]

    try:
        if hours < 1:
            raise ValueError("hours must be at least 1")

        current_block = subtensor.get_current_block()
        current_time_utc = cached_block_time(current_block)
        assert current_time_utc is not None
        report_end_utc = now_utc or current_time_utc

        rows: list[HourlyHealthCheckResult] = []
        for hours_ago in range(hours - 1, -1, -1):
            target_time_utc = report_end_utc - timedelta(hours=hours_ago)
            checked_at_block = find_block_for_target_time(
                subtensor=subtensor,
                target_time_utc=target_time_utc,
                current_block=current_block,
            )
            checked_at_time_utc = cached_block_time(checked_at_block)
            assert checked_at_time_utc is not None

            result = _check_validator_health_with_subtensor(
                subtensor=subtensor,
                netuid=netuid,
                hotkey=hotkey,
                checked_at_block=checked_at_block,
                current_epoch=False,
            )

            rows.append(
                HourlyHealthCheckResult(
                    hours_ago=hours_ago,
                    target_time_utc=target_time_utc,
                    checked_at_block=checked_at_block,
                    checked_at_time_utc=checked_at_time_utc,
                    status=result.status,
                    epoch_start=result.epoch_start,
                    epoch_end=result.epoch_end,
                    last_update_block=result.last_update_block,
                    last_update_time_utc=cached_block_time(result.last_update_block),
                )
            )

        return rows
    finally:
        close_fn = getattr(subtensor, "close", None)
        if callable(close_fn):
            close_fn()


def _format_utc(value: datetime | None) -> str:
    if value is None:
        return "-"
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def render_hourly_health_table(rows: list[HourlyHealthCheckResult]) -> str:
    headers = (
        "hrs_ago",
        "target_time_utc",
        "block",
        "block_time_utc",
        "status",
        "epoch",
        "last_upd",
        "last_upd_time_utc",
    )
    lines = [
        "{:>7} {:>20} {:>10} {:>20} {:>6} {:>23} {:>10} {:>20}".format(*headers)
    ]

    for row in rows:
        epoch_label = (
            "-"
            if row.epoch_start < 0 or row.epoch_end < 0
            else f"{row.epoch_start}-{row.epoch_end}"
        )
        lines.append(
            "{:>7} {:>20} {:>10} {:>20} {:>6} {:>23} {:>10} {:>20}".format(
                row.hours_ago,
                _format_utc(row.target_time_utc),
                row.checked_at_block,
                _format_utc(row.checked_at_time_utc),
                row.status,
                epoch_label,
                "-" if row.last_update_block is None else row.last_update_block,
                _format_utc(row.last_update_time_utc),
            )
        )

    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check health for validator 5FF6... on subnet 124."
    )
    parser.add_argument(
        "--netuid",
        type=int,
        default=DEFAULT_NETUID,
        help=f"Subnet netuid. Default: {DEFAULT_NETUID}.",
    )
    parser.add_argument(
        "--hotkey",
        default=DEFAULT_HOTKEY,
        help="Validator hotkey ss58. Defaults to the production validator hotkey.",
    )
    parser.add_argument(
        "--network",
        default=DEFAULT_NETWORK,
        help=f"Bittensor network name or websocket endpoint. Default: {DEFAULT_NETWORK}.",
    )
    parser.add_argument(
        "--chain-endpoint",
        default="",
        help="Explicit websocket endpoint. Overrides --network.",
    )
    parser.add_argument(
        "--block",
        type=int,
        default=None,
        help="Optional block to evaluate instead of the live chain tip.",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=DEFAULT_REPORT_HOURS,
        help=f"Number of hourly checks to print in table mode. Default: {DEFAULT_REPORT_HOURS}.",
    )
    parser.add_argument(
        "--last-epochs",
        type=int,
        default=None,
        help="Print true/false depending on whether all of the last N completed epochs were healthy.",
    )
    parser.add_argument(
        "--current-epoch",
        action="store_true",
        help="Check the current in-progress epoch instead of the latest completed epoch.",
    )
    parser.add_argument(
        "--single-check",
        action="store_true",
        help="Print a single health check instead of the default hourly table.",
    )
    return parser.parse_args(argv)


def format_health_check_error(exc: Exception) -> str:
    message = str(exc)
    if "UnknownBlock" in message or "State already discarded" in message:
        return (
            "Historical chain state is unavailable from the selected endpoint. "
            "Use --network archive or an archive websocket endpoint for old blocks."
        )
    return message


def main() -> int:
    args = parse_args()
    try:
        if args.last_epochs is not None:
            ok = were_last_epochs_healthy(
                epochs=args.last_epochs,
                netuid=args.netuid,
                hotkey=args.hotkey,
                network=args.network,
                chain_endpoint=args.chain_endpoint,
            )
            print(str(ok).lower())
            return 0 if ok else 1

        if args.single_check or args.block is not None or args.current_epoch:
            result = check_validator_health(
                netuid=args.netuid,
                hotkey=args.hotkey,
                network=args.network,
                chain_endpoint=args.chain_endpoint,
                block=args.block,
                current_epoch=args.current_epoch,
            )
            print(result.message)
            return 0 if result.ok else 1

        rows = collect_recent_hourly_health_checks(
            netuid=args.netuid,
            hotkey=args.hotkey,
            network=args.network,
            chain_endpoint=args.chain_endpoint,
            hours=args.hours,
        )
    except Exception as exc:
        print(f"ERROR: Health check failed: {format_health_check_error(exc)}")
        return 2

    print(render_hourly_health_table(rows))
    return 0 if all(row.status == "OK" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
