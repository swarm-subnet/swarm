#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence


U16_MAX = 65535.0


@dataclass
class EpochRecord:
    row: int
    epoch_start: int
    epoch_end: int
    tempo: int
    validator_uid: int | None
    registered: bool
    registered_this_epoch: bool
    last_update_block: int | None
    last_update_in_epoch: bool
    weight_row_present: bool
    weight_entries: int
    nonzero_weight_entries: int
    changed_since_prev_epoch: bool | None
    weight_hash: str | None
    top_weights: list[dict[str, float | int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect historical validator weights per epoch. "
            "For more than ~300 blocks of history you should use the archive network."
        )
    )
    parser.add_argument("--netuid", type=int, required=True, help="Subnet netuid.")
    parser.add_argument(
        "--hotkey",
        required=True,
        help="Validator hotkey ss58 address.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs to inspect. Default: 100.",
    )
    parser.add_argument(
        "--network",
        default="archive",
        help="Bittensor network name or websocket endpoint. Default: archive.",
    )
    parser.add_argument(
        "--chain-endpoint",
        default="",
        help=(
            "Explicit websocket endpoint. Overrides --network. "
            "Example: wss://archive.chain.opentensor.ai:443"
        ),
    )
    parser.add_argument(
        "--mechid",
        type=int,
        default=0,
        help="Mechanism id for newer SDKs. Default: 0.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many top targets to display per epoch. Default: 5.",
    )
    parser.add_argument(
        "--include-current-epoch",
        action="store_true",
        help="Include the current in-progress epoch as the newest row.",
    )
    parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="Output format. Default: table.",
    )
    return parser.parse_args()


def build_subtensor(target: str):
    import bittensor as bt

    subtensor_ctor = getattr(bt, "Subtensor", None) or getattr(bt, "subtensor", None)
    if subtensor_ctor is None:
        raise RuntimeError("Could not locate a bittensor subtensor constructor.")
    return subtensor_ctor(network=target)


def scalar_value(result: Any) -> Any:
    return getattr(result, "value", result)


def query_subtensor_value(subtensor: Any, name: str, block: int | None, params: list[Any]) -> Any:
    result = subtensor.query_subtensor(name=name, block=block, params=params)
    return scalar_value(result)


def normalize_uid(value: Any) -> int | None:
    if value is None:
        return None
    uid = int(value)
    if uid < 0 or uid > 1_000_000_000:
        return None
    return uid


def get_uid_for_hotkey(subtensor: Any, netuid: int, hotkey: str, block: int) -> int | None:
    uid = subtensor.get_uid_for_hotkey_on_subnet(hotkey, netuid, block=block)
    return normalize_uid(uid)


def get_subnet_epoch_state(subtensor: Any, netuid: int, block: int) -> tuple[int, int]:
    subnet_info_fn = getattr(subtensor, "get_subnet_info", None)
    if callable(subnet_info_fn):
        subnet_info = subnet_info_fn(netuid, block=block)
        if subnet_info is not None:
            tempo = int(getattr(subnet_info, "tempo"))
            blocks_since = getattr(
                subnet_info,
                "blocks_since_epoch",
                getattr(subnet_info, "blocks_since_last_step", None),
            )
            if blocks_since is not None:
                return tempo, int(blocks_since)

    tempo = getattr(subtensor, "tempo")(netuid, block=block)
    if tempo is None:
        raise RuntimeError(f"Could not fetch tempo for netuid {netuid} at block {block}.")

    blocks_since = None
    blocks_since_fn = getattr(subtensor, "blocks_since_last_step", None)
    if callable(blocks_since_fn):
        blocks_since = blocks_since_fn(netuid, block=block)

    if blocks_since is None:
        legacy_blocks_since_fn = getattr(subtensor, "blocks_since_epoch", None)
        if callable(legacy_blocks_since_fn):
            blocks_since = legacy_blocks_since_fn(netuid, block=block)

    if blocks_since is None:
        for storage_name in ("BlocksSinceLastStep", "BlocksSinceEpoch"):
            try:
                blocks_since = query_subtensor_value(
                    subtensor, storage_name, block=block, params=[netuid]
                )
                if blocks_since is not None:
                    break
            except Exception:
                continue

    if blocks_since is None:
        raise RuntimeError(
            f"Could not fetch blocks-since-epoch data for netuid {netuid} at block {block}."
        )

    return int(tempo), int(blocks_since)


def get_last_update_vector(subtensor: Any, netuid: int, block: int) -> list[int]:
    raw = query_subtensor_value(subtensor, "LastUpdate", block=block, params=[netuid])
    if raw is None:
        return []
    if hasattr(raw, "tolist"):
        return [int(value) for value in raw.tolist()]
    if isinstance(raw, Sequence):
        return [int(value) for value in raw]
    raise RuntimeError(
        f"Unexpected LastUpdate payload type at block {block}: {type(raw).__name__}"
    )


def get_weight_map(subtensor: Any, netuid: int, mechid: int, block: int) -> list[tuple[int, Any]]:
    try:
        rows = subtensor.weights(netuid=netuid, mechid=mechid, block=block)
    except TypeError:
        rows = subtensor.weights(netuid=netuid, block=block)
    return list(rows or [])


def normalize_weight_row(raw_pairs: Any) -> list[tuple[int, int]]:
    if raw_pairs is None:
        return []

    normalized: list[tuple[int, int]] = []
    for pair in raw_pairs:
        if isinstance(pair, dict):
            target_uid = pair.get("uid", pair.get("target_uid"))
            value = pair.get("weight", pair.get("value"))
        else:
            try:
                target_uid, value = pair
            except Exception as exc:
                raise RuntimeError(f"Unexpected weight pair payload: {pair!r}") from exc

        normalized.append((int(target_uid), int(value)))

    normalized.sort(key=lambda item: item[0])
    return normalized


def extract_weight_row(weight_map: Iterable[tuple[Any, Any]], validator_uid: int | None) -> list[tuple[int, int]]:
    if validator_uid is None:
        return []

    for row_uid, raw_pairs in weight_map:
        if int(row_uid) == validator_uid:
            return normalize_weight_row(raw_pairs)
    return []


def hash_weight_row(weight_row: list[tuple[int, int]]) -> str | None:
    if not weight_row:
        return None
    encoded = json.dumps(weight_row, separators=(",", ":"), sort_keys=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def summarize_top_weights(weight_row: list[tuple[int, int]], top_k: int) -> list[dict[str, float | int]]:
    top_pairs = [pair for pair in weight_row if pair[1] > 0]
    top_pairs.sort(key=lambda item: (-item[1], item[0]))
    top_pairs = top_pairs[:top_k]
    return [
        {
            "uid": target_uid,
            "raw": raw_weight,
            "normalized": round(raw_weight / U16_MAX, 6),
        }
        for target_uid, raw_weight in top_pairs
    ]


def format_top_weights(top_weights: list[dict[str, float | int]]) -> str:
    if not top_weights:
        return "-"
    return ", ".join(
        f"{item['uid']}:{item['normalized']:.6f}"
        for item in top_weights
    )


def build_epoch_blocks(
    subtensor: Any,
    netuid: int,
    epochs: int,
    include_current_epoch: bool,
) -> list[tuple[int, int, int]]:
    current_block = subtensor.get_current_block()
    _, current_blocks_since = get_subnet_epoch_state(subtensor, netuid, current_block)
    current_epoch_start = current_block - current_blocks_since

    cursor = current_block if include_current_epoch else current_epoch_start - 1
    epoch_blocks: list[tuple[int, int, int]] = []

    while cursor >= 0 and len(epoch_blocks) < epochs:
        tempo, blocks_since = get_subnet_epoch_state(subtensor, netuid, cursor)
        epoch_start = cursor - blocks_since
        epoch_end = cursor if include_current_epoch and len(epoch_blocks) == 0 else epoch_start + tempo
        if epoch_end > cursor:
            epoch_end = cursor
        epoch_blocks.append((epoch_start, epoch_end, tempo))
        cursor = epoch_start - 1

    epoch_blocks.reverse()
    return epoch_blocks


def collect_history(
    subtensor: Any,
    netuid: int,
    hotkey: str,
    epochs: int,
    mechid: int,
    top_k: int,
    include_current_epoch: bool,
) -> list[EpochRecord]:
    epoch_blocks = build_epoch_blocks(
        subtensor=subtensor,
        netuid=netuid,
        epochs=epochs,
        include_current_epoch=include_current_epoch,
    )

    records: list[EpochRecord] = []
    previous_registered = False
    previous_hash: str | None = None

    if epoch_blocks and epoch_blocks[0][0] > 0:
        previous_registered = (
            get_uid_for_hotkey(subtensor, netuid, hotkey, block=epoch_blocks[0][0] - 1)
            is not None
        )

    for row_number, (epoch_start, epoch_end, tempo) in enumerate(epoch_blocks, start=1):
        validator_uid = get_uid_for_hotkey(subtensor, netuid, hotkey, block=epoch_end)
        registered = validator_uid is not None
        registered_this_epoch = registered and not previous_registered

        last_update_block = None
        weight_row: list[tuple[int, int]] = []

        if registered and validator_uid is not None:
            last_updates = get_last_update_vector(subtensor, netuid, block=epoch_end)
            if validator_uid < len(last_updates):
                last_update_block = int(last_updates[validator_uid])

            weight_map = get_weight_map(subtensor, netuid, mechid=mechid, block=epoch_end)
            weight_row = extract_weight_row(weight_map, validator_uid)

        row_hash = hash_weight_row(weight_row)
        record = EpochRecord(
            row=row_number,
            epoch_start=epoch_start,
            epoch_end=epoch_end,
            tempo=tempo,
            validator_uid=validator_uid,
            registered=registered,
            registered_this_epoch=registered_this_epoch,
            last_update_block=last_update_block,
            last_update_in_epoch=(
                last_update_block is not None
                and epoch_start <= last_update_block <= epoch_end
            ),
            weight_row_present=bool(weight_row),
            weight_entries=len(weight_row),
            nonzero_weight_entries=sum(1 for _, weight in weight_row if weight > 0),
            changed_since_prev_epoch=(
                None if previous_hash is None else row_hash != previous_hash
            ),
            weight_hash=row_hash,
            top_weights=summarize_top_weights(weight_row, top_k=top_k),
        )
        records.append(record)
        previous_registered = registered
        previous_hash = row_hash

    return records


def render_table(records: list[EpochRecord]) -> str:
    headers = (
        "row",
        "start",
        "end",
        "tempo",
        "uid",
        "reg",
        "reg_new",
        "last_upd",
        "upd_in_ep",
        "entries",
        "nonzero",
        "changed",
        "top_weights",
    )
    lines = [
        "{:>4} {:>10} {:>10} {:>6} {:>5} {:>4} {:>7} {:>10} {:>9} {:>7} {:>7} {:>7}  {}".format(
            *headers
        )
    ]

    for record in records:
        lines.append(
            "{:>4} {:>10} {:>10} {:>6} {:>5} {:>4} {:>7} {:>10} {:>9} {:>7} {:>7} {:>7}  {}".format(
                record.row,
                record.epoch_start,
                record.epoch_end,
                record.tempo,
                "-" if record.validator_uid is None else record.validator_uid,
                "Y" if record.registered else "N",
                "Y" if record.registered_this_epoch else "N",
                "-" if record.last_update_block is None else record.last_update_block,
                "Y" if record.last_update_in_epoch else "N",
                record.weight_entries,
                record.nonzero_weight_entries,
                "-" if record.changed_since_prev_epoch is None else ("Y" if record.changed_since_prev_epoch else "N"),
                format_top_weights(record.top_weights),
            )
        )

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    target = args.chain_endpoint or args.network

    try:
        subtensor = build_subtensor(target)
    except Exception as exc:
        print(f"Failed to initialize subtensor for '{target}': {exc}", file=sys.stderr)
        return 1

    try:
        records = collect_history(
            subtensor=subtensor,
            netuid=args.netuid,
            hotkey=args.hotkey,
            epochs=args.epochs,
            mechid=args.mechid,
            top_k=args.top_k,
            include_current_epoch=args.include_current_epoch,
        )
    except Exception as exc:
        print(f"Failed to collect weight history: {exc}", file=sys.stderr)
        print(
            "Hint: if you are requesting more than ~300 blocks of history, use --network archive "
            "or --chain-endpoint wss://archive.chain.opentensor.ai:443",
            file=sys.stderr,
        )
        return 1
    finally:
        close_fn = getattr(subtensor, "close", None)
        if callable(close_fn):
            close_fn()

    if args.format == "json":
        print(json.dumps([asdict(record) for record in records], indent=2))
    else:
        print(render_table(records))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
