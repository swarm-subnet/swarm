import hashlib
import hmac
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import bittensor as bt

from swarm.constants import (
    BENCHMARK_VERSION,
    BENCHMARK_TOTAL_SEED_COUNT,
    BENCHMARK_SCREENING_SEED_COUNT,
    EPOCH_DURATION_SECONDS,
    EPOCH_ANCHOR_UTC,
)

STATE_DIR = Path(__file__).parent.parent.parent / "state"
EPOCH_SEEDS_DIR = STATE_DIR / "epoch_seeds"
EPOCH_ORIGIN_FILE = EPOCH_SEEDS_DIR / "epoch_origin.json"


def _compute_raw_week(ts: Optional[float] = None) -> int:
    if ts is None:
        ts = time.time()
    anchor_ts = EPOCH_ANCHOR_UTC.timestamp()
    return int((ts - anchor_ts) // EPOCH_DURATION_SECONDS)


def _load_or_create_origin() -> int:
    EPOCH_SEEDS_DIR.mkdir(parents=True, exist_ok=True)
    if EPOCH_ORIGIN_FILE.exists():
        try:
            data = json.loads(EPOCH_ORIGIN_FILE.read_text())
            return data["origin_raw_week"]
        except (json.JSONDecodeError, KeyError):
            pass

    raw = _compute_raw_week()
    data = {
        "origin_raw_week": raw,
        "set_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp = EPOCH_ORIGIN_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, separators=(",", ":")))
    tmp.replace(EPOCH_ORIGIN_FILE)
    bt.logging.info(f"Epoch origin set: raw_week={raw} (epoch 1 starts now)")
    return raw


def _derive_seeds(secret: str, epoch_number: int) -> List[int]:
    seeds = []
    key = secret.encode()
    for i in range(BENCHMARK_TOTAL_SEED_COUNT):
        msg = f"epoch_{epoch_number}_seed_{i}".encode()
        h = hmac.new(key, msg, hashlib.sha256)
        seeds.append(int.from_bytes(h.digest()[:4], "big"))
    return seeds


class BenchmarkSeedManager:

    def __init__(self, secret: str = None):
        self.secret = secret or os.getenv("SWARM_PRIVATE_BENCHMARK_SECRET")
        if not self.secret:
            raise ValueError(
                "SWARM_PRIVATE_BENCHMARK_SECRET env var required"
            )

        EPOCH_SEEDS_DIR.mkdir(parents=True, exist_ok=True)
        self._origin = _load_or_create_origin()
        self.epoch_number = self._raw_to_epoch(_compute_raw_week())
        self.seeds: List[int] = []

        self._publish_unpublished_epochs()
        self._load_or_generate_seeds()

        bt.logging.info(
            f"BenchmarkSeedManager: epoch={self.epoch_number}, "
            f"{len(self.seeds)} seeds ({BENCHMARK_SCREENING_SEED_COUNT} screening + "
            f"{BENCHMARK_TOTAL_SEED_COUNT - BENCHMARK_SCREENING_SEED_COUNT} benchmark)"
        )

    def _raw_to_epoch(self, raw_week: int) -> int:
        return raw_week - self._origin + 1

    def _epoch_to_raw(self, epoch: int) -> int:
        return self._origin + epoch - 1

    def _epoch_file(self, epoch: int) -> Path:
        return EPOCH_SEEDS_DIR / f"epoch_{epoch}.json"

    def _load_or_generate_seeds(self) -> None:
        path = self._epoch_file(self.epoch_number)
        if path.exists():
            try:
                data = json.loads(path.read_text())
                if (
                    data.get("epoch_number") == self.epoch_number
                    and len(data.get("seeds", [])) == BENCHMARK_TOTAL_SEED_COUNT
                ):
                    self.seeds = data["seeds"]
                    bt.logging.info(f"Loaded seeds from {path.name}")
                    return
            except (json.JSONDecodeError, KeyError):
                bt.logging.warning(f"Corrupt epoch file {path.name}, regenerating")

        self.seeds = _derive_seeds(self.secret, self.epoch_number)
        self._save_epoch_file(self.epoch_number, self.seeds, published=False)
        bt.logging.info(f"Generated {len(self.seeds)} seeds for epoch {self.epoch_number}")

    def _save_epoch_file(self, epoch: int, seeds: List[int], published: bool) -> None:
        start, end = self.epoch_time_range(epoch)
        data = {
            "epoch_number": epoch,
            "started_at": start.isoformat(),
            "ended_at": end.isoformat(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "seed_count": len(seeds),
            "benchmark_version": BENCHMARK_VERSION,
            "published": published,
            "seeds": seeds,
        }
        path = self._epoch_file(epoch)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, separators=(",", ":")))
        tmp.replace(path)

    def _publish_unpublished_epochs(self) -> None:
        self._pending_publications: List[dict] = []
        for f in sorted(EPOCH_SEEDS_DIR.glob("epoch_*.json")):
            if f.name == "epoch_origin.json":
                continue
            try:
                data = json.loads(f.read_text())
                ep = data.get("epoch_number")
                if ep is not None and ep < self.epoch_number and not data.get("published", False):
                    self._pending_publications.append(data)
            except (json.JSONDecodeError, KeyError):
                continue

    def get_pending_publications(self) -> List[dict]:
        return list(self._pending_publications)

    def mark_epoch_published(self, epoch: int) -> None:
        path = self._epoch_file(epoch)
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            data["published"] = True
            data["published_at"] = datetime.now(timezone.utc).isoformat()
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, separators=(",", ":")))
            tmp.replace(path)
        except (json.JSONDecodeError, KeyError):
            pass
        self._pending_publications = [
            p for p in self._pending_publications if p.get("epoch_number") != epoch
        ]

    def check_epoch_transition(self) -> bool:
        current = self._raw_to_epoch(_compute_raw_week())
        return current != self.epoch_number

    def advance_to_new_epoch(self) -> int:
        old_epoch = self.epoch_number
        self.epoch_number = self._raw_to_epoch(_compute_raw_week())

        old_file = self._epoch_file(old_epoch)
        if old_file.exists():
            try:
                data = json.loads(old_file.read_text())
                if not data.get("published", False):
                    self._pending_publications.append(data)
            except (json.JSONDecodeError, KeyError):
                pass

        self._load_or_generate_seeds()
        bt.logging.info(f"Epoch transition: {old_epoch} → {self.epoch_number}")
        return old_epoch

    def epoch_time_range(self, epoch: int) -> tuple[datetime, datetime]:
        raw = self._epoch_to_raw(epoch)
        anchor_ts = EPOCH_ANCHOR_UTC.timestamp()
        start_ts = anchor_ts + raw * EPOCH_DURATION_SECONDS
        end_ts = start_ts + EPOCH_DURATION_SECONDS
        start = datetime.fromtimestamp(start_ts, tz=timezone.utc)
        end = datetime.fromtimestamp(end_ts, tz=timezone.utc)
        return start, end

    def get_screening_seeds(self) -> List[int]:
        return self.seeds[:BENCHMARK_SCREENING_SEED_COUNT]

    def get_benchmark_seeds(self) -> List[int]:
        return self.seeds[BENCHMARK_SCREENING_SEED_COUNT:]

    def get_all_seeds(self) -> List[int]:
        return list(self.seeds)

    def get_current_seeds_data(self) -> dict:
        start, end = self.epoch_time_range(self.epoch_number)
        return {
            "epoch_number": self.epoch_number,
            "started_at": start.isoformat(),
            "ended_at": end.isoformat(),
            "seed_count": len(self.seeds),
            "seeds": list(self.seeds),
        }
