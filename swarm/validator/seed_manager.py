import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import bittensor as bt

from swarm.challenge_families import DEFAULT_RUNTIME_FAMILY_ID
from swarm.constants import (
    BENCHMARK_SCREENING_SEED_COUNT,
    BENCHMARK_TOTAL_SEED_COUNT,
    BENCHMARK_VERSION,
    EPOCH_ANCHOR_UTC,
    EPOCH_DURATION_SECONDS,
)

STATE_DIR = Path(__file__).parent.parent.parent / "state"
EPOCH_SEEDS_DIR = STATE_DIR / "epoch_seeds"

_MAX_SEED = 2**32 - 1
_EPOCH_FILE_RE = re.compile(r"^epoch_(\d+)(?:__(.+))?\.json$")


def _generate_random_seeds(count: int) -> List[int]:
    rng = random.SystemRandom()
    return [rng.randint(0, _MAX_SEED) for _ in range(count)]


class BenchmarkSeedManager:
    """Per-epoch seed management with family-specific seed sets.

    Benchmark epoch remains global across the network. Within a given epoch,
    each challenge family owns an independent seed set and publication record.
    """

    def __init__(self) -> None:
        EPOCH_SEEDS_DIR.mkdir(parents=True, exist_ok=True)
        self.seeds: List[int] = []
        self.current_epoch_requires_state_invalidation = False
        self._pending_publications: List[dict] = []
        self._family_seeds: Dict[str, List[int]] = {}

        self.epoch_number = self._latest_local_epoch()
        if self.epoch_number > 0:
            self._publish_unpublished_epochs()
            self._load_or_generate_seeds(invalidate_local_state_on_regenerate=True)

        bt.logging.info(
            f"BenchmarkSeedManager: epoch={self.epoch_number}, "
            f"{len(self.seeds)} seeds for {DEFAULT_RUNTIME_FAMILY_ID} "
            f"({BENCHMARK_SCREENING_SEED_COUNT} screening + "
            f"{BENCHMARK_TOTAL_SEED_COUNT - BENCHMARK_SCREENING_SEED_COUNT} benchmark)"
        )

    def _latest_local_epoch(self) -> int:
        """Return the highest epoch number found in EPOCH_SEEDS_DIR, or 0."""
        best = 0
        for path in EPOCH_SEEDS_DIR.glob("epoch_*.json"):
            parsed = self._parse_epoch_file_path(path)
            if parsed is None:
                continue
            candidate, _family_id = parsed
            if candidate > best:
                best = candidate
        return best

    def _epoch_to_raw(self, epoch: int) -> int:
        return epoch - 1

    def _epoch_file(self, epoch: int, family_id: str = DEFAULT_RUNTIME_FAMILY_ID) -> Path:
        if family_id == DEFAULT_RUNTIME_FAMILY_ID:
            return EPOCH_SEEDS_DIR / f"epoch_{epoch}.json"
        return EPOCH_SEEDS_DIR / f"epoch_{epoch}__{family_id}.json"

    def _parse_epoch_file_path(self, path: Path) -> Tuple[int, str] | None:
        match = _EPOCH_FILE_RE.match(path.name)
        if not match:
            return None
        try:
            epoch_number = int(match.group(1))
        except ValueError:
            return None
        family_id = match.group(2) or DEFAULT_RUNTIME_FAMILY_ID
        return epoch_number, family_id

    def _load_epoch_payload(self, path: Path) -> dict:
        data = json.loads(path.read_text())
        data.setdefault("family_id", DEFAULT_RUNTIME_FAMILY_ID)
        return data

    def _queue_pending_publication(self, data: dict) -> None:
        family_id = str(data.get("family_id") or DEFAULT_RUNTIME_FAMILY_ID)
        epoch_number = data.get("epoch_number")
        if epoch_number is None:
            return
        key = (int(epoch_number), family_id)
        if any(
            int(item.get("epoch_number", -1)) == key[0]
            and str(item.get("family_id") or DEFAULT_RUNTIME_FAMILY_ID) == key[1]
            for item in self._pending_publications
        ):
            return
        normalized = dict(data)
        normalized["family_id"] = family_id
        self._pending_publications.append(normalized)

    def _ensure_epoch_family_seeds(
        self,
        epoch: int,
        family_id: str,
        *,
        invalidate_local_state_on_regenerate: bool,
    ) -> List[int]:
        path = self._epoch_file(epoch, family_id)
        if path.exists():
            try:
                data = self._load_epoch_payload(path)
                if (
                    data.get("epoch_number") == epoch
                    and str(data.get("family_id") or DEFAULT_RUNTIME_FAMILY_ID) == family_id
                    and len(data.get("seeds", [])) == BENCHMARK_TOTAL_SEED_COUNT
                ):
                    seeds = [int(seed) for seed in data["seeds"]]
                    self._family_seeds[family_id] = seeds
                    if family_id == DEFAULT_RUNTIME_FAMILY_ID:
                        self.seeds = list(seeds)
                        self.current_epoch_requires_state_invalidation = False
                    bt.logging.info(f"Loaded seeds from {path.name}")
                    return seeds
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                bt.logging.warning(f"Corrupt epoch file {path.name}, regenerating")

        seeds = _generate_random_seeds(BENCHMARK_TOTAL_SEED_COUNT)
        self._family_seeds[family_id] = seeds
        if family_id == DEFAULT_RUNTIME_FAMILY_ID:
            self.seeds = list(seeds)
            self.current_epoch_requires_state_invalidation = (
                invalidate_local_state_on_regenerate
            )
        self._save_epoch_file(epoch, family_id, seeds, published=False)
        bt.logging.info(
            f"Generated {len(seeds)} random seeds for epoch {epoch} family {family_id}"
        )
        return seeds

    def _load_or_generate_seeds(
        self,
        *,
        invalidate_local_state_on_regenerate: bool,
    ) -> None:
        self._ensure_epoch_family_seeds(
            self.epoch_number,
            DEFAULT_RUNTIME_FAMILY_ID,
            invalidate_local_state_on_regenerate=invalidate_local_state_on_regenerate,
        )

    def _save_epoch_file(
        self,
        epoch: int,
        family_id: str,
        seeds: List[int],
        published: bool,
    ) -> None:
        start, end = self.epoch_time_range(epoch)
        data = {
            "epoch_number": epoch,
            "family_id": family_id,
            "started_at": start.isoformat(),
            "ended_at": end.isoformat(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "seed_count": len(seeds),
            "benchmark_version": BENCHMARK_VERSION,
            "published": published,
            "seeds": seeds,
        }
        path = self._epoch_file(epoch, family_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, separators=(",", ":")))
        tmp.replace(path)

    def _publish_unpublished_epochs(self) -> None:
        pending: List[dict] = []
        for path in sorted(EPOCH_SEEDS_DIR.glob("epoch_*.json")):
            parsed = self._parse_epoch_file_path(path)
            if parsed is None:
                continue
            epoch_number, _family_id = parsed
            if epoch_number >= self.epoch_number:
                continue
            try:
                data = self._load_epoch_payload(path)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            if not data.get("published", False):
                pending.append(data)
        self._pending_publications = []
        for item in pending:
            self._queue_pending_publication(item)

    def get_pending_publications(self, family_id: str | None = None) -> List[dict]:
        publications = list(self._pending_publications)
        if family_id is None:
            return publications
        return [
            item
            for item in publications
            if str(item.get("family_id") or DEFAULT_RUNTIME_FAMILY_ID) == family_id
        ]

    def mark_epoch_published(
        self,
        epoch: int,
        family_id: str = DEFAULT_RUNTIME_FAMILY_ID,
    ) -> None:
        path = self._epoch_file(epoch, family_id)
        if not path.exists():
            return
        try:
            data = self._load_epoch_payload(path)
            data["published"] = True
            data["published_at"] = datetime.now(timezone.utc).isoformat()
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, separators=(",", ":")))
            tmp.replace(path)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        self._pending_publications = [
            publication
            for publication in self._pending_publications
            if not (
                int(publication.get("epoch_number", -1)) == epoch
                and str(publication.get("family_id") or DEFAULT_RUNTIME_FAMILY_ID) == family_id
            )
        ]

    def align_to_epoch(self, epoch: int) -> int | None:
        """Align local seed state to the epoch reported by ``/sync``.

        Global epoch remains shared. Family-specific seeds for the old epoch stay
        pending until published, even when the validator realigns backward.
        """
        if epoch <= 0 or epoch == self.epoch_number:
            return None

        old_epoch = self.epoch_number
        for path in EPOCH_SEEDS_DIR.glob(f"epoch_{old_epoch}*.json"):
            parsed = self._parse_epoch_file_path(path)
            if parsed is None:
                continue
            try:
                data = self._load_epoch_payload(path)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            if not data.get("published", False):
                self._queue_pending_publication(data)

        self.epoch_number = epoch
        self._family_seeds = {}
        self.seeds = []
        self._publish_unpublished_epochs()
        self._load_or_generate_seeds(invalidate_local_state_on_regenerate=False)
        bt.logging.info(
            f"BenchmarkSeedManager aligned to backend epoch: {old_epoch} -> {self.epoch_number}"
        )
        return old_epoch

    def epoch_time_range(self, epoch: int) -> tuple[datetime, datetime]:
        raw = self._epoch_to_raw(epoch)
        anchor_ts = EPOCH_ANCHOR_UTC.timestamp()
        start_ts = anchor_ts + raw * EPOCH_DURATION_SECONDS
        end_ts = start_ts + EPOCH_DURATION_SECONDS
        start = datetime.fromtimestamp(start_ts, tz=timezone.utc)
        end = datetime.fromtimestamp(end_ts, tz=timezone.utc)
        return start, end

    def seconds_until_epoch_end(self) -> float:
        _, end = self.epoch_time_range(self.epoch_number)
        return max(0.0, end.timestamp() - time.time())

    def _ensure_current_family_seeds(
        self,
        family_id: str = DEFAULT_RUNTIME_FAMILY_ID,
    ) -> List[int]:
        if self.epoch_number <= 0:
            return []
        seeds = self._family_seeds.get(family_id)
        if seeds is not None:
            return list(seeds)
        return self._ensure_epoch_family_seeds(
            self.epoch_number,
            family_id,
            invalidate_local_state_on_regenerate=False,
        )

    def get_screening_seeds(
        self,
        family_id: str = DEFAULT_RUNTIME_FAMILY_ID,
    ) -> List[int]:
        return self._ensure_current_family_seeds(family_id)[:BENCHMARK_SCREENING_SEED_COUNT]

    def get_benchmark_seeds(
        self,
        family_id: str = DEFAULT_RUNTIME_FAMILY_ID,
    ) -> List[int]:
        return self._ensure_current_family_seeds(family_id)[BENCHMARK_SCREENING_SEED_COUNT:]

    def get_all_seeds(
        self,
        family_id: str = DEFAULT_RUNTIME_FAMILY_ID,
    ) -> List[int]:
        return list(self._ensure_current_family_seeds(family_id))
