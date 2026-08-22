import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import bittensor as bt

from swarm.challenge_families import DEFAULT_RUNTIME_FAMILY_ID
from swarm.constants import (
    BENCHMARK_SCREENING_SEED_COUNT,
    BENCHMARK_TOTAL_SEED_COUNT,
    BENCHMARK_VERSION,
    EPOCH_ANCHOR_UTC,
    EPOCH_DURATION_LONG_SECONDS,
    EPOCH_DURATION_SECONDS,
    EPOCH_SWITCH_NUMBER,
    EPOCH_SWITCH_TS,
)

STATE_DIR = Path(__file__).parent.parent.parent / "state"
EPOCH_SEEDS_DIR = STATE_DIR / "epoch_seeds"
# Kept out of the epoch_*.json namespace so a restart cannot read it as the rollover having happened.
PREEVAL_SEEDS_DIR = STATE_DIR / "preeval_seeds"

_MAX_SEED = 2**32 - 1
_EPOCH_FILE_RE = re.compile(r"^epoch_(\d+)(?:__(.+))?\.json$")
_PREEVAL_FILE_RE = re.compile(r"^preeval_(\d+)(?:__(.+))?\.json$")


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

    def _seed_file(self, directory: Path, prefix: str, epoch: int, family_id: str) -> Path:
        suffix = "" if family_id == DEFAULT_RUNTIME_FAMILY_ID else f"__{family_id}"
        return directory / f"{prefix}_{epoch}{suffix}.json"

    def _epoch_file(self, epoch: int, family_id: str = DEFAULT_RUNTIME_FAMILY_ID) -> Path:
        return self._seed_file(EPOCH_SEEDS_DIR, "epoch", epoch, family_id)

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

    def _read_seed_file(self, path: Path, epoch: int, family_id: str) -> List[int] | None:
        """Seeds from a stored file, or None when it is absent, corrupt or for another scope."""
        if not path.exists():
            return None
        try:
            data = self._load_epoch_payload(path)
            if (
                data.get("epoch_number") == epoch
                and str(data.get("family_id") or DEFAULT_RUNTIME_FAMILY_ID) == family_id
                and len(data.get("seeds", [])) == BENCHMARK_TOTAL_SEED_COUNT
            ):
                return [int(seed) for seed in data["seeds"]]
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            bt.logging.warning(f"Corrupt epoch file {path.name}, regenerating")
        return None

    def _ensure_epoch_family_seeds(
        self,
        epoch: int,
        family_id: str,
        *,
        invalidate_local_state_on_regenerate: bool,
    ) -> List[int]:
        path = self._epoch_file(epoch, family_id)
        seeds = self._read_seed_file(path, epoch, family_id)
        if seeds is not None:
            self._family_seeds[family_id] = seeds
            if family_id == DEFAULT_RUNTIME_FAMILY_ID:
                self.seeds = list(seeds)
                self.current_epoch_requires_state_invalidation = False
            bt.logging.info(f"Loaded seeds from {path.name}")
            return seeds

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
        path: Path | None = None,
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
        path = path or self._epoch_file(epoch, family_id)
        path.parent.mkdir(parents=True, exist_ok=True)
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
        self._promote_preeval_seeds(epoch)
        self._publish_unpublished_epochs()
        self._load_or_generate_seeds(invalidate_local_state_on_regenerate=False)
        bt.logging.info(
            f"BenchmarkSeedManager aligned to backend epoch: {old_epoch} -> {self.epoch_number}"
        )
        return old_epoch

    def _epoch_start_ts(self, epoch: int) -> float:
        if epoch < EPOCH_SWITCH_NUMBER:
            return EPOCH_ANCHOR_UTC.timestamp() + (epoch - 1) * EPOCH_DURATION_SECONDS
        return EPOCH_SWITCH_TS + (epoch - EPOCH_SWITCH_NUMBER) * EPOCH_DURATION_LONG_SECONDS

    def epoch_time_range(self, epoch: int) -> tuple[datetime, datetime]:
        start = datetime.fromtimestamp(self._epoch_start_ts(epoch), tz=timezone.utc)
        end = datetime.fromtimestamp(self._epoch_start_ts(epoch + 1), tz=timezone.utc)
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

    def _seeds_for(self, family_id: str, epoch: Optional[int]) -> List[int]:
        if epoch is None or epoch == self.epoch_number:
            return self._ensure_current_family_seeds(family_id)
        return self.seeds_for_epoch(epoch, family_id)

    def get_screening_seeds(
        self,
        family_id: str = DEFAULT_RUNTIME_FAMILY_ID,
        epoch: Optional[int] = None,
    ) -> List[int]:
        return self._seeds_for(family_id, epoch)[:BENCHMARK_SCREENING_SEED_COUNT]

    def get_benchmark_seeds(
        self,
        family_id: str = DEFAULT_RUNTIME_FAMILY_ID,
        epoch: Optional[int] = None,
    ) -> List[int]:
        return self._seeds_for(family_id, epoch)[BENCHMARK_SCREENING_SEED_COUNT:]

    def get_all_seeds(
        self,
        family_id: str = DEFAULT_RUNTIME_FAMILY_ID,
        epoch: Optional[int] = None,
    ) -> List[int]:
        return list(self._seeds_for(family_id, epoch))

    def _preeval_file(self, epoch: int, family_id: str) -> Path:
        return self._seed_file(PREEVAL_SEEDS_DIR, "preeval", epoch, family_id)

    def _promote_preeval_seeds(self, epoch: int) -> None:
        """Adopt the seeds already flown for this epoch so they publish like any other."""
        for path in PREEVAL_SEEDS_DIR.glob("preeval_*.json"):
            match = _PREEVAL_FILE_RE.match(path.name)
            if match is None:
                continue
            file_epoch = int(match.group(1))
            if file_epoch < epoch:
                # An epoch the backend skipped past; its seeds can never be flown.
                path.unlink()
                continue
            if file_epoch > epoch:
                continue
            family_id = match.group(2) or DEFAULT_RUNTIME_FAMILY_ID
            target = self._epoch_file(epoch, family_id)
            if target.exists():
                path.unlink()
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            path.replace(target)
            bt.logging.info(f"Promoted pre-eval seeds for epoch {epoch} family {family_id}")

    def seeds_for_epoch(
        self,
        epoch: int,
        family_id: str = DEFAULT_RUNTIME_FAMILY_ID,
    ) -> List[int]:
        """Seeds for any epoch; a future epoch is generated and kept out of the published set."""
        if epoch <= self.epoch_number:
            return list(self._ensure_epoch_family_seeds(
                epoch, family_id, invalidate_local_state_on_regenerate=False,
            ))
        path = self._preeval_file(epoch, family_id)
        seeds = self._read_seed_file(path, epoch, family_id)
        if seeds is None:
            seeds = _generate_random_seeds(BENCHMARK_TOTAL_SEED_COUNT)
            self._save_epoch_file(epoch, family_id, seeds, published=False, path=path)
            bt.logging.info(f"Generated pre-eval seeds for epoch {epoch} family {family_id}")
        return seeds
