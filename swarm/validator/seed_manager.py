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

"""Per-epoch benchmark seeds: generation, storage, publication and the engine's collision-tree cache folder."""

import json
import os
import random
import re
import shutil
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
from swarm.validator.seed_scheme import (
    DEFAULT_SEED_SCHEME_MIN_VERSION,
    SEED_SCHEME_VERSION,
    derive_seeds,
    key_fingerprint,
    seed_set_id,
    uses_derived_seeds,
)

STATE_DIR = Path(__file__).parent.parent.parent / "state"
EPOCH_SEEDS_DIR = STATE_DIR / "epoch_seeds"
# Kept out of the epoch_*.json namespace so a restart cannot read it as the rollover having happened.
PREEVAL_SEEDS_DIR = STATE_DIR / "preeval_seeds"
# Where the engine reads the collision trees it saved per mesh and scale; the folder follows the epoch.
BVH_CACHE_ENV = "SWARM_BVH_CACHE_DIR"

_MAX_SEED = 2**32 - 1
_EPOCH_FILE_RE = re.compile(r"^epoch_(\d+)(?:__(.+))?\.json$")
_PREEVAL_FILE_RE = re.compile(r"^preeval_(\d+)(?:__(.+))?\.json$")
# Written into every derived seed file. A file without it predates the shared scheme.
_RANDOM_SCHEME = "random_per_validator"


class SeedsNotReady(RuntimeError):
    """The shared scheme is live but this epoch's key has not arrived yet.

    Raised rather than falling back to random seeds: a silent fallback is exactly the
    divergence this whole mechanism exists to remove, and it would be invisible.
    """


def _generate_random_seeds(count: int) -> List[int]:
    """Fresh seeds from the system's random source."""
    rng = random.SystemRandom()
    return [rng.randint(0, _MAX_SEED) for _ in range(count)]


class BenchmarkSeedManager:
    """Per-epoch seed management with family-specific seed sets.

    Benchmark epoch remains global across the network. Within a given epoch,
    each challenge family owns an independent seed set and publication record.
    """

    def __init__(self) -> None:
        """Recover the latest local epoch and its seeds from disk, generating them when absent."""
        EPOCH_SEEDS_DIR.mkdir(parents=True, exist_ok=True)
        self.seeds: List[int] = []
        self.current_epoch_requires_state_invalidation = False
        self._pending_publications: List[dict] = []
        self._family_seeds: Dict[str, List[int]] = {}
        # Set from /sync, which is created after this manager, so it starts keyless.
        self._scheme_min_version = DEFAULT_SEED_SCHEME_MIN_VERSION
        self._epoch_keys: Dict[int, str] = {}

        self.epoch_number = self._latest_local_epoch()
        if self.epoch_number > 0:
            self._publish_unpublished_epochs()
            try:
                self._load_or_generate_seeds(invalidate_local_state_on_regenerate=True)
            except SeedsNotReady:
                # Constructed before the backend client exists, so under the shared scheme
                # there is no key yet. The first sync fills it in.
                bt.logging.info("Seed manager waiting for this epoch's key from the backend")

        bt.logging.info(
            f"BenchmarkSeedManager: epoch={self.epoch_number}, "
            f"{len(self.seeds)} seeds for {DEFAULT_RUNTIME_FAMILY_ID} "
            f"({BENCHMARK_SCREENING_SEED_COUNT} screening + "
            f"{BENCHMARK_TOTAL_SEED_COUNT - BENCHMARK_SCREENING_SEED_COUNT} benchmark)"
        )

    def apply_backend_scheme(
        self,
        min_version: Optional[str],
        epoch_keys: Optional[Dict[str, dict]],
    ) -> None:
        """Adopt the seed scheme and epoch keys reported by ``/sync``.

        A key arriving for an epoch whose cached list was built without it drops that cache,
        because checking the file alone would leave a stale list live in memory.
        """
        if min_version:
            self._scheme_min_version = str(min_version)
        for raw_epoch, entry in (epoch_keys or {}).items():
            try:
                epoch = int(raw_epoch)
                key = str(entry["key"])
            except (KeyError, TypeError, ValueError):
                continue
            if entry.get("commitment") and key_fingerprint(key) != entry["commitment"]:
                # The commitment is published before the key is used; a key that does not
                # match it is not the one the network committed to.
                bt.logging.error(
                    f"Refusing the epoch {epoch} key: it does not match its published commitment"
                )
                continue
            if self._epoch_keys.get(epoch) != key:
                self._epoch_keys[epoch] = key
                if epoch == self.epoch_number:
                    self._family_seeds = {}
                    self.seeds = []

    def uses_derived_seeds(self) -> bool:
        """Whether this validator's own version is scored on the shared seed list."""
        return uses_derived_seeds(BENCHMARK_VERSION, self._scheme_min_version)

    def seeds_ready(self, epoch: Optional[int] = None) -> bool:
        """Whether seeds can be built for an epoch, so the caller knows not to take work."""
        if not self.uses_derived_seeds():
            return True
        target = self.epoch_number if epoch is None else epoch
        return target > 0 and target in self._epoch_keys

    def seed_set_id_for(
        self,
        epoch: Optional[int] = None,
        family_id: str = DEFAULT_RUNTIME_FAMILY_ID,
    ) -> Optional[str]:
        """Identity of the list a phase is flying, sent with its scores; None under the random scheme."""
        target = self.epoch_number if epoch is None else epoch
        key = self._epoch_keys.get(target)
        if not self.uses_derived_seeds() or key is None:
            return None
        return seed_set_id(key_fingerprint(key), target, family_id)

    def _build_seeds(self, epoch: int, family_id: str) -> List[int]:
        """The epoch's seed list for one family: derived from the key, or rolled locally.

        Under the shared scheme a missing key is an error, never a quiet roll of the dice.
        """
        if not self.uses_derived_seeds():
            return _generate_random_seeds(BENCHMARK_TOTAL_SEED_COUNT)
        key = self._epoch_keys.get(epoch)
        if key is None:
            raise SeedsNotReady(f"no key held for epoch {epoch}")
        return derive_seeds(key, epoch, family_id, BENCHMARK_TOTAL_SEED_COUNT)

    def _file_stamp(self, epoch: int) -> Tuple[str, Optional[str]]:
        """The scheme name and key fingerprint a seed file for this epoch must carry."""
        key = self._epoch_keys.get(epoch)
        if not self.uses_derived_seeds() or key is None:
            return _RANDOM_SCHEME, None
        return SEED_SCHEME_VERSION, key_fingerprint(key)

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
        """Path of a seed file for one epoch and family under the given folder."""
        suffix = "" if family_id == DEFAULT_RUNTIME_FAMILY_ID else f"__{family_id}"
        return directory / f"{prefix}_{epoch}{suffix}.json"

    def _epoch_file(self, epoch: int, family_id: str = DEFAULT_RUNTIME_FAMILY_ID) -> Path:
        """Path of the published-seeds file for one epoch and family."""
        return self._seed_file(EPOCH_SEEDS_DIR, "epoch", epoch, family_id)

    def _parse_epoch_file_path(self, path: Path) -> Tuple[int, str] | None:
        """Epoch number and family id encoded in a seed file name, or None for a foreign file."""
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
        """The seed file's JSON, with the family id filled in for old files."""
        data = json.loads(path.read_text())
        data.setdefault("family_id", DEFAULT_RUNTIME_FAMILY_ID)
        return data

    def _queue_pending_publication(self, data: dict) -> None:
        """Remember a seed set that still has to be published, once per epoch and family."""
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
        """Seeds from a stored file, or None when it is absent, corrupt or for another scope.

        The scheme and key stamps are part of the scope. ``state/`` survives an update, so
        without them a validator that upgrades mid-epoch would keep flying the seeds it rolled
        before the switch, and nothing would say so.
        """
        if not path.exists():
            return None
        try:
            data = self._load_epoch_payload(path)
            scheme, fingerprint = self._file_stamp(epoch)
            if (
                data.get("epoch_number") == epoch
                and str(data.get("family_id") or DEFAULT_RUNTIME_FAMILY_ID) == family_id
                and str(data.get("scheme") or _RANDOM_SCHEME) == scheme
                and data.get("key_fingerprint") == fingerprint
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
        """Seeds for one epoch and family, read from disk or generated and saved."""
        path = self._epoch_file(epoch, family_id)
        seeds = self._read_seed_file(path, epoch, family_id)
        if seeds is not None:
            self._family_seeds[family_id] = seeds
            if family_id == DEFAULT_RUNTIME_FAMILY_ID:
                self.seeds = list(seeds)
                self.current_epoch_requires_state_invalidation = False
            bt.logging.info(f"Loaded seeds from {path.name}")
            return seeds

        seeds = self._build_seeds(epoch, family_id)
        self._family_seeds[family_id] = seeds
        if family_id == DEFAULT_RUNTIME_FAMILY_ID:
            self.seeds = list(seeds)
            self.current_epoch_requires_state_invalidation = (
                invalidate_local_state_on_regenerate
            )
        self._save_epoch_file(epoch, family_id, seeds, published=False)
        source = "derived" if self.uses_derived_seeds() else "random"
        bt.logging.info(
            f"Built {len(seeds)} {source} seeds for epoch {epoch} family {family_id}"
        )
        return seeds

    def _load_or_generate_seeds(
        self,
        *,
        invalidate_local_state_on_regenerate: bool,
    ) -> None:
        """Load or generate the current epoch's seeds and activate the epoch's cache folder."""
        self._ensure_epoch_family_seeds(
            self.epoch_number,
            DEFAULT_RUNTIME_FAMILY_ID,
            invalidate_local_state_on_regenerate=invalidate_local_state_on_regenerate,
        )
        self._activate_bvh_cache()

    def _activate_bvh_cache(self) -> None:
        """Point the engine's collision-tree cache at this epoch's folder and drop the older epochs' folders."""
        cache_root = STATE_DIR / "bvh_cache"
        epoch_dir = cache_root / f"epoch_{self.epoch_number}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        for old in cache_root.iterdir():
            if old.is_dir() and old != epoch_dir:
                shutil.rmtree(old, ignore_errors=True)
        os.environ[BVH_CACHE_ENV] = str(epoch_dir)

    def _save_epoch_file(
        self,
        epoch: int,
        family_id: str,
        seeds: List[int],
        published: bool,
        path: Path | None = None,
    ) -> None:
        """Write a seed set with its epoch window and publication state, atomically."""
        start, end = self.epoch_time_range(epoch)
        scheme, fingerprint = self._file_stamp(epoch)
        data = {
            "epoch_number": epoch,
            "family_id": family_id,
            "started_at": start.isoformat(),
            "ended_at": end.isoformat(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "seed_count": len(seeds),
            "benchmark_version": BENCHMARK_VERSION,
            "scheme": scheme,
            "key_fingerprint": fingerprint,
            "published": published,
            "seeds": seeds,
        }
        path = path or self._epoch_file(epoch, family_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, separators=(",", ":")))
        tmp.replace(path)

    def _publish_unpublished_epochs(self) -> None:
        """Queue every stored seed set from an earlier epoch that was never published."""
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
        """Seed sets waiting to be published, optionally for one family."""
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
        """Record that an epoch's seeds are published and drop them from the pending queue."""
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
        try:
            self._load_or_generate_seeds(invalidate_local_state_on_regenerate=False)
        except SeedsNotReady:
            # The rollover reached us before the new epoch's key did; the next sync builds them.
            bt.logging.info(f"Waiting for the epoch {epoch} key before building its seeds")
        bt.logging.info(
            f"BenchmarkSeedManager aligned to backend epoch: {old_epoch} -> {self.epoch_number}"
        )
        return old_epoch

    def _epoch_start_ts(self, epoch: int) -> float:
        """Unix time at which the epoch begins, on either side of the schedule switch."""
        if epoch < EPOCH_SWITCH_NUMBER:
            return EPOCH_ANCHOR_UTC.timestamp() + (epoch - 1) * EPOCH_DURATION_SECONDS
        return EPOCH_SWITCH_TS + (epoch - EPOCH_SWITCH_NUMBER) * EPOCH_DURATION_LONG_SECONDS

    def epoch_time_range(self, epoch: int) -> tuple[datetime, datetime]:
        """Start and end of an epoch as aware datetimes."""
        start = datetime.fromtimestamp(self._epoch_start_ts(epoch), tz=timezone.utc)
        end = datetime.fromtimestamp(self._epoch_start_ts(epoch + 1), tz=timezone.utc)
        return start, end

    def seconds_until_epoch_end(self) -> float:
        """Seconds left in the current epoch, never negative."""
        _, end = self.epoch_time_range(self.epoch_number)
        return max(0.0, end.timestamp() - time.time())

    def _ensure_current_family_seeds(
        self,
        family_id: str = DEFAULT_RUNTIME_FAMILY_ID,
    ) -> List[int]:
        """Seeds of the current epoch for one family, loaded on first use."""
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
        """Seeds for a family in the current epoch or in any other epoch."""
        if epoch is None or epoch == self.epoch_number:
            return self._ensure_current_family_seeds(family_id)
        return self.seeds_for_epoch(epoch, family_id)

    def get_screening_seeds(
        self,
        family_id: str = DEFAULT_RUNTIME_FAMILY_ID,
        epoch: Optional[int] = None,
    ) -> List[int]:
        """The screening slice of a family's seeds."""
        return self._seeds_for(family_id, epoch)[:BENCHMARK_SCREENING_SEED_COUNT]

    def get_benchmark_seeds(
        self,
        family_id: str = DEFAULT_RUNTIME_FAMILY_ID,
        epoch: Optional[int] = None,
    ) -> List[int]:
        """The benchmark slice of a family's seeds."""
        return self._seeds_for(family_id, epoch)[BENCHMARK_SCREENING_SEED_COUNT:]

    def get_all_seeds(
        self,
        family_id: str = DEFAULT_RUNTIME_FAMILY_ID,
        epoch: Optional[int] = None,
    ) -> List[int]:
        """Every seed of a family for the epoch."""
        return list(self._seeds_for(family_id, epoch))

    def _preeval_file(self, epoch: int, family_id: str) -> Path:
        """Path of the pre-evaluation seed file for one epoch and family."""
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
            seeds = self._build_seeds(epoch, family_id)
            self._save_epoch_file(epoch, family_id, seeds, published=False, path=path)
            bt.logging.info(f"Built pre-eval seeds for epoch {epoch} family {family_id}")
        return seeds
