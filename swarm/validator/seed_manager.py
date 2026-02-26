import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import List

import bittensor as bt

from swarm.constants import (
    BENCHMARK_VERSION,
    BENCHMARK_PUBLIC_SEED_COUNT,
    BENCHMARK_PRIVATE_SEED_COUNT,
)


class BenchmarkSeedManager:
    """Manages benchmark seeds: 1000 public (from file) + 200 private (derived from secret)."""

    def __init__(self, private_secret: str = None):
        """Initialize seed manager.

        Args:
            private_secret: Secret for deriving private seeds. If None, reads from
                           SWARM_PRIVATE_BENCHMARK_SECRET env var.
        """
        self.private_secret = private_secret or os.getenv("SWARM_PRIVATE_BENCHMARK_SECRET")
        if not self.private_secret:
            raise ValueError(
                "SWARM_PRIVATE_BENCHMARK_SECRET env var required for benchmark seeds"
            )

        self.public_seeds = self._load_public_seeds()
        self.private_seeds = self._derive_private_seeds()

        bt.logging.info(
            f"BenchmarkSeedManager initialized: {len(self.public_seeds)} public + "
            f"{len(self.private_seeds)} private seeds"
        )

    def _load_public_seeds(self) -> List[int]:
        """Load public seeds from committed artifact."""
        path = Path(__file__).parent.parent / "benchmark" / "public_seeds_v1.json"

        if not path.exists():
            raise FileNotFoundError(
                f"Public seeds artifact not found: {path}. "
                "This file must be committed to the repository."
            )

        with open(path) as f:
            data = json.load(f)

        if data.get("version") != BENCHMARK_VERSION:
            bt.logging.warning(
                f"Public seeds version mismatch: file has {data.get('version')}, "
                f"expected {BENCHMARK_VERSION}"
            )

        seeds = data.get("seeds", [])
        if len(seeds) != BENCHMARK_PUBLIC_SEED_COUNT:
            bt.logging.warning(
                f"Public seed count mismatch: {len(seeds)} != {BENCHMARK_PUBLIC_SEED_COUNT}"
            )

        return seeds

    def _derive_private_seeds(self) -> List[int]:
        """Derive private seeds using HMAC-SHA256.

        Private seeds are deterministic: same secret = same seeds across all validators.
        """
        seeds = []
        for i in range(BENCHMARK_PRIVATE_SEED_COUNT):
            msg = f"private_seed_{i}".encode()
            h = hmac.new(self.private_secret.encode(), msg, hashlib.sha256)
            seed = int.from_bytes(h.digest()[:4], 'big')
            seeds.append(seed)
        return seeds

    def get_screening_seeds(self) -> List[int]:
        """Return 200 private seeds for screening phase."""
        return self.private_seeds.copy()

    def get_public_seeds(self) -> List[int]:
        """Return 1000 public seeds for full benchmark."""
        return self.public_seeds.copy()
