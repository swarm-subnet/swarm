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

"""Deriving an epoch's seed list from the key the backend serves.

Every validator on the shared scheme runs this exact arithmetic, so the whole network flies
the same 1100 maps. The encoding is pinned: a change to it is a new SEED_SCHEME_VERSION, never
a quiet edit, because the seeds it produces are the benchmark itself. The known-answer vectors
in the tests exist to make an accidental change fail loudly instead of silently moving the maps.
"""

import hashlib
import hmac
from typing import List, Optional

# The encoding these functions implement. The backend pins the same string.
SEED_SCHEME_VERSION = "v1"
# The first benchmark version scored on the shared list; below it, seeds stay per-validator.
# The backend reports the live value on /sync, so this is only the floor before the first sync.
DEFAULT_SEED_SCHEME_MIN_VERSION = "5.1.6"
_SEED_SET_ID_CHARS = 16


def parse_benchmark_version(value: str) -> tuple:
    """Tuple suitable for version ordering. Numeric parts compare numerically; non-numeric parts trail."""
    parsed: List[tuple] = []
    for part in value.split("."):
        try:
            parsed.append((0, int(part)))
        except ValueError:
            parsed.append((1, part))
    return tuple(parsed)


def uses_derived_seeds(
    benchmark_version: Optional[str],
    min_version: str = DEFAULT_SEED_SCHEME_MIN_VERSION,
) -> bool:
    """True when this benchmark version is scored on the shared per-epoch seed list."""
    if not benchmark_version:
        return False
    return parse_benchmark_version(benchmark_version) >= parse_benchmark_version(min_version)


def key_fingerprint(key: str) -> str:
    """The key's public commitment: the SHA-256 of its raw bytes, hex encoded."""
    return hashlib.sha256(bytes.fromhex(key)).hexdigest()


def seed_set_id(commitment: str, epoch_number: int, family_id: str) -> str:
    """Identity of one epoch and family's seed list, safe to send with scores.

    Built from the commitment rather than the key, so it reveals nothing about the secret.
    """
    material = f"{SEED_SCHEME_VERSION}|{commitment}|{epoch_number}|{family_id}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:_SEED_SET_ID_CHARS]


def derive_seed(key: str, epoch_number: int, family_id: str, index: int) -> int:
    """The seed at one index of an epoch's list for one family."""
    message = f"{SEED_SCHEME_VERSION}|{family_id}|{epoch_number}|{index}".encode("utf-8")
    digest = hmac.new(bytes.fromhex(key), message, hashlib.sha256).digest()
    return int.from_bytes(digest[:4], "big")


def derive_seeds(key: str, epoch_number: int, family_id: str, count: int) -> List[int]:
    """The first ``count`` seeds of an epoch's list for one family."""
    return [derive_seed(key, epoch_number, family_id, i) for i in range(count)]
