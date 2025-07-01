"""
Fast, incremental SHA‑256 helper.

Keeps memory use low by reading the file in fixed‑size blocks.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256sum(fp: Path, buf: int = 1 << 20) -> str:
    """
    Compute the SHA‑256 hex digest of *fp*.

    Parameters
    ----------
    fp : Path
        File to hash.
    buf : int, optional
        Block size in bytes (default = 1 MiB).

    Returns
    -------
    str
        64‑character lowercase hexadecimal digest.
    """
    h = hashlib.sha256()
    with fp.open("rb") as f:
        while blk := f.read(buf):
            h.update(blk)
    return h.hexdigest()
