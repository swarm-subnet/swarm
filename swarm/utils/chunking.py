"""Utility helpers for streaming large binary artefacts in fixed‑size pieces."""

from pathlib import Path
from typing import Iterator

# Default chunk size (256 KiB).  Feel free to tune if your network/storage profile differs.
CHUNK: int = 1 << 18   # 256 KiB


def iter_chunks(fp: Path, chunk_size: int = CHUNK) -> Iterator[bytes]:
    """
    Lazily yield *chunk_size* bytes from *fp* until EOF.

    Parameters
    ----------
    fp : Path
        Path to the file you want to stream.
    chunk_size : int, optional
        Number of bytes per chunk (defaults to ``CHUNK`` – 256 KiB).

    Yields
    ------
    bytes
        The next slice of raw file data.  Iteration ends automatically at EOF.
    """
    with fp.open("rb") as f:
        while (buf := f.read(chunk_size)):
            yield buf


__all__ = ["CHUNK", "iter_chunks"]
