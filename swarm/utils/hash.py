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
