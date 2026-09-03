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

from __future__ import annotations

import hashlib

from swarm.utils.hash import sha256sum


def test_sha256sum_matches_hashlib_for_small_file(tmp_path):
    fp = tmp_path / "payload.bin"
    data = b"swarm-subnet-test-data"
    fp.write_bytes(data)

    expected = hashlib.sha256(data).hexdigest()
    assert sha256sum(fp) == expected


def test_sha256sum_matches_hashlib_for_chunked_reads(tmp_path):
    fp = tmp_path / "large.bin"
    data = b"0123456789abcdef" * 10000
    fp.write_bytes(data)

    expected = hashlib.sha256(data).hexdigest()
    assert sha256sum(fp, buf=64) == expected
