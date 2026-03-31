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
