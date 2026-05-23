from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO = Path(__file__).resolve().parents[2]


_DOC_FILES = [
    _REPO / "README.md",
    _REPO / "docs" / "miner.md",
    _REPO / "ARCHITECTURE.md",
]

_FORBIDDEN = [
    re.compile(r"landing platform", re.IGNORECASE),
    re.compile(r"land on the platform", re.IGNORECASE),
    re.compile(r"moving platforms?", re.IGNORECASE),
]


@pytest.mark.parametrize("path", _DOC_FILES, ids=lambda p: p.name)
def test_no_landing_references(path):
    if not path.is_file():
        pytest.skip(f"missing doc file {path}")
    text = path.read_text()
    for pattern in _FORBIDDEN:
        match = pattern.search(text)
        assert not match, f"{path.name} still contains {pattern.pattern}: {match.group(0) if match else ''}"
