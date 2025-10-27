"""Pytest configuration for ensuring local package imports."""
from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_on_syspath() -> None:
    repo_root = Path(__file__).resolve().parent
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


_ensure_repo_on_syspath()
