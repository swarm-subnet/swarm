#!/usr/bin/env python3
"""Redirect — benchmark engine moved to swarm.benchmark.engine."""
import sys as _sys
from pathlib import Path as _Path

_repo_root = str(_Path(__file__).resolve().parent.parent)
if _repo_root not in _sys.path:
    _sys.path.insert(0, _repo_root)

from swarm.benchmark import engine as _engine

_mod = _sys.modules[__name__]
for _attr in dir(_engine):
    if not _attr.startswith("__"):
        setattr(_mod, _attr, getattr(_engine, _attr))

main = _engine.main

if __name__ == "__main__":
    main()
