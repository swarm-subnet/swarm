#!/usr/bin/env python3
"""Run the tests the miner manifest names.

The workflow calls this rather than listing selectors of its own, so the manifest
stays the only place the miner test set is written down.
"""
from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

MANIFEST = Path(__file__).resolve().parent / "miner_tests.toml"


def selectors(manifest_path: Path = MANIFEST) -> list[str]:
    with manifest_path.open("rb") as fh:
        files = tomllib.load(fh)["files"]
    named = [s for key, group in files.items() if key != "extra_tests" for s in group]
    ordered = dict.fromkeys(named + list(files.get("extra_tests", [])))
    return list(ordered)


def main() -> int:
    chosen = selectors()
    if not chosen:
        print("the manifest names no tests", file=sys.stderr)
        return 1
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-n4", "--dist", "worksteal",
         "-p", "no:cacheprovider", *chosen]
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
