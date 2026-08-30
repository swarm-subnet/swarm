#!/usr/bin/env python3
"""Decide whether a pull request runs the miner tests alone or the whole suite.

The manifest lists exact paths, so anything it does not name - a new file, a
shared module, a rename out of the miner side - runs the whole suite. Every
uncertainty resolves the same way: running everything is only slow, while
running the miner tests alone on a change they do not cover would report a pass
nobody checked.

Usage:
    classify_changes.py --manifest ci/miner_tests.toml --diff <file with git
    diff --name-status -z output>

Prints ``miner`` or ``full``.
"""
from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path


def paths_from_name_status(payload: bytes) -> list[str] | None:
    """Every path in a ``git diff --name-status -z -M`` record set.

    Only edits to files that already existed can take the short lane. Adding,
    deleting, renaming or turning a file into a symlink all change what the
    listed tests were written against, and none of them is worth the minutes
    the whole suite costs, so the payload is refused and everything runs.
    """
    fields = [
        f.decode(errors="surrogateescape") for f in payload.split(b"\0") if f != b""
    ]
    paths: list[str] = []
    i = 0
    while i < len(fields):
        status = fields[i]
        if status != "M" or i + 1 >= len(fields):
            return None
        paths.append(fields[i + 1])
        i += 2
    return paths or None


def within_bounds(path: str, bounds: dict) -> bool:
    return path in set(bounds.get("exact", ())) or any(
        path.startswith(p) for p in bounds.get("prefixes", ())
    )


def classify(paths: list[str] | None, manifest: dict) -> str:
    if not paths:
        return "full"
    listed = set(manifest.get("files", {}))
    protected = tuple(manifest.get("protected", ()))
    for path in paths:
        if path.startswith(protected) or path not in listed:
            return "full"
    return "miner"


def load_manifest(path: Path) -> dict:
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    files = {k: v for k, v in raw.get("files", {}).items() if isinstance(v, list) and k != "extra_tests"}
    return {
        "files": files,
        "extra_tests": raw.get("files", {}).get("extra_tests", []),
        "bounds": raw.get("bounds", {}),
        "protected": raw.get("bounds", {}).get("protected", []),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--diff", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
        payload = args.diff.read_bytes()
    except (OSError, tomllib.TOMLDecodeError):
        print("full")
        return 0

    print(classify(paths_from_name_status(payload), manifest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
