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

import argparse
import json
import sys
from pathlib import Path

BACKEND_MIRROR = Path("app") / "family_registry.json"
WEBSITE_MIRROR = Path("src") / "family_registry.json"


def _validate_registry(payload: dict) -> None:
    """Refuse to mirror a registry whose enum-valued fields carry a typo.

    Every reader falls back to a default for an unknown value, so a misspelt
    visibility would silently reopen a family instead of failing loudly here."""
    enums = payload["enum_types"]
    checks = (
        ("visibility", set(enums["visibility"])),
        ("family_state", set(enums["family_state"])),
        ("emissions_state", set(enums["emissions_state"])),
    )
    for family_id, family in payload["challenge_families"].items():
        for field, allowed in checks:
            value = family.get(field)
            if value not in allowed:
                raise SystemExit(
                    f"{family_id}: {field} {value!r} is not one of {sorted(allowed)}"
                )


def _mirror_path(checkout: Path, mirror: Path) -> Path:
    target = checkout / mirror
    if not target.is_file():
        raise SystemExit(f"{target} does not exist; is {checkout} the right checkout?")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Copy the family registry from the swarm schema into the backend and "
            "website mirrors. Both checkouts must be named explicitly: a workspace "
            "can hold several worktrees of the same repo on different branches, and "
            "a directory name cannot tell them apart."
        )
    )
    parser.add_argument("--backend", type=Path, required=True, help="path to the swarm-backend checkout")
    parser.add_argument("--website", type=Path, required=True, help="path to the Swarm-Website checkout")
    parser.add_argument(
        "--check",
        action="store_true",
        help="report drift without writing; exits non-zero when a mirror is stale",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    source_path = repo_root / "swarm" / "domain_model" / "benchmark_domain_model.schema.json"
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    _validate_registry(payload)
    rendered = json.dumps(payload, indent=2) + "\n"

    target_paths = (
        _mirror_path(args.backend, BACKEND_MIRROR),
        _mirror_path(args.website, WEBSITE_MIRROR),
    )

    drifted = False
    for target_path in target_paths:
        if target_path.read_text(encoding="utf-8") == rendered:
            print(f"in sync {target_path}")
            continue
        drifted = True
        if args.check:
            print(f"DRIFTED {target_path}")
            continue
        target_path.write_text(rendered, encoding="utf-8")
        print(f"synced  {target_path}")

    return 1 if (args.check and drifted) else 0


if __name__ == "__main__":
    sys.exit(main())
