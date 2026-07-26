from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_MIRROR = Path("app") / "family_registry.json"
WEBSITE_MIRROR = Path("src") / "family_registry.json"


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

    repo_root = Path(__file__).resolve().parents[1]
    source_path = repo_root / "swarm" / "domain_model" / "benchmark_domain_model.schema.json"
    payload = json.loads(source_path.read_text(encoding="utf-8"))
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
