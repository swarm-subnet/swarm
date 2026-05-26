from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workspace_root = repo_root.parent
    source_path = repo_root / "swarm" / "domain_model" / "benchmark_domain_model.schema.json"
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    rendered = json.dumps(payload, indent=2) + "\n"

    target_paths = (
        workspace_root / "swarm_backend" / "swarm-backend" / "app" / "family_registry.json",
        workspace_root / "swarm_website" / "Swarm-Website" / "src" / "family_registry.json",
    )
    for target_path in target_paths:
        target_path.write_text(rendered, encoding="utf-8")
        print(f"synced {target_path}")


if __name__ == "__main__":
    main()
