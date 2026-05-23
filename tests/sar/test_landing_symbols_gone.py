"""D.4 guardrail: landing-mode body and state-machine symbols must be deleted.

This pins the deletion so a future revert or merge cannot silently put the
legacy landing pipeline back into production code.
"""
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]

PROD_DIRS = ("swarm", "scripts")

FORBIDDEN_SYMBOLS = (
    "_update_landing_state",
    "_is_landing_floor_body",
    "_landing_history",
    "LANDING_PLATFORM_RADIUS",
    "LANDING_PLATFORM_HEIGHT",
)


def _iter_prod_files():
    for d in PROD_DIRS:
        for path in (ROOT / d).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            yield path


def test_landing_state_machine_symbols_removed():
    pattern = re.compile("|".join(re.escape(s) for s in FORBIDDEN_SYMBOLS))
    offenders = []
    for path in _iter_prod_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for ln, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{ln}: {line.strip()}")
    assert not offenders, "Landing-mode symbols leaked back:\n" + "\n".join(offenders)


def test_constants_module_has_no_landing_constants():
    from swarm import constants

    for name in (
        "LANDING_PLATFORM_RADIUS",
        "LANDING_PLATFORM_HEIGHT",
        "LANDING_PAD_RGBA",
        "LANDING_HISTORY_LEN",
    ):
        assert not hasattr(constants, name), f"constants.{name} should be gone"
