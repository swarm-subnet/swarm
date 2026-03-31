"""Pytest configuration for ensuring local package imports."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


def _ensure_repo_on_syspath() -> None:
    repo_root = Path(__file__).resolve().parent
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


_ensure_repo_on_syspath()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-e2e",
        action="store_true",
        default=False,
        help="Run e2e/runtime tests that are skipped by default.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    run_e2e = config.getoption("--run-e2e") or os.getenv("SWARM_RUN_E2E") == "1"
    if run_e2e:
        return

    skip_e2e = pytest.mark.skip(
        reason="e2e/runtime tests are opt-in; use --run-e2e or SWARM_RUN_E2E=1"
    )
    for item in items:
        if item.get_closest_marker("e2e") is not None:
            item.add_marker(skip_e2e)
