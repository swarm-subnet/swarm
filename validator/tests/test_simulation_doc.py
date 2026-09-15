"""docs/simulation.md against the source: every family option on the runtime class and every
engine-related environment variable the simulator reads has a row. Reads the source as text, so it
runs on any wheel."""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC = REPO_ROOT / "docs" / "simulation.md"
RUNTIME = REPO_ROOT / "swarm" / "challenge_families" / "base.py"
# Where the simulator reads its switches: the simulator package, the seed manager that hands the
# engine its cache folder, and the worker module that owns the container start.
ENGINE_SOURCES = (
    REPO_ROOT / "swarm" / "core",
    REPO_ROOT / "swarm" / "validator" / "seed_manager.py",
    REPO_ROOT / "swarm" / "benchmark" / "engine_parts" / "workers.py",
)
# Class attributes that name the family rather than steer the engine.
IDENTITY_FIELDS = {"family_id", "runtime_supported"}
_ENV_READ = re.compile(r'(?:environ\.get|getenv|env_bool|environ\[|_ENV = )\(?"(SWARM_\w+)"')


def _documented() -> set[str]:
    """Every backticked identifier in the document."""
    return set(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", DOC.read_text()))


def _runtime_fields() -> list[str]:
    """The annotated class attributes of ChallengeFamilyRuntime, in source order."""
    tree = ast.parse(RUNTIME.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ChallengeFamilyRuntime":
            return [
                stmt.target.id
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
            ]
    raise AssertionError("ChallengeFamilyRuntime not found in %s" % RUNTIME)


def _engine_env_vars() -> set[str]:
    """Every SWARM_ variable the engine-facing sources read."""
    found: set[str] = set()
    for source in ENGINE_SOURCES:
        files = source.rglob("*.py") if source.is_dir() else [source]
        for path in files:
            found.update(_ENV_READ.findall(path.read_text(errors="ignore")))
    return found


def test_every_family_option_has_a_row():
    """Each annotated attribute of the family runtime, apart from its identity, appears in the document."""
    names = _documented()
    missing = [f for f in _runtime_fields() if f not in IDENTITY_FIELDS and f not in names]
    assert missing == [], f"family options without a row in docs/simulation.md: {missing}"


def test_every_engine_environment_variable_has_a_row():
    """Each SWARM_ variable read by the simulator sources appears in the document."""
    names = _documented()
    found = _engine_env_vars()
    assert found, "no environment reads found; the pattern or the sources moved"
    missing = sorted(v for v in found if v not in names)
    assert missing == [], f"environment variables without a row in docs/simulation.md: {missing}"
