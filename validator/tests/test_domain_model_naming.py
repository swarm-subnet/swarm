from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_PATHS = [
    REPO_ROOT / "validator" / "docs",
    REPO_ROOT / "ARCHITECTURE.md",
    REPO_ROOT / "docs",
    REPO_ROOT / "validator" / "scripts",
    REPO_ROOT / "swarm" / "core" / "maps",
    REPO_ROOT / "validator" / "tests" / "sar",
]
_LEGACY_MAP_FAMILY = " ".join(("map", "family"))
_LEGACY_MAP_FAMILY_HYPHEN = "-".join(("map", "family"))
BANNED_PHRASES = (_LEGACY_MAP_FAMILY, _LEGACY_MAP_FAMILY_HYPHEN)


def _iter_text_files():
    for base_path in SCAN_PATHS:
        if base_path.is_file():
            yield base_path
            continue
        for path in base_path.rglob("*"):
            if path.is_dir():
                continue
            if path.suffix.lower() not in {".md", ".py", ".txt"}:
                continue
            yield path


@pytest.mark.parametrize("base", SCAN_PATHS, ids=lambda p: str(p.name))
def test_each_scanned_root_contributes_files(base):
    """Per root, not in aggregate: one root going empty would otherwise hide
    behind the others still finding files."""
    assert base.exists(), f"{base} is scanned but not there"
    if base.is_file():
        return
    found = [p for p in base.rglob("*") if p.suffix.lower() in {".md", ".py", ".txt"}]
    assert found, f"{base} contributed no files to the scan"


def test_benchmark_domain_docs_do_not_use_ambiguous_family_wording():
    offenders = []
    scanned = 0
    for path in _iter_text_files():
        scanned += 1
        content = path.read_text(encoding="utf-8").lower()
        if any(phrase in content for phrase in BANNED_PHRASES):
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert scanned, "scanned no files at all, so finding no offenders proves nothing"
    assert offenders == []
