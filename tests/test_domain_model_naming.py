from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_PATHS = [
    REPO_ROOT / "ARCHITECTURE.md",
    REPO_ROOT / "docs",
    REPO_ROOT / "scripts",
    REPO_ROOT / "swarm" / "core" / "maps",
    REPO_ROOT / "tests" / "sar",
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


def test_benchmark_domain_docs_do_not_use_ambiguous_family_wording():
    offenders = []
    for path in _iter_text_files():
        content = path.read_text(encoding="utf-8").lower()
        if any(phrase in content for phrase in BANNED_PHRASES):
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert offenders == []
