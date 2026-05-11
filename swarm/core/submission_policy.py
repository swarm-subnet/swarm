"""Static rules for what counts as a valid model submission.

Single source of truth shared by backend chain scanning and validator
verification. The matching backend copy lives at
``app/submission_policy.py`` in the swarm-backend repo and must be kept
in sync — changing a rule here without mirroring it there will let the
backend accept submissions the validator cannot run.
"""

import stat
from pathlib import Path
from zipfile import BadZipFile, ZipFile

REQUIRED_ROOT_FILES: tuple[str, ...] = ("drone_agent.py",)
FORBIDDEN_SUFFIXES: tuple[str, ...] = (".exe", ".so", ".dll", ".sh", ".bat", ".pyc")
MAX_UNCOMPRESSED_BYTES: int = 50 * 1024 * 1024

STRUCTURE_FAILURE_REASON_PREFIXES: tuple[str, ...] = (
    "missing_required_file:",
    "forbidden_suffix:",
)


def check_safety(
    zip_path: Path, *, max_uncompressed: int = MAX_UNCOMPRESSED_BYTES
) -> tuple[bool, str]:
    """Reject path traversal, symlinks, corrupt archives, or zip bombs."""
    try:
        with ZipFile(zip_path) as zf:
            total = 0
            for info in zf.infolist():
                name = info.filename
                if name.startswith(("/", "\\")) or ".." in Path(name).parts:
                    return False, f"Path traversal detected: {name}"
                if stat.S_ISLNK(info.external_attr >> 16):
                    return False, f"Symlink not allowed: {name}"
                total += info.file_size
                if total > max_uncompressed:
                    return False, f"Uncompressed size too large ({total / 1e6:.1f} MB)"
            return True, "ok"
    except BadZipFile:
        return False, "Corrupted ZIP archive"
    except Exception as exc:
        return False, f"ZIP inspection error: {exc}"


def check_structure(zip_path: Path) -> tuple[bool, str]:
    """Reject missing required files or forbidden suffixes."""
    try:
        with ZipFile(zip_path) as zf:
            names = zf.namelist()
    except BadZipFile:
        return False, "Corrupted ZIP archive"
    except Exception as exc:
        return False, f"ZIP inspection error: {exc}"

    for required in REQUIRED_ROOT_FILES:
        if required not in names:
            return False, f"missing_required_file:{required}"

    forbidden = sorted({n for n in names if n.endswith(FORBIDDEN_SUFFIXES)})
    if forbidden:
        return False, f"forbidden_suffix:{','.join(forbidden)}"

    return True, "ok"


def validate_submission_zip(zip_path: Path) -> tuple[bool, str]:
    """Run safety then structure checks against ``zip_path``."""
    ok, reason = check_safety(zip_path)
    if not ok:
        return ok, reason
    return check_structure(zip_path)
