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

"""Static rules for what counts as a valid model submission.

Single source of truth shared by backend chain scanning and validator
verification. The matching backend copy lives at
``app/submission_policy.py`` in the swarm-backend repo and must be kept
in sync — changing a rule here without mirroring it there will let the
backend accept submissions the validator cannot run. ``miner/src/miner.py``
inlines these rules for its pre-commit check and must be updated too.
"""

import hashlib
import stat
from pathlib import Path
from zipfile import BadZipFile, ZipFile

SUBMISSION_INTERFACE_VERSION: str = "submission_zip.v1"
EXECUTION_PROFILE_ID: str = "swarm.code-agent.cpu.v1"
RUNNER_ABI: str = "agent_rpc.v1"
VALIDATOR_CONTRACT: str = "agent_rpc.v1"

REQUIRED_ROOT_FILES: tuple[str, ...] = ("drone_agent.py",)
FORBIDDEN_SUFFIXES: tuple[str, ...] = (".exe", ".so", ".dll", ".sh", ".bat", ".pyc")
MAX_UNCOMPRESSED_BYTES: int = 50 * 1024 * 1024


def profile_digest() -> str:
    """Fingerprint of the admission rules a score was produced under.

    Any change to what the lane accepts changes this digest, so scores stay
    attributable to the exact contract that admitted the submission.
    """
    canonical = "|".join(
        (
            EXECUTION_PROFILE_ID,
            SUBMISSION_INTERFACE_VERSION,
            RUNNER_ABI,
            ",".join(REQUIRED_ROOT_FILES),
            ",".join(FORBIDDEN_SUFFIXES),
            str(MAX_UNCOMPRESSED_BYTES),
        )
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


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
