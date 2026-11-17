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
import io
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
# Every entry is unpacked on each batch, so the count is bounded even when the bytes are not.
MAX_ARCHIVE_ENTRIES: int = 1000

# Compiled code is refused by its header, so no name or nested archive hides it.
# Left out of profile_digest() so stored scores stay valid.
NATIVE_HEADER_PROBE_BYTES: int = 4096
NESTED_ARCHIVE_MAX_DEPTH: int = 3
NESTED_ARCHIVE_MAX_BYTES: int = MAX_UNCOMPRESSED_BYTES
NESTED_ARCHIVE_MAX_ENTRIES: int = 10 * MAX_ARCHIVE_ENTRIES
NESTED_DIRECTORY_MAX_BYTES: int = 1024 * 1024

_ELF_MAGIC = b"\x7fELF"
# thin 32- and 64-bit Mach-O in both byte orders, then the fat headers no class file shares
_MACHO_MAGICS = (
    b"\xfe\xed\xfa\xce",
    b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf",
    b"\xcf\xfa\xed\xfe",
    b"\xca\xfe\xba\xbf",
    b"\xbf\xba\xfe\xca",
    b"\xbe\xba\xfe\xca",
)
_MACHO_FAT_MAGIC = b"\xca\xfe\xba\xbe"
# Java class files share the fat magic, but their major version is never below 45.
_JAVA_CLASS_MIN_MAJOR = 45
_PE_OFFSET_FIELD = slice(0x3C, 0x40)
_PE_SIGNATURE = b"PE\x00\x00"
_ZIP_LOCAL_MAGIC = b"PK\x03\x04"
_ZIP_END_MAGIC = b"PK\x05\x06"
_ZIP64_LOCATOR_MAGIC = b"PK\x06\x07"
_ZIP_END_RECORD_BYTES = 22
_ZIP64_LOCATOR_BYTES = 20
_ZIP_END_SEARCH_BYTES = _ZIP_END_RECORD_BYTES + 0xFFFF


def profile_digest() -> str:
    """Fingerprint of the admission rules a score was produced under.

    Any change to what the lane accepts changes this digest, so scores stay
    attributable to the exact contract that admitted the submission. The
    native-code check and the entry cap are left out, so stored scores stay valid.
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
    """Reject path traversal, symlinks, corrupt archives, zip bombs, or too many entries."""
    try:
        with ZipFile(zip_path) as zf:
            infos = zf.infolist()
            if len(infos) > MAX_ARCHIVE_ENTRIES:
                return False, f"Too many files ({len(infos)}, limit {MAX_ARCHIVE_ENTRIES})"
            total = 0
            for info in infos:
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


def is_native_executable(head: bytes) -> bool:
    """True when ``head``, the first bytes of a file, opens with an ELF, Mach-O or PE header."""
    if head.startswith(_ELF_MAGIC) or head.startswith(_MACHO_MAGICS):
        return True
    if head.startswith(_MACHO_FAT_MAGIC):
        return int.from_bytes(head[4:8], "big") < _JAVA_CLASS_MIN_MAJOR
    if head.startswith(b"MZ") and len(head) >= _PE_OFFSET_FIELD.stop:
        offset = int.from_bytes(head[_PE_OFFSET_FIELD], "little")
        return head[offset : offset + len(_PE_SIGNATURE)] == _PE_SIGNATURE
    return False


class _NestedScanBudget:
    """What the native-code scan may still spend on archives inside the archive."""

    def __init__(self) -> None:
        """Start with the full byte and entry allowance."""
        self.bytes = NESTED_ARCHIVE_MAX_BYTES
        self.entries = NESTED_ARCHIVE_MAX_ENTRIES


def _directory_is_small(data: bytes) -> bool:
    """True when an in-memory zip's central directory is small enough to parse cheaply."""
    end = data.rfind(_ZIP_END_MAGIC, max(0, len(data) - _ZIP_END_SEARCH_BYTES))
    if end < 0 or end + _ZIP_END_RECORD_BYTES > len(data):
        return False
    # a zip64 record can declare a larger directory than the plain end record shows
    locator = end - _ZIP64_LOCATOR_BYTES
    if locator >= 0 and data[locator : locator + 4] == _ZIP64_LOCATOR_MAGIC:
        return False
    return int.from_bytes(data[end + 12 : end + 16], "little") <= NESTED_DIRECTORY_MAX_BYTES


def _scan_nested(
    data: bytes, prefix: str, depth: int, budget: _NestedScanBudget, found: list[str]
) -> None:
    """Scan an in-memory archive; an unreadable one is left alone, as nothing executes from it."""
    if not _directory_is_small(data):
        return
    try:
        with ZipFile(io.BytesIO(data)) as zf:
            _scan_entries(zf, prefix, depth, budget, found)
    except Exception:
        return


def _scan_entries(
    zf: ZipFile, prefix: str, depth: int, budget: _NestedScanBudget, found: list[str]
) -> None:
    """Record every entry of ``zf`` holding native code, nested archives included within budget."""
    for info in zf.infolist():
        if info.is_dir():
            continue
        if budget.entries <= 0:
            return
        budget.entries -= 1
        name = f"{prefix}{info.filename}"
        with zf.open(info) as handle:
            head = handle.read(NATIVE_HEADER_PROBE_BYTES)
        if is_native_executable(head):
            found.append(name)
        elif (
            head.startswith(_ZIP_LOCAL_MAGIC)
            and depth < NESTED_ARCHIVE_MAX_DEPTH
            and info.file_size <= budget.bytes
        ):
            budget.bytes -= info.file_size
            _scan_nested(zf.read(info), f"{name}!", depth + 1, budget, found)


def check_native_code(zip_path: Path) -> tuple[bool, str]:
    """Reject any entry, here or in a nested zip, that is a compiled executable or library."""
    found: list[str] = []
    try:
        with ZipFile(zip_path) as zf:
            _scan_entries(zf, "", 0, _NestedScanBudget(), found)
    except BadZipFile:
        return False, "Corrupted ZIP archive"
    except Exception as exc:
        return False, f"ZIP inspection error: {exc}"

    if found:
        return False, f"native_code:{','.join(sorted(set(found)))}"
    return True, "ok"


def validate_submission_zip(zip_path: Path) -> tuple[bool, str]:
    """Run safety, structure, then native-code checks against ``zip_path``."""
    ok, reason = check_safety(zip_path)
    if not ok:
        return ok, reason
    ok, reason = check_structure(zip_path)
    if not ok:
        return ok, reason
    return check_native_code(zip_path)
