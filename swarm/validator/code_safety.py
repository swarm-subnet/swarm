import ast
import re
from pathlib import Path
from typing import Dict, List, Tuple
from zipfile import ZipFile, BadZipFile

import bittensor as bt

from swarm.constants import (
    TRAINING_CODE_MAX_SIZE,
    TRAINING_CODE_MAX_FILES,
)


class CodeSafetyValidator:

    FORBIDDEN_IMPORTS = {
        "subprocess", "os.system", "eval", "exec",
        "socket", "urllib", "requests", "__import__",
        "importlib.import_module", "pty", "atexit",
    }

    ALLOWED_EXTENSIONS = {".py", ".json", ".yaml", ".yml", ".txt", ".pkl", ".md"}

    DANGEROUS_PATTERNS = [
        b"os.system", b"subprocess.", b"socket.",
        b"urllib.", b"requests.", b"__import__",
        b"eval(", b"exec(", b"compile(",
        b"open(", b"file(",
    ]

    @classmethod
    def validate_training_code(cls, training_code_path: Path) -> Tuple[bool, float, List[str]]:
        """
        Validate training code for safety.
        Returns (is_safe, safety_score, findings)
        """
        findings = []

        try:
            if not training_code_path.exists():
                return False, 0.0, ["Training code file does not exist"]

            if training_code_path.stat().st_size > TRAINING_CODE_MAX_SIZE:
                return False, 0.0, [f"Training code exceeds {TRAINING_CODE_MAX_SIZE} bytes"]

            if not cls._is_safe_zip(training_code_path):
                return False, 0.0, ["Training code ZIP is unsafe"]

            with ZipFile(training_code_path, 'r') as zf:
                file_list = zf.namelist()

                if "train.py" not in file_list:
                    return False, 0.0, ["Missing required train.py"]

                if len(file_list) > TRAINING_CODE_MAX_FILES:
                    return False, 0.0, [f"Too many files: {len(file_list)} > {TRAINING_CODE_MAX_FILES}"]

                for filename in file_list:
                    ext = Path(filename).suffix
                    if ext and ext not in cls.ALLOWED_EXTENSIONS:
                        findings.append(f"Disallowed file extension: {filename}")

                train_py_content = zf.read("train.py")

                for pattern in cls.DANGEROUS_PATTERNS:
                    if pattern in train_py_content:
                        findings.append(f"Dangerous pattern found: {pattern.decode('utf-8', errors='ignore')}")

                try:
                    train_py_str = train_py_content.decode('utf-8')

                    tree = ast.parse(train_py_str)

                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                if any(forbidden in alias.name for forbidden in cls.FORBIDDEN_IMPORTS):
                                    findings.append(f"Forbidden import: {alias.name}")

                        elif isinstance(node, ast.ImportFrom):
                            if node.module and any(forbidden in node.module for forbidden in cls.FORBIDDEN_IMPORTS):
                                findings.append(f"Forbidden import from: {node.module}")

                except SyntaxError as e:
                    findings.append(f"Syntax error in train.py: {e}")
                except Exception as e:
                    findings.append(f"Failed to parse train.py: {e}")

            if findings:
                critical_count = sum(1 for f in findings if any(x in f.lower() for x in ["forbidden", "dangerous"]))
                if critical_count > 0:
                    return False, 0.0, findings

                safety_score = max(0.0, 1.0 - (len(findings) * 0.1))
                return True, safety_score, findings

            return True, 1.0, []

        except BadZipFile:
            return False, 0.0, ["Corrupted ZIP file"]
        except Exception as e:
            return False, 0.0, [f"Validation error: {e}"]

    @classmethod
    def _is_safe_zip(cls, path: Path) -> bool:
        """Check if ZIP file is safe (no path traversal, size limits)"""
        try:
            with ZipFile(path) as zf:
                total_size = 0
                for info in zf.infolist():
                    if info.filename.startswith(("/", "\\")) or ".." in Path(info.filename).parts:
                        return False
                    total_size += info.file_size
                    if total_size > TRAINING_CODE_MAX_SIZE:
                        return False
            return True
        except Exception:
            return False
