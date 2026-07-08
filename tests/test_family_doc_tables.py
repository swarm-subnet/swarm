import subprocess
import sys
from pathlib import Path


def test_family_io_tables_match_schema():
    repo = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "scripts/gen_family_io_tables.py", "--check"],
        cwd=repo, capture_output=True, text=True,
    )
    assert result.returncode == 0, f"family doc io-tables drifted:\n{result.stderr}"
