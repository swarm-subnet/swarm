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

"""Which submission lane an artifact belongs to.

New submissions are always code agents; the legacy model-graph lane survives
only so champions crowned before the switch stay runnable for re-evaluation.
The lane is read from the archive itself rather than carried alongside it: the
bytes are pinned to the backend's recorded hash before anything here runs, so
the archive is the authoritative description of how it must be executed.

Kept free of the ONNX dependencies so importing it costs nothing.
"""

import json
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from swarm.core.submission_policy import SUBMISSION_INTERFACE_VERSION

MODEL_GRAPH_INTERFACE_VERSION: str = "model_graph.v1"
GRAPH_MANIFEST_NAME: str = "manifest.json"
CODE_AGENT_ENTRY_POINT: str = "drone_agent.py"

RUNNABLE_INTERFACE_VERSIONS: frozenset[str] = frozenset(
    {SUBMISSION_INTERFACE_VERSION, MODEL_GRAPH_INTERFACE_VERSION}
)

_MAX_MANIFEST_BYTES: int = 1 * 1024 * 1024


def _read_graph_manifest(zip_path: Path) -> dict | None:
    """The graph manifest, or None when this is not a model-graph artifact.

    A code agent always ships ``drone_agent.py``, so its presence settles the
    question without reading anything else.
    """
    try:
        with ZipFile(zip_path) as zf:
            names = zf.namelist()
            if CODE_AGENT_ENTRY_POINT in names or GRAPH_MANIFEST_NAME not in names:
                return None
            if zf.getinfo(GRAPH_MANIFEST_NAME).file_size > _MAX_MANIFEST_BYTES:
                return None
            raw = zf.read(GRAPH_MANIFEST_NAME)
    except (BadZipFile, KeyError, OSError, ValueError):
        return None

    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None

    if not isinstance(manifest, dict):
        return None
    if str(manifest.get("model_graph_version", "")) != MODEL_GRAPH_INTERFACE_VERSION:
        return None
    return manifest


def is_model_graph_artifact(zip_path: Path) -> bool:
    """True when the archive is a legacy model-graph artifact."""
    return _read_graph_manifest(zip_path) is not None


def graph_declared_family(zip_path: Path) -> str | None:
    """Family named by the graph manifest, or None when absent."""
    manifest = _read_graph_manifest(zip_path)
    if manifest is None:
        return None
    return str(manifest.get("family_id") or "") or None
