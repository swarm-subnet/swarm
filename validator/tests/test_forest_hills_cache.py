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

"""The forest hills mesh cache is published only once it is completely written."""

from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from swarm.core.forest_generator_parts import hills


class _FailingWriter:
    """File stand-in that writes through to the real file and raises after a set number of writes."""

    def __init__(self, handle, writes_before_failure: int):
        """Wrap an open handle and arm the failure counter."""
        self._handle = handle
        self._left = writes_before_failure

    def write(self, data):
        """Write through until the counter runs out, then fail as a killed worker would."""
        if self._left <= 0:
            raise OSError("worker died mid-write")
        self._left -= 1
        return self._handle.write(data)

    def __enter__(self):
        """Enter the wrapped handle's context."""
        return self

    def __exit__(self, *exc):
        """Close the wrapped handle whatever happened."""
        self._handle.close()
        return False


@pytest.fixture
def hills_cache(tmp_path, monkeypatch):
    """Point the hills cache at an empty folder and skip when the hill assets are not installed."""
    if not hills._hill_obj_candidates():
        pytest.skip("hill mesh assets are not installed")
    monkeypatch.setattr(hills, "HILLS_MESH_CACHE_DIR", str(tmp_path))
    return tmp_path


def test_interrupted_write_never_leaves_a_mesh_under_the_final_name(hills_cache, monkeypatch):
    """A write that dies partway leaves nothing a parallel worker could load as the finished mesh."""

    def _open(path, mode="r", *args, **kwargs):
        """Fail writes partway, pass reads through."""
        handle = builtins.open(path, mode, *args, **kwargs)
        return _FailingWriter(handle, 50) if "w" in mode else handle

    monkeypatch.setattr(hills, "open", _open, raising=False)
    with pytest.raises(OSError):
        hills._ensure_merged_hills_obj()

    assert not Path(hills._merged_hills_obj_path()).exists()
    assert not list(hills_cache.iterdir()), "a partial file was left in the cache folder"


def test_completed_write_publishes_the_whole_mesh(hills_cache):
    """A finished write lands under the final name, with faces and no temp file beside it."""
    path = Path(hills._ensure_merged_hills_obj())

    text = path.read_text()
    assert path.parent == hills_cache
    assert "\nf " in text and text.endswith("\n")
    assert [p.name for p in hills_cache.iterdir()] == [path.name]
