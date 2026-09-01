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

from __future__ import annotations

from pathlib import Path

from swarm.core.forest_generator_parts import geometry as forest_geometry


def test_material_visual_obj_paths_generate_split_objs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        forest_geometry.tempfile, "gettempdir", lambda: str(tmp_path)
    )
    obj_path = Path(
        "swarm/assets/maps/forest/quaternius_ultimate_nature/normal/CommonTree_1.obj"
    )

    split_paths = forest_geometry._material_visual_obj_paths(str(obj_path))

    assert set(split_paths) == {"Green", "Wood"}
    for path in split_paths.values():
        split_obj = Path(path)
        assert split_obj.exists()
        contents = split_obj.read_text(encoding="utf-8")
        assert "\nv " in contents
        assert "\nvn " in contents
        assert "\nf " in contents
