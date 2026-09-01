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
