from __future__ import annotations

import os

import pytest

from swarm.core.warehouse.shared import (
    MeshKitLoader,
    first_existing_path,
    normalize_mtl_texture_paths,
)


def test_first_existing_path_returns_first_hit(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    b.write_text("x")
    found = first_existing_path([str(a), str(b)])
    assert found == os.path.abspath(b)


def test_mesh_loader_scale_xyz_and_asset_path(tmp_path):
    obj_dir = tmp_path / "objs"
    obj_dir.mkdir()
    (obj_dir / "cube.obj").write_text("v 0 0 0\n")
    loader = MeshKitLoader(str(obj_dir), texture_path="", cli=0)

    assert loader._scale_xyz(2.0) == (2.0, 2.0, 2.0)
    assert loader._scale_xyz([1, 2, 3]) == (1.0, 2.0, 3.0)
    with pytest.raises(ValueError):
        loader._scale_xyz([1, 2])

    assert loader._asset_path("cube.obj").endswith("cube.obj")
    with pytest.raises(FileNotFoundError):
        loader._asset_path("missing.obj")


def test_mesh_loader_vertex_parsing_and_bounds(tmp_path):
    obj_dir = tmp_path / "objs"
    obj_dir.mkdir()
    (obj_dir / "shape.obj").write_text(
        "\n".join(
            [
                "v 1 2 3",
                "v -1 -2 -3",
            ]
        )
        + "\n"
    )
    loader = MeshKitLoader(str(obj_dir), texture_path="", cli=0)

    verts = loader._parse_vertices("shape.obj")
    # conversion: (x, y, z) -> (x, -z, y)
    assert (1.0, -3.0, 2.0) in verts
    assert (-1.0, 3.0, -2.0) in verts

    min_v, max_v = loader._bounds("shape.obj", (2.0, 1.0, 1.0))
    assert min_v[0] == -2.0
    assert max_v[0] == 2.0
    size = loader.model_size("shape.obj", (2.0, 1.0, 1.0))
    assert size[0] == 4.0


def test_normalize_mtl_texture_paths_rewrites_colormap_and_stamps(tmp_path):
    mtl = tmp_path / "material.mtl"
    mtl.write_text("newmtl mat\nmap_Kd some/path/colormap.png\n")

    normalize_mtl_texture_paths(str(tmp_path))
    content = mtl.read_text()
    assert "map_Kd Textures/colormap.png" in content
    stamp = tmp_path / ".mtl_normalize_stamp_v1.json"
    assert stamp.exists()

    before = content
    normalize_mtl_texture_paths(str(tmp_path))
    after = mtl.read_text()
    assert before == after
