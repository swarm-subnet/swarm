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

import struct
import zlib
from pathlib import Path

from validator.scripts import check_blender_export as checker

# A unit cube standing on z = 0, centred on the origin, wound outward.
CUBE_VERTICES = [
    (-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.5, 0.5, 0.0), (-0.5, 0.5, 0.0),
    (-0.5, -0.5, 1.0), (0.5, -0.5, 1.0), (0.5, 0.5, 1.0), (-0.5, 0.5, 1.0),
]
CUBE_NORMALS = [(0, 0, -1), (0, 0, 1), (0, -1, 0), (1, 0, 0), (0, 1, 0), (-1, 0, 0)]
CUBE_QUADS = [
    ((1, 3, 2, 4), 1), ((5, 6, 7, 8), 2), ((1, 2, 6, 5), 3),
    ((2, 3, 7, 6), 4), ((3, 4, 8, 7), 5), ((4, 1, 5, 8), 6),
]
CUBE_TRIANGLES = [((1, 3, 2), 1), ((1, 4, 3), 1), ((5, 6, 7), 2), ((5, 7, 8), 2),
                  ((1, 2, 6), 3), ((1, 6, 5), 3), ((2, 3, 7), 4), ((2, 7, 6), 4),
                  ((3, 4, 8), 5), ((3, 8, 7), 5), ((4, 1, 5), 6), ((4, 5, 8), 6)]


def write_png(path: Path, colour_type: int = 2, depth: int = 8) -> None:
    channels = {0: 1, 2: 3, 4: 2, 6: 4}[colour_type]
    row = b"\x00" + b"\x80" * (channels * depth // 8)

    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    ihdr = struct.pack(">IIBBBBB", 1, 1, depth, colour_type, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(row)) + chunk(b"IEND", b""))


def write_jpeg(path: Path, frame_marker: int = 0xC0) -> None:
    frame = struct.pack(">BBHHB", 8, 1, 1, 1, 1) + b"\x01\x11\x00"
    path.write_bytes(b"\xff\xd8" + bytes([0xFF, frame_marker]) + struct.pack(">H", len(frame) + 2) + frame + b"\xff\xd9")


def write_piece(
    folder: Path,
    name: str = "piece",
    *,
    vertices=CUBE_VERTICES,
    faces=CUBE_TRIANGLES,
    normals=CUBE_NORMALS,
    offset=(0.0, 0.0, 0.0),
    flip_faces=(),
    flip_normals=False,
    with_uvs=True,
    materials=("mat",),
    texture="tex.png",
    extra_lines=(),
) -> Path:
    obj = folder / f"{name}.obj"
    lines = [f"mtllib {name}.mtl", f"o {name}"]
    for x, y, z in vertices:
        lines.append(f"v {x + offset[0]:.6f} {y + offset[1]:.6f} {z + offset[2]:.6f}")
    if with_uvs:
        lines.append("vt 0.5 0.5")
    for nx, ny, nz in normals:
        sign = -1 if flip_normals else 1
        lines.append(f"vn {sign * nx} {sign * ny} {sign * nz}")
    for index, material in enumerate(materials):
        lines.append(f"usemtl {material}")
        for face_index, (corners, normal) in enumerate(faces):
            if index and face_index < len(faces) // 2:
                continue
            if face_index in flip_faces:
                corners = tuple(reversed(corners))
            uv = "1" if with_uvs else ""
            lines.append("f " + " ".join(f"{c}/{uv}/{normal}" for c in corners))
    lines.extend(extra_lines)
    obj.write_text("\n".join(lines) + "\n")
    mtl = ["newmtl mat", "Kd 1 1 1", f"map_Kd {texture}", "newmtl second", "Kd 1 0 0"]
    (folder / f"{name}.mtl").write_text("\n".join(mtl) + "\n")
    tex = folder / texture
    if not tex.exists() and texture.endswith(".png"):
        write_png(tex)
    return obj


def run(folder: Path) -> list:
    return checker.check_export([folder])


def errors(findings) -> list:
    return [f for f in findings if f.level == "error"]


def messages(findings) -> str:
    return "\n".join(f.message for f in findings)


def test_clean_cube_has_no_findings(tmp_path):
    write_piece(tmp_path)
    assert run(tmp_path) == []


def test_non_triangle_face_is_reported_with_its_line(tmp_path):
    write_piece(tmp_path, faces=CUBE_QUADS)
    found = errors(run(tmp_path))
    assert len(found) == 1
    assert "4 corners" in found[0].message
    assert found[0].line == 19


def test_second_material_in_one_file(tmp_path):
    write_piece(tmp_path, materials=("mat", "second"))
    assert "2 materials in one file" in messages(errors(run(tmp_path)))


def test_material_missing_from_mtl(tmp_path):
    write_piece(tmp_path, materials=("ghost",))
    assert "ghost is not in piece.mtl" in messages(errors(run(tmp_path)))


def test_missing_texture(tmp_path):
    write_piece(tmp_path, texture="missing.jpg")
    assert "texture missing.jpg not found" in messages(errors(run(tmp_path)))


def test_texture_with_alpha(tmp_path):
    write_png(tmp_path / "tex.png", colour_type=6)
    write_piece(tmp_path)
    assert "alpha channel" in messages(errors(run(tmp_path)))


def test_sixteen_bit_png(tmp_path):
    write_png(tmp_path / "tex.png", depth=16)
    write_piece(tmp_path)
    assert "16-bit PNG" in messages(errors(run(tmp_path)))


def test_progressive_jpeg_fails_and_baseline_passes(tmp_path):
    write_jpeg(tmp_path / "tex.jpg", frame_marker=0xC2)
    write_piece(tmp_path, texture="tex.jpg")
    assert "progressive JPEG" in messages(errors(run(tmp_path)))
    write_jpeg(tmp_path / "tex.jpg", frame_marker=0xC0)
    assert run(tmp_path) == []


def test_textured_faces_without_uvs(tmp_path):
    write_piece(tmp_path, with_uvs=False)
    assert "no UVs" in messages(errors(run(tmp_path)))


def test_line_over_1023_characters(tmp_path):
    write_piece(tmp_path, extra_lines=["# " + "x" * 1500])
    found = errors(run(tmp_path))
    assert len(found) == 1
    assert found[0].line == 31
    assert "1502 characters" in found[0].message


def test_one_flipped_face_breaks_the_winding(tmp_path):
    write_piece(tmp_path, flip_faces=(0,))
    assert "wind in opposite directions" in messages(errors(run(tmp_path)))


def test_inside_out_shell(tmp_path):
    write_piece(tmp_path, flip_faces=range(12))
    found = errors(run(tmp_path))
    assert "inside out" in messages(found)
    assert "opposite directions" not in messages(found)


def test_normals_against_winding_is_a_warning(tmp_path):
    write_piece(tmp_path, flip_normals=True)
    found = run(tmp_path)
    assert errors(found) == []
    assert "12 faces whose vn points against their winding" in messages(found)


def test_open_plane_is_a_warning(tmp_path):
    write_piece(tmp_path, vertices=CUBE_VERTICES[4:], faces=[((1, 2, 3), 2), ((1, 3, 4), 2)])
    found = run(tmp_path)
    assert errors(found) == []
    assert "4 open edges" in messages(found)


def test_piece_exported_y_up_on_its_origin(tmp_path):
    y_up = [(x, z, -y) for x, y, z in CUBE_VERTICES]
    write_piece(tmp_path, vertices=y_up, faces=[(tuple(reversed(c)), n) for c, n in CUBE_TRIANGLES], normals=[(0, 0, 1)] * 6)
    assert "looks exported Y-up" in messages(errors(run(tmp_path)))


def test_map_floors_facing_y_is_reported_on_the_folder(tmp_path):
    ground = [(-50.0, 0.0, -50.0), (50.0, 0.0, -50.0), (50.0, 0.0, 50.0), (-50.0, 0.0, 50.0)]
    write_piece(tmp_path, "ground", vertices=ground, faces=[((1, 3, 2), 1), ((1, 4, 3), 1)], normals=[(0, 1, 0)])
    write_piece(tmp_path, "shed", offset=(10.0, 0.0, 10.0))
    found = errors(run(tmp_path))
    assert any(f.path == tmp_path and "looks exported Y-up" in f.message for f in found)


def test_piece_away_from_map_and_origin_is_a_transform_warning(tmp_path):
    write_piece(tmp_path, "room")
    write_piece(tmp_path, "stray", offset=(40.0, 40.0, 0.0))
    found = run(tmp_path)
    assert errors(found) == []
    assert [f.path.name for f in found] == ["stray.obj"]
    assert "transform was not applied" in found[0].message


def test_piece_in_a_subfolder_is_judged_against_the_whole_map(tmp_path):
    room = [(-0.5 + 20 * (i in (1, 2)), -0.5 + 8 * (i in (2, 3)), 0.0) for i in range(4)]
    write_piece(tmp_path, "shell", vertices=room + [(x, y, 3.0) for x, y, _ in room])
    (tmp_path / "fixed").mkdir()
    write_piece(tmp_path / "fixed", "window", offset=(19.0, 4.0, 1.0))
    assert run(tmp_path) == []


def test_main_exit_code_and_report(tmp_path, capsys):
    write_piece(tmp_path)
    assert checker.main([str(tmp_path)]) == 0
    assert capsys.readouterr().out.strip() == "0 errors, 0 warnings"
    write_piece(tmp_path, faces=CUBE_QUADS)
    assert checker.main([str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "piece.obj" in out and "line 19" in out and "fix:" in out
    assert out.strip().endswith("1 errors, 0 warnings")
