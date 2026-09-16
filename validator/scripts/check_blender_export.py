#!/usr/bin/env python3
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

"""Check a Blender OBJ export against the rules the Bullet importer enforces silently.

    python3 validator/scripts/check_blender_export.py <folder> [<folder> ...]

Every OBJ under the folder is read with its MTL and textures, and every broken
rule is printed with the file, the line and the fix. Errors set the exit code;
warnings do not. The rules come from the swarm Bullet fork and are the reasons a
map loads without complaint and still looks wrong:

- one material per OBJ file: the importer keeps the first texture it finds and
  paints every face with it (b3ImportMeshUtility.cpp, TinyRendererVisualShapeConverter.cpp)
- triangles only: a face with more corners is fanned from its first corner with
  no concavity check (tiny_obj_loader.cpp, exportFaceGroupToShape)
- no line over 1023 characters: the parser reads lines into a 1024 byte buffer and
  treats the tail as a new line (tiny_obj_loader.cpp, b3BulletDefaultFileIO.h)
- Z-up, transforms applied: nothing is converted on import, the file is used as is
- outward, closed faces: a triangle is hidden when the camera is behind its vertex
  order, and the double-sided flag is not honoured for map bodies (TinyRenderer.cpp)
- baseline JPEG or 8-bit PNG, no alpha: the decoder rejects progressive JPEG and
  16-bit PNG, and drops the alpha channel of anything it accepts (stb_image.cpp,
  b3ImportMeshUtility.cpp)

Pieces in one folder are either placed in the map frame together, or each
exported on its own origin with its base on z = 0 for the builder to place. A
piece that is neither is reported as an unapplied transform.
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

MAX_LINE = 1023
BASE_TOLERANCE = 0.05
AXIS_ALIGNED = 0.9
JPEG_BASELINE = (0xC0, 0xC1)
JPEG_PROGRESSIVE = 0xC2
PNG_ALPHA_COLOUR_TYPES = (4, 6)
RECALCULATE = "in Blender select all, Mesh > Normals > Recalculate Outside"

Vec3 = Tuple[float, float, float]


@dataclass
class Finding:
    """One broken rule: where it is, how bad it is, and what to do in Blender."""

    path: Path
    level: str
    message: str
    fix: str
    line: Optional[int] = None

    def render(self) -> str:
        """The two report lines for this finding, message then fix."""
        where = f"line {self.line}: " if self.line else ""
        return f"  {self.level.upper():<8}{where}{self.message}\n          fix: {self.fix}"


@dataclass
class ObjFile:
    """What the checker keeps from one OBJ file: geometry, material names and over-long lines."""

    path: Path
    vertices: List[Vec3] = field(default_factory=list)
    normals: List[Vec3] = field(default_factory=list)
    has_texcoords: bool = False
    faces: List[Tuple[int, List[Tuple[int, int, int]]]] = field(default_factory=list)
    materials: Dict[str, int] = field(default_factory=dict)
    mtllib: Optional[Tuple[int, str]] = None
    long_lines: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def bounds(self) -> Tuple[Vec3, Vec3]:
        """Axis-aligned box around every vertex, as (min, max)."""
        xs, ys, zs = zip(*self.vertices)
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def _parse_index(token: str, count: int) -> int:
    """OBJ indices are 1-based, negative ones count from the end, missing is -1."""
    if not token:
        return -1
    value = int(token)
    return value - 1 if value > 0 else count + value


def parse_obj(path: Path) -> ObjFile:
    """Read the vertices, normals, faces, material names and line lengths of one OBJ file."""
    obj = ObjFile(path)
    texcoord_count = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line_no, raw in enumerate(handle, 1):
            line = raw.rstrip("\r\n")
            if len(line) > MAX_LINE:
                obj.long_lines.append((line_no, len(line)))
            parts = line.split()
            if not parts or parts[0].startswith("#"):
                continue
            key = parts[0]
            if key == "v":
                obj.vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif key == "vn":
                obj.normals.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif key == "vt":
                texcoord_count += 1
            elif key == "f":
                corners = []
                for token in parts[1:]:
                    fields = (token.split("/") + ["", ""])[:3]
                    corners.append((
                        _parse_index(fields[0], len(obj.vertices)),
                        _parse_index(fields[1], texcoord_count),
                        _parse_index(fields[2], len(obj.normals)),
                    ))
                obj.faces.append((line_no, corners))
            elif key == "usemtl" and len(parts) > 1:
                obj.materials.setdefault(parts[1], line_no)
            elif key == "mtllib" and len(parts) > 1:
                obj.mtllib = (line_no, line.split(None, 1)[1].strip())
    obj.has_texcoords = texcoord_count > 0
    return obj


def parse_mtl(path: Path) -> Tuple[Dict[str, Optional[str]], List[Tuple[int, int]]]:
    """Material name to its map_Kd (None when untextured), plus the over-long lines."""
    materials: Dict[str, Optional[str]] = {}
    long_lines: List[Tuple[int, int]] = []
    current: Optional[str] = None
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line_no, raw in enumerate(handle, 1):
            line = raw.rstrip("\r\n")
            if len(line) > MAX_LINE:
                long_lines.append((line_no, len(line)))
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            if parts[0] == "newmtl":
                current = parts[1].strip()
                materials[current] = None
            elif parts[0] == "map_Kd" and current is not None:
                materials[current] = parts[1].strip()
    return materials, long_lines


def texture_problem(path: Path) -> Optional[str]:
    """Why the engine's decoder would reject or degrade this texture, or None."""
    with open(path, "rb") as handle:
        data = handle.read()
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        depth, colour_type = struct.unpack(">BB", data[24:26])
        if depth != 8:
            return f"{depth}-bit PNG, the decoder only reads 8-bit PNG so the piece renders white"
        if colour_type in PNG_ALPHA_COLOUR_TYPES or b"tRNS" in data[33:]:
            return "PNG with an alpha channel, alpha is dropped at load so see-through parts turn solid"
        return None
    if data[:2] == b"\xff\xd8":
        pos = 2
        while pos + 4 <= len(data):
            if data[pos] != 0xFF:
                pos += 1
                continue
            marker = data[pos + 1]
            if marker in (0xFF, 0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                pos += 2
                continue
            if marker in JPEG_BASELINE:
                return None
            if marker == JPEG_PROGRESSIVE:
                return "progressive JPEG, the decoder only reads baseline JPEG so the piece renders white"
            if marker == 0xDA:
                break
            pos += 2 + struct.unpack(">H", data[pos + 2:pos + 4])[0]
        return "JPEG frame type the decoder does not read, the piece renders white"
    return "not a JPEG or PNG file, the engine cannot decode it so the piece renders white"


def _sub(a: Vec3, b: Vec3) -> Vec3:
    """Vector a minus b."""
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a: Vec3, b: Vec3) -> Vec3:
    """Cross product of a and b."""
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a: Vec3, b: Vec3) -> float:
    """Dot product of a and b."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def check_lines(path: Path, long_lines: Iterable[Tuple[int, int]]) -> List[Finding]:
    """One error per line longer than the parser's buffer."""
    return [
        Finding(path, "error", f"line is {length} characters, the parser reads {MAX_LINE} and treats the rest as a new line",
                "export triangulated faces and keep file names short", line_no)
        for line_no, length in long_lines
    ]


def check_faces(obj: ObjFile) -> List[Finding]:
    """An error on the first face that is not a triangle."""
    for line_no, corners in obj.faces:
        if len(corners) != 3:
            return [Finding(obj.path, "error", f"face has {len(corners)} corners, the engine fans it from the first corner without checking the shape",
                            "switch on Export Triangulated Mesh in the Blender OBJ exporter", line_no)]
    return []


def check_materials(obj: ObjFile) -> List[Finding]:
    """Material count, MTL and texture presence, texture format and UVs for one OBJ."""
    findings: List[Finding] = []
    if not obj.mtllib:
        return [Finding(obj.path, "warning", "no mtllib line, the piece has no material and renders in the builder's colour only",
                        "export with materials on, one material per file")]
    mtl_line, mtl_name = obj.mtllib
    mtl_path = obj.path.parent / mtl_name
    if not mtl_path.is_file():
        return [Finding(obj.path, "error", f"material file {mtl_name} not found next to the OBJ, the piece renders white",
                        "keep the MTL next to the OBJ under the name the mtllib line gives", mtl_line)]
    materials, long_lines = parse_mtl(mtl_path)
    findings.extend(check_lines(mtl_path, long_lines))
    if len(obj.materials) > 1:
        findings.append(Finding(obj.path, "error", f"{len(obj.materials)} materials in one file ({', '.join(obj.materials)}), the engine keeps one texture for every face",
                                "split the object so each OBJ file carries one material", sorted(obj.materials.values())[1]))
    for name, line_no in obj.materials.items():
        if name not in materials:
            findings.append(Finding(obj.path, "error", f"material {name} is not in {mtl_name}, the engine paints the piece white",
                                    "re-export so the MTL carries the material the OBJ uses", line_no))
            continue
        texture = materials[name]
        if texture is None:
            continue
        if os.path.isabs(texture):
            findings.append(Finding(mtl_path, "error", f"map_Kd {texture} is an absolute path, the engine only looks next to the MTL",
                                    "export with Path Mode = Copy so map_Kd is a bare file name"))
            continue
        texture_path = mtl_path.parent / texture
        if not texture_path.is_file():
            findings.append(Finding(mtl_path, "error", f"texture {texture} not found next to the MTL, the piece renders white",
                                    "export with Path Mode = Copy so the texture lands next to the MTL"))
            continue
        problem = texture_problem(texture_path)
        if problem:
            findings.append(Finding(texture_path, "error", problem, "save the texture as baseline JPEG or 8-bit PNG without alpha"))
        if not obj.has_texcoords:
            findings.append(Finding(obj.path, "error", "textured material but the faces carry no UVs, the engine samples one texel for the whole piece",
                                    "switch on Export UV Coordinates, or unwrap the object", line_no))
    return findings


def check_orientation(obj: ObjFile) -> List[Finding]:
    """Winding against neighbours, inside-out shells, normals against the winding, open edges."""
    findings: List[Finding] = []
    welded: Dict[Vec3, int] = {}
    weld_of = [welded.setdefault(v, len(welded)) for v in obj.vertices]
    positions = list(welded)
    stride = len(positions) + 1
    forward: Counter = Counter()
    backward: Counter = Counter()
    edge_line: Dict[int, int] = {}
    parent = list(range(len(positions)))

    def root(i: int) -> int:
        """The representative vertex of the connected piece i belongs to."""
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    triangles: List[Tuple[int, Tuple[int, int, int]]] = []
    against = 0
    first_against = 0
    for line_no, corners in obj.faces:
        if len(corners) != 3:
            continue
        tri = (weld_of[corners[0][0]], weld_of[corners[1][0]], weld_of[corners[2][0]])
        if len(set(tri)) < 3:
            continue
        triangles.append((line_no, tri))
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = min(a, b) * stride + max(a, b)
            (forward if a < b else backward)[key] += 1
            edge_line.setdefault(key, line_no)
            parent[root(a)] = root(b)
        if obj.normals and all(c[2] >= 0 for c in corners):
            p0, p1, p2 = (positions[i] for i in tri)
            stored = [obj.normals[c[2]] for c in corners]
            summed = (sum(n[0] for n in stored), sum(n[1] for n in stored), sum(n[2] for n in stored))
            if _dot(_cross(_sub(p1, p0), _sub(p2, p0)), summed) < 0:
                against += 1
                first_against = first_against or line_no

    edges = set(forward) | set(backward)
    flipped = [e for e in edges if forward[e] + backward[e] == 2 and (forward[e] == 2 or backward[e] == 2)]
    open_edges = [e for e in edges if forward[e] + backward[e] == 1]
    broken = {root(e // stride) for e in flipped + open_edges}
    broken |= {root(e // stride) for e in edges if forward[e] + backward[e] > 2}
    if flipped:
        findings.append(Finding(obj.path, "error", f"{len(flipped)} edges where neighbouring faces wind in opposite directions, the engine hides the flipped faces from one side",
                                RECALCULATE, min(edge_line[e] for e in flipped)))
    if against:
        findings.append(Finding(obj.path, "warning", f"{against} faces whose vn points against their winding, they are lit from the wrong side",
                                "recalculate normals outside and export normals again", first_against))

    volume: Dict[int, float] = defaultdict(float)
    faces_of: Counter = Counter()
    first_line: Dict[int, int] = {}
    for line_no, tri in triangles:
        component = root(tri[0])
        if component in broken:
            continue
        a, b, c = (positions[i] for i in tri)
        volume[component] += _dot(a, _cross(b, c))
        faces_of[component] += 1
        first_line.setdefault(component, line_no)
    inside_out = [c for c, v in volume.items() if v < 0]
    if inside_out:
        findings.append(Finding(obj.path, "error", f"{len(inside_out)} closed shells are inside out ({sum(faces_of[c] for c in inside_out)} faces), the engine shows them only from inside",
                                RECALCULATE, min(first_line[c] for c in inside_out)))
    if open_edges:
        findings.append(Finding(obj.path, "warning", f"{len(open_edges)} open edges, the back of these faces is invisible in the engine",
                                "give thin parts a thickness or a second flipped copy; a floor or ground plane is fine as it is", min(edge_line[e] for e in open_edges)))
    return findings


def facing_area(obj: ObjFile) -> Dict[str, float]:
    """Area of faces looking up the Y axis and up the Z axis, by their winding."""
    area = {"y": 0.0, "z": 0.0}
    for _, corners in obj.faces:
        if len(corners) != 3:
            continue
        p0, p1, p2 = (obj.vertices[c[0]] for c in corners)
        n = _cross(_sub(p1, p0), _sub(p2, p0))
        length = math.sqrt(_dot(n, n))
        if length == 0.0:
            continue
        if n[1] / length > AXIS_ALIGNED:
            area["y"] += length / 2
        elif n[2] / length > AXIS_ALIGNED:
            area["z"] += length / 2
    return area


def _centred(lo: float, hi: float) -> bool:
    """Whether a range sits on the origin, within a quarter of its own width."""
    return abs(lo + hi) <= 0.25 * (hi - lo)


def _local_up_axis(lo: Vec3, hi: Vec3) -> Optional[str]:
    """Which axis a piece on its own origin stands along: base at zero, centred in the other two."""
    for axis, index, other in (("z", 2, (0, 1)), ("y", 1, (0, 2))):
        if abs(lo[index]) <= BASE_TOLERANCE and hi[index] > BASE_TOLERANCE and all(_centred(lo[i], hi[i]) for i in other):
            return axis
    return None


def _overlaps(a: Tuple[Vec3, Vec3], b: Tuple[Vec3, Vec3], margin: float) -> bool:
    """Whether two boxes touch once each is grown by the margin."""
    return all(a[0][i] - margin <= b[1][i] and b[0][i] <= a[1][i] + margin for i in range(3))


def check_frames(folder: Path, objs: List[ObjFile]) -> List[Finding]:
    """Frame and up-axis rules across every piece of the map."""
    findings: List[Finding] = []
    objs = [o for o in objs if o.vertices]
    boxes = {o.path: o.bounds for o in objs}
    extent = max((hi[i] - lo[i] for lo, hi in boxes.values() for i in range(3)), default=0.0)
    margin = max(0.5, 0.1 * extent)
    world: List[ObjFile] = []
    for obj in objs:
        lo, hi = boxes[obj.path]
        up = _local_up_axis(lo, hi)
        if up == "y":
            findings.append(Finding(obj.path, "error", "piece stands on y = 0 and is centred on the origin in x and z, it looks exported Y-up",
                                    "export with Up = Z and Forward = Y"))
            continue
        if up == "z":
            continue
        others = [boxes[p] for p in boxes if p != obj.path]
        if not others or any(_overlaps((lo, hi), box, margin) for box in others):
            world.append(obj)
            continue
        box = ", ".join(f"{lo[i]:.2f}..{hi[i]:.2f}" for i in range(3))
        findings.append(Finding(obj.path, "warning", f"piece is neither on its own origin nor inside the map ({box}), a transform was not applied",
                                "in Blender apply location, rotation and scale (Ctrl+A) before exporting, or centre the piece on its origin"))
    if world:
        area = Counter()
        for obj in world:
            area.update(facing_area(obj))
        if area["y"] > area["z"]:
            findings.append(Finding(folder, "error", f"the map's floors face +Y ({area['y']:.1f} m2 up Y, {area['z']:.1f} m2 up Z), it looks exported Y-up",
                                    "export with Up = Z and Forward = Y"))
    return findings


def check_export(folders: List[Path]) -> List[Finding]:
    """Check every OBJ under the folders, which together hold one map."""
    findings: List[Finding] = []
    objs: List[ObjFile] = []
    for folder in folders:
        for path in sorted(folder.rglob("*.obj")):
            obj = parse_obj(path)
            objs.append(obj)
            findings.extend(check_lines(path, obj.long_lines))
            findings.extend(check_faces(obj))
            findings.extend(check_materials(obj))
            findings.extend(check_orientation(obj))
    findings.extend(check_frames(folders[0], objs))
    return findings


def report(findings: List[Finding], root: Path) -> str:
    """The printed report: findings grouped by file, paths relative to root, totals last."""
    lines = []
    by_path: Dict[Path, List[Finding]] = defaultdict(list)
    for finding in findings:
        by_path[finding.path].append(finding)
    for path in sorted(by_path):
        try:
            shown = path.relative_to(root)
        except ValueError:
            shown = path
        lines.append(str(shown) or ".")
        lines.extend(f.render() for f in by_path[path])
    errors = sum(1 for f in findings if f.level == "error")
    warnings = len(findings) - errors
    lines.append(f"{errors} errors, {warnings} warnings")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """Check the folders given on the command line; exit 1 when any error is found."""
    parser = argparse.ArgumentParser(description="Check a Blender OBJ export against the engine's import rules.")
    parser.add_argument("folders", nargs="+", type=Path, help="folder holding the exported .obj, .mtl and texture files")
    args = parser.parse_args(argv)
    for folder in args.folders:
        if not folder.is_dir():
            parser.error(f"not a folder: {folder}")
    findings = check_export(args.folders)
    print(report(findings, Path.cwd()))
    return 1 if any(f.level == "error" for f in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
