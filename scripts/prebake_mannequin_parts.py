#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Optional


def _safe_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)
    return token or "default"


def _obj_mtl_path(obj_path: str) -> Optional[str]:
    obj_dir = os.path.dirname(obj_path)
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.lower().startswith("mtllib "):
                rel = line.split(None, 1)[1].strip()
                mtl_path = os.path.normpath(os.path.join(obj_dir, rel))
                if os.path.exists(mtl_path):
                    return mtl_path
    fallback = os.path.splitext(obj_path)[0] + ".mtl"
    return fallback if os.path.exists(fallback) else None


def _parse_mtl_materials(mtl_path: Optional[str]) -> dict[str, dict]:
    if not mtl_path or not os.path.exists(mtl_path):
        return {}
    materials: dict[str, dict] = {}
    current: Optional[str] = None
    with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("newmtl "):
                current = line.split(None, 1)[1].strip()
                materials.setdefault(current, {"mtl_lines": [raw]})
                continue
            if current is None:
                continue
            materials[current].setdefault("mtl_lines", []).append(raw)
            if line.lower().startswith("kd "):
                parts = line.split()
                if len(parts) < 4:
                    continue
                materials[current]["rgba"] = [
                    max(0.0, min(1.0, float(parts[1]))),
                    max(0.0, min(1.0, float(parts[2]))),
                    max(0.0, min(1.0, float(parts[3]))),
                    1.0,
                ]
                continue
            if line.lower().startswith("map_kd "):
                rel = line.split(None, 1)[1].strip()
                tex_path = os.path.normpath(os.path.join(os.path.dirname(mtl_path), rel))
                materials[current]["texture_ref"] = rel
                if os.path.exists(tex_path):
                    materials[current]["texture_path"] = tex_path
    return materials


def _prebake(obj_path: str, out_dir: str) -> Iterable[str]:
    mtl_path = _obj_mtl_path(obj_path)
    material_info = _parse_mtl_materials(mtl_path)

    vertices: list[tuple[float, float, float]] = []
    texcoords: list[tuple[float, ...]] = []
    normals: list[tuple[float, float, float]] = []
    faces_by_mat: OrderedDict[str, list[list[tuple[int, Optional[int], Optional[int]]]]] = OrderedDict()
    current_mat = "__default__"

    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
                continue
            if line.startswith("vt "):
                parts = line.split()
                texcoords.append(tuple(float(v) for v in parts[1:]))
                continue
            if line.startswith("vn "):
                parts = line.split()
                normals.append((float(parts[1]), float(parts[2]), float(parts[3])))
                continue
            if line.lower().startswith("usemtl "):
                current_mat = line.split(None, 1)[1].strip()
                faces_by_mat.setdefault(current_mat, [])
                continue
            if line.startswith("f "):
                corners: list[tuple[int, Optional[int], Optional[int]]] = []
                for token in line.split()[1:]:
                    chunks = token.split("/")
                    vi = int(chunks[0])
                    if vi < 0:
                        vi = len(vertices) + 1 + vi
                    vi -= 1
                    vti = None
                    if len(chunks) >= 2 and chunks[1]:
                        vti = int(chunks[1])
                        if vti < 0:
                            vti = len(texcoords) + 1 + vti
                        vti -= 1
                    vni = None
                    if len(chunks) >= 3 and chunks[2]:
                        vni = int(chunks[2])
                        if vni < 0:
                            vni = len(normals) + 1 + vni
                        vni -= 1
                    corners.append((vi, vti, vni))
                if len(corners) < 3:
                    continue
                tris = faces_by_mat.setdefault(current_mat, [])
                for i in range(1, len(corners) - 1):
                    tris.append([corners[0], corners[i], corners[i + 1]])

    if not faces_by_mat:
        raise SystemExit(f"no materials in {obj_path}")

    os.makedirs(out_dir, exist_ok=True)
    written = []
    sanitized: dict[str, str] = {}
    for mat_name in faces_by_mat.keys():
        safe = _safe_token(mat_name)
        if safe in sanitized.values():
            raise SystemExit(
                f"material name collision after sanitisation: {mat_name!r} -> {safe!r}"
                f" (already used by {[k for k,v in sanitized.items() if v == safe][0]!r})"
            )
        sanitized[mat_name] = safe

    for mat_name, tris in faces_by_mat.items():
        if not tris:
            continue
        safe_mat = sanitized[mat_name]
        part_obj_path = os.path.join(out_dir, f"{safe_mat}.obj")
        info = material_info.get(mat_name, {})
        used_v = sorted({corner[0] for tri in tris for corner in tri})
        used_vt = sorted({corner[1] for tri in tris for corner in tri if corner[1] is not None})
        used_vn = sorted({corner[2] for tri in tris for corner in tri if corner[2] is not None})
        remap_v = {old: new + 1 for new, old in enumerate(used_v)}
        remap_vt = {old: new + 1 for new, old in enumerate(used_vt)}
        remap_vn = {old: new + 1 for new, old in enumerate(used_vn)}

        part_mtl_path = os.path.join(out_dir, f"{safe_mat}.mtl")
        texture_ref = info.get("texture_ref") or ""
        if texture_ref:
            src_dir = os.path.dirname(_obj_mtl_path(obj_path) or "")
            abs_tex = os.path.normpath(os.path.join(src_dir, texture_ref))
            rel_tex = os.path.relpath(abs_tex, out_dir).replace("\\", "/")
        else:
            rel_tex = ""

        with open(part_mtl_path, "w", encoding="utf-8") as mtl_out:
            mtl_lines = info.get("mtl_lines") or []
            if mtl_lines:
                for raw in mtl_lines:
                    stripped = raw.strip().lower()
                    if stripped.startswith("map_kd ") and rel_tex:
                        indent = raw[: len(raw) - len(raw.lstrip())]
                        mtl_out.write(f"{indent}map_Kd {rel_tex}\n")
                    else:
                        mtl_out.write(raw)
            else:
                rgba = list(info.get("rgba", [1.0, 1.0, 1.0, 1.0]))
                mtl_out.write(f"newmtl {mat_name}\n")
                mtl_out.write(f"Kd {rgba[0]:.6f} {rgba[1]:.6f} {rgba[2]:.6f}\n")
                if rel_tex:
                    mtl_out.write(f"map_Kd {rel_tex}\n")

        with open(part_obj_path, "w", encoding="utf-8") as out:
            out.write(f"# prebaked split of {os.path.basename(obj_path)} material {mat_name}\n")
            out.write(f"mtllib {os.path.basename(part_mtl_path)}\n")
            out.write(f"usemtl {mat_name}\n")
            for old in used_v:
                vx, vy, vz = vertices[old]
                out.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")
            for old in used_vt:
                tc = texcoords[old]
                if len(tc) >= 3:
                    out.write(f"vt {tc[0]:.6f} {tc[1]:.6f} {tc[2]:.6f}\n")
                else:
                    out.write(f"vt {tc[0]:.6f} {tc[1]:.6f}\n")
            for old in used_vn:
                nx, ny, nz = normals[old]
                out.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")
            for tri in tris:
                tokens = []
                for vi, vti, vni in tri:
                    rv = remap_v[vi]
                    rvt = remap_vt.get(vti) if vti is not None else None
                    rvn = remap_vn.get(vni) if vni is not None else None
                    if rvt is not None and rvn is not None:
                        tokens.append(f"{rv}/{rvt}/{rvn}")
                    elif rvt is not None:
                        tokens.append(f"{rv}/{rvt}")
                    elif rvn is not None:
                        tokens.append(f"{rv}//{rvn}")
                    else:
                        tokens.append(f"{rv}")
                out.write("f " + " ".join(tokens) + "\n")

        written.append(part_obj_path)

    return written


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    default_obj = repo_root / "swarm" / "assets" / "maps" / "custom" / "people" / "open_mannequin_raw" / "mannequin_a_raw.obj"
    default_out = default_obj.parent / "split"

    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", default=str(default_obj), help="source .obj path")
    parser.add_argument("--out-dir", default=str(default_out), help="output directory for split parts")
    args = parser.parse_args(argv)

    obj_path = os.fspath(args.obj)
    out_dir = os.fspath(args.out_dir)
    if not os.path.exists(obj_path):
        print(f"missing source OBJ: {obj_path}", file=sys.stderr)
        return 2

    written = list(_prebake(obj_path, out_dir))
    print(f"wrote {len(written)} parts into {out_dir}")
    for path in written:
        print(f"  {os.path.relpath(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
