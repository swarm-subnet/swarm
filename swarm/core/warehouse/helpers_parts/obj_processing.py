from ._shared import *


def _safe_token_name(name):
    out = []
    for ch in str(name):
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    token = "".join(out).strip("._")
    return token or "mat"


def _purge_generated_model_artifacts(model_path):
    cache_key = os.path.abspath(model_path)
    _OBJ_MTL_SPLIT_CACHE.pop(cache_key, None)
    _OBJ_COLLISION_PROXY_CACHE.pop(cache_key, None)
    _OBJ_MTL_VISUAL_PROXY_CACHE.pop(cache_key, None)
    _OBJ_DOUBLE_SIDED_PROXY_CACHE.pop(cache_key, None)
    _TEXTURE_CACHE.clear()

    split_root = os.path.join(
        os.path.dirname(model_path),
        "_split_by_mtl",
        _safe_token_name(os.path.splitext(os.path.basename(model_path))[0]),
    )
    if os.path.isdir(split_root):
        shutil.rmtree(split_root, ignore_errors=True)
    double_sided_root = os.path.join(os.path.dirname(model_path), "_double_sided")
    if os.path.isdir(double_sided_root):
        shutil.rmtree(double_sided_root, ignore_errors=True)


def _obj_double_sided_proxy_path(model_path):
    cache_key = os.path.abspath(model_path)
    cached = _OBJ_DOUBLE_SIDED_PROXY_CACHE.get(cache_key)
    if cached and os.path.exists(cached):
        return cached
    try:
        with open(model_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        _OBJ_DOUBLE_SIDED_PROXY_CACHE[cache_key] = model_path
        return model_path

    out_lines = []
    reversed_face_count = 0
    for raw in lines:
        out_lines.append(raw)
        stripped = raw.lstrip()
        if not stripped.startswith("f "):
            continue
        tokens = stripped.split()[1:]
        if len(tokens) < 3:
            continue
        indent = raw[: len(raw) - len(stripped)]
        out_lines.append(f"{indent}f " + " ".join(reversed(tokens)) + "\n")
        reversed_face_count += 1

    if reversed_face_count == 0:
        _OBJ_DOUBLE_SIDED_PROXY_CACHE[cache_key] = model_path
        return model_path

    out_root = os.path.join(os.path.dirname(model_path), "_double_sided")
    os.makedirs(out_root, exist_ok=True)
    out_name = f"{os.path.splitext(os.path.basename(model_path))[0]}__double.obj"
    out_path = os.path.join(out_root, out_name)
    with open(out_path, "w", encoding="utf-8") as o:
        o.writelines(out_lines)
    _OBJ_DOUBLE_SIDED_PROXY_CACHE[cache_key] = out_path
    return out_path


def _obj_collision_proxy_path(model_path):
    cache_key = os.path.abspath(model_path)
    if cache_key in _OBJ_COLLISION_PROXY_CACHE:
        return _OBJ_COLLISION_PROXY_CACHE[cache_key]
    try:
        with open(model_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        _OBJ_COLLISION_PROXY_CACHE[cache_key] = model_path
        return model_path

    sanitized = []
    for raw in lines:
        stripped = raw.lstrip()
        if stripped.startswith("mtllib ") or stripped.startswith("usemtl "):
            continue
        sanitized.append(raw)
    if not sanitized:
        _OBJ_COLLISION_PROXY_CACHE[cache_key] = model_path
        return model_path

    split_root = os.path.join(
        os.path.dirname(model_path),
        "_split_by_mtl",
        _safe_token_name(os.path.splitext(os.path.basename(model_path))[0]),
    )
    os.makedirs(split_root, exist_ok=True)
    out_path = os.path.join(split_root, "__collision_proxy.obj")
    with open(out_path, "w", encoding="utf-8") as o:
        o.write("# collision-only proxy: stripped mtllib/usemtl\n")
        o.writelines(sanitized)
    _OBJ_COLLISION_PROXY_CACHE[cache_key] = out_path
    return out_path


def _resolve_mtl_texture_path(mtl_path, tex_ref):
    if not tex_ref:
        return ""
    ref = str(tex_ref).strip().strip("\"'").replace("\\", "/")
    if not ref:
        return ""
    mtl_dir = os.path.dirname(os.path.abspath(mtl_path))
    base_name = os.path.basename(ref)
    candidates = []
    if os.path.isabs(ref):
        candidates.append(ref)
    else:
        candidates.append(os.path.join(mtl_dir, ref))
    if base_name:
        candidates.append(os.path.join(mtl_dir, base_name))
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c).replace("\\", "/")
    return ""


def _obj_mtl_visual_proxy_path(model_path):
    cache_key = os.path.abspath(model_path)
    if (
        not MACHINING_FORCE_REFRESH_MTL_PROXY
    ) and cache_key in _OBJ_MTL_VISUAL_PROXY_CACHE:
        return _OBJ_MTL_VISUAL_PROXY_CACHE[cache_key]
    try:
        with open(model_path, "r", encoding="utf-8", errors="ignore") as f:
            obj_lines = f.readlines()
    except Exception:
        _OBJ_MTL_VISUAL_PROXY_CACHE[cache_key] = model_path
        return model_path

    mtllib_name = None
    for raw in obj_lines:
        stripped = raw.strip()
        if stripped.startswith("mtllib "):
            mtllib_name = stripped.split(maxsplit=1)[1].strip()
            break
    if not mtllib_name:
        _OBJ_MTL_VISUAL_PROXY_CACHE[cache_key] = model_path
        return model_path

    mtl_path = os.path.join(os.path.dirname(model_path), mtllib_name)
    if not os.path.exists(mtl_path):
        _OBJ_MTL_VISUAL_PROXY_CACHE[cache_key] = model_path
        return model_path

    split_root = os.path.join(
        os.path.dirname(model_path),
        "_split_by_mtl",
        _safe_token_name(os.path.splitext(os.path.basename(model_path))[0]),
    )
    os.makedirs(split_root, exist_ok=True)
    out_obj_path = os.path.join(split_root, "__visual_proxy.obj")
    out_mtl_name = "__visual_proxy.mtl"
    out_mtl_path = os.path.join(split_root, out_mtl_name)

    rewritten_mtl = []
    with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            stripped = raw.strip()
            if stripped.lower().startswith("map_kd "):
                tex_ref = (
                    stripped.split(maxsplit=1)[1].strip()
                    if len(stripped.split(maxsplit=1)) >= 2
                    else ""
                )
                resolved_tex = _resolve_mtl_texture_path(mtl_path, tex_ref)
                if resolved_tex and os.path.exists(resolved_tex):
                    tex_name = os.path.basename(resolved_tex)
                    dst_tex = os.path.join(split_root, tex_name)
                    if not os.path.exists(dst_tex):
                        shutil.copy2(resolved_tex, dst_tex)
                    indent = raw[: len(raw) - len(raw.lstrip())]
                    rewritten_mtl.append(f"{indent}map_Kd {tex_name}\n")
                    continue
            rewritten_mtl.append(raw)

    with open(out_mtl_path, "w", encoding="utf-8") as f:
        f.writelines(rewritten_mtl)

    out_obj_lines = []
    for raw in obj_lines:
        stripped = raw.strip()
        if stripped.startswith("mtllib "):
            indent = raw[: len(raw) - len(raw.lstrip())]
            out_obj_lines.append(f"{indent}mtllib {out_mtl_name}\n")
        else:
            out_obj_lines.append(raw)
    with open(out_obj_path, "w", encoding="utf-8") as f:
        f.writelines(out_obj_lines)

    _OBJ_MTL_VISUAL_PROXY_CACHE[cache_key] = out_obj_path
    return out_obj_path


def _parse_mtl_colors(mtl_path):
    colors = {}
    texture_by_material = {}
    ka_colors = {}
    map_kd = {}
    alpha = {}
    current = None
    if not os.path.exists(mtl_path):
        return colors, texture_by_material

    def _color_from_texture_ref(tex_ref):
        key = os.path.basename(str(tex_ref).replace("\\", "/")).strip().lower()
        palette = {
            "trak-k3-kmx-left-side-view-zoom.jpg": (0.66, 0.78, 0.72),
            "trak-k3-kmx-front-view-zoom.jpg": (0.66, 0.78, 0.72),
            "staal.jpg": (0.60, 0.61, 0.63),
            "perfo plaat.jpg": (0.44, 0.45, 0.47),
            "donker staal.jpeg": (0.16, 0.16, 0.18),
            "tnt124control-00000248.jpg": (0.24, 0.26, 0.30),
            "img0.jpg": (0.92, 0.92, 0.92),
        }
        return palette.get(key)

    with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            head = parts[0]
            if head == "newmtl" and len(parts) >= 2:
                current = " ".join(parts[1:])
                continue
            if current is None:
                continue
            if head == "Kd" and len(parts) >= 4:
                try:
                    colors[current] = [
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                        alpha.get(current, 1.0),
                    ]
                except Exception:
                    pass
            elif head == "Ka" and len(parts) >= 4:
                try:
                    ka_colors[current] = [
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                    ]
                except Exception:
                    pass
            elif head.lower() == "map_kd":
                tex = (
                    line.split(maxsplit=1)[1].strip()
                    if len(line.split(maxsplit=1)) >= 2
                    else ""
                )
                if tex:
                    map_kd[current] = tex
            elif head == "d" and len(parts) >= 2:
                try:
                    a = float(parts[1])
                    alpha[current] = a
                    if current in colors:
                        colors[current][3] = a
                except Exception:
                    pass
            elif head == "Tr" and len(parts) >= 2:
                try:
                    a = 1.0 - float(parts[1])
                    alpha[current] = a
                    if current in colors:
                        colors[current][3] = a
                except Exception:
                    pass

    for mat, tex_ref in map_kd.items():
        tex_path = _resolve_mtl_texture_path(mtl_path, tex_ref)
        if tex_path:
            texture_by_material[mat] = tex_path

    for mat, tex_ref in map_kd.items():
        if mat in colors:
            continue
        rgb = _color_from_texture_ref(tex_ref)
        if rgb is None:
            continue
        colors[mat] = [rgb[0], rgb[1], rgb[2], alpha.get(mat, 1.0)]
    for mat, ka in ka_colors.items():
        if mat in colors:
            continue
        colors[mat] = [ka[0], ka[1], ka[2], alpha.get(mat, 1.0)]
    for mat in map_kd.keys():
        if mat in colors:
            continue
        colors[mat] = [0.62, 0.64, 0.66, alpha.get(mat, 1.0)]
    for mat in list(colors.keys()):
        if str(mat).strip().lower() == "slang":
            colors[mat] = [0.10, 0.11, 0.12, alpha.get(mat, colors[mat][3])]

    return colors, texture_by_material


def _obj_material_parts(model_path):
    cache_key = os.path.abspath(model_path)
    if cache_key in _OBJ_MTL_SPLIT_CACHE:
        return _OBJ_MTL_SPLIT_CACHE[cache_key]

    split_root = os.path.join(
        os.path.dirname(model_path),
        "_split_by_mtl",
        _safe_token_name(os.path.splitext(os.path.basename(model_path))[0]),
    )
    os.makedirs(split_root, exist_ok=True)

    model_sig = {"size": -1, "mtime_ns": -1}
    try:
        st = os.stat(model_path)
        model_sig = {
            "size": int(st.st_size),
            "mtime_ns": int(
                getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))
            ),
        }
    except Exception:
        pass

    manifest_path = os.path.join(split_root, "_parts_manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f) or {}
            sig = manifest.get("model_sig", {}) or {}
            if (
                int(sig.get("size", -1)) == model_sig["size"]
                and int(sig.get("mtime_ns", -1)) == model_sig["mtime_ns"]
            ):
                cached_parts = []
                valid_manifest = True
                for item in manifest.get("parts", []) or []:
                    rel_or_abs = str(item.get("path", "")).strip()
                    if not rel_or_abs:
                        valid_manifest = False
                        break
                    part_path = (
                        rel_or_abs
                        if os.path.isabs(rel_or_abs)
                        else os.path.join(split_root, rel_or_abs)
                    )
                    part_path = os.path.abspath(part_path)
                    if not os.path.exists(part_path):
                        valid_manifest = False
                        break
                    rgba = item.get("rgba", [0.72, 0.72, 0.72, 1.0])
                    if not isinstance(rgba, (list, tuple)) or len(rgba) < 3:
                        rgba = [0.72, 0.72, 0.72, 1.0]
                    rgba = [
                        float(rgba[0]),
                        float(rgba[1]),
                        float(rgba[2]),
                        float(rgba[3]) if len(rgba) >= 4 else 1.0,
                    ]
                    cached_parts.append(
                        {
                            "path": part_path,
                            "material": str(item.get("material", "default")),
                            "rgba": rgba,
                            "texture_path": str(item.get("texture_path", "")),
                        }
                    )
                if valid_manifest:
                    _OBJ_MTL_SPLIT_CACHE[cache_key] = cached_parts
                    return cached_parts
        except Exception:
            pass

    with open(model_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    vertices = []
    texcoords = []
    normals = []
    mtllib_name = None
    for raw in lines:
        if raw.startswith("v "):
            parts = raw.split()
            if len(parts) >= 4:
                try:
                    vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
                except Exception:
                    pass
        elif raw.startswith("vt "):
            parts = raw.split()
            if len(parts) >= 3:
                try:
                    if len(parts) >= 4:
                        texcoords.append(
                            (float(parts[1]), float(parts[2]), float(parts[3]))
                        )
                    else:
                        texcoords.append((float(parts[1]), float(parts[2])))
                except Exception:
                    pass
        elif raw.startswith("vn "):
            parts = raw.split()
            if len(parts) >= 4:
                try:
                    normals.append((float(parts[1]), float(parts[2]), float(parts[3])))
                except Exception:
                    pass
        elif raw.startswith("mtllib ") and mtllib_name is None:
            mtllib_name = raw.split(maxsplit=1)[1].strip()

    if not vertices:
        _OBJ_MTL_SPLIT_CACHE[cache_key] = []
        return []

    def _resolve_obj_index(token, count):
        if not token:
            return None
        try:
            idx = int(token)
        except Exception:
            return None
        if idx > 0:
            idx0 = idx - 1
        elif idx < 0:
            idx0 = count + idx
        else:
            return None
        return idx0 if 0 <= idx0 < count else None

    material_faces = {}
    current_mtl = "default"
    for raw in lines:
        if raw.startswith("usemtl "):
            current_mtl = raw.split(maxsplit=1)[1].strip() or "default"
            continue
        if not raw.startswith("f "):
            continue
        toks = raw.split()[1:]
        corners = []
        for t in toks:
            chunks = t.split("/")
            vi = _resolve_obj_index(
                chunks[0] if len(chunks) >= 1 else "", len(vertices)
            )
            if vi is None:
                continue
            vti = _resolve_obj_index(
                chunks[1] if len(chunks) >= 2 else "", len(texcoords)
            )
            vni = _resolve_obj_index(
                chunks[2] if len(chunks) >= 3 else "", len(normals)
            )
            corners.append((vi, vti, vni))
        if len(corners) < 3:
            continue
        tris = material_faces.setdefault(current_mtl, [])
        for i in range(1, len(corners) - 1):
            tris.append((corners[0], corners[i], corners[i + 1]))

    mtl_colors = {}
    mtl_textures = {}
    if mtllib_name:
        mtl_path = os.path.join(os.path.dirname(model_path), mtllib_name)
        mtl_colors, mtl_textures = _parse_mtl_colors(mtl_path)

    out_parts = []
    for mtl_name, tris in material_faces.items():
        if not tris:
            continue
        unique_vi = {c[0] for tri in tris for c in tri if c[0] is not None}
        if len(unique_vi) < 3:
            continue
        total_area = 0.0
        for tri in tris:
            try:
                (v0, _t0, _n0), (v1, _t1, _n1), (v2, _t2, _n2) = tri
                p0 = vertices[v0]
                p1 = vertices[v1]
                p2 = vertices[v2]
                ux, uy, uz = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
                vx, vy, vz = (p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2])
                cx = uy * vz - uz * vy
                cy = uz * vx - ux * vz
                cz = ux * vy - uy * vx
                total_area += 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)
            except Exception:
                continue
        if total_area <= 1e-8:
            continue

        used_v = sorted({c[0] for tri in tris for c in tri if c[0] is not None})
        used_vt = sorted({c[1] for tri in tris for c in tri if c[1] is not None})
        used_vn = sorted({c[2] for tri in tris for c in tri if c[2] is not None})
        remap_v = {old_i: (new_i + 1) for new_i, old_i in enumerate(used_v)}
        remap_vt = {old_i: (new_i + 1) for new_i, old_i in enumerate(used_vt)}
        remap_vn = {old_i: (new_i + 1) for new_i, old_i in enumerate(used_vn)}

        out_name = f"{_safe_token_name(mtl_name)}.obj"
        out_path = os.path.join(split_root, out_name)
        with open(out_path, "w", encoding="utf-8") as o:
            o.write(f"# material split: {mtl_name}\n")
            for old_i in used_v:
                vx, vy, vz = vertices[old_i]
                o.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")
            for old_i in used_vt:
                tc = texcoords[old_i]
                if len(tc) >= 3:
                    o.write(f"vt {tc[0]:.6f} {tc[1]:.6f} {tc[2]:.6f}\n")
                else:
                    o.write(f"vt {tc[0]:.6f} {tc[1]:.6f}\n")
            for old_i in used_vn:
                nx, ny, nz = normals[old_i]
                o.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")
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
                o.write("f " + " ".join(tokens) + "\n")

        rgba = mtl_colors.get(mtl_name, [0.72, 0.72, 0.72, 1.0])
        out_parts.append(
            {
                "path": out_path,
                "material": mtl_name,
                "rgba": rgba,
                "texture_path": mtl_textures.get(mtl_name, ""),
            }
        )

    try:
        manifest_parts = []
        for part in out_parts:
            part_path_abs = os.path.abspath(str(part.get("path", "")))
            part_rel = os.path.relpath(part_path_abs, split_root).replace("\\", "/")
            manifest_parts.append(
                {
                    "path": part_rel,
                    "material": str(part.get("material", "default")),
                    "rgba": [
                        float(v)
                        for v in list(part.get("rgba", [0.72, 0.72, 0.72, 1.0]))[:4]
                    ],
                    "texture_path": str(part.get("texture_path", "")),
                }
            )
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(
                {"version": 1, "model_sig": model_sig, "parts": manifest_parts},
                mf,
                ensure_ascii=True,
                indent=0,
            )
    except Exception:
        pass

    _OBJ_MTL_SPLIT_CACHE[cache_key] = out_parts
    return out_parts


# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------
