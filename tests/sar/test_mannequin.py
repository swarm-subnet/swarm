from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_ASSET_DIR = _REPO_ROOT / "swarm" / "assets" / "maps" / "custom" / "people" / "open_mannequin_raw"
_RAW_OBJ = _ASSET_DIR / "mannequin_a_raw.obj"
_RAW_MTL = _ASSET_DIR / "mannequin_a_raw.mtl"
_LICENSE = _ASSET_DIR / "LICENSE.txt"
_SPLIT_DIR = _ASSET_DIR / "split"


def _obj_bounds(path):
    mn = [float("inf")] * 3
    mx = [float("-inf")] * 3
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            mn[0] = min(mn[0], x); mn[1] = min(mn[1], y); mn[2] = min(mn[2], z)
            mx[0] = max(mx[0], x); mx[1] = max(mx[1], y); mx[2] = max(mx[2], z)
    return tuple(mn), tuple(mx)


def _materials_in_mtl(path):
    mats = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.lower().startswith("newmtl "):
                mats.append(line.split(None, 1)[1].strip())
    return mats


def test_loadable():
    assert _RAW_OBJ.is_file(), f"missing raw mannequin: {_RAW_OBJ}"
    mn, mx = _obj_bounds(_RAW_OBJ)
    # OBJ is Y-up so "standing height" is the Y extent
    standing_height = mx[1] - mn[1]
    assert 1.5 <= standing_height <= 2.1, (
        f"unexpected standing height {standing_height:.3f}"
    )
    assert all(b > 0 for b in (mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]))


def test_license_present():
    assert _LICENSE.is_file()
    text = _LICENSE.read_text()
    assert "CC0" in text or "public domain" in text.lower()


def test_prebaked_parts_present():
    assert _RAW_MTL.is_file()
    materials = _materials_in_mtl(_RAW_MTL)
    assert materials, "MTL has no newmtl entries"
    assert _SPLIT_DIR.is_dir()
    for mat in materials:
        safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in mat)
        part_obj = _SPLIT_DIR / f"{safe}.obj"
        part_mtl = _SPLIT_DIR / f"{safe}.mtl"
        assert part_obj.is_file(), f"missing split OBJ for material {mat}: {part_obj}"
        assert part_mtl.is_file(), f"missing split MTL for material {mat}: {part_mtl}"


def test_no_runtime_writes():
    from swarm.core.env_builder import mesh_loader

    with mock.patch("os.makedirs") as mk_mkdir:
        list(mesh_loader.iter_prebaked_parts(_SPLIT_DIR))
        for path in mesh_loader.iter_prebaked_parts(_SPLIT_DIR):
            mesh_loader.obj_bounds(path)
        assert not mk_mkdir.called, "mesh_loader created directories at runtime"
