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

from swarm.core.maps.open import builder as open_builder


class _DummyPyBullet:
    GEOM_MESH = 1
    GEOM_BOX = 2
    GEOM_FORCE_CONCAVE_TRIMESH = 2
    JOINT_FIXED = 4
    VISUAL_SHAPE_DOUBLE_SIDED = 8

    def __init__(self) -> None:
        self.collision_calls: list[dict] = []
        self.visual_calls: list[dict] = []
        self.multibody_calls: list[dict] = []
        self.change_visual_calls: list[dict] = []

    def createCollisionShape(self, *args, **kwargs) -> int:
        self.collision_calls.append(dict(kwargs))
        return 11

    def createVisualShape(self, *args, **kwargs) -> int:
        self.visual_calls.append(dict(kwargs))
        return 12

    def createMultiBody(self, *args, **kwargs) -> int:
        self.multibody_calls.append(dict(kwargs))
        return 13

    def changeVisualShape(self, body_id, link_id, **kwargs) -> None:
        self.change_visual_calls.append(
            {
                "body_id": body_id,
                "link_id": link_id,
                **kwargs,
            }
        )

    def getQuaternionFromEuler(self, values):
        return tuple(values)


def test_spawn_terrain_keeps_grass_tint_when_applying_texture(monkeypatch) -> None:
    dummy_p = _DummyPyBullet()

    monkeypatch.setattr(open_builder, "p", dummy_p)
    monkeypatch.setattr(open_builder, "_generate_terrain_obj", lambda seed, size: "/tmp/open.obj")
    monkeypatch.setattr(open_builder, "_load_texture", lambda cli: 77)

    open_builder._spawn_terrain(cli=5, seed=123)

    assert dummy_p.visual_calls
    assert dummy_p.change_visual_calls
    assert dummy_p.visual_calls[0]["rgbaColor"] == open_builder._TERRAIN_BASE_RGBA
    assert dummy_p.visual_calls[0]["specularColor"] == open_builder._TERRAIN_SPECULAR
    assert dummy_p.change_visual_calls[0]["textureUniqueId"] == 77
    assert dummy_p.change_visual_calls[0]["rgbaColor"] == open_builder._TERRAIN_BASE_RGBA
    assert dummy_p.change_visual_calls[0]["specularColor"] == open_builder._TERRAIN_SPECULAR


def test_open_terrain_cache_defaults_under_repo_state() -> None:
    cache_dir = Path(open_builder._TERRAIN_CACHE_DIR)

    assert cache_dir == Path(open_builder._STATE_DIR) / "open_terrain"
    assert "assets" not in cache_dir.parts


def test_terrain_obj_path_uses_state_cache_dir() -> None:
    terrain_path = Path(open_builder._terrain_obj_path(123))

    assert terrain_path == (
        Path(open_builder._STATE_DIR)
        / "open_terrain"
        / "open_terrain_v3_s123.obj"
    )



def test_terrain_cache_is_never_visible_half_written(tmp_path, monkeypatch):
    """Workers race on the same seed; a reader must never see a partial mesh."""
    import os
    import threading

    monkeypatch.setattr(open_builder, "_TERRAIN_CACHE_DIR", str(tmp_path))
    path = open_builder._terrain_obj_path(4242)
    seen: list[bool] = []
    stop = threading.Event()

    def _watch() -> None:
        # Anything published under the final name must already parse as a whole mesh.
        while not stop.is_set():
            if os.path.exists(path):
                text = Path(path).read_text()
                seen.append(text.endswith("\n") and text.count("v ") > 0 and "f " in text)
                return

    watcher = threading.Thread(target=_watch, daemon=True)
    watcher.start()
    open_builder._generate_terrain_obj(4242)
    stop.set()
    watcher.join(timeout=5)

    assert Path(path).exists()
    assert all(seen), "a partially written terrain mesh became visible under its final name"
    assert not list(tmp_path.glob("*.part")), "temp file left behind"


def test_terrain_cache_reuses_a_completed_mesh(tmp_path, monkeypatch):
    monkeypatch.setattr(open_builder, "_TERRAIN_CACHE_DIR", str(tmp_path))
    first = open_builder._generate_terrain_obj(99)
    stamp = Path(first).stat().st_mtime_ns
    second = open_builder._generate_terrain_obj(99)
    assert first == second
    assert Path(second).stat().st_mtime_ns == stamp, "cached mesh was rewritten"
