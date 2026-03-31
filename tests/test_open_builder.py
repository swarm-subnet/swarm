from __future__ import annotations

from swarm.core.maps.open import builder as open_builder


class _DummyPyBullet:
    GEOM_MESH = 1
    GEOM_FORCE_CONCAVE_TRIMESH = 2

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


def test_spawn_terrain_keeps_grass_tint_when_applying_texture(monkeypatch) -> None:
    dummy_p = _DummyPyBullet()

    monkeypatch.setattr(open_builder, "p", dummy_p)
    monkeypatch.setattr(open_builder, "_generate_terrain_obj", lambda seed: "/tmp/open.obj")
    monkeypatch.setattr(open_builder, "_load_texture", lambda cli: 77)

    open_builder._spawn_terrain(cli=5, seed=123)

    assert dummy_p.visual_calls
    assert dummy_p.change_visual_calls
    assert dummy_p.visual_calls[0]["rgbaColor"] == open_builder._TERRAIN_BASE_RGBA
    assert dummy_p.visual_calls[0]["specularColor"] == open_builder._TERRAIN_SPECULAR
    assert dummy_p.change_visual_calls[0]["textureUniqueId"] == 77
    assert dummy_p.change_visual_calls[0]["rgbaColor"] == open_builder._TERRAIN_BASE_RGBA
    assert dummy_p.change_visual_calls[0]["specularColor"] == open_builder._TERRAIN_SPECULAR
