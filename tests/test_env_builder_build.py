from __future__ import annotations

from swarm.constants import START_PLATFORM_TAKEOFF_BUFFER
from swarm.core.env_builder import build as build_mod


class _DummyPyBullet:
    GEOM_CYLINDER = 1

    def __init__(self) -> None:
        self._next_uid = 10

    def getNumBodies(self, physicsClientId: int) -> int:
        _ = physicsClientId
        return 0

    def createCollisionShape(self, *args, **kwargs) -> int:
        _ = args, kwargs
        self._next_uid += 1
        return self._next_uid

    def createVisualShape(self, *args, **kwargs) -> int:
        _ = args, kwargs
        self._next_uid += 1
        return self._next_uid

    def createMultiBody(self, *args, **kwargs) -> int:
        _ = args, kwargs
        self._next_uid += 1
        return self._next_uid

    def changeDynamics(self, *args, **kwargs) -> None:
        _ = args, kwargs
        return None


def test_build_world_reuses_cleared_forest_start_surface(monkeypatch) -> None:
    dummy_p = _DummyPyBullet()
    start = (1.0, 2.0, 3.0)
    cleared_surface_z = 12.5

    monkeypatch.setattr(build_mod.shared, "p", dummy_p)
    monkeypatch.setattr(build_mod, "_try_load_static_world_cache", lambda *args, **kwargs: False)
    monkeypatch.setattr(build_mod, "_build_static_world", lambda *args, **kwargs: None)
    monkeypatch.setattr(build_mod, "_save_static_world_cache_from_client", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        build_mod,
        "_find_clear_platform_position",
        lambda *args, **kwargs: (7.0, 8.0, cleared_surface_z),
    )

    def _unexpected_random_height(seed: int, challenge_type: int) -> float:
        raise AssertionError(
            f"forest start platform height should come from the cleared surface, got random helper call for seed={seed} type={challenge_type}"
        )

    monkeypatch.setattr(build_mod.shared, "get_platform_height_for_seed", _unexpected_random_height)

    result = build_mod.build_world(
        seed=123,
        cli=99,
        start=start,
        goal=None,
        challenge_type=6,
    )

    _, start_platform_uids, start_surface_z, _, adjusted_start, _ = result

    assert len(start_platform_uids) == 3
    assert start_surface_z == cleared_surface_z
    assert adjusted_start == (7.0, 8.0, cleared_surface_z + START_PLATFORM_TAKEOFF_BUFFER)
