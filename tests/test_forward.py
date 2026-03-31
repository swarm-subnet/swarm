from __future__ import annotations

from types import SimpleNamespace

from swarm.validator import forward as forward_mod


def test_invalidate_local_state_for_regenerated_seeds(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(forward_mod, "clear_normal_model_queue", lambda: calls.append("queue"))
    monkeypatch.setattr(forward_mod, "clear_benchmark_cache", lambda: calls.append("cache"))

    validator = SimpleNamespace(
        seed_manager=SimpleNamespace(current_epoch_requires_state_invalidation=True)
    )

    forward_mod._invalidate_local_state_for_regenerated_seeds(validator)

    assert calls == ["queue", "cache"]
    assert validator.seed_manager.current_epoch_requires_state_invalidation is False


def test_invalidate_local_state_for_regenerated_seeds_noop_when_not_needed(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(forward_mod, "clear_normal_model_queue", lambda: calls.append("queue"))
    monkeypatch.setattr(forward_mod, "clear_benchmark_cache", lambda: calls.append("cache"))

    validator = SimpleNamespace(
        seed_manager=SimpleNamespace(current_epoch_requires_state_invalidation=False)
    )

    forward_mod._invalidate_local_state_for_regenerated_seeds(validator)

    assert calls == []
