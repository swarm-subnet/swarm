from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np

from swarm.validator import forward as forward_mod
from swarm.validator.runtime_telemetry import ValidatorRuntimeTracker


def test_forward_updates_runtime_tracker(monkeypatch, tmp_path) -> None:
    tracker = ValidatorRuntimeTracker(state_dir=tmp_path)
    calls: list[str] = []

    class _Backend:
        async def sync(self):
            return {
                "current_top": {},
                "weights": {},
                "reeval_queue": [],
                "pending_models": [],
                "leaderboard_version": 3,
            }

    class _DockerEvaluator:
        def cleanup(self):
            calls.append("cleanup")

    validator = SimpleNamespace(
        forward_count=0,
        wallet=SimpleNamespace(hotkey=SimpleNamespace(ss58_address="validator_hotkey")),
        metagraph=SimpleNamespace(n=1, hotkeys=["validator_hotkey"], S=np.array([1.0])),
        scores=np.zeros(1, dtype=np.float32),
        seed_manager=SimpleNamespace(
            epoch_number=22,
            get_pending_publications=lambda: [],
            check_epoch_transition=lambda: False,
            seconds_until_epoch_end=lambda: 10_000.0,
        ),
        backend_api=_Backend(),
        docker_evaluator=_DockerEvaluator(),
        runtime_tracker=tracker,
    )

    async def _ensure_models_from_backend(self, pending):
        _ = self, pending
        return {}

    monkeypatch.setattr(forward_mod, "_get_validator_stake", lambda self: 12.0)
    monkeypatch.setattr(forward_mod, "_apply_backend_weights_to_scores", lambda self, weights: None)
    monkeypatch.setattr(forward_mod, "_ensure_models_from_backend", _ensure_models_from_backend)
    monkeypatch.setattr(forward_mod, "_detect_new_models", lambda self, paths: {})
    monkeypatch.setattr(forward_mod, "load_normal_model_queue", lambda: {"items": {}})
    monkeypatch.setattr(forward_mod, "save_normal_model_queue", lambda queue: None)
    monkeypatch.setattr(forward_mod.DockerSecureEvaluator, "_base_ready", True)

    asyncio.run(forward_mod.forward(validator))

    snapshot = tracker.snapshot_copy()
    assert snapshot["forward"]["last_completed_forward_count"] == 1
    assert snapshot["backend"]["pending_models_count"] == 0
    assert snapshot["backend"]["leaderboard_version"] == 3
    assert snapshot["epoch"]["epoch_number"] == 22
    assert snapshot["docker"]["cleanup_count"] == 1
    assert calls == ["cleanup"]
