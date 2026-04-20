from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from swarm.validator import seed_manager as seed_manager_mod
from swarm.validator.utils_parts.backend_submission import (
    _publish_pending_epoch_seeds,
)


def _run(coro):
    return asyncio.run(coro)


def _write_epoch_file(seeds_dir, epoch: int, seeds: list[int], *, published: bool = False) -> None:
    seeds_dir.mkdir(parents=True, exist_ok=True)
    (seeds_dir / f"epoch_{epoch}.json").write_text(json.dumps({
        "epoch_number": epoch,
        "seeds": seeds,
        "started_at": "2026-04-01T00:00:00+00:00",
        "ended_at": "2026-04-01T01:00:00+00:00",
        "generated_at": "2026-04-01T00:00:00+00:00",
        "seed_count": len(seeds),
        "benchmark_version": "v4.0.2.5",
        "published": published,
    }))


def _make_manager(monkeypatch, tmp_path):
    seeds_dir = tmp_path / "state" / "epoch_seeds"
    monkeypatch.setattr(seed_manager_mod, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(seed_manager_mod, "EPOCH_SEEDS_DIR", seeds_dir)
    return seeds_dir


def _make_validator(manager, response_or_exc):
    calls: list[dict] = []

    async def _publish_epoch_seeds(**kwargs):
        calls.append(kwargs)
        if isinstance(response_or_exc, Exception):
            raise response_or_exc
        if callable(response_or_exc):
            return response_or_exc(kwargs)
        return response_or_exc

    return SimpleNamespace(
        seed_manager=manager,
        backend_api=SimpleNamespace(publish_epoch_seeds=_publish_epoch_seeds),
    ), calls


def _is_published_on_disk(seeds_dir, epoch: int) -> bool:
    data = json.loads((seeds_dir / f"epoch_{epoch}.json").read_text())
    return bool(data.get("published", False))


def test_marks_published_when_backend_returns_published_true(monkeypatch, tmp_path):
    seeds_dir = _make_manager(monkeypatch, tmp_path)
    _write_epoch_file(seeds_dir, 11, [1, 2, 3], published=False)
    _write_epoch_file(seeds_dir, 12, [9, 8, 7], published=False)

    manager = seed_manager_mod.BenchmarkSeedManager()
    validator, calls = _make_validator(manager, {"published": True})

    _run(_publish_pending_epoch_seeds(validator))

    assert len(calls) == 1
    assert calls[0]["epoch_number"] == 11
    assert _is_published_on_disk(seeds_dir, 11) is True


def test_marks_published_when_backend_returns_accepted_true(monkeypatch, tmp_path):
    seeds_dir = _make_manager(monkeypatch, tmp_path)
    _write_epoch_file(seeds_dir, 5, [4, 5, 6], published=False)
    _write_epoch_file(seeds_dir, 6, [1, 1, 1], published=False)

    manager = seed_manager_mod.BenchmarkSeedManager()
    validator, _ = _make_validator(manager, {"accepted": True})

    _run(_publish_pending_epoch_seeds(validator))

    assert _is_published_on_disk(seeds_dir, 5) is True


def test_leaves_file_pending_when_backend_returns_error_dict(monkeypatch, tmp_path):
    seeds_dir = _make_manager(monkeypatch, tmp_path)
    _write_epoch_file(seeds_dir, 21, [7, 7, 7], published=False)
    _write_epoch_file(seeds_dir, 22, [8, 8, 8], published=False)

    manager = seed_manager_mod.BenchmarkSeedManager()
    validator, _ = _make_validator(manager, {"error": "epoch already closed", "status_code": 409})

    _run(_publish_pending_epoch_seeds(validator))

    assert _is_published_on_disk(seeds_dir, 21) is False
    assert any(p.get("epoch_number") == 21 for p in manager.get_pending_publications())


def test_leaves_file_pending_when_backend_returns_empty_dict(monkeypatch, tmp_path):
    seeds_dir = _make_manager(monkeypatch, tmp_path)
    _write_epoch_file(seeds_dir, 33, [3, 3], published=False)
    _write_epoch_file(seeds_dir, 34, [4, 4], published=False)

    manager = seed_manager_mod.BenchmarkSeedManager()
    validator, _ = _make_validator(manager, {})

    _run(_publish_pending_epoch_seeds(validator))

    assert _is_published_on_disk(seeds_dir, 33) is False


def test_leaves_file_pending_when_publish_raises(monkeypatch, tmp_path):
    seeds_dir = _make_manager(monkeypatch, tmp_path)
    _write_epoch_file(seeds_dir, 41, [0, 0], published=False)
    _write_epoch_file(seeds_dir, 42, [1, 1], published=False)

    manager = seed_manager_mod.BenchmarkSeedManager()
    validator, _ = _make_validator(manager, RuntimeError("boom"))

    _run(_publish_pending_epoch_seeds(validator))

    assert _is_published_on_disk(seeds_dir, 41) is False


def test_independent_results_for_mixed_responses(monkeypatch, tmp_path):
    seeds_dir = _make_manager(monkeypatch, tmp_path)
    _write_epoch_file(seeds_dir, 51, [1], published=False)
    _write_epoch_file(seeds_dir, 52, [2], published=False)
    _write_epoch_file(seeds_dir, 60, [0], published=True)

    manager = seed_manager_mod.BenchmarkSeedManager()

    def _responder(kwargs):
        if kwargs["epoch_number"] == 51:
            return {"published": True}
        return {"error": "backend rejected", "status_code": 422}

    validator, calls = _make_validator(manager, _responder)

    _run(_publish_pending_epoch_seeds(validator))

    assert {c["epoch_number"] for c in calls} == {51, 52}
    assert _is_published_on_disk(seeds_dir, 51) is True
    assert _is_published_on_disk(seeds_dir, 52) is False


def test_rejected_publish_retries_on_next_call_and_eventually_succeeds(monkeypatch, tmp_path):
    seeds_dir = _make_manager(monkeypatch, tmp_path)
    _write_epoch_file(seeds_dir, 77, [5, 5, 5], published=False)
    _write_epoch_file(seeds_dir, 78, [6, 6, 6], published=False)

    manager = seed_manager_mod.BenchmarkSeedManager()

    attempts = {"count": 0}

    def _responder(_kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            return {"error": "backend transient", "status_code": 503}
        return {"published": True}

    validator, _ = _make_validator(manager, _responder)

    _run(_publish_pending_epoch_seeds(validator))
    assert _is_published_on_disk(seeds_dir, 77) is False

    _run(_publish_pending_epoch_seeds(validator))
    assert _is_published_on_disk(seeds_dir, 77) is True
    assert all(p.get("epoch_number") != 77 for p in manager.get_pending_publications())
