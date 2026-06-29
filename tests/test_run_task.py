from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from swarm.validator.utils_parts import run_task as run_task_module


def _seed_manager(epoch: int = 5) -> SimpleNamespace:
    return SimpleNamespace(epoch_number=epoch)


def _validator(*, backend_api: Any, seed_manager: Any | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        backend_api=backend_api,
        seed_manager=seed_manager or _seed_manager(),
    )


class _RecordingBackend:
    def __init__(self):
        self.submissions: list[dict] = []
        self.next_response: dict = {"recorded": True, "task_status": "SUBMITTED"}

    async def submit_task_result(self, **kwargs):
        self.submissions.append(kwargs)
        return self.next_response


def _patch_helpers(
    monkeypatch,
    *,
    fetch_paths: dict[int, tuple] | None = None,
    fetch_hash_ok: bool = True,
    screening_result: tuple | None = None,
    benchmark_result: tuple | None = None,
):
    async def _fake_fetch(_self, _models):
        return fetch_paths or {}
    monkeypatch.setattr(run_task_module, "_ensure_models_from_backend", _fake_fetch)

    monkeypatch.setattr(
        run_task_module,
        "sha256sum",
        lambda _path: ("abc" * 22)[:64] if fetch_hash_ok else "0" * 64,
    )

    if screening_result is not None:
        async def _fake_screening(_self, _uid, _path, **_kw):
            return screening_result
        monkeypatch.setattr(run_task_module, "_run_screening", _fake_screening)

    if benchmark_result is not None:
        async def _fake_benchmark(_self, _uid, _path, **_kw):
            return benchmark_result
        monkeypatch.setattr(run_task_module, "_run_full_benchmark", _fake_benchmark)


@pytest.fixture
def fake_model_path(tmp_path) -> Path:
    p = tmp_path / "UID_42.zip"
    p.write_bytes(b"fake")
    return p


@pytest.mark.asyncio
async def test_run_task_screening_happy_path_submits_score(
    monkeypatch, fake_model_path,
):
    target_hash = ("abc" * 22)[:64]
    backend = _RecordingBackend()
    _patch_helpers(
        monkeypatch,
        fetch_paths={42: (fake_model_path, "https://github.com/x/y")},
        screening_result=(
            0.42,
            [0.4] * 200,
            {"city": [0.4] * 200},
            None,
            False,
        ),
    )

    cancel, wake = asyncio.Event(), asyncio.Event()
    await run_task_module.run_task(
        _validator(backend_api=backend),
        {
            "task_id": 100,
            "uid": 42,
            "phase": "SCREENING",
            "seeds_from": 0,
            "seeds_to": 200,
            "model_hash": target_hash,
            "github_url": "https://github.com/x/y",
            "epoch_number": 5,
            "early_fail_rules": {"threshold": 0.6, "checkpoints": {"50": 0.5}},
        },
        cancel_flag=cancel,
        wake_flag=wake,
    )

    assert len(backend.submissions) == 1
    s = backend.submissions[0]
    assert s["task_id"] == 100
    assert s["seeds_evaluated"] == 200
    assert s["score"] == pytest.approx(0.42)
    assert s["early_failed"] is False


@pytest.mark.asyncio
async def test_run_task_screening_early_fail_marks_flag(
    monkeypatch, fake_model_path,
):
    target_hash = ("abc" * 22)[:64]
    backend = _RecordingBackend()
    _patch_helpers(
        monkeypatch,
        fetch_paths={42: (fake_model_path, "https://github.com/x/y")},
        screening_result=(
            0.1,
            [0.1] * 50,
            {"city": [0.1] * 50},
            "early_fail_at_50",
            True,
        ),
    )

    await run_task_module.run_task(
        _validator(backend_api=backend),
        {
            "task_id": 101,
            "uid": 42,
            "phase": "SCREENING",
            "seeds_from": 0,
            "seeds_to": 200,
            "model_hash": target_hash,
            "github_url": "https://github.com/x/y",
            "epoch_number": 5,
        },
        cancel_flag=asyncio.Event(),
        wake_flag=asyncio.Event(),
    )

    s = backend.submissions[0]
    assert s["early_failed"] is True
    assert s["seeds_evaluated"] == 50


@pytest.mark.asyncio
async def test_run_task_aborts_when_cancel_flag_signal(
    monkeypatch, fake_model_path,
):
    target_hash = ("abc" * 22)[:64]
    backend = _RecordingBackend()
    _patch_helpers(
        monkeypatch,
        fetch_paths={42: (fake_model_path, "https://github.com/x/y")},
        screening_result=(
            0.0,
            [],
            {},
            "cancel_flag_set",
            False,
        ),
    )

    await run_task_module.run_task(
        _validator(backend_api=backend),
        {
            "task_id": 102,
            "uid": 42,
            "phase": "SCREENING",
            "seeds_from": 0,
            "seeds_to": 200,
            "model_hash": target_hash,
            "github_url": "https://github.com/x/y",
            "epoch_number": 5,
        },
        cancel_flag=asyncio.Event(),
        wake_flag=asyncio.Event(),
    )

    assert backend.submissions == []


@pytest.mark.asyncio
async def test_run_task_benchmark_happy_path(monkeypatch, fake_model_path):
    target_hash = ("abc" * 22)[:64]
    backend = _RecordingBackend()
    _patch_helpers(
        monkeypatch,
        fetch_paths={42: (fake_model_path, "https://github.com/x/y")},
        benchmark_result=(
            0.7,
            {"city": 0.7},
            [0.7] * 800,
            {"city": [0.7] * 800},
            None,
        ),
    )

    await run_task_module.run_task(
        _validator(backend_api=backend),
        {
            "task_id": 200,
            "uid": 42,
            "phase": "BENCHMARK",
            "seeds_from": 300,
            "seeds_to": 1100,
            "model_hash": target_hash,
            "github_url": "https://github.com/x/y",
            "epoch_number": 5,
        },
        cancel_flag=asyncio.Event(),
        wake_flag=asyncio.Event(),
    )

    s = backend.submissions[0]
    assert s["task_id"] == 200
    assert s["score"] == pytest.approx(0.7)
    assert s["seeds_evaluated"] == 1100


@pytest.mark.asyncio
async def test_run_task_reeval_evaluates_full_1100_seeds(
    monkeypatch, fake_model_path,
):
    """Plan §2.10: REEVAL covers all 1100 seeds (0..1099), not just the
    benchmark 800. Validators receive a REEVAL task with seeds_from=0
    and the streaming evaluator must iterate the full epoch seed list."""
    target_hash = ("abc" * 22)[:64]
    backend = _RecordingBackend()

    captured = {}

    async def _fake_benchmark(_self, _uid, _path, **kwargs):
        captured["seeds_from"] = kwargs.get("seeds_from")
        return (
            0.6,
            {"city": 0.6},
            [0.6] * 1100,
            {"city": [0.6] * 1100},
            None,
        )

    async def _fake_fetch(_self, _models):
        return {42: (fake_model_path, "https://github.com/x/y")}
    monkeypatch.setattr(run_task_module, "_ensure_models_from_backend", _fake_fetch)
    monkeypatch.setattr(run_task_module, "sha256sum", lambda _path: target_hash)
    monkeypatch.setattr(run_task_module, "_run_full_benchmark", _fake_benchmark)

    await run_task_module.run_task(
        _validator(backend_api=backend),
        {
            "task_id": 300,
            "uid": 42,
            "phase": "REEVAL",
            "seeds_from": 0,
            "seeds_to": 1100,
            "model_hash": target_hash,
            "github_url": "https://github.com/x/y",
            "epoch_number": 5,
        },
        cancel_flag=asyncio.Event(),
        wake_flag=asyncio.Event(),
    )

    assert captured["seeds_from"] == 0
    s = backend.submissions[0]
    assert s["seeds_evaluated"] == 1100
    assert s["early_failed"] is False


@pytest.mark.asyncio
async def test_run_task_drops_when_model_fetch_fails(monkeypatch):
    backend = _RecordingBackend()
    _patch_helpers(monkeypatch, fetch_paths={})

    await run_task_module.run_task(
        _validator(backend_api=backend),
        {
            "task_id": 1,
            "uid": 99,
            "phase": "SCREENING",
            "seeds_from": 0,
            "seeds_to": 200,
            "model_hash": "h" * 64,
            "github_url": "https://github.com/x/y",
            "epoch_number": 5,
        },
        cancel_flag=asyncio.Event(),
        wake_flag=asyncio.Event(),
    )

    assert backend.submissions == []


@pytest.mark.asyncio
async def test_run_task_drops_on_hash_mismatch(monkeypatch, fake_model_path):
    backend = _RecordingBackend()
    _patch_helpers(
        monkeypatch,
        fetch_paths={42: (fake_model_path, "https://github.com/x/y")},
        fetch_hash_ok=False,
    )

    await run_task_module.run_task(
        _validator(backend_api=backend),
        {
            "task_id": 1,
            "uid": 42,
            "phase": "SCREENING",
            "seeds_from": 0,
            "seeds_to": 200,
            "model_hash": ("abc" * 22)[:64],
            "github_url": "https://github.com/x/y",
            "epoch_number": 5,
        },
        cancel_flag=asyncio.Event(),
        wake_flag=asyncio.Event(),
    )

    assert backend.submissions == []


@pytest.mark.asyncio
async def test_run_task_logs_seed_gap_response_without_raising(
    monkeypatch, fake_model_path, caplog,
):
    target_hash = ("abc" * 22)[:64]
    backend = _RecordingBackend()
    backend.next_response = {
        "recorded": False, "reason": "seed_gap_at_index_57",
        "task_status": "RUNNING",
    }
    _patch_helpers(
        monkeypatch,
        fetch_paths={42: (fake_model_path, "https://github.com/x/y")},
        screening_result=(
            0.4, [0.4] * 200, {"city": [0.4] * 200}, None, False,
        ),
    )

    await run_task_module.run_task(
        _validator(backend_api=backend),
        {
            "task_id": 5,
            "uid": 42,
            "phase": "SCREENING",
            "seeds_from": 0,
            "seeds_to": 200,
            "model_hash": target_hash,
            "github_url": "https://github.com/x/y",
            "epoch_number": 5,
        },
        cancel_flag=asyncio.Event(),
        wake_flag=asyncio.Event(),
    )

    assert len(backend.submissions) == 1
