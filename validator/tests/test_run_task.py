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

"""What one backend task does end to end: fetch the model, check its hash, run the phase, submit or drop."""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from swarm.validator.backend_api import BackendRejectedError, BackendTransportError
from swarm.validator.utils_parts import run_task as run_task_module


def _seed_manager(epoch: int = 5) -> SimpleNamespace:
    """Return a stub carrying only the epoch number the phase falls back to."""
    return SimpleNamespace(epoch_number=epoch)


def _validator(*, backend_api: Any, seed_manager: Any | None = None) -> SimpleNamespace:
    """Return the validator stand-in a task needs: a backend client and a seed manager."""
    return SimpleNamespace(
        backend_api=backend_api,
        seed_manager=seed_manager or _seed_manager(),
    )


class _RecordingBackend:
    """A backend client that files every submission it is handed and answers with a canned reply."""
    def __init__(self):
        """Start with nothing filed and a reply that accepts the next submission."""
        self.submissions: list[dict] = []
        self.next_response: dict = {"recorded": True, "task_status": "SUBMITTED"}

    async def submit_task_result(self, **kwargs):
        """File the keyword arguments as one submission and answer with the canned reply."""
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
    """Swap the model fetch, the digest and the two evaluators for outcomes fixed by the caller."""
    async def _fake_fetch(_self, _models):
        """Return the prepared uid-to-path map rather than downloading anything."""
        return fetch_paths or {}
    monkeypatch.setattr(run_task_module, "_ensure_models_from_backend", _fake_fetch)

    monkeypatch.setattr(
        run_task_module,
        "sha256sum",
        lambda _path: ("abc" * 22)[:64] if fetch_hash_ok else "0" * 64,
    )

    if screening_result is not None:
        async def _fake_screening(_self, _uid, _path, **_kw):
            """Return the screening outcome the caller fixed, without flying a seed."""
            return screening_result
        monkeypatch.setattr(run_task_module, "_run_screening", _fake_screening)

    if benchmark_result is not None:
        async def _fake_benchmark(_self, _uid, _path, **_kw):
            """Return the benchmark outcome the caller fixed, without flying a seed."""
            return benchmark_result
        monkeypatch.setattr(run_task_module, "_run_full_benchmark", _fake_benchmark)


@pytest.fixture
def fake_model_path(tmp_path) -> Path:
    """A file standing in for UID 42's fetched submission on disk."""
    p = tmp_path / "UID_42.zip"
    p.write_bytes(b"fake")
    return p


@pytest.mark.asyncio
async def test_run_task_screening_happy_path_submits_score(
    monkeypatch, fake_model_path,
):
    """One screening ends in exactly one submission: the averaged score, the seeds flown, no early failure."""
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
    """A screening cut short carries the early-fail flag and counts only the seeds actually flown."""
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
    """A phase that stops on the cancel signal reaches the backend with nothing: no partial score is filed."""
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
async def test_run_task_cancelled_during_the_download_never_starts_the_phase(
    monkeypatch, fake_model_path,
):
    """A cancel that lands while the model downloads ends the task there: no phase runs and nothing is filed."""
    target_hash = ("abc" * 22)[:64]
    backend = _RecordingBackend()
    cancel = asyncio.Event()
    phases_run = []
    _patch_helpers(monkeypatch)

    async def _fetch_then_cancel(_self, _models):
        """Hand back the model, with the cancel arriving while it was on its way."""
        cancel.set()
        return {42: (fake_model_path, "https://github.com/x/y")}

    async def _record_phase(*_args, **_kwargs):
        """Note that a phase started, which a cancelled task must never reach."""
        phases_run.append(True)

    monkeypatch.setattr(run_task_module, "_ensure_models_from_backend", _fetch_then_cancel)
    monkeypatch.setattr(run_task_module, "_run_phase", _record_phase)

    await run_task_module.run_task(
        _validator(backend_api=backend),
        {
            "task_id": 103,
            "uid": 42,
            "phase": "BENCHMARK",
            "seeds_from": 200,
            "seeds_to": 1200,
            "model_hash": target_hash,
            "github_url": "https://github.com/x/y",
            "epoch_number": 5,
        },
        cancel_flag=cancel,
        wake_flag=asyncio.Event(),
    )

    assert phases_run == []
    assert backend.submissions == []


@pytest.mark.asyncio
async def test_run_task_benchmark_happy_path(monkeypatch, fake_model_path):
    """A benchmark reports the window offset plus the seeds it flew, so seeds 200 to 1000 submit as 1000."""
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
            "seeds_from": 200,
            "seeds_to": 1000,
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
    assert s["seeds_evaluated"] == 1000


@pytest.mark.asyncio
async def test_run_task_reeval_evaluates_full_1000_seeds(
    monkeypatch, fake_model_path,
):
    """REEVAL covers all 1000 seeds (0..999), not just the
    benchmark 700. Validators receive a REEVAL task with seeds_from=0
    and the streaming evaluator must iterate the full epoch seed list."""
    target_hash = ("abc" * 22)[:64]
    backend = _RecordingBackend()

    captured = {}

    async def _fake_benchmark(_self, _uid, _path, **kwargs):
        """Keep the seeds_from it was called with and answer with a full thousand-seed result."""
        captured["seeds_from"] = kwargs.get("seeds_from")
        return (
            0.6,
            {"city": 0.6},
            [0.6] * 1000,
            {"city": [0.6] * 1000},
            None,
        )

    async def _fake_fetch(_self, _models):
        """Hand back the staged file as UID 42's model rather than downloading one."""
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
            "seeds_to": 1000,
            "model_hash": target_hash,
            "github_url": "https://github.com/x/y",
            "epoch_number": 5,
        },
        cancel_flag=asyncio.Event(),
        wake_flag=asyncio.Event(),
    )

    assert captured["seeds_from"] == 0
    s = backend.submissions[0]
    assert s["seeds_evaluated"] == 1000
    assert s["early_failed"] is False


@pytest.mark.asyncio
async def test_run_task_drops_when_model_fetch_fails(monkeypatch):
    """A UID whose bytes never arrive is abandoned before any phase starts, with nothing filed for it."""
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
    """A file whose digest is not the one the task named is never evaluated and never scored."""
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
    """A backend reply reporting a missing seed is only logged: the run still ends cleanly after one send."""
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


class _ClaimBackend(_RecordingBackend):
    """A backend whose seed claims play a script of replies and exceptions."""
    def __init__(self, claims):
        """Queue the claim outcomes to play, in order."""
        super().__init__()
        self._claims = list(claims)

    async def claim_seeds(self, task_id, count=1):
        """Play the next scripted claim: raise it if it is an exception, else return it."""
        step = self._claims.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


async def _feeder_outcomes(monkeypatch, fake_model_path, backend, calls: int) -> list:
    """Run a seed-flow benchmark whose evaluator only calls the feeder, and return what the feeder answered."""
    outcomes: list = []

    async def _fake_benchmark(_self, _uid, _path, *, seed_feeder, **_kw):
        """Ask the feeder the given number of times, then report one flown seed so a result is submitted."""
        for _ in range(calls):
            outcomes.append(await seed_feeder(2))
        return 0.5, {"city": 0.5}, [0.5], {"city": [0.5]}, None

    _patch_helpers(monkeypatch, fetch_paths={42: (fake_model_path, "https://github.com/x/y")})
    monkeypatch.setattr(run_task_module, "_run_full_benchmark", _fake_benchmark)
    await run_task_module.run_task(
        _validator(backend_api=backend),
        {
            "task_id": 9, "uid": 42, "phase": "BENCHMARK", "flow": "seed",
            "model_hash": ("abc" * 22)[:64],
            "github_url": "https://github.com/x/y", "epoch_number": 5,
        },
        cancel_flag=asyncio.Event(),
        wake_flag=asyncio.Event(),
    )
    return outcomes


@pytest.mark.asyncio
async def test_seed_feeder_outage_is_not_a_drained_pool(monkeypatch, fake_model_path):
    """A claim that fails keeps the task alive, and the grant after it is still flown."""
    backend = _ClaimBackend([
        BackendTransportError("backend 500"),
        {"granted": [3, 8], "pending": 5, "leased_other": 0, "done": 0},
    ])
    outcomes = await _feeder_outcomes(monkeypatch, fake_model_path, backend, calls=2)
    assert outcomes == [([], False), ([3, 8], False)]


@pytest.mark.asyncio
async def test_seed_feeder_stops_when_the_backend_closed_the_task(monkeypatch, fake_model_path):
    """A refused claim ends the feeding at once instead of being asked again."""
    backend = _ClaimBackend([BackendRejectedError("/claim", 409, {"detail": "task_not_running"})])
    outcomes = await _feeder_outcomes(monkeypatch, fake_model_path, backend, calls=1)
    assert outcomes == [([], True)]


@pytest.mark.asyncio
async def test_seed_feeder_gives_up_after_a_long_outage(monkeypatch, fake_model_path):
    """Once the backend has been unreachable past the give-up window, the feeder stops asking."""
    monkeypatch.setattr(run_task_module, "CLAIM_GIVE_UP_SEC", 100.0)
    clock = iter([0.0, 50.0, 50.0, 120.0, 120.0])
    monkeypatch.setattr(run_task_module, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    backend = _ClaimBackend([BackendTransportError("down")] * 3)
    outcomes = await _feeder_outcomes(monkeypatch, fake_model_path, backend, calls=3)
    assert outcomes == [([], False), ([], False), ([], True)]


@pytest.mark.asyncio
async def test_seed_feeder_outage_clock_restarts_after_a_good_claim(monkeypatch, fake_model_path):
    """Two short outages with a real reply between them never add up to a give-up."""
    monkeypatch.setattr(run_task_module, "CLAIM_GIVE_UP_SEC", 100.0)
    clock = iter([0.0, 0.0, 500.0, 500.0])
    monkeypatch.setattr(run_task_module, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    backend = _ClaimBackend([
        BackendTransportError("down"),
        {"granted": [], "pending": 4, "leased_other": 2, "done": 0},
        BackendTransportError("down"),
    ])
    outcomes = await _feeder_outcomes(monkeypatch, fake_model_path, backend, calls=3)
    assert outcomes == [([], False), ([], False), ([], False)]


@pytest.mark.asyncio
async def test_seed_feeder_reply_without_counts_is_not_a_drained_pool(monkeypatch, fake_model_path):
    """A reply that does not state the pool's size cannot declare the pool empty."""
    backend = _ClaimBackend([{"message": "ok"}])
    outcomes = await _feeder_outcomes(monkeypatch, fake_model_path, backend, calls=1)
    assert outcomes == [([], False)]


@pytest.mark.asyncio
async def test_seed_feeder_real_empty_pool_still_drains(monkeypatch, fake_model_path):
    """A real reply of nothing granted and nothing pending still ends the run."""
    backend = _ClaimBackend([{"granted": [], "pending": 0, "leased_other": 0, "done": 1000}])
    outcomes = await _feeder_outcomes(monkeypatch, fake_model_path, backend, calls=1)
    assert outcomes == [([], True)]


@pytest.mark.parametrize("failure", [
    BackendTransportError("backend 503"),
    BackendRejectedError("/result", 410, {"reason": "task_finalized"}),
])
@pytest.mark.asyncio
async def test_run_task_survives_a_result_that_never_lands(monkeypatch, fake_model_path, failure):
    """A result the backend cannot take, or refuses, ends the task cleanly instead of crashing the forward loop."""
    class _FailingBackend:
        """A backend whose result endpoint always fails the same way."""
        async def submit_task_result(self, **_kwargs):
            """Raise the configured failure."""
            raise failure

    _patch_helpers(
        monkeypatch,
        fetch_paths={42: (fake_model_path, "https://github.com/x/y")},
        screening_result=(0.4, [0.4] * 200, {"city": [0.4] * 200}, None, False),
    )
    await run_task_module.run_task(
        _validator(backend_api=_FailingBackend()),
        {
            "task_id": 5, "uid": 42, "phase": "SCREENING",
            "seeds_from": 0, "seeds_to": 200,
            "model_hash": ("abc" * 22)[:64],
            "github_url": "https://github.com/x/y", "epoch_number": 5,
        },
        cancel_flag=asyncio.Event(),
        wake_flag=asyncio.Event(),
    )
