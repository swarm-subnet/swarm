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

"""Streaming evaluation phase: chunked seed uploads, re-authorization, cancellation and heartbeat stops."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from swarm.validator import utils as validator_utils
from swarm.validator.utils_parts import evaluation as validator_evaluation
from swarm.validator.utils_parts.heartbeat import HeartbeatManager

_FAKE_MODEL_DIR = tempfile.TemporaryDirectory(prefix="swarm_streaming_phase_")
_FAKE_MODEL_ZIP = Path(_FAKE_MODEL_DIR.name) / "fake_model.zip"
_FAKE_MODEL_ZIP.write_bytes(b"streaming-phase fixture artifact")


def _make_validator(
    upload_results=None,
    heartbeat_calls=None,
) -> SimpleNamespace:
    """A validator stub carrying a backend client and a docker evaluator with fixed image hashes."""
    heartbeat_calls = heartbeat_calls if heartbeat_calls is not None else []

    async def _post_heartbeat(**kwargs):
        """Record the heartbeat payload and acknowledge it."""
        heartbeat_calls.append(kwargs)
        return {"ok": True}

    upload_sequence = iter(upload_results) if upload_results is not None else None

    async def _post_seed_scores_batch(**kwargs):
        """Answer from the scripted upload results, falling back to a recorded acknowledgement."""
        if upload_sequence is None:
            return {"recorded": True}
        try:
            return next(upload_sequence)
        except StopIteration:
            return {"recorded": True}

    async def _authorize_task(*_args, **_kwargs):
        """Approve every task check, with reason ok."""
        return {"authorized": True, "reason": "ok"}

    return SimpleNamespace(
        backend_api=SimpleNamespace(
            post_heartbeat=_post_heartbeat,
            post_seed_scores_batch=_post_seed_scores_batch,
            authorize_task=_authorize_task,
        ),
        docker_evaluator=SimpleNamespace(
            _get_image_hash_label=lambda: "test-image-hash",
            _calculate_docker_hash=lambda: "test-image-hash",
        ),
    )


def _heartbeat(validator) -> HeartbeatManager:
    """Return a HeartbeatManager wired to the stub backend and the running event loop."""
    return HeartbeatManager(validator.backend_api, asyncio.get_event_loop())


_ALL_MAP_TYPES = ("city", "open", "mountain", "village", "warehouse", "forest")


def _make_evaluate_stub(
    score_per_seed: float = 0.75,
    map_type: str = "city",
    per_seed_delay: float = 0.0,
    detail_fn=None,
):
    """Rolling-contract stub: one call for all seeds, honoring the
    ``should_stop`` poll before each seed and firing ``on_seed_result``
    with a per-seed detail dict — the same protocol as the real
    ``_evaluate_seeds``."""

    async def _evaluate(_self, _uid, _model_path, seeds, *args, **kwargs):
        """Walk the seeds, halting on ``should_stop``, and return the scores, the per-type buckets and the details."""
        on_seed_result = kwargs.get("on_seed_result")
        should_stop = kwargs.get("should_stop")
        scores: list = []
        per_type = {name: [] for name in _ALL_MAP_TYPES}
        details: list = []
        for i, _seed in enumerate(seeds):
            if should_stop is not None and should_stop():
                break
            if per_seed_delay:
                await asyncio.sleep(per_seed_delay)
            detail = (
                dict(detail_fn(i))
                if detail_fn is not None
                else {"score": score_per_seed, "map_type": map_type,
                      "failure_reason": "NONE"}
            )
            detail.setdefault("failure_reason", "NONE")
            detail.setdefault("metric_key", detail["map_type"])
            scores.append(detail["score"])
            details.append(detail)
            if detail["map_type"] in per_type:
                per_type[detail["map_type"]].append(detail["score"])
            if on_seed_result is not None:
                on_seed_result(i, dict(detail))
        return scores, per_type, details
    return _evaluate


def _make_evaluate_stub_with_infra(infra_local_index: int, score_per_seed: float = 0.75, map_type: str = "city"):
    """An evaluate stub where every tenth seed, at ``infra_local_index``, fails with INFRA."""
    def _detail(i):
        """Return one seed's result dict, marked INFRA at the chosen slot of every ten."""
        return {
            "score": score_per_seed,
            "map_type": map_type,
            "failure_reason": "INFRA" if i % 10 == infra_local_index else "NONE",
        }
    return _make_evaluate_stub(score_per_seed, map_type, detail_fn=_detail)


def test_streaming_phase_excludes_infra_seeds_from_upload(monkeypatch):
    """A seed that failed on infrastructure never reaches the backend; every other index does."""
    validator = _make_validator()
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub_with_infra(2))

    uploaded_indices: list = []

    async def _capture(**kwargs):
        """Collect the seed indexes of every uploaded row and acknowledge the batch."""
        uploaded_indices.extend(s["seed_index"] for s in kwargs.get("scores", []))
        return {"recorded": True}

    validator.backend_api.post_seed_scores_batch = _capture

    async def _run():
        """Stream 20 seeds through the phase in two chunks of ten."""
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=_FAKE_MODEL_ZIP,
                seeds=list(range(20)),
                phase_description="benchmark",
                seed_offset=0,
                epoch_number=1,
                hb=hb,
                chunk_size=10,
            )
        finally:
            hb.finish()

    asyncio.run(_run())
    # Two chunks of 10; local index 2 -> global seed_index 2 and 12 are infra -> not uploaded.
    assert sorted(uploaded_indices) == [i for i in range(20) if i not in (2, 12)]


def test_streaming_phase_uploads_slow_act_strikes_as_valid_zero(monkeypatch):
    """A SLOW_ACT_STRIKES seed uploads as a genuine zero score, not as an excluded infra failure."""
    validator = _make_validator()

    def _detail(i):
        """Return a struck-out mountain seed at index 1, a scoring city seed otherwise."""
        if i == 1:
            return {"score": 0.0, "map_type": "mountain",
                    "failure_reason": "SLOW_ACT_STRIKES"}
        return {"score": 0.6, "map_type": "city"}

    monkeypatch.setattr(
        validator_utils, "_evaluate_seeds", _make_evaluate_stub(detail_fn=_detail),
    )

    rows: list[dict] = []

    async def _capture(**kwargs):
        """Accumulate the uploaded rows and report them recorded."""
        rows.extend(kwargs["scores"])
        return {"recorded": True}

    validator.backend_api.post_seed_scores_batch = _capture

    async def _run():
        """Stream three seeds in one chunk and hand back the phase result."""
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=_FAKE_MODEL_ZIP,
                seeds=list(range(3)),
                phase_description="benchmark",
                seed_offset=0,
                epoch_number=1,
                hb=hb,
                chunk_size=3,
            )
        finally:
            hb.finish()

    scores, _per_type, _details, cancel = asyncio.run(_run())

    assert cancel is None
    assert len(scores) == 3
    striked = [r for r in rows if r["seed_index"] == 1]
    assert striked == [
        {
            "seed_index": 1,
            "score": 0.0,
            "metric_key": "mountain",
            "map_type": "mountain",
            "failure_reason": "SLOW_ACT_STRIKES",
        }
    ]


def test_streaming_phase_forwards_task_id_to_upload(monkeypatch):
    """Every score batch carries the task id the phase was started with."""
    validator = _make_validator()
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    posted_task_ids: list = []

    async def _capture(**kwargs):
        """Record the task id seen on each upload call and acknowledge it."""
        posted_task_ids.append(kwargs.get("task_id"))
        return {"recorded": True}

    validator.backend_api.post_seed_scores_batch = _capture

    async def _run():
        """Stream 20 seeds with task id 555 attached."""
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=_FAKE_MODEL_ZIP,
                seeds=list(range(20)),
                phase_description="benchmark",
                seed_offset=0,
                epoch_number=1,
                hb=hb,
                task_id=555,
                chunk_size=10,
            )
        finally:
            hb.finish()

    asyncio.run(_run())
    assert posted_task_ids == [555, 555]


def test_streaming_phase_final_retry_carries_task_id(monkeypatch):
    """A batch that failed once keeps the task id on its retry."""
    validator = _make_validator()
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    attempts: list[int | None] = []

    async def _flaky(**kwargs):
        """Reject the first upload as transient, then acknowledge, logging the task id each time."""
        attempts.append(kwargs.get("task_id"))
        if len(attempts) == 1:
            return {"recorded": False, "detail": "transient"}
        return {"recorded": True}

    validator.backend_api.post_seed_scores_batch = _flaky

    async def _run():
        """Stream ten seeds as one chunk with task id 777."""
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=_FAKE_MODEL_ZIP,
                seeds=list(range(10)),
                phase_description="benchmark",
                seed_offset=0,
                epoch_number=1,
                hb=hb,
                task_id=777,
                chunk_size=10,
            )
        finally:
            hb.finish()

    asyncio.run(_run())
    assert attempts == [777, 777]


def test_streaming_phase_happy_path(monkeypatch):
    """Twenty-five seeds upload as chunks of 10, 10 and 5, their indexes offset by 100."""
    posted_batches: list[list[dict]] = []
    validator = _make_validator()

    async def _capture_upload(**kwargs):
        """Store each posted batch of score rows and acknowledge it."""
        posted_batches.append(list(kwargs["scores"]))
        return {"recorded": True}

    validator.backend_api.post_seed_scores_batch = _capture_upload
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    async def _run():
        """Stream 25 seeds from offset 100 in chunks of ten."""
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=_FAKE_MODEL_ZIP,
                seeds=list(range(25)),
                phase_description="benchmark",
                seed_offset=100,
                epoch_number=42,
                hb=hb,
                chunk_size=10,
            )
        finally:
            hb.finish()

    scores, per_type, details, cancel = asyncio.run(_run())

    assert cancel is None
    assert len(scores) == 25
    assert len(details) == 25
    assert sum(len(v) for v in per_type.values()) == 25

    assert len(posted_batches) == 3
    assert [len(b) for b in posted_batches] == [10, 10, 5]
    all_indices = [entry["seed_index"] for batch in posted_batches for entry in batch]
    assert all_indices == list(range(100, 125))


def test_streaming_phase_re_authorize_cancels(monkeypatch):
    """A denied re-authorization stops dispatch mid-run and returns the partial scores with its reason."""
    validator = _make_validator()
    monkeypatch.setattr(
        validator_utils, "_evaluate_seeds", _make_evaluate_stub(per_seed_delay=0.02),
    )

    authorize_calls = {"n": 0}

    async def _re_authorize():
        """Approve the first check, deny the second with an epoch rotation."""
        authorize_calls["n"] += 1
        if authorize_calls["n"] >= 2:
            return {"authorized": False, "reason": "epoch rotated"}
        return {"authorized": True}

    async def _run():
        """Stream 30 slow seeds while re-authorizing every 10 ms."""
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=_FAKE_MODEL_ZIP,
                seeds=list(range(30)),
                phase_description="benchmark",
                seed_offset=0,
                epoch_number=1,
                hb=hb,
                chunk_size=10,
                re_authorize=_re_authorize,
                re_auth_interval_sec=0.01,
            )
        finally:
            hb.finish()

    scores, _per_type, _details, cancel = asyncio.run(_run())

    assert cancel == "epoch rotated"
    assert 0 < len(scores) < 30
    assert authorize_calls["n"] == 2


def test_streaming_phase_retries_failed_batches(monkeypatch):
    """An upload the backend did not record is posted again, unchanged and in full."""
    validator = _make_validator()
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    attempts: list[int] = []

    async def _flaky_upload(**kwargs):
        """Report the first batch as not recorded, every later one as recorded."""
        attempts.append(len(kwargs["scores"]))
        if len(attempts) == 1:
            return {"recorded": False, "detail": "transient backend error"}
        return {"recorded": True}

    validator.backend_api.post_seed_scores_batch = _flaky_upload

    async def _run():
        """Stream ten seeds as a single chunk against the flaky backend."""
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=_FAKE_MODEL_ZIP,
                seeds=list(range(10)),
                phase_description="benchmark",
                seed_offset=0,
                epoch_number=1,
                hb=hb,
                chunk_size=10,
            )
        finally:
            hb.finish()

    scores, _per_type, _details, cancel = asyncio.run(_run())

    assert cancel is None
    assert len(scores) == 10
    assert attempts == [10, 10]


def test_streaming_phase_retries_upload_exception(monkeypatch):
    """An upload that raises is retried rather than lost, and the phase still completes."""
    validator = _make_validator()
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    calls: list[str] = []

    async def _failing_then_succeed(**kwargs):
        """Raise on the first call, acknowledge on every call after it."""
        calls.append("call")
        if len(calls) == 1:
            raise RuntimeError("network down")
        return {"recorded": True}

    validator.backend_api.post_seed_scores_batch = _failing_then_succeed

    async def _run():
        """Stream five seeds against a backend that throws once."""
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=_FAKE_MODEL_ZIP,
                seeds=list(range(5)),
                phase_description="benchmark",
                seed_offset=0,
                epoch_number=1,
                hb=hb,
                chunk_size=10,
            )
        finally:
            hb.finish()

    scores, _per_type, _details, cancel = asyncio.run(_run())

    assert cancel is None
    assert len(scores) == 5
    assert len(calls) == 2


def test_streaming_phase_respects_inflight_cap(monkeypatch):
    """No more than ``max_inflight`` uploads are ever in the air at once."""
    validator = _make_validator()
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    inflight_counts: list[int] = []
    currently_inflight = {"n": 0}

    async def _slow_upload(**kwargs):
        """Hold the upload for 10 ms while counting how many are running together."""
        currently_inflight["n"] += 1
        inflight_counts.append(currently_inflight["n"])
        await asyncio.sleep(0.01)
        currently_inflight["n"] -= 1
        return {"recorded": True}

    validator.backend_api.post_seed_scores_batch = _slow_upload

    async def _run():
        """Stream 80 seeds with at most two uploads allowed in flight."""
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=_FAKE_MODEL_ZIP,
                seeds=list(range(80)),
                phase_description="benchmark",
                seed_offset=0,
                epoch_number=1,
                hb=hb,
                chunk_size=10,
                max_inflight=2,
            )
        finally:
            hb.finish()

    scores, _per_type, _details, cancel = asyncio.run(_run())

    assert cancel is None
    assert len(scores) == 80
    assert max(inflight_counts) <= 2


def test_streaming_phase_invokes_on_chunk_complete(monkeypatch):
    """The chunk callback fires once per upload group, each time with the running and final seed counts."""
    validator = _make_validator()
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    records: list[dict] = []

    def _on_chunk(**info):
        """Keep the evaluated and total counts reported by the callback."""
        records.append({"evaluated": info["evaluated"], "total": info["total"]})

    async def _run():
        """Stream 23 screening seeds in chunks of ten with the callback attached."""
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=_FAKE_MODEL_ZIP,
                seeds=list(range(23)),
                phase_description="screening",
                seed_offset=0,
                epoch_number=5,
                hb=hb,
                chunk_size=10,
                on_chunk_complete=_on_chunk,
            )
        finally:
            hb.finish()

    asyncio.run(_run())

    assert [r["evaluated"] for r in records] == [10, 20, 23]
    assert all(r["total"] == 23 for r in records)


def test_streaming_phase_empty_seeds():
    """An empty seed list returns empty scores and details with no cancel reason."""
    validator = _make_validator()

    async def _run():
        """Stream nothing at all and hand back the phase result."""
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=_FAKE_MODEL_ZIP,
                seeds=[],
                phase_description="benchmark",
                seed_offset=0,
                epoch_number=1,
                hb=hb,
                chunk_size=10,
            )
        finally:
            hb.finish()

    scores, per_type, details, cancel = asyncio.run(_run())
    assert scores == []
    assert details == []
    assert cancel is None
    assert all(v == [] for v in per_type.values())


def test_streaming_phase_filters_unknown_map_type_from_uploads(monkeypatch):
    """A seed whose map type is not one of the six known families is scored but never uploaded."""
    validator = _make_validator()
    uploads: list[list[dict]] = []

    async def _capture_upload(**kwargs):
        """Save the rows of every posted batch and acknowledge it."""
        uploads.append(list(kwargs["scores"]))
        return {"recorded": True}

    validator.backend_api.post_seed_scores_batch = _capture_upload

    async def _evaluate(_self, _uid, _model_path, seeds, *args, **kwargs):
        """Score even seeds as city and odd seeds as an unknown map type."""
        scores = [0.5] * len(seeds)
        per_type = {name: [] for name in (
            "city", "open", "mountain", "village", "warehouse", "forest",
        )}
        details = []
        for i, _seed in enumerate(seeds):
            if i % 2 == 0:
                details.append({"score": 0.5, "map_type": "city"})
                per_type["city"].append(0.5)
            else:
                details.append({"score": 0.0, "map_type": "unknown"})
        return scores, per_type, details

    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _evaluate)

    async def _run():
        """Stream ten seeds from offset 100 in a single chunk."""
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=_FAKE_MODEL_ZIP,
                seeds=list(range(10)),
                phase_description="benchmark",
                seed_offset=100,
                epoch_number=1,
                hb=hb,
                chunk_size=10,
            )
        finally:
            hb.finish()

    scores, _per_type, details, cancel = asyncio.run(_run())

    assert cancel is None
    assert len(scores) == 10
    assert len(details) == 10
    assert uploads == [
        [
            {"seed_index": 100, "score": 0.5, "metric_key": "city", "map_type": "city", "failure_reason": "NONE"},
            {"seed_index": 102, "score": 0.5, "metric_key": "city", "map_type": "city", "failure_reason": "NONE"},
            {"seed_index": 104, "score": 0.5, "metric_key": "city", "map_type": "city", "failure_reason": "NONE"},
            {"seed_index": 106, "score": 0.5, "metric_key": "city", "map_type": "city", "failure_reason": "NONE"},
            {"seed_index": 108, "score": 0.5, "metric_key": "city", "map_type": "city", "failure_reason": "NONE"},
        ]
    ]


def test_streaming_phase_reauthorize_passes_first_then_fails(monkeypatch):
    """Evaluation continues while re-authorization passes and stops the moment it is refused."""
    validator = _make_validator()
    monkeypatch.setattr(
        validator_utils, "_evaluate_seeds", _make_evaluate_stub(per_seed_delay=0.02),
    )

    calls = {"n": 0}

    async def _re_authorize():
        """Approve the first two checks, then refuse with a banned model."""
        calls["n"] += 1
        if calls["n"] <= 2:
            return {"authorized": True}
        return {"authorized": False, "reason": "model banned"}

    async def _run():
        """Stream 40 delayed seeds with re-authorization every 10 ms."""
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=_FAKE_MODEL_ZIP,
                seeds=list(range(40)),
                phase_description="benchmark",
                seed_offset=0,
                epoch_number=1,
                hb=hb,
                chunk_size=10,
                re_authorize=_re_authorize,
                re_auth_interval_sec=0.01,
            )
        finally:
            hb.finish()

    scores, _per_type, _details, cancel = asyncio.run(_run())
    assert cancel == "model banned"
    assert 0 < len(scores) < 40
    assert calls["n"] == 3


def test_streaming_phase_final_retry_also_fails_does_not_raise(monkeypatch):
    """A batch the backend never records is parked quietly; the phase still returns all its scores."""
    validator = _make_validator()
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    async def _always_fail(**kwargs):
        """Report every batch as not recorded because the backend is offline."""
        return {"recorded": False, "detail": "backend offline"}

    validator.backend_api.post_seed_scores_batch = _always_fail

    async def _run():
        """Stream ten seeds against a backend that records nothing."""
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=_FAKE_MODEL_ZIP,
                seeds=list(range(10)),
                phase_description="benchmark",
                seed_offset=0,
                epoch_number=1,
                hb=hb,
                chunk_size=10,
            )
        finally:
            hb.finish()

    scores, _per_type, _details, cancel = asyncio.run(_run())
    assert cancel is None
    assert len(scores) == 10


def test_streaming_phase_forwards_evaluator_prior_done(monkeypatch):
    """The evaluator is told how many seeds were already done and how many the whole run holds."""
    validator = _make_validator()

    evaluator_calls: list[dict] = []

    async def _evaluate(_self, _uid, _model_path, seeds, *args, **kwargs):
        """Record the prior-progress kwargs, then return city scores for every seed."""
        evaluator_calls.append({
            "seeds_len": len(seeds),
            "prior_seeds_done": kwargs.get("prior_seeds_done"),
            "prior_total_seeds": kwargs.get("prior_total_seeds"),
        })
        scores = [0.5] * len(seeds)
        per_type = {name: [] for name in (
            "city", "open", "mountain", "village", "warehouse", "forest",
        )}
        per_type["city"] = list(scores)
        details = [{"score": 0.5, "map_type": "city"} for _ in seeds]
        return scores, per_type, details

    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _evaluate)

    async def _run():
        """Stream 20 seeds declaring 300 already done out of 1000."""
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=_FAKE_MODEL_ZIP,
                seeds=list(range(20)),
                phase_description="benchmark",
                seed_offset=0,
                epoch_number=1,
                hb=hb,
                chunk_size=10,
                evaluator_prior_done=300,
                evaluator_total_seeds=1000,
            )
        finally:
            hb.finish()

    asyncio.run(_run())

    assert evaluator_calls == [
        {"seeds_len": 20, "prior_seeds_done": 300, "prior_total_seeds": 1000}
    ]


def test_streaming_phase_passes_all_pre_built_tasks_in_one_call(monkeypatch):
    """The whole task list goes to the evaluator in a single call, never sliced per chunk."""
    validator = _make_validator()
    slices: list[list[object]] = []

    async def _evaluate(_self, _uid, _model_path, seeds, *args, **kwargs):
        """Record the pre-built task slice it was handed and score every seed as city."""
        slices.append(list(kwargs.get("pre_built_tasks") or []))
        scores = [0.5] * len(seeds)
        per_type = {name: [] for name in (
            "city", "open", "mountain", "village", "warehouse", "forest",
        )}
        per_type["city"] = list(scores)
        details = [{"score": 0.5, "map_type": "city"} for _ in seeds]
        return scores, per_type, details

    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _evaluate)

    tasks = [f"task-{i}" for i in range(25)]

    async def _run():
        """Stream 25 screening seeds with all 25 tasks supplied up front."""
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=_FAKE_MODEL_ZIP,
                seeds=list(range(25)),
                phase_description="screening",
                seed_offset=0,
                epoch_number=1,
                hb=hb,
                chunk_size=10,
                pre_built_tasks=tasks,
            )
        finally:
            hb.finish()

    asyncio.run(_run())

    assert slices == [tasks]


def test_run_full_benchmark_streams_reeval_seeds(monkeypatch):
    """A re-evaluation over 20 supplied seeds averages 0.75 and uploads indexes 0 to 19."""
    validator = _make_validator()
    validator.seed_manager = SimpleNamespace(
        epoch_number=11,
        get_benchmark_seeds=lambda: list(range(20)),
    )

    uploads: list[list[dict]] = []

    async def _capture_upload(**kwargs):
        """Keep every posted batch of score rows and acknowledge it."""
        uploads.append(list(kwargs["scores"]))
        return {"recorded": True}

    validator.backend_api.post_seed_scores_batch = _capture_upload
    monkeypatch.setattr(
        validator_utils, "_evaluate_seeds", _make_evaluate_stub(map_type="open"),
    )

    async def _run():
        """Benchmark UID 42 over 20 seeds as a re-evaluation."""
        return await validator_evaluation._run_full_benchmark(
            validator,
            uid=42,
            model_path=_FAKE_MODEL_ZIP,
            seeds=list(range(20)),
            reeval=True,
        )

    avg, per_type_avgs, scores, per_type_raw, cancel = asyncio.run(_run())

    assert cancel is None
    assert len(scores) == 20
    assert avg == pytest.approx(0.75)
    assert per_type_avgs["open"] == pytest.approx(0.75)
    assert len(uploads) == 2
    all_indices = [entry["seed_index"] for batch in uploads for entry in batch]
    assert all_indices == list(range(20))


def test_run_full_benchmark_uses_offset_when_seeds_none(monkeypatch):
    """With no seeds supplied the benchmark indexes start at 300, past the screening range."""
    validator = _make_validator()
    validator.seed_manager = SimpleNamespace(
        epoch_number=11,
        get_benchmark_seeds=lambda: list(range(10)),
    )

    uploads: list[list[dict]] = []

    async def _capture_upload(**kwargs):
        """Collect the rows of every uploaded batch and acknowledge it."""
        uploads.append(list(kwargs["scores"]))
        return {"recorded": True}

    validator.backend_api.post_seed_scores_batch = _capture_upload
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    async def _run():
        """Benchmark UID 42 without naming any seeds."""
        return await validator_evaluation._run_full_benchmark(
            validator,
            uid=42,
            model_path=_FAKE_MODEL_ZIP,
        )

    asyncio.run(_run())

    all_indices = [entry["seed_index"] for batch in uploads for entry in batch]
    assert all_indices == list(range(300, 310))


def test_run_full_benchmark_covers_full_range_from_seed_zero(monkeypatch):
    """A range starting at seed 0 spans both halves: screening tasks then benchmark tasks, offsets 0 to 4 each."""
    validator = _make_validator()
    validator.seed_manager = SimpleNamespace(
        epoch_number=11,
        get_all_seeds=lambda: list(range(10)),
        get_screening_seeds=lambda: list(range(5)),
    )

    monkeypatch.setattr(validator_evaluation, "BENCHMARK_SCREENING_SEED_COUNT", 5)
    monkeypatch.setattr(validator_evaluation, "BENCHMARK_FULL_SEED_COUNT", 5)

    screening_offsets: list[int] = []
    benchmark_offsets: list[int] = []

    def _screening_tasks(sim_dt, seeds, family_id, offset, total_seed_count):
        """Record the offset asked for and return one placeholder task."""
        screening_offsets.append(offset)
        return [f"scr-{offset}"]

    def _benchmark_tasks(sim_dt, seeds, family_id, offset, total_seed_count):
        """Note the offset requested and hand back a single placeholder task."""
        benchmark_offsets.append(offset)
        return [f"bench-{offset}"]

    monkeypatch.setattr(validator_evaluation, "build_screening_tasks", _screening_tasks)
    monkeypatch.setattr(validator_evaluation, "build_benchmark_tasks", _benchmark_tasks)

    uploads: list[list[dict]] = []

    async def _capture_upload(**kwargs):
        """Keep the rows of each posted batch and acknowledge it."""
        uploads.append(list(kwargs["scores"]))
        return {"recorded": True}

    validator.backend_api.post_seed_scores_batch = _capture_upload
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    async def _run():
        """Benchmark UID 42 across the whole seed range, 0 through 10."""
        return await validator_evaluation._run_full_benchmark(
            validator,
            uid=42,
            model_path=_FAKE_MODEL_ZIP,
            seeds_from=0,
            seeds_to=10,
        )

    avg, _per_type_avgs, scores, _per_type_raw, cancel = asyncio.run(_run())

    assert cancel is None
    assert len(scores) == 10
    assert screening_offsets == [0, 1, 2, 3, 4]
    assert benchmark_offsets == [0, 1, 2, 3, 4]
    all_indices = [entry["seed_index"] for batch in uploads for entry in batch]
    assert all_indices == list(range(10))


def test_run_screening_streams_with_unified_chunks(monkeypatch):
    """Screening 25 seeds uploads them as three chunks, indexes 0 to 24, all scored on forest maps."""
    validator = _make_validator()
    validator.seed_manager = SimpleNamespace(
        epoch_number=3,
        get_screening_seeds=lambda: list(range(25)),
    )

    uploads: list[list[dict]] = []

    async def _capture_upload(**kwargs):
        """Store every uploaded batch of rows and acknowledge it."""
        uploads.append(list(kwargs["scores"]))
        return {"recorded": True}

    validator.backend_api.post_seed_scores_batch = _capture_upload
    monkeypatch.setattr(
        validator_utils, "_evaluate_seeds", _make_evaluate_stub(map_type="forest"),
    )

    async def _run():
        """Screen UID 99 over the epoch's screening seeds."""
        return await validator_evaluation._run_screening(
            validator,
            uid=99,
            model_path=_FAKE_MODEL_ZIP,
        )

    avg, scores, per_type, cancel, _early = asyncio.run(_run())

    assert cancel is None
    assert len(scores) == 25
    assert avg == pytest.approx(0.75)
    assert len(uploads) == 3
    all_indices = [entry["seed_index"] for batch in uploads for entry in batch]
    assert all_indices == list(range(25))
    assert per_type["forest"] == [0.75] * 25


# ── Real-flow integration tests ──────────────────────────────────────────
# These exercise the full streaming path: _evaluate_seeds runs real, only
# the docker evaluator (process-spawning layer) and backend HTTP are mocked.


def _make_docker_evaluator(score: float = 0.73):
    """A docker evaluator stub that returns one successful ValidationResult per task at a fixed score."""
    from swarm.protocol import ValidationResult

    async def _evaluate_seeds_parallel(tasks, uid, model_path, **kwargs):
        """Return a passing result for each task, padding with None once ``should_stop`` fires."""
        on_seed_result = kwargs.get("on_seed_result")
        should_stop = kwargs.get("should_stop")
        results: list = []
        for i, _task in enumerate(tasks):
            if should_stop is not None and should_stop():
                results.extend([None] * (len(tasks) - len(results)))
                break
            result = ValidationResult(int(uid), True, 1.0, float(score))
            results.append(result)
            if on_seed_result is not None:
                on_seed_result(i, result, "seed_done")
        return results

    return SimpleNamespace(
        evaluate_seeds_parallel=_evaluate_seeds_parallel,
        _get_image_hash_label=lambda: "test-image-hash",
        _calculate_docker_hash=lambda: "test-image-hash",
    )


def test_run_full_benchmark_real_flow_streams_chunks(tmp_path):
    """The real evaluation path, only docker and HTTP faked, uploads 25 scores as 10, 10 and 5 from index 300."""
    model_path = tmp_path / "UID_42.zip"
    model_path.write_bytes(b"fake-model")

    uploads: list[list[dict]] = []

    async def _capture_upload(**kwargs):
        """Keep each posted batch of score rows and acknowledge it."""
        uploads.append(list(kwargs["scores"]))
        return {"recorded": True}

    async def _post_heartbeat(**kwargs):
        """Acknowledge the heartbeat without asking for a stop."""
        return {"ok": True}

    validator = SimpleNamespace(
        docker_evaluator=_make_docker_evaluator(score=0.81),
        backend_api=SimpleNamespace(
            post_heartbeat=_post_heartbeat,
            post_seed_scores_batch=_capture_upload,
        ),
        seed_manager=SimpleNamespace(
            epoch_number=7,
            get_benchmark_seeds=lambda: [900001 + i for i in range(25)],
        ),
    )

    async def _run():
        """Benchmark UID 42 through the real streaming path."""
        return await validator_evaluation._run_full_benchmark(
            validator, uid=42, model_path=model_path,
        )

    avg, per_type_avgs, scores, per_type_raw, cancel = asyncio.run(_run())

    assert cancel is None
    assert len(scores) == 25
    assert avg == pytest.approx(0.81)
    assert [len(b) for b in uploads] == [10, 10, 5]
    all_indices = [entry["seed_index"] for batch in uploads for entry in batch]
    assert all_indices == list(range(300, 325))
    type_totals = sum(len(v) for v in per_type_raw.values())
    assert type_totals == 25


def test_run_screening_real_flow_streams_chunks(tmp_path):
    """Real-path screening of 15 seeds uploads them as 10 then 5, averaging the evaluator's score."""
    model_path = tmp_path / "UID_55.zip"
    model_path.write_bytes(b"fake-model")

    uploads: list[list[dict]] = []

    async def _capture_upload(**kwargs):
        """Save every posted batch of rows and acknowledge it."""
        uploads.append(list(kwargs["scores"]))
        return {"recorded": True}

    async def _post_heartbeat(**kwargs):
        """Acknowledge the heartbeat and ask for no stop."""
        return {"ok": True}

    validator = SimpleNamespace(
        docker_evaluator=_make_docker_evaluator(score=0.64),
        backend_api=SimpleNamespace(
            post_heartbeat=_post_heartbeat,
            post_seed_scores_batch=_capture_upload,
        ),
        seed_manager=SimpleNamespace(
            epoch_number=3,
            get_screening_seeds=lambda: [800001 + i for i in range(15)],
        ),
    )

    async def _run():
        """Screen UID 55 through the real streaming path."""
        return await validator_evaluation._run_screening(
            validator, uid=55, model_path=model_path,
        )

    avg, scores, per_type, cancel, _early = asyncio.run(_run())

    assert cancel is None
    assert len(scores) == 15
    assert avg == pytest.approx(0.64)
    assert [len(b) for b in uploads] == [10, 5]
    all_indices = [entry["seed_index"] for batch in uploads for entry in batch]
    assert all_indices == list(range(15))


# ── Re-eval kill-switch tests ────────────────────────────────────────────


def test_run_full_benchmark_reeval_does_not_authorize_per_chunk(monkeypatch):
    """REEVAL no longer calls /tasks/authorize between chunks.

    The legacy authorize endpoint returned 410 to consensus hotkeys,
    which froze the queue at every epoch transition. Cancellation now
    flows over SSE via the supplied ``cancel_flag``.
    """
    validator = _make_validator()
    validator.seed_manager = SimpleNamespace(
        epoch_number=11,
        get_benchmark_seeds=lambda: list(range(30)),
    )

    auth_calls: list[str] = []

    async def _authorize(*args, **kwargs):
        """Record that the legacy authorize endpoint was called and allow the task."""
        auth_calls.append("called")
        return {"authorized": True, "reason": "ok"}

    validator.backend_api.authorize_task = _authorize
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    async def _run():
        """Re-evaluate UID 42 over 30 supplied seeds."""
        return await validator_evaluation._run_full_benchmark(
            validator, uid=42, model_path=_FAKE_MODEL_ZIP,
            seeds=list(range(30)), reeval=True,
        )

    _avg, _per_type_avgs, scores, _per_type_raw, cancel = asyncio.run(_run())

    assert cancel is None
    assert len(scores) == 30
    assert auth_calls == []


def test_run_full_benchmark_reeval_cancels_via_cancel_flag(monkeypatch):
    """Setting the SSE cancel flag mid-run halts the re-evaluation at the next seed, with 20 scores kept."""
    validator = _make_validator()
    validator.seed_manager = SimpleNamespace(
        epoch_number=11,
        get_benchmark_seeds=lambda: list(range(30)),
    )

    uploads: list[list[dict]] = []

    async def _capture_upload(**kwargs):
        """Keep every batch of uploaded rows and acknowledge it."""
        uploads.append(list(kwargs["scores"]))
        return {"recorded": True}

    validator.backend_api.post_seed_scores_batch = _capture_upload

    cancel_flag = asyncio.Event()

    def _detail(i):
        """Score a city seed, setting the cancel flag once seed 19 is reached."""
        if i >= 19:
            cancel_flag.set()
        return {"score": 0.5, "map_type": "city"}

    monkeypatch.setattr(
        validator_utils, "_evaluate_seeds",
        _make_evaluate_stub(detail_fn=_detail),
    )

    async def _run():
        """Re-evaluate 30 seeds with the cancel flag wired in."""
        return await validator_evaluation._run_full_benchmark(
            validator, uid=42, model_path=_FAKE_MODEL_ZIP,
            seeds=list(range(30)), reeval=True,
            cancel_flag=cancel_flag,
        )

    _avg, _per_type_avgs, scores, _per_type_raw, cancel = asyncio.run(_run())

    assert cancel == "backend stop_required: cancel_flag_set"
    assert len(scores) == 20


def test_run_full_benchmark_non_reeval_skips_authorize(monkeypatch):
    """An ordinary benchmark never calls the authorize endpoint either."""
    validator = _make_validator()
    validator.seed_manager = SimpleNamespace(
        epoch_number=11,
        get_benchmark_seeds=lambda: list(range(20)),
    )

    auth_calls: list[str] = []

    async def _authorize(*args, **kwargs):
        """Note the call and answer authorized."""
        auth_calls.append("called")
        return {"authorized": True}

    validator.backend_api.authorize_task = _authorize
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    async def _run():
        """Benchmark UID 42 with no re-evaluation flag."""
        return await validator_evaluation._run_full_benchmark(
            validator, uid=42, model_path=_FAKE_MODEL_ZIP,
        )

    _avg, _per_type_avgs, scores, _per_type_raw, cancel = asyncio.run(_run())

    assert cancel is None
    assert len(scores) == 20
    assert auth_calls == []


def test_run_screening_reeval_does_not_authorize_per_chunk(monkeypatch):
    """Screening a re-evaluation streams all 25 seeds without one authorize call."""
    validator = _make_validator()
    validator.seed_manager = SimpleNamespace(
        epoch_number=3,
        get_screening_seeds=lambda: list(range(25)),
    )

    auth_calls: list[str] = []

    async def _authorize(*args, **kwargs):
        """Log the call and report the task authorized."""
        auth_calls.append("called")
        return {"authorized": True}

    validator.backend_api.authorize_task = _authorize
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    async def _run():
        """Screen UID 99 as a re-evaluation."""
        return await validator_evaluation._run_screening(
            validator, uid=99, model_path=_FAKE_MODEL_ZIP, reeval=True,
        )

    _avg, scores, _per_type, cancel, _early = asyncio.run(_run())

    assert cancel is None
    assert len(scores) == 25
    assert auth_calls == []


def test_run_screening_reeval_cancels_via_cancel_flag(monkeypatch):
    """The SSE cancel flag stops screening after 10 seeds and reports the backend's stop reason."""
    validator = _make_validator()
    validator.seed_manager = SimpleNamespace(
        epoch_number=3,
        get_screening_seeds=lambda: list(range(25)),
    )

    cancel_flag = asyncio.Event()

    def _detail(i):
        """Score a city seed, raising the cancel flag from seed 9 on."""
        if i >= 9:
            cancel_flag.set()
        return {"score": 0.5, "map_type": "city"}

    monkeypatch.setattr(
        validator_utils, "_evaluate_seeds",
        _make_evaluate_stub(detail_fn=_detail),
    )

    async def _run():
        """Screen 25 seeds as a re-evaluation with the cancel flag wired in."""
        return await validator_evaluation._run_screening(
            validator, uid=99, model_path=_FAKE_MODEL_ZIP, reeval=True,
            cancel_flag=cancel_flag,
        )

    _avg, scores, _per_type, cancel, _early = asyncio.run(_run())

    assert cancel == "backend stop_required: cancel_flag_set"
    assert len(scores) == 10


def test_run_screening_non_reeval_skips_authorize(monkeypatch):
    """A plain screening run reaches all 15 seeds with the authorize endpoint untouched."""
    validator = _make_validator()
    validator.seed_manager = SimpleNamespace(
        epoch_number=3,
        get_screening_seeds=lambda: list(range(15)),
    )

    auth_calls: list[str] = []

    async def _authorize(*args, **kwargs):
        """Count the call and answer that the task may proceed."""
        auth_calls.append("called")
        return {"authorized": True}

    validator.backend_api.authorize_task = _authorize
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    async def _run():
        """Screen UID 99 with no re-evaluation flag set."""
        return await validator_evaluation._run_screening(
            validator, uid=99, model_path=_FAKE_MODEL_ZIP,
        )

    _avg, scores, _per_type, cancel, _early = asyncio.run(_run())

    assert cancel is None
    assert len(scores) == 15
    assert auth_calls == []


def test_heartbeat_manager_honors_stop_required():
    """A heartbeat answering ``stop_required`` latches a reason naming the backend's conflict code and message."""
    responses = [
        {"recorded": True, "stop_required": False},
        {
            "recorded": True,
            "stop_required": True,
            "conflicts": [
                {
                    "severity": "CRITICAL",
                    "code": "INVALID_SCREENING_IN_FLIGHT",
                    "message": "Validator running UID 66 while backend status is SCREENING_FAILED",
                }
            ],
        },
    ]

    async def _run():
        """Send two heartbeats and hand back the stop reason after the second."""
        class _Api:
            """A backend stub that walks the scripted heartbeat responses in order."""
            def __init__(self):
                """Start the response cursor at the first scripted reply."""
                self._idx = 0

            async def post_heartbeat(self, **_kwargs):
                """Return the next scripted response, repeating the last one forever."""
                resp = responses[min(self._idx, len(responses) - 1)]
                self._idx += 1
                return resp

        hb = HeartbeatManager(_Api(), asyncio.get_event_loop())
        hb.start("evaluating_screening", uid=66, total=200)
        try:
            await hb._safe_heartbeat(0, hb._session_id)
            assert hb.should_stop() is None
            await hb._safe_heartbeat(10, hb._session_id)
            return hb.should_stop()
        finally:
            hb._stop_timer()

    reason = asyncio.run(_run())
    assert reason is not None
    assert "INVALID_SCREENING_IN_FLIGHT" in reason
    assert "SCREENING_FAILED" in reason


def test_heartbeat_manager_start_resets_stop_flag():
    """Opening a new session clears a stop latched during the previous one."""
    async def _run():
        """Latch a stop, open a second session, and report the flag afterwards."""
        class _Api:
            """A backend stub that always demands a stop."""
            async def post_heartbeat(self, **_kwargs):
                """Answer every post with a stop_required conflict."""
                return {"stop_required": True, "conflicts": [{"code": "X", "message": "y"}]}

        hb = HeartbeatManager(_Api(), asyncio.get_event_loop())
        hb.start("evaluating_screening", uid=1, total=10)
        try:
            await hb._safe_heartbeat(0, hb._session_id)
            assert hb.should_stop() is not None

            hb.start("evaluating_screening", uid=2, total=10)
            return hb.should_stop()
        finally:
            hb._stop_timer()

    assert asyncio.run(_run()) is None


def test_streaming_phase_stops_when_should_stop_fires(monkeypatch):
    """A stop reason from the poll ends dispatch at the next seed and surfaces as the cancel reason."""
    validator = _make_validator()
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    call_count = {"n": 0}

    def _should_stop():
        """Stay clear for two polls, then report a failed benchmark."""
        call_count["n"] += 1
        if call_count["n"] >= 3:
            return "INVALID_BENCHMARK_IN_FLIGHT: model failed"
        return None

    async def _run():
        """Stream 40 seeds behind a stop poll that trips on the third call."""
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=_FAKE_MODEL_ZIP,
                seeds=list(range(40)),
                phase_description="benchmark",
                seed_offset=0,
                epoch_number=1,
                hb=hb,
                should_stop=_should_stop,
                chunk_size=10,
            )
        finally:
            hb.finish()

    scores, _per_type, _details, cancel = asyncio.run(_run())

    assert cancel == "backend stop_required: INVALID_BENCHMARK_IN_FLIGHT: model failed"
    assert len(scores) == 2


def test_streaming_phase_runs_when_should_stop_clear(monkeypatch):
    """A poll that never returns a reason lets all 20 seeds evaluate."""
    validator = _make_validator()
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    async def _run():
        """Stream 20 seeds behind a poll that always answers None."""
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=_FAKE_MODEL_ZIP,
                seeds=list(range(20)),
                phase_description="benchmark",
                seed_offset=0,
                epoch_number=1,
                hb=hb,
                should_stop=lambda: None,
                chunk_size=10,
            )
        finally:
            hb.finish()

    scores, _per_type, _details, cancel = asyncio.run(_run())

    assert cancel is None
    assert len(scores) == 20


def test_run_full_benchmark_stops_on_heartbeat_stop_required(tmp_path, monkeypatch):
    """A heartbeat stop mid-benchmark ends the run after one seed and carries the backend's code out."""
    model_path = tmp_path / "UID_66.zip"
    model_path.write_bytes(b"fake-model")

    uploads: list[list[dict]] = []

    async def _capture_upload(**kwargs):
        """Keep the posted batches of score rows and acknowledge each."""
        uploads.append(list(kwargs["scores"]))
        return {"recorded": True}

    async def _post_heartbeat(**_kwargs):
        """Acknowledge the heartbeat with no stop requested."""
        return {"recorded": True, "stop_required": False}

    validator = SimpleNamespace(
        docker_evaluator=_make_docker_evaluator(score=0.5),
        backend_api=SimpleNamespace(
            post_heartbeat=_post_heartbeat,
            post_seed_scores_batch=_capture_upload,
        ),
        seed_manager=SimpleNamespace(
            epoch_number=4,
            get_benchmark_seeds=lambda: list(range(900001, 900031)),
        ),
    )

    should_stop_calls = {"n": 0}
    original_should_stop = HeartbeatManager.should_stop

    def _should_stop(self):
        """Defer to the real check once, then demand a benchmark stop."""
        should_stop_calls["n"] += 1
        if should_stop_calls["n"] >= 2:
            return "INVALID_BENCHMARK_IN_FLIGHT: BENCHMARK_FAILED"
        return original_should_stop(self)

    monkeypatch.setattr(HeartbeatManager, "should_stop", _should_stop)

    async def _run():
        """Benchmark UID 66 through the real streaming path."""
        return await validator_evaluation._run_full_benchmark(
            validator, uid=66, model_path=model_path,
        )

    _avg, _per_type_avgs, scores, _per_type_raw, cancel = asyncio.run(_run())

    assert cancel is not None
    assert "INVALID_BENCHMARK_IN_FLIGHT" in cancel
    assert "BENCHMARK_FAILED" in cancel
    assert len(scores) == 1


def test_run_screening_stops_on_heartbeat_stop_required(tmp_path, monkeypatch):
    """A heartbeat stop during screening ends the run after one seed, carrying the conflict code out."""
    model_path = tmp_path / "UID_66.zip"
    model_path.write_bytes(b"fake-model")

    uploads: list[list[dict]] = []

    async def _capture_upload(**kwargs):
        """Save the posted batches of score rows and acknowledge each."""
        uploads.append(list(kwargs["scores"]))
        return {"recorded": True}

    async def _post_heartbeat(**_kwargs):
        """Acknowledge the heartbeat and request no stop."""
        return {"recorded": True, "stop_required": False}

    validator = SimpleNamespace(
        docker_evaluator=_make_docker_evaluator(score=0.4),
        backend_api=SimpleNamespace(
            post_heartbeat=_post_heartbeat,
            post_seed_scores_batch=_capture_upload,
        ),
        seed_manager=SimpleNamespace(
            epoch_number=2,
            get_screening_seeds=lambda: list(range(800001, 800026)),
        ),
    )

    should_stop_calls = {"n": 0}
    original_should_stop = HeartbeatManager.should_stop

    def _should_stop(self):
        """Defer to the real check once, then demand a screening stop."""
        should_stop_calls["n"] += 1
        if should_stop_calls["n"] >= 2:
            return "INVALID_SCREENING_IN_FLIGHT: SCREENING_FAILED"
        return original_should_stop(self)

    monkeypatch.setattr(HeartbeatManager, "should_stop", _should_stop)

    async def _run():
        """Screen UID 66 through the real streaming path."""
        return await validator_evaluation._run_screening(
            validator, uid=66, model_path=model_path,
        )

    _avg, scores, _per_type, cancel, _early = asyncio.run(_run())

    assert cancel is not None
    assert "INVALID_SCREENING_IN_FLIGHT" in cancel
    assert len(scores) == 1


def test_heartbeat_manager_ignores_none_response():
    """A backend that answers nothing at all leaves the evaluation running."""
    async def _run():
        """Post one heartbeat and report the stop state afterwards."""
        class _Api:
            """A backend stub whose heartbeat post returns nothing."""
            async def post_heartbeat(self, **_kwargs):
                """Answer with None instead of a response body."""
                return None

        hb = HeartbeatManager(_Api(), asyncio.get_event_loop())
        hb.start("evaluating_benchmark", uid=5, total=10)
        try:
            await hb._safe_heartbeat(0, hb._session_id)
            return hb.should_stop()
        finally:
            hb._stop_timer()

    assert asyncio.run(_run()) is None


def test_heartbeat_manager_ignores_response_without_stop_required():
    """A response with no ``stop_required`` key leaves the evaluation running."""
    async def _run():
        """Post one heartbeat and hand back the stop state."""
        class _Api:
            """A backend stub that acknowledges without ever asking for a halt."""
            async def post_heartbeat(self, **_kwargs):
                """Answer recorded and accepted, with no stop key."""
                return {"recorded": True, "accepted": True}

        hb = HeartbeatManager(_Api(), asyncio.get_event_loop())
        hb.start("evaluating_benchmark", uid=5, total=10)
        try:
            await hb._safe_heartbeat(0, hb._session_id)
            return hb.should_stop()
        finally:
            hb._stop_timer()

    assert asyncio.run(_run()) is None


def test_heartbeat_manager_stop_without_conflicts_uses_default_reason():
    """A stop carrying no conflicts still latches, under the bare reason ``stop_required``."""
    async def _run():
        """Post one heartbeat and return the latched reason."""
        class _Api:
            """A backend stub demanding a stop but listing no conflicts."""
            async def post_heartbeat(self, **_kwargs):
                """Answer with stop_required set and no conflict list."""
                return {"recorded": True, "stop_required": True}

        hb = HeartbeatManager(_Api(), asyncio.get_event_loop())
        hb.start("evaluating_benchmark", uid=5, total=10)
        try:
            await hb._safe_heartbeat(0, hb._session_id)
            return hb.should_stop()
        finally:
            hb._stop_timer()

    assert asyncio.run(_run()) == "stop_required"


def test_heartbeat_manager_handles_post_exception():
    """A heartbeat post that raises is swallowed and never latches a stop."""
    async def _run():
        """Post one heartbeat into a failing backend and report the stop state."""
        class _Api:
            """A backend stub whose heartbeat post raises a network error."""
            async def post_heartbeat(self, **_kwargs):
                """Raise a RuntimeError instead of answering."""
                raise RuntimeError("network down")

        hb = HeartbeatManager(_Api(), asyncio.get_event_loop())
        hb.start("evaluating_benchmark", uid=5, total=10)
        try:
            await hb._safe_heartbeat(0, hb._session_id)
            return hb.should_stop()
        finally:
            hb._stop_timer()

    assert asyncio.run(_run()) is None


def test_heartbeat_manager_ignores_stale_session_response():
    """A stop that arrives for a superseded session is discarded."""
    async def _run():
        """Post under an old session id after a new session opened, then report the stop state."""
        class _Api:
            """A backend stub that always answers with a stale conflict."""
            async def post_heartbeat(self, **_kwargs):
                """Answer with stop_required and a STALE conflict."""
                return {"recorded": True, "stop_required": True,
                        "conflicts": [{"code": "STALE", "message": "old"}]}

        hb = HeartbeatManager(_Api(), asyncio.get_event_loop())
        hb.start("evaluating_benchmark", uid=5, total=10)
        try:
            stale_session = hb._session_id
            hb.start("evaluating_benchmark", uid=6, total=10)
            await hb._safe_heartbeat(0, stale_session)
            return hb.should_stop()
        finally:
            hb._stop_timer()

    assert asyncio.run(_run()) is None


def test_heartbeat_manager_stop_latches_until_next_session():
    """Once latched, a stop survives later clean heartbeats until a new session opens."""
    async def _run():
        """Post a stopping heartbeat then a clean one, reporting the latched reason after each."""
        responses = [
            {"stop_required": True, "conflicts": [{"code": "X", "message": "first"}]},
            {"stop_required": False},
            {"stop_required": False},
        ]

        class _Api:
            """A backend stub replaying one stop then clean acknowledgements."""
            def __init__(self):
                """Start the cursor at the first scripted response."""
                self._idx = 0

            async def post_heartbeat(self, **_kwargs):
                """Return the next scripted reply, holding on the last one."""
                resp = responses[min(self._idx, len(responses) - 1)]
                self._idx += 1
                return resp

        hb = HeartbeatManager(_Api(), asyncio.get_event_loop())
        hb.start("evaluating_benchmark", uid=5, total=10)
        try:
            await hb._safe_heartbeat(0, hb._session_id)
            first = hb.should_stop()
            await hb._safe_heartbeat(10, hb._session_id)
            second = hb.should_stop()
            return first, second
        finally:
            hb._stop_timer()

    first, second = asyncio.run(_run())
    assert first is not None
    assert "X" in first
    assert second is not None
    assert "X" in second


def test_streaming_phase_polls_stop_per_seed(monkeypatch):
    """The stop poll is consulted once for every seed, 50 times over 50 seeds."""
    validator = _make_validator()
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    call_log: list[int] = []

    def _should_stop():
        """Log the poll and never ask for a halt."""
        call_log.append(len(call_log))
        return None

    async def _run():
        """Stream 50 seeds behind a poll that counts its calls."""
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=_FAKE_MODEL_ZIP,
                seeds=list(range(50)),
                phase_description="benchmark",
                seed_offset=0,
                epoch_number=1,
                hb=hb,
                should_stop=_should_stop,
                chunk_size=10,
            )
        finally:
            hb.finish()

    scores, _per_type, _details, cancel = asyncio.run(_run())

    assert cancel is None
    assert len(scores) == 50
    assert len(call_log) == 50


def _patch_fast_authorize(monkeypatch):
    """Strip the backoff out of ``authorize_with_retry`` so retry tests do not sleep."""
    real_authorize = validator_evaluation.authorize_with_retry

    async def _fast_authorize(auth_fn, **kwargs):
        """Call the real retry wrapper with the backoff delay forced to zero."""
        kwargs["base_delay"] = 0.0
        return await real_authorize(auth_fn, **kwargs)

    monkeypatch.setattr(validator_evaluation, "authorize_with_retry", _fast_authorize)


def test_streaming_phase_reauthorize_recovers_after_transport_failure(monkeypatch):
    """A 502 on the authorization check is retried, not treated as a denial, and the run finishes."""
    validator = _make_validator()
    monkeypatch.setattr(
        validator_utils, "_evaluate_seeds", _make_evaluate_stub(per_seed_delay=0.02),
    )
    _patch_fast_authorize(monkeypatch)

    calls = {"n": 0}

    async def _re_authorize():
        """Fail twice as a transport error, then approve."""
        calls["n"] += 1
        if calls["n"] <= 2:
            return {"error": "502 Bad Gateway", "transport_failure": True}
        return {"authorized": True}

    async def _run():
        """Stream 20 delayed seeds with a flaky authorization check."""
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=_FAKE_MODEL_ZIP,
                seeds=list(range(20)),
                phase_description="benchmark",
                seed_offset=0,
                epoch_number=1,
                hb=hb,
                chunk_size=10,
                re_authorize=_re_authorize,
                re_auth_interval_sec=0.01,
            )
        finally:
            hb.finish()

    scores, _per_type, _details, cancel = asyncio.run(_run())

    assert cancel is None
    assert len(scores) == 20
    assert calls["n"] >= 3


def test_streaming_phase_reauthorize_transport_exhaustion_raises(monkeypatch):
    """Transport failures that never clear raise BackendTransportError rather than cancelling the task."""
    from swarm.validator.backend_api import BackendTransportError

    validator = _make_validator()
    monkeypatch.setattr(
        validator_utils, "_evaluate_seeds", _make_evaluate_stub(per_seed_delay=0.02),
    )
    _patch_fast_authorize(monkeypatch)

    calls = {"n": 0}

    async def _re_authorize():
        """Report a transport failure on every attempt."""
        calls["n"] += 1
        return {"error": "connection reset", "transport_failure": True}

    async def _run():
        """Stream 20 delayed seeds against an authorization check that never recovers."""
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=_FAKE_MODEL_ZIP,
                seeds=list(range(20)),
                phase_description="benchmark",
                seed_offset=0,
                epoch_number=1,
                hb=hb,
                chunk_size=10,
                re_authorize=_re_authorize,
                re_auth_interval_sec=0.01,
            )
        finally:
            hb.finish()

    try:
        asyncio.run(_run())
    except BackendTransportError as exc:
        assert "connection reset" in str(exc)
    else:
        raise AssertionError("expected BackendTransportError to be raised")

    assert calls["n"] >= 3


def test_streaming_phase_reauthorize_real_denial_still_cancels(monkeypatch):
    """A genuine denial is not retried: the first refusal cancels the run."""
    validator = _make_validator()
    monkeypatch.setattr(
        validator_utils, "_evaluate_seeds", _make_evaluate_stub(per_seed_delay=0.02),
    )
    _patch_fast_authorize(monkeypatch)

    calls = {"n": 0}

    async def _re_authorize():
        """Refuse the task outright, blaming a rotated epoch."""
        calls["n"] += 1
        return {"authorized": False, "reason": "epoch rotated"}

    async def _run():
        """Stream 30 delayed seeds against a check that refuses at once."""
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=_FAKE_MODEL_ZIP,
                seeds=list(range(30)),
                phase_description="benchmark",
                seed_offset=0,
                epoch_number=1,
                hb=hb,
                chunk_size=10,
                re_authorize=_re_authorize,
                re_auth_interval_sec=0.01,
            )
        finally:
            hb.finish()

    scores, _per_type, _details, cancel = asyncio.run(_run())

    assert cancel == "epoch rotated"
    assert len(scores) < 30
    assert calls["n"] == 1


def test_run_screening_heartbeat_includes_assignment_id(monkeypatch):
    """Heartbeats sent during screening must carry assignment_id so the
    backend's consistency report does not flag ACTIVE_HEARTBEAT_MISSING_
    ASSIGNMENT_ID for task-lease validators."""
    heartbeat_calls: list[dict] = []
    validator = _make_validator(heartbeat_calls=heartbeat_calls)
    validator.seed_manager = SimpleNamespace(
        epoch_number=12,
        get_screening_seeds=lambda: list(range(20)),
    )
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    async def _run():
        """Screen UID 314 under assignment 4242."""
        return await validator_evaluation._run_screening(
            validator, uid=314, model_path=_FAKE_MODEL_ZIP,
            task_id=4242,
        )

    asyncio.run(_run())

    sent_with_active = [c for c in heartbeat_calls if c.get("active_task")]
    assert sent_with_active, "expected at least one heartbeat with active_task"
    active = sent_with_active[0]["active_task"]
    assert active.get("assignment_id") == 4242
    assert active.get("uid") == 314
    assert active.get("phase") == "SCREENING"
    assert active.get("family_id") == "cf_autopilot"
    assert active.get("epoch_number") == 12


def test_run_full_benchmark_heartbeat_includes_assignment_id(monkeypatch):
    """Benchmark heartbeats name the assignment, the UID, the BENCHMARK phase, the family and the epoch."""
    heartbeat_calls: list[dict] = []
    validator = _make_validator(heartbeat_calls=heartbeat_calls)
    validator.seed_manager = SimpleNamespace(
        epoch_number=12,
        get_benchmark_seeds=lambda: list(range(20)),
    )
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    async def _run():
        """Benchmark UID 271 under assignment 8888."""
        return await validator_evaluation._run_full_benchmark(
            validator, uid=271, model_path=_FAKE_MODEL_ZIP,
            task_id=8888,
        )

    asyncio.run(_run())

    sent_with_active = [c for c in heartbeat_calls if c.get("active_task")]
    assert sent_with_active, "expected at least one heartbeat with active_task"
    active = sent_with_active[0]["active_task"]
    assert active.get("assignment_id") == 8888
    assert active.get("uid") == 271
    assert active.get("phase") == "BENCHMARK"
    assert active.get("family_id") == "cf_autopilot"
    assert active.get("epoch_number") == 12


def test_run_full_benchmark_reeval_heartbeat_includes_assignment_id(monkeypatch):
    """A re-evaluation labels its heartbeat phase REEVAL while still carrying the assignment and family."""
    heartbeat_calls: list[dict] = []
    validator = _make_validator(heartbeat_calls=heartbeat_calls)
    validator.seed_manager = SimpleNamespace(
        epoch_number=12,
        get_benchmark_seeds=lambda: list(range(20)),
    )
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    async def _run():
        """Re-evaluate UID 42 under assignment 999."""
        return await validator_evaluation._run_full_benchmark(
            validator, uid=42, model_path=_FAKE_MODEL_ZIP,
            task_id=999, reeval=True,
        )

    asyncio.run(_run())

    sent_with_active = [c for c in heartbeat_calls if c.get("active_task")]
    assert sent_with_active
    active = sent_with_active[0]["active_task"]
    assert active.get("phase") == "REEVAL"
    assert active.get("assignment_id") == 999
    assert active.get("family_id") == "cf_autopilot"


def test_run_full_benchmark_resume_reports_cumulative_progress(monkeypatch):
    """When resuming benchmark with seeds_from > 300, the heartbeat must
    report the FULL benchmark range (800) as total and the offset
    (seeds_from - 300) as already-done. Otherwise the dashboard shows
    a misleading 0/(remaining) right after a validator restart."""
    heartbeat_calls: list[dict] = []
    validator = _make_validator(heartbeat_calls=heartbeat_calls)
    validator.seed_manager = SimpleNamespace(
        epoch_number=12,
        get_benchmark_seeds=lambda: list(range(800)),
    )
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    async def _run():
        """Benchmark UID 42 resuming from seed 400 of 800."""
        return await validator_evaluation._run_full_benchmark(
            validator, uid=42, model_path=_FAKE_MODEL_ZIP,
            task_id=8888, seeds_from=400,
        )

    asyncio.run(_run())

    sent = [c for c in heartbeat_calls if c.get("active_task")]
    assert sent
    initial = sent[0]
    assert initial["total_seeds"] == 800
    assert initial["progress"] == 100  # 400 - 300 already done


def test_run_screening_resume_reports_cumulative_progress(monkeypatch):
    """A screening resumed at seed 50 reports 200 total and 50 already done, not a fresh start."""
    heartbeat_calls: list[dict] = []
    validator = _make_validator(heartbeat_calls=heartbeat_calls)
    validator.seed_manager = SimpleNamespace(
        epoch_number=12,
        get_screening_seeds=lambda: list(range(200)),
    )
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    async def _run():
        """Screen UID 314 resuming from seed 50 of 200."""
        return await validator_evaluation._run_screening(
            validator, uid=314, model_path=_FAKE_MODEL_ZIP,
            task_id=4242, seeds_from=50,
        )

    asyncio.run(_run())

    sent = [c for c in heartbeat_calls if c.get("active_task")]
    assert sent
    initial = sent[0]
    assert initial["total_seeds"] == 200
    assert initial["progress"] == 50


def test_run_screening_resume_uses_family_specific_seed_slice(monkeypatch):
    """Resuming screening slices the seed list of the named family, not the default one."""
    validator = _make_validator()
    validator.seed_manager = SimpleNamespace(
        epoch_number=12,
        get_screening_seeds=lambda family_id="cf_search_and_rescue": (
            list(range(100, 160)) if family_id == "cf_autopilot" else list(range(200))
        ),
    )
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    captured_tasks: list[tuple[list[int], str]] = []

    def _fake_screening_tasks(*, sim_dt, seeds, family_id, offset, total_seed_count):
        """Record the seeds and family it was asked for and return one task each."""
        _ = sim_dt, offset, total_seed_count
        captured_tasks.append((list(seeds), family_id))
        return [SimpleNamespace(challenge_type=1) for _ in seeds]

    monkeypatch.setattr(validator_evaluation, "build_screening_tasks", _fake_screening_tasks)

    async def _run():
        """Screen the cf_autopilot family from seed 50 onward."""
        return await validator_evaluation._run_screening(
            validator,
            uid=314,
            model_path=_FAKE_MODEL_ZIP,
            family_id="cf_autopilot",
            task_id=4242,
            seeds_from=50,
        )

    asyncio.run(_run())
    assert captured_tasks == [(list(range(150, 160)), "cf_autopilot")]


def test_run_full_benchmark_resume_uses_family_specific_seed_slice(monkeypatch):
    """Resuming a benchmark reads the named family's own seed list, giving five seeds from index 305."""
    validator = _make_validator()
    validator.seed_manager = SimpleNamespace(
        epoch_number=12,
        get_benchmark_seeds=lambda family_id="cf_search_and_rescue": (
            list(range(1000, 1010)) if family_id == "cf_autopilot" else list(range(800))
        ),
    )
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    async def _run():
        """Benchmark the cf_autopilot family from seed 305 onward."""
        return await validator_evaluation._run_full_benchmark(
            validator,
            uid=42,
            model_path=_FAKE_MODEL_ZIP,
            family_id="cf_autopilot",
            task_id=8888,
            seeds_from=305,
        )

    avg, _per_type, scores, _raw, _cancel = asyncio.run(_run())
    assert avg == pytest.approx(0.75)
    assert len(scores) == 5


def test_run_streaming_phase_uploads_family_local_seed_indices(monkeypatch):
    """Uploaded indexes are family-local: the seed offset is added, giving 205 through 209."""
    validator = _make_validator()
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    uploads: list[list[int]] = []

    async def _capture(**kwargs):
        """Collect the seed indexes of one upload and acknowledge it."""
        uploads.append([item["seed_index"] for item in kwargs["scores"]])
        return {"recorded": True}

    validator.backend_api.post_seed_scores_batch = _capture

    async def _run():
        """Stream five seeds for cf_autopilot at offset 205."""
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=_FAKE_MODEL_ZIP,
                seeds=list(range(5)),
                phase_description="benchmark",
                family_id="cf_autopilot",
                seed_offset=205,
                epoch_number=1,
                hb=hb,
                chunk_size=5,
            )
        finally:
            hb.finish()

    asyncio.run(_run())
    assert uploads == [[205, 206, 207, 208, 209]]


def test_a_scored_seed_stays_in_flight_until_the_backend_acks_it(monkeypatch):
    """The engine lets a seed go the moment it finishes, but its score is only
    queued here; reporting it idle would hand the seed back before it lands."""
    validator = _make_validator()
    reported: list = []
    monkeypatch.setattr(
        HeartbeatManager,
        "set_in_flight",
        lambda _self, indexes: reported.append(list(indexes)),
    )

    detail = {
        "score": 0.9, "map_type": "city", "metric_key": "city",
        "failure_reason": "NONE",
    }

    async def _evaluate(_self, _uid, _model_path, seeds, *args, **kwargs):
        """Hold seed 0, deliver its score, release it, and return the single result."""
        on_held = kwargs["on_held_seeds"]
        on_held([0])
        kwargs["on_seed_result"](0, dict(detail))
        on_held([])
        return [detail["score"]], {"city": [detail["score"]]}, [dict(detail)]

    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _evaluate)

    async def _feeder(_free_slots):
        """Offer no further seeds and report the source exhausted."""
        return [], True

    async def _run():
        """Stream one seed with a feeder that has nothing left to give."""
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=_FAKE_MODEL_ZIP,
                seeds=[0],
                phase_description="benchmark",
                seed_offset=0,
                epoch_number=1,
                hb=hb,
                chunk_size=2,
                seed_feeder=_feeder,
                initial_pending=[],
            )
        finally:
            hb.finish()

    asyncio.run(_run())

    assert reported[0] == [0]
    assert reported[1] == [0], "seed dropped from the report while its score was unsent"
    assert reported[-1] == []
