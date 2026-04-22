from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from swarm.validator import utils as validator_utils
from swarm.validator.utils_parts import evaluation as validator_evaluation
from swarm.validator.utils_parts.heartbeat import HeartbeatManager


def _make_validator(
    upload_results=None,
    heartbeat_calls=None,
) -> SimpleNamespace:
    heartbeat_calls = heartbeat_calls if heartbeat_calls is not None else []

    async def _post_heartbeat(**kwargs):
        heartbeat_calls.append(kwargs)
        return {"ok": True}

    upload_sequence = iter(upload_results) if upload_results is not None else None

    async def _post_seed_scores_batch(**kwargs):
        if upload_sequence is None:
            return {"recorded": True}
        try:
            return next(upload_sequence)
        except StopIteration:
            return {"recorded": True}

    async def _authorize_task(*_args, **_kwargs):
        return {"authorized": True, "reason": "ok"}

    return SimpleNamespace(
        backend_api=SimpleNamespace(
            post_heartbeat=_post_heartbeat,
            post_seed_scores_batch=_post_seed_scores_batch,
            authorize_task=_authorize_task,
        ),
    )


def _heartbeat(validator) -> HeartbeatManager:
    return HeartbeatManager(validator.backend_api, asyncio.get_event_loop())


def _make_evaluate_stub(score_per_seed: float = 0.75, map_type: str = "city"):
    async def _evaluate(_self, _uid, _model_path, seeds, *args, **kwargs):
        scores = [score_per_seed] * len(seeds)
        per_type = {"city": [], "open": [], "mountain": [], "village": [],
                    "warehouse": [], "forest": [], "moving_platform": []}
        if map_type in per_type:
            per_type[map_type] = list(scores)
        details = [{"score": score_per_seed, "map_type": map_type} for _ in seeds]
        return scores, per_type, details
    return _evaluate


def test_streaming_phase_forwards_task_id_to_upload(monkeypatch):
    validator = _make_validator()
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    posted_task_ids: list = []

    async def _capture(**kwargs):
        posted_task_ids.append(kwargs.get("task_id"))
        return {"recorded": True}

    validator.backend_api.post_seed_scores_batch = _capture

    async def _run():
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=Path("/tmp/fake.zip"),
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
    validator = _make_validator()
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    attempts: list[int | None] = []

    async def _flaky(**kwargs):
        attempts.append(kwargs.get("task_id"))
        if len(attempts) == 1:
            return {"recorded": False, "detail": "transient"}
        return {"recorded": True}

    validator.backend_api.post_seed_scores_batch = _flaky

    async def _run():
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=Path("/tmp/fake.zip"),
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
    posted_batches: list[list[dict]] = []
    validator = _make_validator()

    async def _capture_upload(**kwargs):
        posted_batches.append(list(kwargs["scores"]))
        return {"recorded": True}

    validator.backend_api.post_seed_scores_batch = _capture_upload
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    async def _run():
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=Path("/tmp/fake.zip"),
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
    validator = _make_validator()
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    authorize_calls = {"n": 0}

    async def _re_authorize():
        authorize_calls["n"] += 1
        if authorize_calls["n"] >= 2:
            return {"authorized": False, "reason": "epoch rotated"}
        return {"authorized": True}

    async def _run():
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=Path("/tmp/fake.zip"),
                seeds=list(range(30)),
                phase_description="benchmark",
                seed_offset=0,
                epoch_number=1,
                hb=hb,
                chunk_size=10,
                re_authorize=_re_authorize,
            )
        finally:
            hb.finish()

    scores, _per_type, _details, cancel = asyncio.run(_run())

    assert cancel == "epoch rotated"
    assert len(scores) == 10


def test_streaming_phase_retries_failed_batches(monkeypatch):
    validator = _make_validator()
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    attempts: list[int] = []

    async def _flaky_upload(**kwargs):
        attempts.append(len(kwargs["scores"]))
        if len(attempts) == 1:
            return {"recorded": False, "detail": "transient backend error"}
        return {"recorded": True}

    validator.backend_api.post_seed_scores_batch = _flaky_upload

    async def _run():
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=Path("/tmp/fake.zip"),
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
    validator = _make_validator()
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    calls: list[str] = []

    async def _failing_then_succeed(**kwargs):
        calls.append("call")
        if len(calls) == 1:
            raise RuntimeError("network down")
        return {"recorded": True}

    validator.backend_api.post_seed_scores_batch = _failing_then_succeed

    async def _run():
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=Path("/tmp/fake.zip"),
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
    validator = _make_validator()
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    inflight_counts: list[int] = []
    currently_inflight = {"n": 0}

    async def _slow_upload(**kwargs):
        currently_inflight["n"] += 1
        inflight_counts.append(currently_inflight["n"])
        await asyncio.sleep(0.01)
        currently_inflight["n"] -= 1
        return {"recorded": True}

    validator.backend_api.post_seed_scores_batch = _slow_upload

    async def _run():
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=Path("/tmp/fake.zip"),
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
    validator = _make_validator()
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    records: list[dict] = []

    def _on_chunk(**info):
        records.append({"evaluated": info["evaluated"], "total": info["total"]})

    async def _run():
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=Path("/tmp/fake.zip"),
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
    validator = _make_validator()

    async def _run():
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=Path("/tmp/fake.zip"),
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
    validator = _make_validator()
    uploads: list[list[dict]] = []

    async def _capture_upload(**kwargs):
        uploads.append(list(kwargs["scores"]))
        return {"recorded": True}

    validator.backend_api.post_seed_scores_batch = _capture_upload

    async def _evaluate(_self, _uid, _model_path, seeds, *args, **kwargs):
        scores = [0.5] * len(seeds)
        per_type = {name: [] for name in (
            "city", "open", "mountain", "village", "warehouse", "forest", "moving_platform",
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
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=Path("/tmp/fake.zip"),
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
            {"seed_index": 100, "score": 0.5, "map_type": "city"},
            {"seed_index": 102, "score": 0.5, "map_type": "city"},
            {"seed_index": 104, "score": 0.5, "map_type": "city"},
            {"seed_index": 106, "score": 0.5, "map_type": "city"},
            {"seed_index": 108, "score": 0.5, "map_type": "city"},
        ]
    ]


def test_streaming_phase_reauthorize_passes_first_then_fails(monkeypatch):
    validator = _make_validator()
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    calls = {"n": 0}

    async def _re_authorize():
        calls["n"] += 1
        if calls["n"] == 1:
            return {"authorized": True}
        if calls["n"] == 2:
            return {"authorized": True}
        return {"authorized": False, "reason": "model banned"}

    async def _run():
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=Path("/tmp/fake.zip"),
                seeds=list(range(40)),
                phase_description="benchmark",
                seed_offset=0,
                epoch_number=1,
                hb=hb,
                chunk_size=10,
                re_authorize=_re_authorize,
            )
        finally:
            hb.finish()

    scores, _per_type, _details, cancel = asyncio.run(_run())
    assert cancel == "model banned"
    assert len(scores) == 20
    assert calls["n"] == 3


def test_streaming_phase_final_retry_also_fails_does_not_raise(monkeypatch):
    validator = _make_validator()
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    async def _always_fail(**kwargs):
        return {"recorded": False, "detail": "backend offline"}

    validator.backend_api.post_seed_scores_batch = _always_fail

    async def _run():
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=Path("/tmp/fake.zip"),
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
    validator = _make_validator()

    evaluator_calls: list[dict] = []

    async def _evaluate(_self, _uid, _model_path, seeds, *args, **kwargs):
        evaluator_calls.append({
            "seeds_len": len(seeds),
            "prior_seeds_done": kwargs.get("prior_seeds_done"),
            "prior_total_seeds": kwargs.get("prior_total_seeds"),
        })
        scores = [0.5] * len(seeds)
        per_type = {name: [] for name in (
            "city", "open", "mountain", "village", "warehouse", "forest", "moving_platform",
        )}
        per_type["city"] = list(scores)
        details = [{"score": 0.5, "map_type": "city"} for _ in seeds]
        return scores, per_type, details

    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _evaluate)

    async def _run():
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=Path("/tmp/fake.zip"),
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

    assert evaluator_calls[0]["prior_seeds_done"] == 300
    assert evaluator_calls[0]["prior_total_seeds"] == 1000
    assert evaluator_calls[1]["prior_seeds_done"] == 310
    assert evaluator_calls[1]["prior_total_seeds"] == 1000


def test_streaming_phase_slices_pre_built_tasks_per_chunk(monkeypatch):
    validator = _make_validator()
    slices: list[list[object]] = []

    async def _evaluate(_self, _uid, _model_path, seeds, *args, **kwargs):
        slices.append(list(kwargs.get("pre_built_tasks") or []))
        scores = [0.5] * len(seeds)
        per_type = {name: [] for name in (
            "city", "open", "mountain", "village", "warehouse", "forest", "moving_platform",
        )}
        per_type["city"] = list(scores)
        details = [{"score": 0.5, "map_type": "city"} for _ in seeds]
        return scores, per_type, details

    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _evaluate)

    tasks = [f"task-{i}" for i in range(25)]

    async def _run():
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=Path("/tmp/fake.zip"),
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

    assert [len(s) for s in slices] == [10, 10, 5]
    assert slices[0] == tasks[:10]
    assert slices[1] == tasks[10:20]
    assert slices[2] == tasks[20:25]


def test_run_full_benchmark_streams_reeval_seeds(monkeypatch):
    validator = _make_validator()
    validator.seed_manager = SimpleNamespace(
        epoch_number=11,
        get_benchmark_seeds=lambda: list(range(20)),
    )

    uploads: list[list[dict]] = []

    async def _capture_upload(**kwargs):
        uploads.append(list(kwargs["scores"]))
        return {"recorded": True}

    validator.backend_api.post_seed_scores_batch = _capture_upload
    monkeypatch.setattr(
        validator_utils, "_evaluate_seeds", _make_evaluate_stub(map_type="open"),
    )

    async def _run():
        return await validator_evaluation._run_full_benchmark(
            validator,
            uid=42,
            model_path=Path("/tmp/fake.zip"),
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
    validator = _make_validator()
    validator.seed_manager = SimpleNamespace(
        epoch_number=11,
        get_benchmark_seeds=lambda: list(range(10)),
    )

    uploads: list[list[dict]] = []

    async def _capture_upload(**kwargs):
        uploads.append(list(kwargs["scores"]))
        return {"recorded": True}

    validator.backend_api.post_seed_scores_batch = _capture_upload
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    async def _run():
        return await validator_evaluation._run_full_benchmark(
            validator,
            uid=42,
            model_path=Path("/tmp/fake.zip"),
        )

    asyncio.run(_run())

    all_indices = [entry["seed_index"] for batch in uploads for entry in batch]
    assert all_indices == list(range(200, 210))


def test_run_screening_streams_with_unified_chunks(monkeypatch):
    validator = _make_validator()
    validator.seed_manager = SimpleNamespace(
        epoch_number=3,
        get_screening_seeds=lambda: list(range(25)),
    )

    uploads: list[list[dict]] = []

    async def _capture_upload(**kwargs):
        uploads.append(list(kwargs["scores"]))
        return {"recorded": True}

    validator.backend_api.post_seed_scores_batch = _capture_upload
    monkeypatch.setattr(
        validator_utils, "_evaluate_seeds", _make_evaluate_stub(map_type="forest"),
    )

    async def _run():
        return await validator_evaluation._run_screening(
            validator,
            uid=99,
            model_path=Path("/tmp/fake.zip"),
        )

    avg, scores, per_type, cancel = asyncio.run(_run())

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
    from swarm.protocol import ValidationResult

    async def _evaluate_seeds_parallel(tasks, uid, model_path, **kwargs):
        return [ValidationResult(int(uid), True, 1.0, float(score)) for _ in tasks]

    return SimpleNamespace(evaluate_seeds_parallel=_evaluate_seeds_parallel)


def test_run_full_benchmark_real_flow_streams_chunks(tmp_path):
    model_path = tmp_path / "UID_42.zip"
    model_path.write_bytes(b"fake-model")

    uploads: list[list[dict]] = []

    async def _capture_upload(**kwargs):
        uploads.append(list(kwargs["scores"]))
        return {"recorded": True}

    async def _post_heartbeat(**kwargs):
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
        return await validator_evaluation._run_full_benchmark(
            validator, uid=42, model_path=model_path,
        )

    avg, per_type_avgs, scores, per_type_raw, cancel = asyncio.run(_run())

    assert cancel is None
    assert len(scores) == 25
    assert avg == pytest.approx(0.81)
    assert [len(b) for b in uploads] == [10, 10, 5]
    all_indices = [entry["seed_index"] for batch in uploads for entry in batch]
    assert all_indices == list(range(200, 225))
    type_totals = sum(len(v) for v in per_type_raw.values())
    assert type_totals == 25


def test_run_screening_real_flow_streams_chunks(tmp_path):
    model_path = tmp_path / "UID_55.zip"
    model_path.write_bytes(b"fake-model")

    uploads: list[list[dict]] = []

    async def _capture_upload(**kwargs):
        uploads.append(list(kwargs["scores"]))
        return {"recorded": True}

    async def _post_heartbeat(**kwargs):
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
        return await validator_evaluation._run_screening(
            validator, uid=55, model_path=model_path,
        )

    avg, scores, per_type, cancel = asyncio.run(_run())

    assert cancel is None
    assert len(scores) == 15
    assert avg == pytest.approx(0.64)
    assert [len(b) for b in uploads] == [10, 5]
    all_indices = [entry["seed_index"] for batch in uploads for entry in batch]
    assert all_indices == list(range(15))


def test_queue_worker_real_flow_streams_and_cancels_on_auth(tmp_path):
    """End-to-end: real _process_normal_queue_item → real helper → real
    _evaluate_seeds → mocked docker evaluator + HTTP. Verifies auth cancel
    stops the run between chunks and preserves partial uploads.
    """
    model_path = tmp_path / "UID_77.zip"
    model_path.write_bytes(b"fake-model")

    uploads: list[list[dict]] = []
    auth_calls: list[str] = []

    async def _capture_upload(**kwargs):
        uploads.append(list(kwargs["scores"]))
        return {"recorded": True}

    async def _post_heartbeat(**kwargs):
        return {"ok": True}

    async def _authorize_task(uid, phase, **kwargs):
        auth_calls.append(str(phase))
        # Outer auth + helper chunk_0 auth + helper chunk_1 auth = 3 calls
        # The 4th BENCHMARK auth (before chunk_2) returns unauthorized.
        if phase == "BENCHMARK" and len(
            [p for p in auth_calls if p == "BENCHMARK"]
        ) > 3:
            return {"authorized": False, "reason": "epoch rotated",
                    "decision_version": 1, "task_id": 1}
        return {"authorized": True, "reason": "ok",
                "decision_version": 1, "task_id": 1}

    validator = SimpleNamespace(
        docker_evaluator=_make_docker_evaluator(score=0.5),
        backend_api=SimpleNamespace(
            post_heartbeat=_post_heartbeat,
            post_seed_scores_batch=_capture_upload,
            authorize_task=_authorize_task,
        ),
        seed_manager=SimpleNamespace(
            epoch_number=5,
            get_benchmark_seeds=lambda: list(range(500001, 500051)),
        ),
        metagraph=SimpleNamespace(hotkeys=["hotkey0", "hotkey1", "hotkey2"]),
    )

    key = "1:hash1"
    item = {
        "uid": 1,
        "model_hash": "hash1",
        "model_path": str(model_path),
        "status": "pending",
        "registered": False,
        "screening_recorded": False,
        "score_recorded": False,
        "retry_attempts": 0,
        "next_retry_at": 0,
        "last_error": "",
        "created_at": 0,
        "updated_at": 0,
    }
    queue = {"items": {key: item}}

    # Stub out everything NOT under test so the benchmark streaming path
    # is the only thing exercised.
    from swarm.validator import utils as validator_utils

    async def _register(*args, **kwargs):
        return True, False, ""

    async def _submit_screening(*args, **kwargs):
        return True, False, ""

    async def _submit_score(*args, **kwargs):
        return True, False, ""

    async def _run_screening(*args, **kwargs):
        return 0.85, [0.85], {"city": [0.85]}, None

    save_calls: list[bool] = []

    def _save_queue(_q):
        save_calls.append(True)

    import pytest as _pytest
    monkey = _pytest.MonkeyPatch()
    try:
        monkey.setattr(validator_utils, "sha256sum", lambda _p: "hash1")
        monkey.setattr(validator_utils, "has_cached_score", lambda *_: False)
        monkey.setattr(validator_utils, "set_cached_score",
                       lambda *args, **kwargs: None)
        monkey.setattr(validator_utils, "mark_model_hash_processed",
                       lambda *args, **kwargs: None)
        monkey.setattr(validator_utils, "save_normal_model_queue", _save_queue)
        monkey.setattr(validator_utils, "_register_new_model_with_ack", _register)
        monkey.setattr(validator_utils, "_run_screening", _run_screening)
        monkey.setattr(validator_utils, "_submit_screening_with_ack", _submit_screening)
        monkey.setattr(validator_utils, "_submit_score_with_ack", _submit_score)

        asyncio.run(validator_utils._process_normal_queue_item(
            validator,
            queue=queue,
            key=key,
            validator_hotkey="validator_hotkey",
            validator_stake=123.0,
        ))
    finally:
        monkey.undo()

    # Helper should have streamed 2 full chunks (20 seeds) then hit the
    # third BENCHMARK auth check which returns unauthorized.
    assert item["status"] == "cancelled"
    assert "epoch rotated" in item["last_error"]
    assert len(uploads) == 2
    assert [len(b) for b in uploads] == [10, 10]
    all_indices = [entry["seed_index"] for batch in uploads for entry in batch]
    assert all_indices == list(range(200, 220))
    assert item["benchmark_partial_scores"] == [0.5] * 20


def test_queue_worker_real_flow_streams_full_run(tmp_path):
    """End-to-end: normal queue item completes 30 benchmark seeds with
    streaming uploads in 10-seed chunks. Verifies uploads land during the
    run (not post-facto) and item state ends at completed.
    """
    model_path = tmp_path / "UID_88.zip"
    model_path.write_bytes(b"fake-model")

    uploads: list[list[dict]] = []
    heartbeats: list[dict] = []

    async def _capture_upload(**kwargs):
        uploads.append(list(kwargs["scores"]))
        return {"recorded": True}

    async def _post_heartbeat(**kwargs):
        heartbeats.append(kwargs)
        return {"ok": True}

    async def _authorize_task(uid, phase, **kwargs):
        return {"authorized": True, "reason": "ok",
                "decision_version": 1, "task_id": 1}

    validator = SimpleNamespace(
        docker_evaluator=_make_docker_evaluator(score=0.9),
        backend_api=SimpleNamespace(
            post_heartbeat=_post_heartbeat,
            post_seed_scores_batch=_capture_upload,
            authorize_task=_authorize_task,
        ),
        seed_manager=SimpleNamespace(
            epoch_number=5,
            get_benchmark_seeds=lambda: list(range(600001, 600031)),
        ),
        metagraph=SimpleNamespace(hotkeys=["hotkey0", "hotkey1", "hotkey2"]),
    )

    key = "1:hash1"
    item = {
        "uid": 1,
        "model_hash": "hash1",
        "model_path": str(model_path),
        "status": "pending",
        "registered": False,
        "screening_recorded": False,
        "score_recorded": False,
        "retry_attempts": 0,
        "next_retry_at": 0,
        "last_error": "",
        "created_at": 0,
        "updated_at": 0,
    }
    queue = {"items": {key: item}}

    from swarm.validator import utils as validator_utils

    async def _register(*args, **kwargs):
        return True, False, ""

    async def _submit_screening(*args, **kwargs):
        return True, False, ""

    async def _submit_score(*args, **kwargs):
        return True, False, ""

    async def _run_screening(*args, **kwargs):
        return 0.85, [0.85], {"city": [0.85]}, None

    import pytest as _pytest
    monkey = _pytest.MonkeyPatch()
    try:
        monkey.setattr(validator_utils, "sha256sum", lambda _p: "hash1")
        monkey.setattr(validator_utils, "has_cached_score", lambda *_: False)
        monkey.setattr(validator_utils, "set_cached_score",
                       lambda *args, **kwargs: None)
        monkey.setattr(validator_utils, "mark_model_hash_processed",
                       lambda *args, **kwargs: None)
        monkey.setattr(validator_utils, "save_normal_model_queue",
                       lambda _q: None)
        monkey.setattr(validator_utils, "_register_new_model_with_ack", _register)
        monkey.setattr(validator_utils, "_run_screening", _run_screening)
        monkey.setattr(validator_utils, "_submit_screening_with_ack", _submit_screening)
        monkey.setattr(validator_utils, "_submit_score_with_ack", _submit_score)

        asyncio.run(validator_utils._process_normal_queue_item(
            validator,
            queue=queue,
            key=key,
            validator_hotkey="validator_hotkey",
            validator_stake=123.0,
        ))
    finally:
        monkey.undo()

    assert item["status"] == "completed"
    assert item["screening_recorded"] is True
    assert item["score_recorded"] is True
    assert [len(b) for b in uploads] == [10, 10, 10]
    all_indices = [entry["seed_index"] for batch in uploads for entry in batch]
    assert all_indices == list(range(200, 230))
    assert item["seeds_evaluated"] == 31  # 1 screening + 30 benchmark
    assert item["total_score"] == pytest.approx((0.85 + 0.9 * 30) / 31)


# ── Re-eval kill-switch tests ────────────────────────────────────────────


def test_run_full_benchmark_reeval_authorizes_every_chunk(monkeypatch):
    validator = _make_validator()
    validator.seed_manager = SimpleNamespace(
        epoch_number=11,
        get_benchmark_seeds=lambda: list(range(30)),
    )

    auth_calls: list[tuple] = []

    async def _authorize(uid, phase, **kwargs):
        auth_calls.append((int(uid), str(phase), kwargs.get("epoch_number")))
        return {"authorized": True, "reason": "ok"}

    validator.backend_api.authorize_task = _authorize
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    async def _run():
        return await validator_evaluation._run_full_benchmark(
            validator, uid=42, model_path=Path("/tmp/fake.zip"),
            seeds=list(range(30)), reeval=True,
        )

    _avg, _per_type_avgs, scores, _per_type_raw, cancel = asyncio.run(_run())

    assert cancel is None
    assert len(scores) == 30
    assert len(auth_calls) == 3
    assert all(c == (42, "REEVAL", 11) for c in auth_calls)


def test_run_full_benchmark_reeval_cancels_mid_flight(monkeypatch):
    validator = _make_validator()
    validator.seed_manager = SimpleNamespace(
        epoch_number=11,
        get_benchmark_seeds=lambda: list(range(30)),
    )

    uploads: list[list[dict]] = []

    async def _capture_upload(**kwargs):
        uploads.append(list(kwargs["scores"]))
        return {"recorded": True}

    validator.backend_api.post_seed_scores_batch = _capture_upload
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    call_count = {"n": 0}

    async def _authorize(uid, phase, **kwargs):
        call_count["n"] += 1
        if call_count["n"] >= 3:
            return {"authorized": False, "reason": "epoch rotated"}
        return {"authorized": True, "reason": "ok"}

    validator.backend_api.authorize_task = _authorize

    async def _run():
        return await validator_evaluation._run_full_benchmark(
            validator, uid=42, model_path=Path("/tmp/fake.zip"),
            seeds=list(range(30)), reeval=True,
        )

    _avg, _per_type_avgs, scores, _per_type_raw, cancel = asyncio.run(_run())

    assert cancel == "epoch rotated"
    assert len(scores) == 20
    assert [len(b) for b in uploads] == [10, 10]


def test_run_full_benchmark_non_reeval_skips_authorize(monkeypatch):
    validator = _make_validator()
    validator.seed_manager = SimpleNamespace(
        epoch_number=11,
        get_benchmark_seeds=lambda: list(range(20)),
    )

    auth_calls: list[str] = []

    async def _authorize(*args, **kwargs):
        auth_calls.append("called")
        return {"authorized": True}

    validator.backend_api.authorize_task = _authorize
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    async def _run():
        return await validator_evaluation._run_full_benchmark(
            validator, uid=42, model_path=Path("/tmp/fake.zip"),
        )

    _avg, _per_type_avgs, scores, _per_type_raw, cancel = asyncio.run(_run())

    assert cancel is None
    assert len(scores) == 20
    assert auth_calls == []


def test_run_screening_reeval_authorizes_every_chunk(monkeypatch):
    validator = _make_validator()
    validator.seed_manager = SimpleNamespace(
        epoch_number=3,
        get_screening_seeds=lambda: list(range(25)),
    )

    auth_calls: list[tuple] = []

    async def _authorize(uid, phase, **kwargs):
        auth_calls.append((int(uid), str(phase), kwargs.get("epoch_number")))
        return {"authorized": True}

    validator.backend_api.authorize_task = _authorize
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    async def _run():
        return await validator_evaluation._run_screening(
            validator, uid=99, model_path=Path("/tmp/fake.zip"), reeval=True,
        )

    _avg, scores, _per_type, cancel = asyncio.run(_run())

    assert cancel is None
    assert len(scores) == 25
    assert len(auth_calls) == 3
    assert all(c == (99, "REEVAL", 3) for c in auth_calls)


def test_run_screening_reeval_cancels_mid_flight(monkeypatch):
    validator = _make_validator()
    validator.seed_manager = SimpleNamespace(
        epoch_number=3,
        get_screening_seeds=lambda: list(range(25)),
    )

    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    call_count = {"n": 0}

    async def _authorize(uid, phase, **kwargs):
        call_count["n"] += 1
        if call_count["n"] >= 2:
            return {"authorized": False, "reason": "model banned"}
        return {"authorized": True}

    validator.backend_api.authorize_task = _authorize

    async def _run():
        return await validator_evaluation._run_screening(
            validator, uid=99, model_path=Path("/tmp/fake.zip"), reeval=True,
        )

    _avg, scores, _per_type, cancel = asyncio.run(_run())

    assert cancel == "model banned"
    assert len(scores) == 10


def test_run_screening_non_reeval_skips_authorize(monkeypatch):
    validator = _make_validator()
    validator.seed_manager = SimpleNamespace(
        epoch_number=3,
        get_screening_seeds=lambda: list(range(15)),
    )

    auth_calls: list[str] = []

    async def _authorize(*args, **kwargs):
        auth_calls.append("called")
        return {"authorized": True}

    validator.backend_api.authorize_task = _authorize
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    async def _run():
        return await validator_evaluation._run_screening(
            validator, uid=99, model_path=Path("/tmp/fake.zip"),
        )

    _avg, scores, _per_type, cancel = asyncio.run(_run())

    assert cancel is None
    assert len(scores) == 15
    assert auth_calls == []


def test_run_full_benchmark_reeval_real_flow_cancel(tmp_path):
    model_path = tmp_path / "UID_77.zip"
    model_path.write_bytes(b"fake-model")

    uploads: list[list[dict]] = []
    auth_calls: list[str] = []

    async def _capture_upload(**kwargs):
        uploads.append(list(kwargs["scores"]))
        return {"recorded": True}

    async def _post_heartbeat(**kwargs):
        return {"ok": True}

    async def _authorize(uid, phase, **kwargs):
        auth_calls.append(str(phase))
        if len(auth_calls) >= 3:
            return {"authorized": False, "reason": "admin override"}
        return {"authorized": True}

    validator = SimpleNamespace(
        docker_evaluator=_make_docker_evaluator(score=0.7),
        backend_api=SimpleNamespace(
            post_heartbeat=_post_heartbeat,
            post_seed_scores_batch=_capture_upload,
            authorize_task=_authorize,
        ),
        seed_manager=SimpleNamespace(
            epoch_number=7,
            get_benchmark_seeds=lambda: list(range(700001, 700031)),
        ),
    )

    async def _run():
        return await validator_evaluation._run_full_benchmark(
            validator, uid=42, model_path=model_path,
            seeds=list(range(700001, 700031)), reeval=True,
        )

    _avg, _per_type_avgs, scores, _per_type_raw, cancel = asyncio.run(_run())

    assert cancel == "admin override"
    assert len(scores) == 20
    assert [len(b) for b in uploads] == [10, 10]
    assert auth_calls == ["REEVAL", "REEVAL", "REEVAL"]


def test_heartbeat_manager_honors_stop_required():
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
        class _Api:
            def __init__(self):
                self._idx = 0

            async def post_heartbeat(self, **_kwargs):
                resp = responses[min(self._idx, len(responses) - 1)]
                self._idx += 1
                return resp

        hb = HeartbeatManager(_Api(), asyncio.get_event_loop())
        hb.start("evaluating_screening", uid=66, total=200)
        await hb._safe_heartbeat(0, hb._session_id)
        assert hb.should_stop() is None
        await hb._safe_heartbeat(10, hb._session_id)
        return hb.should_stop()

    reason = asyncio.run(_run())
    assert reason is not None
    assert "INVALID_SCREENING_IN_FLIGHT" in reason
    assert "SCREENING_FAILED" in reason


def test_heartbeat_manager_start_resets_stop_flag():
    async def _run():
        class _Api:
            async def post_heartbeat(self, **_kwargs):
                return {"stop_required": True, "conflicts": [{"code": "X", "message": "y"}]}

        hb = HeartbeatManager(_Api(), asyncio.get_event_loop())
        hb.start("evaluating_screening", uid=1, total=10)
        await hb._safe_heartbeat(0, hb._session_id)
        assert hb.should_stop() is not None

        hb.start("evaluating_screening", uid=2, total=10)
        return hb.should_stop()

    assert asyncio.run(_run()) is None


def test_streaming_phase_stops_when_should_stop_fires(monkeypatch):
    validator = _make_validator()
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    call_count = {"n": 0}

    def _should_stop():
        call_count["n"] += 1
        if call_count["n"] >= 3:
            return "INVALID_BENCHMARK_IN_FLIGHT: model failed"
        return None

    async def _run():
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=Path("/tmp/fake.zip"),
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
    assert len(scores) == 20


def test_streaming_phase_runs_when_should_stop_clear(monkeypatch):
    validator = _make_validator()
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    async def _run():
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=Path("/tmp/fake.zip"),
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
    model_path = tmp_path / "UID_66.zip"
    model_path.write_bytes(b"fake-model")

    uploads: list[list[dict]] = []

    async def _capture_upload(**kwargs):
        uploads.append(list(kwargs["scores"]))
        return {"recorded": True}

    async def _post_heartbeat(**_kwargs):
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
        should_stop_calls["n"] += 1
        if should_stop_calls["n"] >= 2:
            return "INVALID_BENCHMARK_IN_FLIGHT: BENCHMARK_FAILED"
        return original_should_stop(self)

    monkeypatch.setattr(HeartbeatManager, "should_stop", _should_stop)

    async def _run():
        return await validator_evaluation._run_full_benchmark(
            validator, uid=66, model_path=model_path,
        )

    _avg, _per_type_avgs, scores, _per_type_raw, cancel = asyncio.run(_run())

    assert cancel is not None
    assert "INVALID_BENCHMARK_IN_FLIGHT" in cancel
    assert "BENCHMARK_FAILED" in cancel
    assert len(scores) == 10


def test_run_screening_stops_on_heartbeat_stop_required(tmp_path, monkeypatch):
    model_path = tmp_path / "UID_66.zip"
    model_path.write_bytes(b"fake-model")

    uploads: list[list[dict]] = []

    async def _capture_upload(**kwargs):
        uploads.append(list(kwargs["scores"]))
        return {"recorded": True}

    async def _post_heartbeat(**_kwargs):
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
        should_stop_calls["n"] += 1
        if should_stop_calls["n"] >= 2:
            return "INVALID_SCREENING_IN_FLIGHT: SCREENING_FAILED"
        return original_should_stop(self)

    monkeypatch.setattr(HeartbeatManager, "should_stop", _should_stop)

    async def _run():
        return await validator_evaluation._run_screening(
            validator, uid=66, model_path=model_path,
        )

    _avg, scores, _per_type, cancel = asyncio.run(_run())

    assert cancel is not None
    assert "INVALID_SCREENING_IN_FLIGHT" in cancel
    assert len(scores) == 10


def test_heartbeat_manager_ignores_none_response():
    async def _run():
        class _Api:
            async def post_heartbeat(self, **_kwargs):
                return None

        hb = HeartbeatManager(_Api(), asyncio.get_event_loop())
        hb.start("evaluating_benchmark", uid=5, total=10)
        await hb._safe_heartbeat(0, hb._session_id)
        return hb.should_stop()

    assert asyncio.run(_run()) is None


def test_heartbeat_manager_ignores_response_without_stop_required():
    async def _run():
        class _Api:
            async def post_heartbeat(self, **_kwargs):
                return {"recorded": True, "accepted": True}

        hb = HeartbeatManager(_Api(), asyncio.get_event_loop())
        hb.start("evaluating_benchmark", uid=5, total=10)
        await hb._safe_heartbeat(0, hb._session_id)
        return hb.should_stop()

    assert asyncio.run(_run()) is None


def test_heartbeat_manager_stop_without_conflicts_uses_default_reason():
    async def _run():
        class _Api:
            async def post_heartbeat(self, **_kwargs):
                return {"recorded": True, "stop_required": True}

        hb = HeartbeatManager(_Api(), asyncio.get_event_loop())
        hb.start("evaluating_benchmark", uid=5, total=10)
        await hb._safe_heartbeat(0, hb._session_id)
        return hb.should_stop()

    assert asyncio.run(_run()) == "stop_required"


def test_heartbeat_manager_handles_post_exception():
    async def _run():
        class _Api:
            async def post_heartbeat(self, **_kwargs):
                raise RuntimeError("network down")

        hb = HeartbeatManager(_Api(), asyncio.get_event_loop())
        hb.start("evaluating_benchmark", uid=5, total=10)
        await hb._safe_heartbeat(0, hb._session_id)
        return hb.should_stop()

    assert asyncio.run(_run()) is None


def test_heartbeat_manager_ignores_stale_session_response():
    async def _run():
        class _Api:
            async def post_heartbeat(self, **_kwargs):
                return {"recorded": True, "stop_required": True,
                        "conflicts": [{"code": "STALE", "message": "old"}]}

        hb = HeartbeatManager(_Api(), asyncio.get_event_loop())
        hb.start("evaluating_benchmark", uid=5, total=10)
        stale_session = hb._session_id
        hb.start("evaluating_benchmark", uid=6, total=10)
        await hb._safe_heartbeat(0, stale_session)
        return hb.should_stop()

    assert asyncio.run(_run()) is None


def test_heartbeat_manager_stop_latches_until_next_session():
    async def _run():
        responses = [
            {"stop_required": True, "conflicts": [{"code": "X", "message": "first"}]},
            {"stop_required": False},
            {"stop_required": False},
        ]

        class _Api:
            def __init__(self):
                self._idx = 0

            async def post_heartbeat(self, **_kwargs):
                resp = responses[min(self._idx, len(responses) - 1)]
                self._idx += 1
                return resp

        hb = HeartbeatManager(_Api(), asyncio.get_event_loop())
        hb.start("evaluating_benchmark", uid=5, total=10)
        await hb._safe_heartbeat(0, hb._session_id)
        first = hb.should_stop()
        await hb._safe_heartbeat(10, hb._session_id)
        second = hb.should_stop()
        return first, second

    first, second = asyncio.run(_run())
    assert first is not None
    assert "X" in first
    assert second is not None
    assert "X" in second


def test_streaming_phase_checks_stop_each_chunk(monkeypatch):
    validator = _make_validator()
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())

    call_log: list[int] = []

    def _should_stop():
        call_log.append(len(call_log))
        return None

    async def _run():
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=Path("/tmp/fake.zip"),
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
    assert len(call_log) == 5


def test_streaming_phase_reauthorize_recovers_after_transport_failure(monkeypatch):
    validator = _make_validator()
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())
    monkeypatch.setattr(
        "swarm.validator.backend_api.AUTHORIZE_RETRY_BASE_DELAY_SEC", 0.0,
    )

    calls = {"n": 0}

    async def _re_authorize():
        calls["n"] += 1
        if calls["n"] <= 2:
            return {"error": "502 Bad Gateway", "transport_failure": True}
        return {"authorized": True}

    async def _run():
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=Path("/tmp/fake.zip"),
                seeds=list(range(20)),
                phase_description="benchmark",
                seed_offset=0,
                epoch_number=1,
                hb=hb,
                chunk_size=10,
                re_authorize=_re_authorize,
            )
        finally:
            hb.finish()

    scores, _per_type, _details, cancel = asyncio.run(_run())

    assert cancel is None
    assert len(scores) == 20
    assert calls["n"] >= 3


def test_streaming_phase_reauthorize_transport_exhaustion_raises(monkeypatch):
    from swarm.validator.backend_api import BackendTransportError

    validator = _make_validator()
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())
    monkeypatch.setattr(
        "swarm.validator.backend_api.AUTHORIZE_RETRY_BASE_DELAY_SEC", 0.0,
    )

    calls = {"n": 0}

    async def _re_authorize():
        calls["n"] += 1
        return {"error": "connection reset", "transport_failure": True}

    async def _run():
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=Path("/tmp/fake.zip"),
                seeds=list(range(20)),
                phase_description="benchmark",
                seed_offset=0,
                epoch_number=1,
                hb=hb,
                chunk_size=10,
                re_authorize=_re_authorize,
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
    validator = _make_validator()
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _make_evaluate_stub())
    monkeypatch.setattr(
        "swarm.validator.backend_api.AUTHORIZE_RETRY_BASE_DELAY_SEC", 0.0,
    )

    calls = {"n": 0}

    async def _re_authorize():
        calls["n"] += 1
        return {"authorized": False, "reason": "epoch rotated"}

    async def _run():
        hb = _heartbeat(validator)
        try:
            return await validator_evaluation._run_streaming_phase(
                validator,
                uid=7,
                model_path=Path("/tmp/fake.zip"),
                seeds=list(range(30)),
                phase_description="benchmark",
                seed_offset=0,
                epoch_number=1,
                hb=hb,
                chunk_size=10,
                re_authorize=_re_authorize,
            )
        finally:
            hb.finish()

    scores, _per_type, _details, cancel = asyncio.run(_run())

    assert cancel == "epoch rotated"
    assert calls["n"] == 1
