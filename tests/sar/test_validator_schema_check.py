from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from swarm.protocol import (
    FailureReason,
    MapTask,
    SCHEMA_VERSION,
    is_supported_schema,
    normalize_version,
)
from swarm.validator.docker.docker_evaluator_parts.batch import (
    _BatchContext,
    _BatchHelpers,
    _validate_inputs,
)


def test_normalize_v_prefix():
    assert normalize_version("V5.0.0") == "5.0.0"
    assert normalize_version("v5.0.0") == "5.0.0"
    assert normalize_version("5.0.0") == "5.0.0"
    assert normalize_version("1") == "1"


def _make_task(version: str) -> MapTask:
    return MapTask(
        map_seed=1,
        start=(0.0, 0.0, 1.0),
        goal=(5.0, 0.0, 1.0),
        sim_dt=1 / 240,
        horizon=60.0,
        challenge_type=1,
        version=version,
    )


def _make_ctx(tasks):
    helpers = MagicMock(spec=_BatchHelpers)
    helpers.notify_all_failed = MagicMock()
    ctx = MagicMock(spec=_BatchContext)
    ctx.uid = 42
    ctx.worker_id = 0
    ctx.tasks = tasks
    ctx.model_path = Path("/nonexistent/model.zip")
    ctx.helpers = helpers
    return ctx, helpers


def test_v1_now_rejected_post_cutover():
    """D.3.1 cutover: allow-list collapsed to {SCHEMA_VERSION}; V1 tasks are
    rejected with EVAL_ERROR instead of warned-then-passed."""
    tasks = [_make_task("1")]
    ctx, helpers = _make_ctx(tasks)
    result = _validate_inputs(ctx)
    helpers.notify_all_failed.assert_called_once_with(status="unsupported_schema_version")
    assert result is not None
    assert all(r.failure_reason == FailureReason.EVAL_ERROR.value for r in result)


def test_v5_accepted():
    tasks = [_make_task(SCHEMA_VERSION)]
    ctx, helpers = _make_ctx(tasks)
    result = _validate_inputs(ctx)
    helpers.notify_all_failed.assert_called_once()
    status = helpers.notify_all_failed.call_args.kwargs.get("status")
    assert status != "unsupported_schema_version"
    assert result is not None
    assert all(r.failure_reason == "NONE" for r in result)


def test_missing_version_skipped():
    """Absent task.version is permitted so existing test mocks (SimpleNamespace
    without a version attr) keep working. Production tasks always carry a
    version since task_gen.random_task emits SCHEMA_VERSION."""
    class _NoVersion:
        pass
    ctx, helpers = _make_ctx([_NoVersion()])
    result = _validate_inputs(ctx)
    assert result is not None
    helpers.notify_all_failed.assert_called_once()
    status = helpers.notify_all_failed.call_args.kwargs.get("status")
    assert status != "unsupported_schema_version"
    assert all(r.failure_reason == "NONE" for r in result)


def test_unknown_rejected():
    tasks = [_make_task("9.9.9")]
    ctx, helpers = _make_ctx(tasks)
    result = _validate_inputs(ctx)
    assert result is not None
    helpers.notify_all_failed.assert_called_once_with(status="unsupported_schema_version")
    assert all(r.failure_reason == FailureReason.EVAL_ERROR.value for r in result)
