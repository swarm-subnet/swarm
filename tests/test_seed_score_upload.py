"""Test seed score upload fixes: map_type resolution, retry logic."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from swarm.validator.backend_api import BackendApiClient


class FakeWallet:
    class hotkey:
        ss58_address = "5FakeHotkey"
        @staticmethod
        def sign(msg):
            return b"\x00" * 64


@pytest.fixture
def client():
    c = BackendApiClient.__new__(BackendApiClient)
    c.base_url = "http://fake"
    c.timeout = 1.0
    c.wallet = FakeWallet()
    c.client = MagicMock()
    c._runtime_state = {}
    return c


@pytest.mark.asyncio
@patch("asyncio.sleep", return_value=None)
async def test_retry_succeeds_on_second_attempt(mock_sleep, client):
    call_count = 0

    async def mock_post(endpoint, data):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {"error": "timeout"}
        return {"recorded": 5, "message": "ok"}

    client._post_signed = mock_post

    result = await client.post_seed_scores_batch(
        model_uid=1, epoch_number=1, scores=[{"seed_index": 0, "score": 0.5, "map_type": "city"}],
    )
    assert result.get("recorded") == 5
    assert call_count == 2


@pytest.mark.asyncio
@patch("asyncio.sleep", return_value=None)
async def test_retry_exhausted_returns_error(mock_sleep, client):
    async def mock_post(endpoint, data):
        return {"error": "connection refused"}

    client._post_signed = mock_post

    result = await client.post_seed_scores_batch(
        model_uid=1, epoch_number=1, scores=[{"seed_index": 0, "score": 0.5, "map_type": "city"}],
        retries=2,
    )
    assert "error" in result


@pytest.mark.asyncio
@patch("asyncio.sleep", return_value=None)
async def test_retry_on_detail_key(mock_sleep, client):
    call_count = 0

    async def mock_post(endpoint, data):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {"detail": "Invalid map_type: unknown"}
        return {"recorded": 1, "message": "ok"}

    client._post_signed = mock_post

    result = await client.post_seed_scores_batch(
        model_uid=1, epoch_number=1, scores=[{"seed_index": 0, "score": 0.5, "map_type": "city"}],
    )
    assert result.get("recorded") == 1
    assert call_count == 2


@pytest.mark.asyncio
async def test_no_retry_on_success(client):
    call_count = 0

    async def mock_post(endpoint, data):
        nonlocal call_count
        call_count += 1
        return {"recorded": 1, "message": "ok"}

    client._post_signed = mock_post

    result = await client.post_seed_scores_batch(
        model_uid=1, epoch_number=1, scores=[{"seed_index": 0, "score": 0.5, "map_type": "city"}],
    )
    assert call_count == 1
    assert result["recorded"] == 1


def test_evaluate_seeds_failed_result_gets_real_map_type():
    challenge_type_to_name = {
        1: "city", 2: "open", 3: "mountain",
        4: "village", 5: "warehouse", 6: "forest",
    }

    class FakeTask:
        def __init__(self, ct):
            self.challenge_type = ct
            self.moving_platform = False

    tasks = [FakeTask(1), FakeTask(3), FakeTask(5)]
    results = [MagicMock(score=0.8)]

    seed_details = []
    all_scores = []
    task_idx = 0
    for i, task in enumerate(tasks):
        if task is None:
            all_scores.append(0.0)
            seed_details.append({"score": 0.0, "map_type": "unknown"})
            continue

        if task_idx < len(results):
            result = results[task_idx]
            score = result.score if result else 0.0
            all_scores.append(score)
            type_name = challenge_type_to_name.get(task.challenge_type, "unknown")
            seed_details.append({"score": score, "map_type": type_name})
            task_idx += 1
        else:
            type_name = challenge_type_to_name.get(task.challenge_type, "unknown")
            all_scores.append(0.0)
            seed_details.append({"score": 0.0, "map_type": type_name})

    assert len(seed_details) == 3
    assert seed_details[0] == {"score": 0.8, "map_type": "city"}
    assert seed_details[1] == {"score": 0.0, "map_type": "mountain"}
    assert seed_details[2] == {"score": 0.0, "map_type": "warehouse"}
    assert all(d["map_type"] != "unknown" for d in seed_details)
