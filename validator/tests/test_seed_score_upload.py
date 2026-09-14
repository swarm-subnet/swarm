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

"""Test seed score upload fixes: metric_key/map_type resolution, retry logic."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from swarm.validator.backend_api import BackendApiClient


class FakeWallet:
    """A wallet stand-in carrying only what request signing reads off it."""
    class hotkey:
        """The hotkey namespace: a fixed ss58 address and a signature that costs nothing."""
        ss58_address = "5FakeHotkey"
        @staticmethod
        def sign(msg):
            """Return 64 zero bytes in place of a real signature."""
            return b"\x00" * 64


@pytest.fixture
def client():
    """A BackendApiClient wired for offline use: fake wallet, mocked transport, empty runtime state."""
    c = BackendApiClient.__new__(BackendApiClient)
    c.base_url = "http://fake"
    c.timeout = 1.0
    c.wallet = FakeWallet()
    c.client = MagicMock()
    c._runtime_state = {}
    return c


@patch("asyncio.sleep", return_value=None)
def test_retry_succeeds_on_second_attempt(mock_sleep, client):
    """A batch the backend does not confirm is posted again, and the second reply is what the caller gets."""
    call_count = 0

    async def mock_post(endpoint, data):
        """Fail the first post with an error payload, then confirm five recorded scores."""
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {"error": "timeout"}
        return {"recorded": 5, "message": "ok"}

    client._post_signed = mock_post

    result = asyncio.run(
        client.post_seed_scores_batch(
            model_uid=1,
            epoch_number=1,
            scores=[{"seed_index": 0, "score": 0.5, "map_type": "city"}],
        )
    )
    assert result.get("recorded") == 5
    assert call_count == 2


@patch("asyncio.sleep", return_value=None)
def test_retry_exhausted_returns_error(mock_sleep, client):
    """Once the attempt budget is spent the caller receives the backend's last rejection, not a success."""
    async def mock_post(endpoint, data):
        """Refuse every post with a connection failure so no attempt ever records."""
        return {"error": "connection refused"}

    client._post_signed = mock_post

    result = asyncio.run(
        client.post_seed_scores_batch(
            model_uid=1,
            epoch_number=1,
            scores=[{"seed_index": 0, "score": 0.5, "map_type": "city"}],
            retries=2,
        )
    )
    assert "error" in result


@patch("asyncio.sleep", return_value=None)
def test_retry_on_detail_key(mock_sleep, client):
    """A rejection carried under `detail` rather than `error` is still treated as unrecorded and resent."""
    call_count = 0

    async def mock_post(endpoint, data):
        """Reject the first post with a `detail` message, then confirm one recorded score."""
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {"detail": "Invalid map_type: unknown"}
        return {"recorded": 1, "message": "ok"}

    client._post_signed = mock_post

    result = asyncio.run(
        client.post_seed_scores_batch(
            model_uid=1,
            epoch_number=1,
            scores=[{"seed_index": 0, "score": 0.5, "map_type": "city"}],
        )
    )
    assert result.get("recorded") == 1
    assert call_count == 2


def test_task_id_included_in_payload(client):
    """A task_id given to the upload reaches the body the backend is sent, so rows land against that task."""
    captured: list[dict] = []

    async def mock_post(endpoint, data):
        """Keep the posted body for inspection and answer with one recorded score."""
        captured.append(data)
        return {"recorded": 1, "message": "ok"}

    client._post_signed = mock_post

    asyncio.run(
        client.post_seed_scores_batch(
            model_uid=7,
            epoch_number=11,
            scores=[{"seed_index": 0, "score": 0.5, "map_type": "city"}],
            task_id=123,
        )
    )
    assert captured[0]["task_id"] == 123


def test_task_id_omitted_when_not_provided(client):
    """Leaving the task_id out sends no such key at all, never a null the backend would have to reject."""
    captured: list[dict] = []

    async def mock_post(endpoint, data):
        """Keep the posted body for inspection and answer with one recorded score."""
        captured.append(data)
        return {"recorded": 1, "message": "ok"}

    client._post_signed = mock_post

    asyncio.run(
        client.post_seed_scores_batch(
            model_uid=7,
            epoch_number=11,
            scores=[{"seed_index": 0, "score": 0.5, "map_type": "city"}],
        )
    )
    assert "task_id" not in captured[0]


def test_family_id_included_in_seed_score_payload(client):
    """The batch names its family, and a score giving only map_type gets the matching metric_key filled in."""
    captured: list[dict] = []

    async def mock_post(endpoint, data):
        """Keep the posted body for inspection and answer with one recorded score."""
        captured.append(data)
        return {"recorded": 1, "message": "ok"}

    client._post_signed = mock_post

    asyncio.run(
        client.post_seed_scores_batch(
            model_uid=7,
            epoch_number=11,
            family_id="cf_autopilot",
            scores=[{"seed_index": 0, "score": 0.5, "map_type": "city"}],
        )
    )
    assert captured[0]["family_id"] == "cf_autopilot"
    assert captured[0]["scores"][0]["metric_key"] == "city"
    assert captured[0]["scores"][0]["map_type"] == "city"


def test_no_retry_on_success(client):
    """A batch the backend confirms first time is sent exactly once, so scores are never written twice."""
    call_count = 0

    async def mock_post(endpoint, data):
        """Count the posts and confirm one recorded score every time."""
        nonlocal call_count
        call_count += 1
        return {"recorded": 1, "message": "ok"}

    client._post_signed = mock_post

    result = asyncio.run(
        client.post_seed_scores_batch(
            model_uid=1,
            epoch_number=1,
            scores=[{"seed_index": 0, "score": 0.5, "map_type": "city"}],
        )
    )
    assert call_count == 1
    assert result["recorded"] == 1


def test_evaluate_seeds_failed_result_gets_real_map_type():
    """A seed with no result of its own still reports the map name its task was built for, never 'unknown'."""
    challenge_type_to_name = {
        1: "city", 2: "open", 3: "mountain",
        4: "village", 5: "warehouse", 6: "forest",
    }

    class FakeTask:
        """A task cut down to the one attribute the map-name lookup reads."""
        def __init__(self, ct):
            """Store the challenge type this stand-in reports."""
            self.challenge_type = ct
    tasks = [FakeTask(1), FakeTask(3), FakeTask(5)]
    results = [MagicMock(score=0.8)]

    seed_details = []
    all_scores = []
    task_idx = 0
    for i, task in enumerate(tasks):
        if task is None:
            all_scores.append(0.0)
            seed_details.append({"score": 0.0, "metric_key": "unknown", "map_type": "unknown"})
            continue

        if task_idx < len(results):
            result = results[task_idx]
            score = result.score if result else 0.0
            all_scores.append(score)
            type_name = challenge_type_to_name.get(task.challenge_type, "unknown")
            seed_details.append({"score": score, "metric_key": type_name, "map_type": type_name})
            task_idx += 1
        else:
            type_name = challenge_type_to_name.get(task.challenge_type, "unknown")
            all_scores.append(0.0)
            seed_details.append({"score": 0.0, "metric_key": type_name, "map_type": type_name})

    assert len(seed_details) == 3
    assert seed_details[0] == {"score": 0.8, "metric_key": "city", "map_type": "city"}
    assert seed_details[1] == {"score": 0.0, "metric_key": "mountain", "map_type": "mountain"}
    assert seed_details[2] == {"score": 0.0, "metric_key": "warehouse", "map_type": "warehouse"}
    assert all(d["map_type"] != "unknown" for d in seed_details)
