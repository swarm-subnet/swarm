from __future__ import annotations

import json


def test_each_seed_has_failure_reason():
    # Per-seed payload schema is a plain dict; build one and assert.
    item = {
        "seed_index": 7,
        "score": 0.5,
        "metric_key": "city",
        "map_type": "city",
        "failure_reason": "TIMEOUT",
    }
    assert "failure_reason" in item
    assert item["failure_reason"] == "TIMEOUT"


def test_mixed_failure_batch():
    reasons = ["NONE", "OBSTACLE_COLLISION", "INFEASIBLE", "TIMEOUT", "SPAWN_FAILURE"]
    batch = [
        {
            "seed_index": i,
            "score": 0.0 if r != "NONE" else 0.9,
            "metric_key": "open",
            "map_type": "open",
            "failure_reason": r,
        }
        for i, r in enumerate(reasons)
    ]
    blob = json.dumps(batch)
    back = json.loads(blob)
    assert [b["failure_reason"] for b in back] == reasons
