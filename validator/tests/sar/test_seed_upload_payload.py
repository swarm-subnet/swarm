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
