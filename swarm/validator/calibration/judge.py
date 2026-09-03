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

"""Per-act scoring decision in baseline-equivalent time.

Kept as a pure function so the fairness logic can be unit-tested without the
Docker/RPC harness. The validator measures the real ``act()`` time, converts it
to baseline-equivalent compute, and decides the step against a fixed budget, so
two validators of different speed reach the same verdict on the same model.
"""
from __future__ import annotations

from dataclasses import dataclass


def act_hard_cap_sec(
    speed_factor: float, overhead_sec: float, *, ref_sec: float, margin_sec: float
) -> float:
    """Liveness ceiling for one act() in local seconds (generous, DoS-only)."""
    return ref_sec * float(speed_factor) + float(overhead_sec) + float(margin_sec)


@dataclass(frozen=True)
class StepVerdict:
    strike: bool            # discard the returned action and count a strike
    hard_cap_hit: bool      # exceeded the liveness ceiling (harder failure)
    normalized_sec: float   # baseline-equivalent compute time


def judge_act(
    elapsed_sec: float,
    *,
    overhead_sec: float,
    speed_factor: float,
    budget_sec: float,
    hard_cap_sec: float,
) -> StepVerdict:
    """Judge a single act() call.

    The action is accepted only when its baseline-equivalent compute is within
    budget; a returned-but-too-slow action is discarded (strike). Exceeding the
    liveness hard cap is a separate, harder failure handled by the caller.
    """
    if speed_factor <= 0:
        raise ValueError("speed_factor must be positive")
    hard_cap_hit = float(elapsed_sec) >= float(hard_cap_sec)
    normalized = max(0.0, float(elapsed_sec) - float(overhead_sec)) / float(speed_factor)
    strike = hard_cap_hit or normalized > float(budget_sec)
    return StepVerdict(strike=strike, hard_cap_hit=hard_cap_hit, normalized_sec=normalized)
