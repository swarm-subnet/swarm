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

from swarm.protocol import MapTask


def _base(**over):
    base = dict(
        map_seed=1,
        start=(0.0, 0.0, 1.0),
        goal=(5.0, 5.0, 1.0),
        sim_dt=1 / 240,
        horizon=60.0,
        challenge_type=1,
    )
    base.update(over)
    return base


def test_round_trip():
    task = MapTask(**_base(search_centre=(7.5, -3.25)))
    blob = task.pack()
    back = MapTask.unpack(blob)
    assert tuple(back.search_centre) == (7.5, -3.25)
    assert back.map_seed == task.map_seed
    assert tuple(back.goal) == task.goal


def test_default_zero():
    task = MapTask(**_base())
    assert task.search_centre == (0.0, 0.0)
    back = MapTask.unpack(task.pack())
    assert tuple(back.search_centre) == (0.0, 0.0)
