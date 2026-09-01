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
