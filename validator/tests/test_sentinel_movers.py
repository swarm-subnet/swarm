"""Solar movers start at step zero and update together on every step."""

from swarm.core.maps.solar.builder import SolarMovers


def test_movers_pose_at_reset_and_on_every_step(monkeypatch):
    """Reset poses both movers at zero before the first advance repeats that step."""
    calls = []
    monkeypatch.setattr(SolarMovers, "_load", lambda *args: None)
    monkeypatch.setattr(SolarMovers, "_advance_pickup", lambda self, step: calls.append(("pickup", step)))
    monkeypatch.setattr(SolarMovers, "_advance_bird", lambda self, step: calls.append(("bird", step)))
    movers = SolarMovers(0, "", {}, [], {})
    assert movers.step == -1
    assert calls == [("pickup", 0), ("bird", 0)]
    for step in range(3):
        movers.advance()
        assert calls[-2:] == [("pickup", step), ("bird", step)]
    movers.advance(9)
    assert calls[-2:] == [("pickup", 9), ("bird", 9)]
