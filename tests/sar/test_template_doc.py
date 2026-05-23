from __future__ import annotations

from pathlib import Path


_TEMPLATE = Path(__file__).resolve().parents[2] / "swarm" / "submission_template" / "drone_agent.py"


def test_mentions_v5_and_drops_z():
    text = _TEMPLATE.read_text()
    assert "V5" in text
    assert "Search-and-Rescue" in text
    assert "search_clue_z" not in text
    assert "(Δx, Δy)" in text or "2D" in text
