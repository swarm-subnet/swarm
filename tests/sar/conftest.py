from __future__ import annotations

import pytest
import pybullet as p


@pytest.fixture
def sar_pybullet():
    cli = p.connect(p.DIRECT)
    try:
        yield cli
    finally:
        try:
            p.disconnect(cli)
        except Exception:
            pass
