from __future__ import annotations

import swarm
from swarm import protocol


def test_schema_and_package_versions_aligned():
    assert protocol.SCHEMA_VERSION == "5.0.0"
    assert swarm.__version__.startswith("5.")
