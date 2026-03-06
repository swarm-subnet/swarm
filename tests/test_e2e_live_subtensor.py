from __future__ import annotations

import os

import pytest


@pytest.mark.e2e
@pytest.mark.integration
@pytest.mark.slow
def test_e2e_live_subtensor_metagraph_query():
    import bittensor as bt  # type: ignore

    network = os.getenv("SWARM_LIVE_BT_NETWORK", "finney")
    netuid = int(os.getenv("SWARM_LIVE_BT_NETUID", "124"))

    subtensor = bt.subtensor(network=network)
    metagraph = subtensor.metagraph(netuid=netuid)

    assert hasattr(metagraph, "hotkeys")
    assert hasattr(metagraph, "S")
    assert len(metagraph.hotkeys) > 0
    assert len(metagraph.S) == len(metagraph.hotkeys)
