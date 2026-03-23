from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from swarm.base import validator as base_validator_mod
from swarm.base.neuron import BaseNeuron
from swarm.validator.runtime_telemetry import ValidatorRuntimeTracker


def test_base_neuron_sync_updates_chain_sync_tracker(tmp_path) -> None:
    tracker = ValidatorRuntimeTracker(state_dir=tmp_path)
    saved: list[str] = []
    dummy = SimpleNamespace(
        runtime_tracker=tracker,
        check_registered=lambda: None,
        should_sync_metagraph=lambda: False,
        should_set_weights=lambda: False,
        save_state=lambda: saved.append("saved"),
    )

    BaseNeuron.sync(dummy)

    snapshot = tracker.snapshot_copy()
    assert saved == ["saved"]
    assert snapshot["chain_sync"]["last_success_at"] is not None
    assert snapshot["chain_sync"]["last_error"] == ""


def test_set_weights_updates_weight_tracker(monkeypatch, tmp_path) -> None:
    tracker = ValidatorRuntimeTracker(state_dir=tmp_path)
    dummy = SimpleNamespace(
        runtime_tracker=tracker,
        scores=np.array([1.0, 0.0], dtype=np.float32),
        metagraph=SimpleNamespace(uids=np.array([0, 1], dtype=np.int64)),
        config=SimpleNamespace(netuid=124),
        subtensor=SimpleNamespace(set_weights=lambda **kwargs: (True, "")),
        wallet=object(),
        spec_version=1,
    )

    monkeypatch.setattr(
        base_validator_mod,
        "process_weights_for_netuid",
        lambda **kwargs: (np.array([0, 1], dtype=np.int64), np.array([1.0, 0.0], dtype=np.float32)),
    )
    monkeypatch.setattr(
        base_validator_mod,
        "convert_weights_and_uids_for_emit",
        lambda **kwargs: ([0, 1], [65535, 0]),
    )

    base_validator_mod.BaseValidatorNeuron.set_weights(dummy)

    snapshot = tracker.snapshot_copy()
    assert snapshot["weights"]["last_attempt_at"] is not None
    assert snapshot["weights"]["last_success_at"] is not None
    assert snapshot["weights"]["last_nonzero_uids"] == 1
