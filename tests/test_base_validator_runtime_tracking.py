from __future__ import annotations

import threading
from types import MethodType, SimpleNamespace

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


def test_base_neuron_sync_delegates_validator_weight_helper(tmp_path) -> None:
    tracker = ValidatorRuntimeTracker(state_dir=tmp_path)
    saved: list[str] = []
    weight_calls: list[dict] = []
    dummy = SimpleNamespace(
        runtime_tracker=tracker,
        check_registered=lambda: None,
        should_sync_metagraph=lambda: False,
        _maybe_set_weights=lambda **kwargs: weight_calls.append(kwargs),
        save_state=lambda: saved.append("saved"),
    )

    BaseNeuron.sync(dummy)

    assert weight_calls == [{"source": "sync"}]
    assert saved == ["saved"]


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


def _weight_helper_dummy(*, ready: bool, set_result: bool = True):
    calls: list[str] = []
    dummy = SimpleNamespace(
        neuron_type="ValidatorNeuron",
        config=SimpleNamespace(
            neuron=SimpleNamespace(disable_set_weights=False, epoch_length=10)
        ),
        step=0,
        block=100,
        metagraph=SimpleNamespace(last_update=np.array([0], dtype=np.int64)),
        uid=0,
        _weights_ready_for_setting=ready,
        _set_weights_lock=threading.Lock(),
        _last_weight_set_attempt_at=0.0,
        _last_successful_weight_set_block=None,
        set_weights=lambda: calls.append("set") or set_result,
    )
    dummy._should_set_weights_due = MethodType(
        base_validator_mod.BaseValidatorNeuron._should_set_weights_due, dummy
    )
    dummy._maybe_set_weights = MethodType(
        base_validator_mod.BaseValidatorNeuron._maybe_set_weights, dummy
    )
    return dummy, calls


def test_background_weight_helper_sets_during_initial_step_when_ready(monkeypatch) -> None:
    monkeypatch.setattr(base_validator_mod, "WEIGHT_SETTER_RETRY_SEC", 0)
    dummy, calls = _weight_helper_dummy(ready=True)

    assert dummy._maybe_set_weights(source="background", allow_initial_step=True) is True
    assert calls == ["set"]
    assert dummy._last_weight_set_attempt_at > 0


def test_background_weight_helper_waits_for_backend_weights(monkeypatch) -> None:
    monkeypatch.setattr(base_validator_mod, "WEIGHT_SETTER_RETRY_SEC", 0)
    dummy, calls = _weight_helper_dummy(ready=False)

    assert dummy._maybe_set_weights(source="background", allow_initial_step=True) is False
    assert calls == []
