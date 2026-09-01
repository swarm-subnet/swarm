from __future__ import annotations

import threading
import time
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
        _subtensor_lock=threading.RLock(),
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
        _subtensor_lock=threading.RLock(),
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


def test_subtensor_lock_serializes_sync_and_set_weights(monkeypatch, tmp_path) -> None:
    """set_weights from background thread blocks while sync holds _subtensor_lock."""
    tracker = ValidatorRuntimeTracker(state_dir=tmp_path)
    lock = threading.RLock()
    gate = threading.Event()
    order: list[str] = []

    def blocking_check_registered():
        order.append("sync_enter")
        gate.wait(timeout=3)
        order.append("sync_exit")

    sync_dummy = SimpleNamespace(
        runtime_tracker=tracker,
        _subtensor_lock=lock,
        check_registered=blocking_check_registered,
        should_sync_metagraph=lambda: False,
        _maybe_set_weights=lambda **kw: None,
        save_state=lambda: None,
    )

    def mock_set_weights(**kwargs):
        order.append("weights_call")
        return (True, "")

    weights_dummy = SimpleNamespace(
        runtime_tracker=tracker,
        _subtensor_lock=lock,
        _scores_lock=threading.RLock(),
        scores=np.array([1.0, 0.0], dtype=np.float32),
        metagraph=SimpleNamespace(uids=np.array([0, 1], dtype=np.int64)),
        config=SimpleNamespace(netuid=124),
        subtensor=SimpleNamespace(set_weights=mock_set_weights),
        wallet=object(),
        spec_version=1,
        block=100,
    )

    monkeypatch.setattr(
        base_validator_mod, "process_weights_for_netuid",
        lambda **kw: (np.array([0, 1], dtype=np.int64), np.array([1.0, 0.0], dtype=np.float32)),
    )
    monkeypatch.setattr(
        base_validator_mod, "convert_weights_and_uids_for_emit",
        lambda **kw: ([0, 1], [65535, 0]),
    )

    sync_thread = threading.Thread(target=lambda: BaseNeuron.sync(sync_dummy))
    sync_thread.start()
    time.sleep(0.05)

    weights_thread = threading.Thread(
        target=lambda: base_validator_mod.BaseValidatorNeuron.set_weights(weights_dummy),
    )
    weights_thread.start()
    time.sleep(0.05)

    assert "sync_enter" in order
    assert "weights_call" not in order, "set_weights must wait for sync to release the lock"

    gate.set()
    sync_thread.join(timeout=3)
    weights_thread.join(timeout=3)

    assert "sync_exit" in order
    assert "weights_call" in order
    sync_idx = order.index("sync_exit")
    weights_idx = order.index("weights_call")
    assert sync_idx < weights_idx, "set_weights must run after sync releases the lock"


def test_maybe_set_weights_blocked_by_sync(monkeypatch, tmp_path) -> None:
    """_maybe_set_weights (background path) blocks while sync holds _subtensor_lock."""
    monkeypatch.setattr(base_validator_mod, "WEIGHT_SETTER_RETRY_SEC", 0)
    tracker = ValidatorRuntimeTracker(state_dir=tmp_path)
    lock = threading.RLock()
    gate = threading.Event()
    order: list[str] = []

    def blocking_check_registered():
        order.append("sync_enter")
        gate.wait(timeout=3)
        order.append("sync_exit")

    sync_dummy = SimpleNamespace(
        runtime_tracker=tracker,
        _subtensor_lock=lock,
        check_registered=blocking_check_registered,
        should_sync_metagraph=lambda: False,
        _maybe_set_weights=lambda **kw: None,
        save_state=lambda: None,
    )

    bg_dummy = SimpleNamespace(
        neuron_type="ValidatorNeuron",
        config=SimpleNamespace(
            neuron=SimpleNamespace(disable_set_weights=False, epoch_length=10),
        ),
        step=0,
        block=100,
        metagraph=SimpleNamespace(last_update=np.array([0], dtype=np.int64)),
        uid=0,
        _weights_ready_for_setting=True,
        _subtensor_lock=lock,
        _set_weights_lock=threading.Lock(),
        _last_weight_set_attempt_at=0.0,
        _last_successful_weight_set_block=None,
        set_weights=lambda: order.append("bg_weights") or True,
    )
    bg_dummy._should_set_weights_due = MethodType(
        base_validator_mod.BaseValidatorNeuron._should_set_weights_due, bg_dummy,
    )
    bg_dummy._maybe_set_weights = MethodType(
        base_validator_mod.BaseValidatorNeuron._maybe_set_weights, bg_dummy,
    )

    sync_thread = threading.Thread(target=lambda: BaseNeuron.sync(sync_dummy))
    sync_thread.start()
    time.sleep(0.05)

    bg_thread = threading.Thread(
        target=lambda: bg_dummy._maybe_set_weights(source="background", allow_initial_step=True),
    )
    bg_thread.start()
    time.sleep(0.05)

    assert "sync_enter" in order
    assert "bg_weights" not in order, "_maybe_set_weights must block on _subtensor_lock"

    gate.set()
    sync_thread.join(timeout=3)
    bg_thread.join(timeout=3)

    assert "sync_exit" in order
    assert "bg_weights" in order
    assert order.index("sync_exit") < order.index("bg_weights")
