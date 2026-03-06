from __future__ import annotations

import argparse
from types import SimpleNamespace

from swarm.utils import config as config_mod


def test_is_cuda_available_prefers_nvidia_smi(monkeypatch):
    def _check_output(cmd, stderr=None):
        _ = stderr
        if cmd[:2] == ["nvidia-smi", "-L"]:
            return b"GPU 0: NVIDIA A100"
        raise RuntimeError("unexpected")

    monkeypatch.setattr(config_mod.subprocess, "check_output", _check_output)
    assert config_mod.is_cuda_available() == "cuda"


def test_is_cuda_available_falls_back_to_nvcc(monkeypatch):
    calls = {"nvidia": 0}

    def _check_output(cmd, stderr=None):
        _ = stderr
        if cmd[:2] == ["nvidia-smi", "-L"]:
            calls["nvidia"] += 1
            raise RuntimeError("missing nvidia-smi")
        return b"Cuda compilation tools, release 12.4"

    monkeypatch.setattr(config_mod.subprocess, "check_output", _check_output)
    assert config_mod.is_cuda_available() == "cuda"
    assert calls["nvidia"] == 1


def test_is_cuda_available_returns_cpu_when_checks_fail(monkeypatch):
    monkeypatch.setattr(config_mod.subprocess, "check_output", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no")))
    assert config_mod.is_cuda_available() == "cpu"


def test_check_config_sets_full_path_and_registers_events_logger(monkeypatch, tmp_path, bt_stub):
    registered = {"name": None}
    checked = {"called": False}

    def _check_config(_cfg):
        checked["called"] = True

    def _register(name):
        registered["name"] = name

    monkeypatch.setattr(bt_stub.logging, "check_config", _check_config)
    monkeypatch.setattr(bt_stub.logging, "register_primary_logger", _register)
    monkeypatch.setattr(
        config_mod,
        "setup_events_logger",
        lambda full_path, events_retention_size: SimpleNamespace(name="events"),
    )

    cfg = SimpleNamespace(
        logging=SimpleNamespace(logging_dir=str(tmp_path / "logs")),
        wallet=SimpleNamespace(name="cold", hotkey="hot"),
        netuid=124,
        neuron=SimpleNamespace(
            name="validator",
            dont_save_events=False,
            events_retention_size=1024,
        ),
    )

    config_mod.check_config(object, cfg)
    assert checked["called"] is True
    assert cfg.neuron.full_path.endswith("cold/hot/netuid124/validator")
    assert registered["name"] == "events"


def test_add_args_registers_common_flags(monkeypatch):
    monkeypatch.setattr(config_mod, "is_cuda_available", lambda: "cpu")
    parser = argparse.ArgumentParser()
    config_mod.add_args(object, parser)
    ns = parser.parse_args([])
    assert ns.netuid == 1
    assert ns.mock is False
    assert ns.__dict__["neuron.device"] == "cpu"


def test_add_miner_args_defaults():
    parser = argparse.ArgumentParser()
    config_mod.add_miner_args(object, parser)
    ns = parser.parse_args([])
    assert ns.__dict__["blacklist.force_validator_permit"] is True
    assert ns.__dict__["blacklist.minimum_stake_requirement"] == 1000


def test_add_validator_args_defaults():
    parser = argparse.ArgumentParser()
    config_mod.add_validator_args(object, parser)
    ns = parser.parse_args([])
    assert ns.__dict__["neuron.timeout"] == 10
    assert ns.__dict__["neuron.vpermit_tao_limit"] == 4096


def test_config_builds_namespace_from_cls_add_args(monkeypatch):
    class _Dummy:
        @staticmethod
        def add_args(parser):
            parser.add_argument("--custom-flag", type=int, default=7)

    ns = config_mod.config(_Dummy)
    assert ns.custom_flag == 7
