# The MIT License (MIT)
# Copyright © 2026 Swarm

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Neuron config surface: the CUDA probe, the log path it builds, and the argparse defaults."""
from __future__ import annotations

import argparse
from types import SimpleNamespace

from swarm.utils import config as config_mod


def test_is_cuda_available_prefers_nvidia_smi(monkeypatch):
    """A machine whose nvidia-smi lists a GPU resolves to the cuda device without consulting nvcc."""
    def _check_output(cmd, stderr=None):
        """Answer the nvidia-smi probe with one listed GPU and raise on anything else."""
        _ = stderr
        if cmd[:2] == ["nvidia-smi", "-L"]:
            return b"GPU 0: NVIDIA A100"
        raise RuntimeError("unexpected")

    monkeypatch.setattr(config_mod.subprocess, "check_output", _check_output)
    assert config_mod.is_cuda_available() == "cuda"


def test_is_cuda_available_falls_back_to_nvcc(monkeypatch):
    """A missing nvidia-smi is not fatal: the nvcc release banner still yields the cuda device, probed once."""
    calls = {"nvidia": 0}

    def _check_output(cmd, stderr=None):
        """Fail the nvidia-smi probe, counting it, and answer nvcc with a release banner."""
        _ = stderr
        if cmd[:2] == ["nvidia-smi", "-L"]:
            calls["nvidia"] += 1
            raise RuntimeError("missing nvidia-smi")
        return b"Cuda compilation tools, release 12.4"

    monkeypatch.setattr(config_mod.subprocess, "check_output", _check_output)
    assert config_mod.is_cuda_available() == "cuda"
    assert calls["nvidia"] == 1


def test_is_cuda_available_returns_cpu_when_checks_fail(monkeypatch):
    """Both probes raising leaves the device string at cpu rather than propagating the error."""
    monkeypatch.setattr(config_mod.subprocess, "check_output", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no")))
    assert config_mod.is_cuda_available() == "cpu"


def test_check_config_sets_full_path_and_registers_events_logger(monkeypatch, tmp_path, bt_stub):
    """The neuron log path nests cold key, hotkey, netuid and neuron name, and the events logger becomes the primary one."""
    registered = {"name": None}
    checked = {"called": False}

    def _check_config(_cfg):
        """Record that bittensor was asked to validate the config namespace."""
        checked["called"] = True

    def _register(name):
        """Record the logger name handed to bittensor as the primary logger."""
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
    """Parsing an empty command line gives netuid 1, mock off, and the device the CUDA probe reported."""
    monkeypatch.setattr(config_mod, "is_cuda_available", lambda: "cpu")
    parser = argparse.ArgumentParser()
    config_mod.add_args(object, parser)
    ns = parser.parse_args([])
    assert ns.netuid == 1
    assert ns.mock is False
    assert ns.__dict__["neuron.device"] == "cpu"


def test_add_miner_args_defaults():
    """A miner parser demands a validator permit and 1000 TAO of stake unless told otherwise."""
    parser = argparse.ArgumentParser()
    config_mod.add_miner_args(object, parser)
    ns = parser.parse_args([])
    assert ns.__dict__["blacklist.force_validator_permit"] is True
    assert ns.__dict__["blacklist.minimum_stake_requirement"] == 1000


def test_add_validator_args_defaults():
    """A validator parser starts at a 10 second forward timeout and a 4096 TAO vpermit ceiling."""
    parser = argparse.ArgumentParser()
    config_mod.add_validator_args(object, parser)
    ns = parser.parse_args([])
    assert ns.__dict__["neuron.timeout"] == 10
    assert ns.__dict__["neuron.vpermit_tao_limit"] == 4096


def test_config_builds_namespace_from_cls_add_args(monkeypatch):
    """Flags a neuron class registers for itself survive into the returned config namespace."""
    class _Dummy:
        """Stand-in neuron class that registers a single integer flag."""
        @staticmethod
        def add_args(parser):
            """Register the custom flag, which defaults to 7."""
            parser.add_argument("--custom-flag", type=int, default=7)

    ns = config_mod.config(_Dummy)
    assert ns.custom_flag == 7
