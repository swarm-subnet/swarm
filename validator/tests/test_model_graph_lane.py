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

"""The legacy model-graph lane: champions crowned before the code-agent switch
stay runnable for re-evaluation, while new submissions remain code-agent only.
"""
from __future__ import annotations

import json
import zipfile
from types import SimpleNamespace

import pytest

from swarm.core.model_verify import classify_model_validity, inspect_model_structure
from swarm.core.submission_lane import (
    MODEL_GRAPH_INTERFACE_VERSION,
    RUNNABLE_INTERFACE_VERSIONS,
    graph_declared_family,
    is_model_graph_artifact,
)
from swarm.core.submission_policy import SUBMISSION_INTERFACE_VERSION
from swarm.validator.docker.docker_evaluator_parts import batch


def _graph_manifest(family_id: str = "cf_autopilot") -> dict:
    return {
        "model_graph_version": MODEL_GRAPH_INTERFACE_VERSION,
        "family_id": family_id,
        "execution_profile": "swarm.onnx-neural.cpu.v1",
        "runner_abi": "graph_runner.v1",
        "models": [{"id": "policy", "file": "models/policy.onnx", "sha256": "00" * 32}],
        "memory": [],
        "nodes": [{"id": "policy", "model": "policy", "inputs": {}, "every_n_steps": 1}],
        "action": "policy.action",
    }


def _write_graph_artifact(path, manifest: dict | None = None) -> object:
    body = _graph_manifest() if manifest is None else manifest
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("manifest.json", json.dumps(body))
        zf.writestr("models/policy.onnx", b"onnx-bytes")
    return path


def _write_code_agent(path, extra: dict | None = None) -> object:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("drone_agent.py", "class DroneFlightController:\n    pass\n")
        for name, body in (extra or {}).items():
            zf.writestr(name, body)
    return path


# ── detection ────────────────────────────────────────────────────────────────

def test_graph_artifact_is_detected(tmp_path):
    artifact = _write_graph_artifact(tmp_path / "graph.zip")
    assert is_model_graph_artifact(artifact) is True


def test_code_agent_is_not_a_graph_artifact(tmp_path):
    agent = _write_code_agent(tmp_path / "agent.zip")
    assert is_model_graph_artifact(agent) is False


def test_drone_agent_wins_over_a_planted_graph_manifest(tmp_path):
    """A code agent shipping a graph manifest must still run as a code agent."""
    agent = _write_code_agent(
        tmp_path / "mixed.zip", {"manifest.json": json.dumps(_graph_manifest())}
    )
    assert is_model_graph_artifact(agent) is False


@pytest.mark.parametrize(
    "manifest",
    [
        {"model_graph_version": "model_graph.v2"},
        {"family_id": "cf_autopilot"},
        [],
    ],
)
def test_manifest_without_the_graph_marker_is_not_a_graph_artifact(tmp_path, manifest):
    path = tmp_path / "other.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest))
    assert is_model_graph_artifact(path) is False


def test_unreadable_or_corrupt_archive_is_not_a_graph_artifact(tmp_path):
    corrupt = tmp_path / "corrupt.zip"
    corrupt.write_bytes(b"not a zip at all")
    assert is_model_graph_artifact(corrupt) is False
    assert is_model_graph_artifact(tmp_path / "absent.zip") is False


def test_oversized_manifest_is_refused_without_reading_it(tmp_path):
    path = tmp_path / "bomb.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", " " * (2 * 1024 * 1024))
    assert is_model_graph_artifact(path) is False


def test_declared_family_comes_from_the_graph_manifest(tmp_path):
    artifact = _write_graph_artifact(
        tmp_path / "sar.zip", _graph_manifest("cf_search_and_rescue")
    )
    assert graph_declared_family(artifact) == "cf_search_and_rescue"
    assert graph_declared_family(_write_code_agent(tmp_path / "agent.zip")) is None


def test_both_lanes_are_runnable():
    assert RUNNABLE_INTERFACE_VERSIONS == {
        SUBMISSION_INTERFACE_VERSION,
        MODEL_GRAPH_INTERFACE_VERSION,
    }


# ── verification ─────────────────────────────────────────────────────────────

def test_graph_artifact_verifies_instead_of_failing_as_missing_drone_agent(tmp_path):
    """The champions predate drone_agent.py; rejecting them here would strand them."""
    artifact = _write_graph_artifact(tmp_path / "graph.zip")
    inspection = inspect_model_structure(artifact)
    assert inspection.get("missing_drone_agent") is None
    status, _reason = classify_model_validity(inspection)
    assert status == "legitimate"


def test_a_code_agent_without_its_entry_point_is_still_rejected(tmp_path):
    path = tmp_path / "empty.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("README.md", "nothing here")
    status, _reason = classify_model_validity(inspect_model_structure(path))
    assert status == "missing_drone_agent"


# ── fetch admission ──────────────────────────────────────────────────────────

def test_fetch_admits_a_graph_champion_and_binds_its_family(tmp_path):
    from swarm.validator.utils_parts.model_fetch import _admitted

    artifact = _write_graph_artifact(
        tmp_path / "graph.zip", _graph_manifest("cf_interceptor")
    )
    assert _admitted(artifact, "cf_interceptor") is True
    assert _admitted(artifact, "cf_autopilot") is False


# ── container staging ────────────────────────────────────────────────────────

def test_graph_lane_skips_the_per_model_dependency_image(tmp_path):
    artifact = _write_graph_artifact(tmp_path / "graph.zip")
    assert batch.prepare_model_image(SimpleNamespace(), 1, artifact) is None


def _staged(tmp_path, artifact):
    """Run _setup_workspace against a stub evaluator and return the context."""
    ctx = batch._BatchContext(
        self=SimpleNamespace(
            _resolve_worker_limits=lambda *_a, **_k: {
                "memory": "2g", "cpus": "1", "cpuset_cpus": None,
            },
            _docker_env_overrides=lambda **kwargs: {
                "SWARM_INFERENCE_THREADS": str(kwargs["thread_cap"]),
            },
            _resolve_base_image_for_key=lambda _k: "swarm_evaluator_base:latest",
            _get_docker_host_ip=lambda: "127.0.0.1",
            last_selected_runtime_profile=None,
            last_selected_worker_limits=None,
            last_selected_runtime_env=None,
            last_selected_run_image=None,
        ),
        tasks=[SimpleNamespace(family_id="cf_autopilot")],
        uid=1,
        model_path=artifact,
        worker_id=0,
    )
    batch._init_batch_state(ctx)
    assert batch._setup_workspace(ctx) is None
    return ctx


def test_graph_lane_stages_the_runner_bootstrap_and_never_unpacks_the_archive(tmp_path):
    artifact = _write_graph_artifact(tmp_path / "graph.zip")
    ctx = _staged(tmp_path, artifact)

    staged = {p.name for p in ctx.submission_dir.iterdir()}
    assert staged == {"model_graph.zip", "main.py", "agent.capnp"}
    assert ctx.docker_envs["SWARM_MODEL_GRAPH_ARTIFACT"] == (
        "/workspace/submission/model_graph.zip"
    )
    assert "swarm.model_graph.server" in (ctx.submission_dir / "main.py").read_text()


def test_graph_bootstrap_waits_for_the_start_gate(tmp_path):
    """Miner bytes must never be parsed before the network lockdown."""
    ctx = _staged(tmp_path, _write_graph_artifact(tmp_path / "graph.zip"))
    body = (ctx.submission_dir / "main.py").read_text()
    assert "wait_for_start_gate()" in body
    assert body.index("wait_for_start_gate()") < body.index("return run_server(")
    assert ctx.docker_envs["SWARM_START_GATE"] == batch._START_GATE_PATH


def test_code_agent_lane_still_unpacks_and_stages_the_rpc_server(tmp_path):
    agent = _write_code_agent(tmp_path / "agent.zip")
    ctx = _staged(tmp_path, agent)

    staged = {p.name for p in ctx.submission_dir.iterdir()}
    assert {
        "drone_agent.py",
        "main.py",
        "runtime_caps.py",
        "agent_server.py",
        "agent.capnp",
    } <= staged
    assert "SWARM_MODEL_GRAPH_ARTIFACT" not in ctx.docker_envs
    assert ctx.docker_envs["SWARM_INFERENCE_THREADS"] == "1"
