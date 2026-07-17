from __future__ import annotations

import asyncio
import subprocess
from types import SimpleNamespace

import pytest

from swarm.model_graph import admit_artifact
from swarm.model_graph.action import canonicalize_action
from swarm.validator.calibration import (
    SpeedFactor,
    baseline_model_available,
    baseline_model_path,
)
from swarm.validator.docker.docker_evaluator_parts import batch, rpc

from .fixtures import autopilot_artifact


def _helpers():
    return batch._BatchHelpers(
        phase=lambda *_a, **_k: None,
        on_seed_complete_guarded=lambda *_a, **_k: None,
        build_failure_seed_meta=lambda *_a, **_k: {},
        notify_all_failed=lambda *_a, **_k: None,
        run_docker_cmd_quiet=lambda *_a, **_k: None,
        cleanup_tmpdir_quiet=lambda *_a, **_k: None,
    )


def test_committed_calibration_baseline_is_admitted_graph():
    assert baseline_model_available()
    result = admit_artifact(baseline_model_path())
    assert result.accepted
    assert result.family_id == "cf_autopilot"


def test_graph_workspace_always_selects_single_base_image(monkeypatch, tmp_path):
    evaluator = SimpleNamespace(
        base_image="swarm_model_graph_runner:latest",
        _resolve_worker_limits=lambda *_a, **_k: {"cpus": "2", "memory": "6g", "cpuset_cpus": None},
        _docker_env_overrides=lambda: {},
        _get_docker_host_ip=lambda: "172.17.0.1",
        last_selected_runtime_profile=None,
        last_selected_worker_limits=None,
        last_selected_runtime_env=None,
        last_selected_run_image=None,
    )
    task = SimpleNamespace(family_id="cf_autopilot", challenge_type=1)
    ctx = batch._BatchContext(
        self=evaluator, tasks=[task], uid=1, model_path=tmp_path / "graph.zip",
        runtime_profile_payload=None, helpers=_helpers(),
    )
    assert batch._setup_graph_workspace(ctx) is None
    assert ctx.run_image == "swarm_model_graph_runner:latest"
    assert ctx.docker_envs["SWARM_MODEL_GRAPH_ARTIFACT"] == "/workspace/model_graph.zip"
    assert ctx.docker_envs["SWARM_START_GATE"] == batch._START_GATE_PATH


def test_strike_zero_action_matches_family_contracts():
    single = rpc._strike_zero_action(1, 5)
    assert canonicalize_action(single, "cf_autopilot").shape == (5,)
    swarm_action = rpc._strike_zero_action(4, 5)
    assert canonicalize_action(swarm_action, "cf_swarm_autopilot", 4).shape == (4, 5)


def test_ineligible_host_self_excludes_with_infra_code(tmp_path, monkeypatch):
    artifact = autopilot_artifact(tmp_path)

    async def ineligible_calibration(_self, _worker_count):
        return SpeedFactor(
            raw=5.0, factor=5.0, eligible=False,
            owner_p90_ms=100.0, local_p90_ms=500.0,
        )

    monkeypatch.setattr(batch, "_ensure_host_speed_factor", ineligible_calibration)
    monkeypatch.setattr(
        batch, "_docker_evaluator_facade",
        lambda: SimpleNamespace(DockerSecureEvaluator=SimpleNamespace(_base_ready=True)),
    )
    statuses = []
    tasks = [SimpleNamespace(version=None, map_seed=1, challenge_type=1, horizon=5.0)]
    results = asyncio.run(
        batch.evaluate_seeds_batch(
            SimpleNamespace(), tasks, uid=7, model_path=artifact, worker_id=0,
            on_seed_complete=lambda meta=None: statuses.append((meta or {}).get("status")),
        )
    )
    assert len(results) == 1
    assert results[0].score == 0.0
    assert results[0].failure_reason == "INFRA_CALIBRATION"
    assert statuses == ["INFRA_CALIBRATION"]


def test_host_calibration_uses_average_contended_worker(monkeypatch):
    def fake_normalize(local_p90_ms):
        raw = float(local_p90_ms) / 100.0
        return SpeedFactor(
            raw=raw,
            factor=raw,
            eligible=True,
            owner_p90_ms=100.0,
            local_p90_ms=float(local_p90_ms),
        )

    monkeypatch.setattr(batch, "normalize_speed_factor", fake_normalize)

    speed = batch._average_host_speed(
        [
            SpeedFactor(raw=0.35, factor=0.35, eligible=True, owner_p90_ms=100.0, local_p90_ms=35.0),
            SpeedFactor(raw=0.52, factor=0.52, eligible=True, owner_p90_ms=100.0, local_p90_ms=52.0),
            SpeedFactor(raw=0.41, factor=0.41, eligible=True, owner_p90_ms=100.0, local_p90_ms=41.0),
        ]
    )

    assert speed.factor == pytest.approx(0.4266666666666667)


def test_seed_upload_provenance_matches_backend_schema(tmp_path):
    from swarm.model_graph import EXECUTION_PROFILE_ID, RUNNER_ABI, profile_digest
    from swarm.validator.utils_parts.evaluation import _seed_upload_provenance

    artifact = autopilot_artifact(tmp_path)
    validator = SimpleNamespace(
        docker_evaluator=SimpleNamespace(
            _get_image_hash_label=lambda: "abc123",
            _calculate_docker_hash=lambda: "abc123",
        )
    )
    provenance = _seed_upload_provenance(validator, artifact)
    assert set(provenance) == {
        "artifact_sha256", "execution_profile_id", "execution_profile_digest",
        "runner_abi", "runner_image_digest",
    }
    assert len(provenance["artifact_sha256"]) == 64
    assert provenance["execution_profile_id"] == EXECUTION_PROFILE_ID
    assert provenance["execution_profile_digest"] == profile_digest()
    assert provenance["runner_abi"] == RUNNER_ABI
    assert provenance["runner_image_digest"] == "abc123"


def test_seed_upload_payload_carries_provenance():
    from swarm.validator.backend_api import BackendApiClient

    captured = {}

    class Stub:
        async def _post_signed(self, path, payload):
            captured.update(payload)
            return {"recorded": 1}

    provenance = {
        "artifact_sha256": "a" * 64,
        "execution_profile_id": "swarm.onnx-neural.cpu.v1",
        "execution_profile_digest": "b" * 64,
        "runner_abi": "graph_runner.v1",
        "runner_image_digest": "c" * 16,
    }
    asyncio.run(
        BackendApiClient.post_seed_scores_batch(
            Stub(), model_uid=1, epoch_number=2,
            scores=[{"seed_index": 0, "score": 0.5, "map_type": "city"}],
            task_id=9, provenance=provenance,
        )
    )
    for key, value in provenance.items():
        assert captured[key] == value


def test_host_gate_requires_every_worker_eligible(monkeypatch):
    from swarm.validator import forward

    slow = SpeedFactor(raw=5.0, factor=5.0, eligible=False, owner_p90_ms=100.0, local_p90_ms=500.0)
    fast = SpeedFactor(raw=1.0, factor=1.0, eligible=True, owner_p90_ms=100.0, local_p90_ms=100.0)
    good_image = SimpleNamespace(
        _get_image_hash_label=lambda: "hash1", _calculate_docker_hash=lambda: "hash1"
    )
    validator = SimpleNamespace(docker_evaluator=good_image)

    def run(speed):
        async def fake_ensure(evaluator, worker_count):
            assert worker_count == forward.N_DOCKER_WORKERS
            return speed
        monkeypatch.setattr(forward, "_ensure_host_speed_factor", fake_ensure)
        return asyncio.run(forward._host_may_score(validator))

    assert run(fast) is True
    assert run(slow) is False
    assert run(None) is False


def test_host_gate_fails_closed_on_bad_image_provenance(monkeypatch):
    from swarm.validator import forward

    fast = SpeedFactor(raw=1.0, factor=1.0, eligible=True, owner_p90_ms=100.0, local_p90_ms=100.0)

    async def fake_ensure(evaluator, worker_id):
        return fast
    monkeypatch.setattr(forward, "_ensure_host_speed_factor", fake_ensure)

    stale = SimpleNamespace(
        _get_image_hash_label=lambda: "old", _calculate_docker_hash=lambda: "new"
    )
    missing = SimpleNamespace(
        _get_image_hash_label=lambda: "", _calculate_docker_hash=lambda: "new"
    )
    for evaluator in (stale, missing):
        validator = SimpleNamespace(docker_evaluator=evaluator)
        assert asyncio.run(forward._host_may_score(validator)) is False


def test_batch_guard_fails_closed_without_calibration(tmp_path, monkeypatch):
    artifact = autopilot_artifact(tmp_path)

    async def no_calibration(self, worker_count):
        return None
    monkeypatch.setattr(batch, "_ensure_host_speed_factor", no_calibration)
    monkeypatch.setattr(
        batch, "_docker_evaluator_facade",
        lambda: SimpleNamespace(DockerSecureEvaluator=SimpleNamespace(_base_ready=True)),
    )
    tasks = [SimpleNamespace(version=None, map_seed=1, challenge_type=1, horizon=5.0)]
    results = asyncio.run(
        batch.evaluate_seeds_batch(SimpleNamespace(), tasks, uid=7, model_path=artifact, worker_id=0)
    )
    assert results[0].failure_reason == "INFRA_CALIBRATION"


def test_seed_upload_provenance_refuses_stale_image(tmp_path, monkeypatch):
    import pytest as _pytest

    from swarm.model_graph.errors import ModelGraphError, ReasonCode
    from swarm.validator.utils_parts.evaluation import _seed_upload_provenance

    artifact = autopilot_artifact(tmp_path)
    good = SimpleNamespace(docker_evaluator=SimpleNamespace(
        _get_image_hash_label=lambda: "hash1", _calculate_docker_hash=lambda: "hash1"
    ))
    provenance = _seed_upload_provenance(good, artifact)
    assert provenance["runner_image_digest"] == "hash1"
    assert len(provenance["artifact_sha256"]) == 64

    stale = SimpleNamespace(docker_evaluator=SimpleNamespace(
        _get_image_hash_label=lambda: "old", _calculate_docker_hash=lambda: "new"
    ))
    with _pytest.raises(ModelGraphError) as exc:
        _seed_upload_provenance(stale, artifact)
    assert exc.value.reason == ReasonCode.INFRA_IMAGE_MISMATCH


def test_infra_reason_codes_are_never_uploaded_as_miner_scores():
    from swarm.validator.utils_parts.evaluation import _is_infra_failure

    assert _is_infra_failure("INFRA")
    assert _is_infra_failure("INFRA_DOCKER")
    assert _is_infra_failure("INFRA_CALIBRATION")
    assert _is_infra_failure("INFRA_RUNNER_RESET")
    assert not _is_infra_failure("NONE")
    assert not _is_infra_failure("MG_STEP_HARD_TIMEOUT")
    assert not _is_infra_failure(None)


def test_run_task_forwards_the_discovery_contract_fields(monkeypatch):
    from swarm.validator.utils_parts import run_task as run_task_module

    captured = []

    async def fake_discovery(self, entries):
        captured.extend(entries)
        return {}

    monkeypatch.setattr(run_task_module, "_ensure_models_from_backend", fake_discovery)
    validator = SimpleNamespace(seed_manager=SimpleNamespace(epoch_number=3))
    task = {
        "uid": 9, "phase": "SCREENING", "task_id": 11,
        "model_hash": "a" * 64, "github_url": "https://github.com/m/r",
        "is_private": False, "family_id": "cf_autopilot",
        "interface_version": "model_graph.v1",
        "artifact_path": "artifacts/cf_autopilot/model_graph.zip",
    }
    asyncio.run(run_task_module.run_task(
        validator, task, cancel_flag=asyncio.Event(), wake_flag=asyncio.Event(),
    ))
    assert len(captured) == 1
    entry = captured[0]
    assert entry["family_id"] == "cf_autopilot"
    assert entry["interface_version"] == "model_graph.v1"
    assert entry["artifact_path"] == "artifacts/cf_autopilot/model_graph.zip"


def test_bootstrap_start_gate_blocks_until_lockdown_signal(tmp_path, monkeypatch):
    import pytest as _pytest

    from swarm.submission_template.main import wait_for_start_gate

    monkeypatch.delenv("SWARM_START_GATE", raising=False)
    wait_for_start_gate(timeout_sec=0.1)

    gate = tmp_path / "start.gate"
    monkeypatch.setenv("SWARM_START_GATE", str(gate))
    with _pytest.raises(SystemExit):
        wait_for_start_gate(timeout_sec=0.1)
    gate.touch()
    wait_for_start_gate(timeout_sec=0.1)


def test_network_lockdown_and_gate_precede_rpc_wait(tmp_path, monkeypatch):
    calls = []
    evaluator = SimpleNamespace(
        _get_container_pid=lambda name: 4242,
        _apply_network_lockdown=lambda pid, ip: calls.append("lockdown") or True,
        _check_rpc_ready=lambda port: calls.append("rpc_ready") or True,
    )
    ctx = batch._BatchContext(
        self=evaluator, tasks=[SimpleNamespace()], uid=1,
        model_path=tmp_path / "graph.zip", helpers=_helpers(),
    )
    ctx.container_name = "swarm_eval_test"
    ctx.host_port = 49999
    ctx.validator_ip = "172.17.0.1"

    monkeypatch.setattr(batch, "_open_start_gate", lambda name: calls.append("gate") or True)
    assert asyncio.run(batch._prepare_graph_network_and_rpc(ctx)) is None
    assert calls == ["lockdown", "gate", "rpc_ready"]

    calls.clear()
    monkeypatch.setattr(batch, "_open_start_gate", lambda name: calls.append("gate") or False)
    results = asyncio.run(batch._prepare_graph_network_and_rpc(ctx))
    assert results is not None
    assert results[0].failure_reason == "INFRA_DOCKER"
    assert "rpc_ready" not in calls


def test_startup_timeout_charges_the_right_side(tmp_path, monkeypatch):
    def make_ctx():
        evaluator = SimpleNamespace(
            _get_container_pid=lambda name: 4242,
            _apply_network_lockdown=lambda pid, ip: True,
            _check_rpc_ready=lambda port: False,
        )
        ctx = batch._BatchContext(
            self=evaluator, tasks=[SimpleNamespace()], uid=1,
            model_path=tmp_path / "graph.zip", helpers=_helpers(),
        )
        ctx.container_name = "swarm_eval_test"
        ctx.host_port = 49999
        ctx.validator_ip = "172.17.0.1"
        return ctx

    monkeypatch.setattr(batch, "_open_start_gate", lambda name: True)
    monkeypatch.setattr(batch, "RUNNER_STARTUP_WALL_SEC", 0.2)

    monkeypatch.setattr(batch, "_container_is_gone", lambda name: False)
    stalled = asyncio.run(batch._prepare_graph_network_and_rpc(make_ctx()))
    assert stalled[0].failure_reason == "INFRA_DOCKER"

    monkeypatch.setattr(batch, "_container_is_gone", lambda name: True)
    crashed = asyncio.run(batch._prepare_graph_network_and_rpc(make_ctx()))
    assert crashed[0].failure_reason == "MG_LOAD_FAILED"


def test_container_is_gone_only_on_positive_confirmation(monkeypatch):
    def fake_run(returncode=0, stdout="", raise_timeout=False):
        def _run(cmd, **kwargs):
            if raise_timeout:
                raise subprocess.TimeoutExpired(cmd, 10)
            return SimpleNamespace(returncode=returncode, stdout=stdout)
        return _run

    monkeypatch.setattr(batch.subprocess, "run", fake_run(returncode=1))
    assert batch._container_is_gone("c") is True

    monkeypatch.setattr(batch.subprocess, "run", fake_run(stdout="0\n"))
    assert batch._container_is_gone("c") is True

    monkeypatch.setattr(batch.subprocess, "run", fake_run(stdout="4242\n"))
    assert batch._container_is_gone("c") is False

    monkeypatch.setattr(batch.subprocess, "run", fake_run(raise_timeout=True))
    assert batch._container_is_gone("c") is False


def test_calibration_uses_ping_not_calibrate(monkeypatch):
    class Agent:
        calls = 0

        async def ping(self, message):
            self.calls += 1
            return "pong"

        async def calibrate(self, obs):
            raise AssertionError("artifact calibrate must not be used")

    facade = SimpleNamespace(
        CALIBRATION_ROUNDS=4,
        RPC_STEP_TIMEOUT_SEC=0.5,
        MINER_COMPUTE_BUDGET_SEC=0.5,
        CALIBRATION_OVERHEAD_CAP_SEC=0.1,
        RPC_PING_TIMEOUT_SEC=2.0,
    )
    monkeypatch.setattr(rpc, "_docker_evaluator_facade", lambda: facade)
    agent = Agent()
    overhead, factor = asyncio.run(
        rpc._calibrate_rpc_overhead_async(object(), agent, object(), {}, 1)
    )
    assert agent.calls == 4
    assert overhead >= 0
    assert factor == 1.0
