from __future__ import annotations

import asyncio
import json
import os
import random
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from swarm.benchmark.engine import _find_seeds
from swarm.constants import SIM_DT
from swarm.protocol import ValidationResult
from swarm.utils.env_factory import make_env
from swarm.validator import forward as forward_mod
from swarm.validator import utils as validator_utils
from swarm.validator.backend_api import BackendApiClient
from swarm.validator.docker.docker_evaluator import DockerSecureEvaluator
from swarm.validator.task_gen import random_task

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBMISSION_TEMPLATE = REPO_ROOT / "swarm" / "submission_template"


def _require_docker() -> None:
    strict = os.getenv("SWARM_E2E_REQUIRE_DOCKER", "0") == "1"

    def _skip_or_fail(msg: str) -> None:
        if strict:
            pytest.fail(msg)
        pytest.skip(msg)

    if shutil.which("docker") is None:
        _skip_or_fail("Docker binary not found in PATH.")

    try:
        info = subprocess.run(
            ["docker", "info"], capture_output=True, text=True, timeout=20
        )
    except Exception as exc:
        _skip_or_fail(f"Docker daemon check failed: {exc}")
        return

    if info.returncode != 0:
        stderr = info.stderr.strip() or info.stdout.strip() or "unknown docker error"
        _skip_or_fail(f"Docker daemon unavailable: {stderr}")


def _require_real_capnp() -> Any:
    try:
        import capnp  # type: ignore
    except Exception as exc:
        pytest.skip(f"pycapnp not available: {exc}")
    if not getattr(capnp, "__file__", None):
        pytest.skip("pycapnp not available (capnp stub active).")
    return capnp


def _get_free_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])
    except OSError as exc:
        pytest.skip(f"Local socket binding unavailable in this environment: {exc}")


def _ensure_evaluator_ready() -> DockerSecureEvaluator:
    evaluator = DockerSecureEvaluator()
    assert evaluator._base_ready, "DockerSecureEvaluator base image is not ready."
    return evaluator


def _seed_from_bench(group: str) -> int:
    random.seed(2026)
    seeds = _find_seeds(1)
    assert group in seeds, f"Benchmark group not found: {group}"
    return int(seeds[group][0])


def _build_model_zip_for_e2e(tmp_path: Path) -> Path:
    zip_path = tmp_path / "e2e_model.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(SUBMISSION_TEMPLATE / "drone_agent.py", "drone_agent.py")
    return zip_path


@pytest.mark.e2e
@pytest.mark.integration
@pytest.mark.slow
def test_e2e_docker_run_exec_lifecycle():
    _require_docker()
    evaluator = _ensure_evaluator_ready()

    container_name = f"swarm_e2e_lifecycle_{int(time.time() * 1000)}"
    try:
        run_res = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-d",
                "--name",
                container_name,
                evaluator.base_image,
                "bash",
                "-lc",
                "sleep infinity",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert run_res.returncode == 0, run_res.stderr

        exec_res = subprocess.run(
            ["docker", "exec", container_name, "python", "-c", "print('swarm-e2e-ok')"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert exec_res.returncode == 0, exec_res.stderr
        assert "swarm-e2e-ok" in exec_res.stdout
    finally:
        subprocess.run(
            ["docker", "rm", "-f", container_name], capture_output=True, text=True
        )


@pytest.mark.e2e
@pytest.mark.integration
@pytest.mark.slow
def test_e2e_rpc_capnp_container_ping_and_act(tmp_path: Path):
    _require_docker()
    capnp = _require_real_capnp()
    evaluator = _ensure_evaluator_ready()

    with tempfile.TemporaryDirectory() as td:
        submission_dir = Path(td) / "submission"
        submission_dir.mkdir(parents=True, exist_ok=True)
        for name in ("main.py", "agent_server.py", "agent.capnp", "drone_agent.py"):
            shutil.copy(SUBMISSION_TEMPLATE / name, submission_dir / name)

        host_port = _get_free_port()
        container_name = f"swarm_e2e_rpc_{int(time.time() * 1000)}"
        run_res = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-d",
                "--name",
                container_name,
                "-p",
                f"{host_port}:8000",
                "-v",
                f"{submission_dir}:/workspace/submission:ro",
                evaluator.base_image,
                "python",
                "/workspace/submission/main.py",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert (
            run_res.returncode == 0
        ), f"Failed to start RPC container: {run_res.stderr}"

        async def _run_rpc_check():
            schema = capnp.load(str(SUBMISSION_TEMPLATE / "agent.capnp"))
            for _ in range(30):
                try:
                    async with capnp.kj_loop():
                        stream = await capnp.AsyncIoStream.create_connection(
                            host="127.0.0.1", port=host_port
                        )
                        client = capnp.TwoPartyClient(stream)
                        agent = client.bootstrap().cast_as(schema.Agent)
                        pong = await agent.ping("test")
                        assert pong.response == "pong"

                        obs = schema.Observation.new_message()
                        entry = obs.init("entries", 1)[0]
                        arr = np.zeros((2,), dtype=np.float32)
                        entry.key = "__value__"
                        entry.tensor.data = arr.tobytes()
                        entry.tensor.shape = [2]
                        entry.tensor.dtype = "float32"

                        act_resp = await agent.act(obs)
                        assert len(act_resp.action.shape) >= 1
                        return
                except Exception:
                    await asyncio.sleep(1.0)
            raise AssertionError("RPC server did not become ready in time")

        try:
            asyncio.run(_run_rpc_check())
        finally:
            subprocess.run(
                ["docker", "rm", "-f", container_name], capture_output=True, text=True
            )


@pytest.mark.e2e
@pytest.mark.integration
@pytest.mark.slow
def test_e2e_docker_evaluator_single_bench_seed(tmp_path: Path):
    _require_docker()
    _require_real_capnp()
    _ensure_evaluator_ready()

    seed = _seed_from_bench("type2_open")
    task = random_task(sim_dt=SIM_DT, seed=seed)
    model_zip = _build_model_zip_for_e2e(tmp_path)
    evaluator = DockerSecureEvaluator()

    results = asyncio.run(
        evaluator.evaluate_seeds_batch(
            tasks=[task],
            uid=999999,
            model_path=model_zip,
            worker_id=0,
        )
    )
    assert len(results) == 1
    result = results[0]
    assert isinstance(result, ValidationResult)
    assert result.time_sec >= 0.0
    assert 0.0 <= float(result.score) <= 1.0


@pytest.mark.e2e
@pytest.mark.integration
@pytest.mark.slow
def test_e2e_simulator_workload_all_bench_groups():
    import pybullet as p  # type: ignore

    if not hasattr(p, "ER_DEPTH_ONLY"):
        pytest.skip(
            "Current pybullet build lacks ER_DEPTH_ONLY; "
            "skip the full simulator workload check on this renderer build."
        )
    selected = [
        "type1_city",
        "type2_open",
        "type3_mountain",
        "type4_village",
        "type5_warehouse",
    ]
    random.seed(2026)
    seeds = _find_seeds(1)

    total_steps = 0
    for group in selected:
        seed = int(seeds[group][0])
        task = random_task(sim_dt=SIM_DT, seed=seed)
        env = make_env(task, gui=False)
        try:
            obs, _ = env.reset(seed=task.map_seed)
            assert obs is not None
            lo = env.action_space.low.flatten()
            action = np.zeros_like(lo, dtype=np.float32)
            for _ in range(30):
                obs, _, terminated, truncated, _ = env.step(action[None, :])
                total_steps += 1
                if terminated or truncated:
                    break
        finally:
            env.close()

    assert total_steps >= 30


@pytest.mark.e2e
@pytest.mark.integration
@pytest.mark.slow
def test_e2e_forward_loop_with_local_backend(tmp_path: Path, monkeypatch):
    class _Hotkey:
        ss58_address = "validator_hotkey"

        def sign(self, data: bytes) -> bytes:
            _ = data
            return b"\x01\x02"

    class _Wallet:
        hotkey = _Hotkey()

    class _SeedManager:
        epoch_number = 1

        def get_pending_publications(self):
            return []

        def mark_epoch_published(self, epoch):
            _ = epoch
            return None

        def check_epoch_transition(self):
            return False

        def advance_to_new_epoch(self):
            return 1

        def get_screening_seeds(self):
            return [_seed_from_bench("type2_open")]

        def get_benchmark_seeds(self):
            return [_seed_from_bench("type2_open")]

        def get_all_seeds(self):
            return self.get_screening_seeds() + self.get_benchmark_seeds()

        def seconds_until_epoch_end(self):
            return 21600.0

    class _DockerEval:
        _base_ready = True

        def cleanup(self):
            return None

    server_state: dict[str, Any] = {"calls": []}

    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            _ = format, args
            return None

        def _send_json(self, payload: dict, status: int = 200):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            server_state["calls"].append(("GET", self.path))
            if self.path == "/validators/sync":
                self._send_json(
                    {
                        "current_champion": {
                            "uid": 0,
                            "benchmark_score": 0.0,
                            "model_hash": "",
                        },
                        "weights": {"0": 1.0},
                        "reeval_queue": [],
                        "pending_models": [
                            {
                                "uid": 1,
                                "model_hash": validator_utils.sha256sum(model_path),
                                "github_url": "https://github.com/owner/repo",
                            }
                        ],
                        "leaderboard_version": 1,
                    }
                )
                return
            self._send_json({"error": "not found"}, status=404)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0") or 0)
            body = self.rfile.read(length) if length > 0 else b""
            server_state["calls"].append(
                ("POST", self.path, body.decode("utf-8", errors="ignore"))
            )

            if self.path == "/validators/models/new":
                self._send_json({"model_id": 1})
                return
            if self.path.endswith("/screening"):
                self._send_json({"recorded": True})
                return
            if self.path.endswith("/score"):
                self._send_json({"recorded": True})
                return
            if self.path.endswith("/upload"):
                self._send_json({"stored": True, "released": False})
                return
            if self.path == "/validators/heartbeat":
                self._send_json({"recorded": True})
                return
            if self.path == "/validators/epoch/publish":
                self._send_json({"published": True})
                return

            self._send_json({"ok": True})

    class _QuietThreadingHTTPServer(ThreadingHTTPServer):
        def handle_error(self, request, client_address):
            _ = request, client_address
            return None

    host = "127.0.0.1"
    port = _get_free_port()
    httpd = _QuietThreadingHTTPServer((host, port), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "UID_1.zip"
    with zipfile.ZipFile(model_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(SUBMISSION_TEMPLATE / "drone_agent.py", "drone_agent.py")

    async def _fake_evaluate_seeds(
        self_obj, uid, model_path, seeds, description="benchmark", on_seed_complete=None
    ):
        _ = self_obj, uid, model_path, seeds, description
        if on_seed_complete:
            on_seed_complete()
        per_type = {
            "city": [],
            "open": [0.8],
            "mountain": [],
            "village": [],
            "warehouse": [],
            "moving_platform": [],
        }
        return [0.8], per_type

    validator = SimpleNamespace(
        wallet=_Wallet(),
        metagraph=SimpleNamespace(
            n=3,
            hotkeys=["validator_hotkey", "miner_hotkey", "other"],
            S=np.array([100.0, 10.0, 5.0], dtype=np.float32),
            coldkeys=["validator_cold", "miner_cold", "other_cold"],
        ),
        scores=np.zeros(3, dtype=np.float32),
        seed_manager=_SeedManager(),
        docker_evaluator=_DockerEval(),
    )

    queue_file = tmp_path / "normal_model_queue.json"
    cache_file = tmp_path / "benchmark_cache.json"

    monkeypatch.setattr(forward_mod, "MODEL_DIR", model_dir)
    monkeypatch.setattr(validator_utils, "MODEL_DIR", model_dir)
    monkeypatch.setattr(validator_utils, "STATE_DIR", tmp_path)
    monkeypatch.setattr(validator_utils, "NORMAL_MODEL_QUEUE_FILE", queue_file)
    monkeypatch.setattr(validator_utils, "CACHE_FILE", cache_file)
    monkeypatch.setattr(forward_mod, "FORWARD_SLEEP_SEC", 0.0)
    monkeypatch.setattr(forward_mod, "NORMAL_MODEL_QUEUE_PROCESS_LIMIT", 1)

    model_hash = validator_utils.sha256sum(model_path)
    github_url = "https://github.com/owner/repo"
    monkeypatch.setattr(
        forward_mod,
        "_ensure_models_from_backend",
        lambda self_obj, pending: asyncio.sleep(
            0, result={1: (model_path, github_url)}
        ),
    )
    monkeypatch.setattr(
        forward_mod,
        "_detect_new_models",
        lambda self_obj, model_paths: {1: (model_path, model_hash, github_url)},
    )
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _fake_evaluate_seeds)
    monkeypatch.setattr(DockerSecureEvaluator, "_base_ready", True)

    async def _run_forward_once():
        backend = BackendApiClient(
            wallet=validator.wallet, base_url=f"http://{host}:{port}", timeout=10.0
        )
        monkeypatch.setattr(backend, "_get_miner_hotkey", lambda uid: "miner_hotkey")
        validator.backend_api = backend
        try:
            await forward_mod.forward(validator)
        finally:
            await backend.close()

    try:
        asyncio.run(_run_forward_once())
    finally:
        httpd.shutdown()
        thread.join(timeout=2.0)

    # Check that forward flow performed real HTTP interactions against local backend.
    called_paths = [c[1] for c in server_state["calls"] if c[0] in ("GET", "POST")]
    assert "/validators/sync" in called_paths
    assert any(p.endswith("/screening") for p in called_paths)
    assert any(p.endswith("/score") for p in called_paths)
    assert np.count_nonzero(validator.scores) >= 1


@pytest.mark.e2e
@pytest.mark.integration
@pytest.mark.slow
def test_e2e_forward_single_cycle_local_backend_no_mocks():
    class _Hotkey:
        ss58_address = "validator_hotkey"

        def sign(self, data: bytes) -> bytes:
            _ = data
            return b"\x01\x02"

    class _Wallet:
        hotkey = _Hotkey()

    class _SeedManager:
        epoch_number = 1

        def get_pending_publications(self):
            return []

        def mark_epoch_published(self, epoch):
            _ = epoch
            return None

        def check_epoch_transition(self):
            return False

        def advance_to_new_epoch(self):
            return 1

        def get_all_seeds(self):
            return []

    server_state: dict[str, Any] = {"calls": []}

    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            _ = format, args
            return None

        def _send_json(self, payload: dict, status: int = 200):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            server_state["calls"].append(("GET", self.path))
            if self.path == "/validators/sync":
                self._send_json(
                    {
                        "current_champion": {
                            "uid": 0,
                            "benchmark_score": 0.0,
                            "model_hash": "",
                        },
                        "weights": {"0": 1.0},
                        "reeval_queue": [],
                        "leaderboard_version": 1,
                    }
                )
                return
            self._send_json({"error": "not found"}, status=404)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0") or 0)
            body = self.rfile.read(length) if length > 0 else b""
            server_state["calls"].append(
                ("POST", self.path, body.decode("utf-8", errors="ignore"))
            )
            if self.path == "/validators/heartbeat":
                self._send_json({"recorded": True})
                return
            if self.path == "/validators/epoch/publish":
                self._send_json({"published": True})
                return
            self._send_json({"ok": True})

    class _QuietThreadingHTTPServer(ThreadingHTTPServer):
        def handle_error(self, request, client_address):
            _ = request, client_address
            return None

    host = "127.0.0.1"
    port = _get_free_port()
    httpd = _QuietThreadingHTTPServer((host, port), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    validator = SimpleNamespace(
        wallet=_Wallet(),
        metagraph=SimpleNamespace(
            n=np.array(1),
            hotkeys=["validator_hotkey"],
            S=np.array([100.0], dtype=np.float32),
            coldkeys=["validator_cold"],
            axons=[SimpleNamespace(is_serving=False)],
            validator_permit=np.array([False]),
        ),
        config=SimpleNamespace(neuron=SimpleNamespace(vpermit_tao_limit=4096)),
        scores=np.zeros(1, dtype=np.float32),
        seed_manager=_SeedManager(),
        # We keep an instance to satisfy cleanup call; base readiness is controlled at class level.
        docker_evaluator=SimpleNamespace(cleanup=lambda: None),
    )

    async def _run_once():
        backend = BackendApiClient(
            wallet=validator.wallet, base_url=f"http://{host}:{port}", timeout=10.0
        )
        validator.backend_api = backend
        try:
            original_base_ready = DockerSecureEvaluator._base_ready
            DockerSecureEvaluator._base_ready = True
            try:
                await forward_mod.forward(validator)
            finally:
                DockerSecureEvaluator._base_ready = original_base_ready
        finally:
            await backend.close()

    try:
        asyncio.run(_run_once())
    finally:
        httpd.shutdown()
        thread.join(timeout=2.0)

    called_paths = [c[1] for c in server_state["calls"] if c[0] in ("GET", "POST")]
    assert "/validators/sync" in called_paths
    assert not any(p.endswith("/screening") for p in called_paths)
    assert not any(p.endswith("/score") for p in called_paths)
