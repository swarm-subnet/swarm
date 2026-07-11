from pathlib import Path

from swarm.model_graph.constants import NUMPY_PIN, ONNX_PIN, ONNXRUNTIME_PIN


DOCKER_DIR = Path(__file__).resolve().parents[2] / "swarm" / "validator" / "docker"


def test_runner_image_uses_graph_only_lock_and_cpu_runtime():
    dockerfile = (DOCKER_DIR / "Dockerfile.model_graph").read_text()
    lock = (DOCKER_DIR / "model-graph-requirements.lock").read_text()
    assert "model-graph-requirements.lock" in dockerfile
    assert "--require-hashes" in dockerfile
    assert "submission_template/main.py" in dockerfile
    assert "nvidia" not in dockerfile.lower()
    assert "FROM python:3.11.13-slim-bookworm@sha256:" in dockerfile
    assert "COPY swarm/model_graph /app/swarm/model_graph" in dockerfile
    assert f"onnx=={ONNX_PIN}" in lock
    assert f"onnxruntime=={ONNXRUNTIME_PIN}" in lock
    assert f"numpy=={NUMPY_PIN}" in lock
    assert "torch==" not in lock
    assert "stable-baselines" not in lock


def test_trusted_bootstrap_never_imports_submission_code():
    template = Path(__file__).resolve().parents[2] / "swarm" / "submission_template"
    main = (template / "main.py").read_text()
    server = (template / "agent_server.py").read_text()
    assert "swarm.model_graph" in main
    assert "swarm.model_graph" in server
    assert "/workspace/submission" not in main
