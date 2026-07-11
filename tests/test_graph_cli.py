from pathlib import Path

from swarm import cli
from swarm.model_graph import admit_artifact
from swarm.model_graph_template.generate_template import generate


def test_model_package_and_verify_graph(tmp_path, capsys):
    source = tmp_path / "source"
    generate(source)
    artifact = tmp_path / "graph.zip"
    assert cli.main([
        "model", "package", "--source", str(source), "--output", str(artifact),
        "--family-id", "cf_autopilot",
    ]) == 0
    assert admit_artifact(artifact).accepted
    assert cli.main(["model", "verify", "--model", str(artifact)]) == 0


def test_model_test_uses_runtime_probe(tmp_path):
    source = tmp_path / "source"
    generate(source)
    assert cli.main([
        "model", "test", "--source", str(source), "--family-id", "cf_autopilot",
    ]) == 0
