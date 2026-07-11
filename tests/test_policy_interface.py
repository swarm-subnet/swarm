from pathlib import Path

from swarm.cli import _package_model_artifact
from swarm.model_graph_template.generate_template import generate
from swarm.policy_interface import (
    build_artifact_policy_contract,
    resolve_policy_interface_version,
    smoke_test_policy_package,
    verify_policy_package_contract,
)


def _package(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    generate(source)
    output = tmp_path / "graph.zip"
    _package_model_artifact(
        source_dir=source, output_zip=output, family_id="cf_autopilot",
        interface_version=None, overwrite=True,
    )
    return output


def test_registry_contract_is_graph_only():
    assert resolve_policy_interface_version("cf_autopilot") == "model_graph.v1"
    contract = build_artifact_policy_contract("cf_autopilot")
    assert contract["contract_filename"] == "manifest.json"
    assert contract["execution_profile"] == "swarm.onnx-neural.cpu.v1"
    assert contract["runner_abi"] == "graph_runner.v1"
    assert "entry_point" not in contract


def test_verify_and_probe_generated_graph(tmp_path):
    artifact = _package(tmp_path)
    ok, reason, contract = verify_policy_package_contract(artifact)
    assert (ok, reason) == (True, "ok")
    assert contract["family_id"] == "cf_autopilot"
    assert smoke_test_policy_package(artifact) == (True, "ok")


def test_python_zip_is_not_a_policy_package(tmp_path):
    import zipfile
    artifact = tmp_path / "python.zip"
    with zipfile.ZipFile(artifact, "w") as archive:
        archive.writestr("controller.py", "class Controller: pass")
    ok, reason, contract = verify_policy_package_contract(artifact)
    assert not ok
    assert reason.startswith("MG_")
    assert contract is None
