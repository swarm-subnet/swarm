from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

from swarm.core import model_verify


def _make_zip(path: Path, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)


def test_blacklist_load_save_and_add(tmp_path):
    file_path = tmp_path / "blacklist.txt"
    assert model_verify.load_blacklist(file_path) == set()

    model_verify.save_blacklist({"b_hash", "a_hash"}, file_path)
    assert file_path.read_text().splitlines() == ["a_hash", "b_hash"]

    model_verify.add_to_blacklist("c_hash", file_path)
    assert model_verify.load_blacklist(file_path) == {"a_hash", "b_hash", "c_hash"}


def test_inspect_model_structure_requires_drone_agent(tmp_path):
    zpath = tmp_path / "submission.zip"
    _make_zip(zpath, {"main.py": b"print('x')"})

    result = model_verify.inspect_model_structure(zpath)
    assert result["missing_drone_agent"] is True
    assert "Missing drone_agent.py" in result["error"]


def test_inspect_model_structure_rejects_dangerous_files(tmp_path):
    zpath = tmp_path / "submission.zip"
    _make_zip(zpath, {"drone_agent.py": b"# ok", "payload.sh": b"echo bad"})

    result = model_verify.inspect_model_structure(zpath)
    assert "Dangerous executable files" in result["error"]


def test_inspect_model_structure_accepts_rpc_submission(tmp_path):
    zpath = tmp_path / "submission.zip"
    _make_zip(zpath, {"drone_agent.py": b"class A: pass", "weights.pt": b"bin"})

    result = model_verify.inspect_model_structure(zpath)
    assert result["submission_type"] == "rpc"
    assert result["has_mlp_extractor"] is True


def test_classify_model_validity_paths():
    assert model_verify.classify_model_validity({"missing_drone_agent": True})[0] == "missing_drone_agent"
    assert model_verify.classify_model_validity({"malicious_findings": ["x"]})[0] == "fake"
    assert model_verify.classify_model_validity({"error": "Dangerous executable"})[0] == "fake"
    assert model_verify.classify_model_validity({"submission_type": "rpc"})[0] == "legitimate"


def test_save_fake_model_for_analysis_keeps_last_three(tmp_path, monkeypatch):
    monkeypatch.setattr(model_verify, "MODEL_DIR", tmp_path / "models")
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"fake-model-bytes")

    for i in range(4):
        model_verify.save_fake_model_for_analysis(
            model_path=model_path,
            uid=7,
            model_hash=f"hash_{i}",
            reason=f"reason_{i}",
            inspection_results={"idx": i},
        )

    uid_dir = model_verify.MODEL_DIR / "UID_7_fake"
    dirs = sorted([p.name for p in uid_dir.iterdir() if p.is_dir()])
    assert dirs == ["1", "2", "3"]

    latest_report = json.loads((uid_dir / "3" / "analysis_report.json").read_text())
    assert latest_report["model_hash"] == "hash_3"
    assert (uid_dir / "3" / "model.zip").exists()


def test_get_uid_from_axon_by_hotkey():
    metagraph = SimpleNamespace(
        axons=[
            SimpleNamespace(hotkey="hk1"),
            SimpleNamespace(hotkey="hk2"),
        ]
    )
    axon = SimpleNamespace(hotkey="hk2")
    assert model_verify.get_uid_from_axon(metagraph, axon) == 1


def test_get_uid_from_axon_returns_none_when_not_found():
    metagraph = SimpleNamespace(axons=[SimpleNamespace(hotkey="hk1")])
    axon = SimpleNamespace(hotkey="missing")
    assert model_verify.get_uid_from_axon(metagraph, axon) is None
