from __future__ import annotations

import json
import io
import sys

import pytest

from swarm.core import evaluator


class _NoCloseIO(io.StringIO):
    def close(self):
        return None


def test_evaluator_verify_only_legitimate_writes_success(monkeypatch, tmp_path):
    result_file = tmp_path / "result.json"
    model_file = tmp_path / "model.zip"
    model_file.write_bytes(b"zip")

    monkeypatch.setattr(
        evaluator,
        "inspect_model_structure",
        lambda path: {"submission_type": "rpc", "path": str(path)},
    )
    monkeypatch.setattr(
        evaluator,
        "classify_model_validity",
        lambda _inspection: ("legitimate", "RPC submission validated"),
    )
    monkeypatch.setattr(evaluator.resource, "setrlimit", lambda *a, **k: None)
    monkeypatch.setattr(sys, "stderr", _NoCloseIO())
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluator.py", "VERIFY_ONLY", "7", str(model_file), str(result_file)],
    )

    with pytest.raises(SystemExit) as ex:
        evaluator.main()

    assert ex.value.code == 0
    payload = json.loads(result_file.read_text())
    assert payload["uid"] == 7
    assert payload["success"] is True
    assert payload["score"] == 0.0
    # Current evaluator JSON conversion casts bool extras to float.
    assert payload["is_fake_model"] == 0.0


def test_evaluator_verify_only_fake_writes_fake_flags(monkeypatch, tmp_path):
    result_file = tmp_path / "result.json"
    model_file = tmp_path / "model.zip"
    model_file.write_bytes(b"zip")

    fake_inspection = {"error": "Dangerous executable files detected: ['payload.sh']"}
    monkeypatch.setattr(
        evaluator, "inspect_model_structure", lambda path: fake_inspection
    )
    monkeypatch.setattr(
        evaluator,
        "classify_model_validity",
        lambda _inspection: (
            "fake",
            "Dangerous executable files detected: ['payload.sh']",
        ),
    )
    monkeypatch.setattr(evaluator.resource, "setrlimit", lambda *a, **k: None)
    monkeypatch.setattr(sys, "stderr", _NoCloseIO())
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluator.py", "VERIFY_ONLY", "9", str(model_file), str(result_file)],
    )

    with pytest.raises(SystemExit) as ex:
        evaluator.main()

    assert ex.value.code == 0
    payload = json.loads(result_file.read_text())
    assert payload["uid"] == 9
    assert payload["success"] is False
    # Current evaluator JSON conversion casts bool extras to float.
    assert payload["is_fake_model"] == 1.0
    assert "fake_reason" in payload
    assert payload["inspection_results"] == fake_inspection
