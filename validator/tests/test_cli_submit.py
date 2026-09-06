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

"""`swarm model submit`: package, verify, then hand off to the private submission."""
from __future__ import annotations

import zipfile
from pathlib import Path

from swarm import cli

AGENT = (
    "import numpy as np\n\n\n"
    "class DroneFlightController:\n"
    "    def act(self, observation):\n"
    "        return np.zeros(5, dtype=np.float32)\n\n"
    "    def reset(self):\n"
    "        pass\n"
)


def _report(compliant: bool) -> dict:
    return {
        "model": "x", "compliant": compliant, "size_bytes": 1, "size_limit_bytes": 2,
        "size_ok": True, "zip_safe": True, "status": "legitimate", "reason": "ok",
        "policy_contract_ok": True, "policy_contract_reason": "ok", "runtime_smoke_ok": compliant,
        "runtime_smoke_reason": "ok" if compliant else "agent crashed", "policy_contract": None,
        "inspection": {},
    }


def _capture_submit(monkeypatch, exit_code: int = 0) -> dict:
    calls = {}

    def submit_private(**kwargs):
        calls.update(kwargs)
        return exit_code

    monkeypatch.setattr(cli, "submit_private", submit_private)
    return calls


def test_submit_packages_the_source_verifies_and_submits(monkeypatch, tmp_path):
    source = tmp_path / "agent"
    source.mkdir()
    (source / "drone_agent.py").write_text(AGENT)
    output = tmp_path / "out" / "submission.zip"
    calls = _capture_submit(monkeypatch)
    verified = []
    monkeypatch.setattr(cli, "_verify_model_zip", lambda path, max_uncompressed_mb: verified.append(path) or _report(True))

    code = cli.main([
        "model", "submit", "--source", str(source), "--family-id", "cf_autopilot",
        "--output", str(output), "--wallet.name", "w", "--wallet.hotkey", "h",
        "--backend-url", "http://backend.test",
    ])

    assert code == 0
    assert output.is_file()
    with zipfile.ZipFile(output) as zf:
        assert "drone_agent.py" in zf.namelist()
    assert verified == [output]
    assert calls["family_id"] == "cf_autopilot"
    assert calls["artifact"] == str(output)
    assert calls["backend_url"] == "http://backend.test"
    assert calls["wallet_name"] == "w" and calls["wallet_hotkey"] == "h"
    assert calls["upload_only"] is False


def test_submit_stops_before_the_chain_when_verification_fails(monkeypatch, tmp_path):
    artifact = tmp_path / "submission.zip"
    with zipfile.ZipFile(artifact, "w") as zf:
        zf.writestr("drone_agent.py", AGENT)
    calls = _capture_submit(monkeypatch)
    monkeypatch.setattr(cli, "_verify_model_zip", lambda path, max_uncompressed_mb: _report(False))

    code = cli.main(["model", "submit", "--artifact", str(artifact), "--family-id", "cf_autopilot"])

    assert code == 1
    assert calls == {}


def test_submit_upload_only_skips_packaging_and_verification(monkeypatch, tmp_path):
    artifact = tmp_path / "submission.zip"
    with zipfile.ZipFile(artifact, "w") as zf:
        zf.writestr("drone_agent.py", AGENT)
    calls = _capture_submit(monkeypatch)
    monkeypatch.setattr(cli, "_verify_model_zip", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("no verify")))

    code = cli.main(["model", "submit", "--artifact", str(artifact), "--family-id", "cf_autopilot", "--upload-only"])

    assert code == 0
    assert calls["upload_only"] is True
    assert calls["artifact"] == str(artifact)


def test_submit_requires_a_family_when_there_is_no_terminal(monkeypatch, tmp_path, capsys):
    artifact = tmp_path / "submission.zip"
    with zipfile.ZipFile(artifact, "w") as zf:
        zf.writestr("drone_agent.py", AGENT)
    calls = _capture_submit(monkeypatch)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)

    assert cli.main(["model", "submit", "--artifact", str(artifact)]) == 1
    assert calls == {}
    assert "--family-id is required" in capsys.readouterr().err


def test_repo_commands_are_gone():
    parser = cli.build_parser()
    commands = parser._subparsers._group_actions[0].choices
    assert "repo" not in commands
    assert "submit" in commands["model"]._subparsers._group_actions[0].choices
