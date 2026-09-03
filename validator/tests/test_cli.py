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

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from swarm import cli


def test_doctor_text_output_with_mocked_checks(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "_run_doctor_checks",
        lambda: [
            cli.DoctorCheck("python", True, "3.11.14", True),
            cli.DoctorCheck("docker_binary", True, "Docker version 26", True),
        ],
    )
    assert cli.main(["doctor"]) == 0
    output = capsys.readouterr().out
    assert "Swarm Doctor" in output
    assert "python: 3.11.14" in output


def test_doctor_fails_if_required_check_fails(monkeypatch):
    monkeypatch.setattr(
        cli,
        "_run_doctor_checks",
        lambda: [cli.DoctorCheck("docker_binary", False, "missing", True)],
    )
    assert cli.main(["doctor"]) == 1


def test_doctor_optional_failure_does_not_fail_exit_code(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "_run_doctor_checks",
        lambda: [cli.DoctorCheck("WANDB_API_KEY", False, "not set", False)],
    )
    assert cli.main(["doctor"]) == 0
    assert "WANDB_API_KEY" in capsys.readouterr().out


def test_doctor_checks_runtime_state_dir(monkeypatch):
    captured = []

    def fake_check(path, name):
        captured.append((path, name))
        return cli.DoctorCheck(name, True, str(path), True)

    monkeypatch.setattr(cli, "_check_python_version", lambda: cli.DoctorCheck("python", True, "ok", True))
    monkeypatch.setattr(cli, "_check_docker_binary", lambda: cli.DoctorCheck("docker_binary", True, "ok", True))
    monkeypatch.setattr(cli, "_check_docker_daemon", lambda: cli.DoctorCheck("docker_daemon", True, "ok", True))
    monkeypatch.setattr(cli, "_check_binary_available", lambda name: cli.DoctorCheck(name, True, "ok", True))
    monkeypatch.setattr(cli, "_check_sandbox_lockdown_permissions", lambda: cli.DoctorCheck("sandbox", True, "ok", False))
    monkeypatch.setattr(cli, "_check_module_available", lambda name: cli.DoctorCheck(name, True, "ok", True))
    monkeypatch.setattr(cli, "_check_writable_dir", fake_check)
    monkeypatch.setattr(cli, "_check_submission_template", lambda: cli.DoctorCheck("template", True, "ok", True))
    monkeypatch.setattr(cli, "_check_benchmark_engine", lambda: cli.DoctorCheck("engine", True, "ok", True))
    cli._run_doctor_checks()
    assert captured[0] == (cli.REPO_ROOT / "swarm" / "state", "state_dir")
    assert captured[1][1] == "model_dir"


def test_sandbox_lockdown_permissions_ok_for_root(monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(cli.os, "geteuid", lambda: 0)
    assert cli._check_sandbox_lockdown_permissions().ok is True


def test_benchmark_invokes_engine_directly(monkeypatch, tmp_path):
    model_path = tmp_path / "graph.zip"
    model_path.write_bytes(b"zip")
    captured = {}
    monkeypatch.setattr("swarm.benchmark.engine.main", lambda argv: captured.setdefault("argv", list(argv)))
    assert cli.main(["benchmark", "--model", str(model_path), "--workers", "3"]) == 0
    assert captured["argv"][captured["argv"].index("--workers") + 1] == "3"


def test_benchmark_fails_if_model_missing(capsys, tmp_path):
    assert cli.main(["benchmark", "--model", str(tmp_path / "missing.zip")]) == 1
    assert "Model not found" in capsys.readouterr().err


def test_report_text_output_parses_summary(tmp_path, capsys):
    log_path = tmp_path / "bench.log"
    log_path.write_text(
        "\n".join([
            "    Seeds evaluated:           50",
            "    Success rate:              40/50 (80.0%)",
            "    Total wall-clock:          120.0s (2.0 min)",
            "    Throughput:                25.00 seeds/min",
            "    Workers used:              3",
            "    Estimated wall-clock:      900.0s (15.0 min)",
        ])
    )
    assert cli.main(["report", "--input", str(log_path)]) == 0
    output = capsys.readouterr().out
    assert "Seeds evaluated: 50" in output
    assert "Workers used: 3" in output


def test_report_text_output_contains_results_block(tmp_path, capsys):
    log_path = tmp_path / "bench.log"
    log_path.write_text(
        "[17:28:58] === RESULTS ===\n"
        "  type2_open 323521 0.9439 Y\n"
        "    Seeds evaluated:           20\n"
        "    Total wall-clock:          50.0s (0.8 min)\n"
        "    Workers used:              2\n"
        "[17:28:58] === BENCHMARK COMPLETE ===\n"
    )
    assert cli.main(["report", "--input", str(log_path)]) == 0
    output = capsys.readouterr().out
    assert "=== RESULTS ===" in output
    assert "type2_open" in output


def test_extract_benchmark_results_block_strips_progress_noise():
    raw = (
        "\x1b[34mnoise\x1b[0m\n\rSeed progress: 100%|####|\n"
        "[17:28:58] === RESULTS ===\n"
        "    Seeds evaluated:           5\n"
        "[17:28:58] === BENCHMARK COMPLETE ===\n"
    )
    block = cli.extract_benchmark_results_block(raw)
    assert block is not None
    assert "\x1b" not in block
    assert "Seed progress" not in block


def test_report_fails_for_non_report_log(tmp_path):
    log_path = tmp_path / "bad.log"
    log_path.write_text("nothing useful here\n")
    assert cli.main(["report", "--input", str(log_path)]) == 1


def test_python_module_entrypoint_help_runs():
    result = subprocess.run(
        [sys.executable, "-m", "swarm", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Swarm CLI" in result.stdout


# The doctor check must agree with what the validator actually stages: batch.py
# copies runtime_caps.py into every submission, so a template missing it is broken.


def _template_dir(tmp_path, names):
    d = tmp_path / "swarm" / "submission_template"
    d.mkdir(parents=True)
    for name in names:
        (d / name).write_text("")
    return d


def test_doctor_accepts_a_complete_submission_template(tmp_path, monkeypatch):
    _template_dir(tmp_path, cli.REQUIRED_TEMPLATE_FILES)
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    assert cli._check_submission_template().ok is True


def test_doctor_rejects_a_template_missing_runtime_caps(tmp_path, monkeypatch):
    names = set(cli.REQUIRED_TEMPLATE_FILES) - {"runtime_caps.py"}
    _template_dir(tmp_path, names)
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)

    check = cli._check_submission_template()
    assert check.ok is False
    assert "runtime_caps.py" in check.detail


def test_required_template_files_match_what_the_validator_stages():
    """cli.py and batch.py held two different lists; they must not drift again."""
    staged = (
        Path(__file__).resolve().parents[2]
        / "swarm" / "validator" / "docker" / "docker_evaluator_parts" / "batch.py"
    ).read_text()
    marker = 'for name in ("agent.capnp", "agent_server.py", "main.py", "runtime_caps.py"):'
    assert marker in staged
    assert set(cli.REQUIRED_TEMPLATE_FILES) == {
        "agent.capnp",
        "agent_server.py",
        "main.py",
        "runtime_caps.py",
    }


ZIP_BYTES = b"champion-zip"


def _fake_champion_client(monkeypatch, calls, uid=7):
    class FakeResponse:
        def __init__(self, payload=None, content=b""):
            self.status_code = 200
            self._payload = payload
            self.content = content

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get(self, url, params=None):
            calls.append((url, params))
            if url.endswith("/champion"):
                return FakeResponse(payload={
                    "uid": uid,
                    "benchmark_score": 0.5,
                    "is_released": True,
                    "model_hash": hashlib.sha256(ZIP_BYTES).hexdigest(),
                })
            return FakeResponse(content=ZIP_BYTES)

    monkeypatch.setattr(httpx, "Client", FakeClient)


def test_champion_scopes_both_requests_to_the_family(monkeypatch, tmp_path):
    calls = []
    _fake_champion_client(monkeypatch, calls)
    monkeypatch.chdir(tmp_path)

    assert cli.main(["champion", "--family-id", "cf_search_and_rescue"]) == 0
    assert [params for _, params in calls] == [
        {"family_id": "cf_search_and_rescue"},
        {"family_id": "cf_search_and_rescue"},
    ]
    assert (tmp_path / "champion_cf_search_and_rescue_UID_7.zip").read_bytes() == ZIP_BYTES


def test_champion_without_a_family_keeps_the_old_filename(monkeypatch, tmp_path):
    calls = []
    _fake_champion_client(monkeypatch, calls)
    monkeypatch.chdir(tmp_path)

    assert cli.main(["champion"]) == 0
    assert [params for _, params in calls] == [{}, {}]
    assert (tmp_path / "champion_UID_7.zip").exists()


def test_champion_rejects_an_unknown_family(capsys):
    with pytest.raises(SystemExit):
        cli.main(["champion", "--family-id", "cf_nope"])
    assert "invalid choice" in capsys.readouterr().err


def test_benchmark_auto_download_uses_the_benchmark_family(monkeypatch, tmp_path):
    model_path = tmp_path / "graph.zip"
    model_path.write_bytes(b"zip")
    captured = {}

    def fake_download(family_id=None):
        captured["family_id"] = family_id
        return model_path

    monkeypatch.setattr(cli, "_download_champion_model", fake_download)
    monkeypatch.setattr("swarm.benchmark.engine.main", lambda argv: 0)

    assert cli.main(["benchmark", "--family-id", "cf_search_and_rescue"]) == 0
    assert captured["family_id"] == "cf_search_and_rescue"
