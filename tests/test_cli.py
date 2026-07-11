from __future__ import annotations

import subprocess
import sys

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
