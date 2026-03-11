from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from types import SimpleNamespace

from swarm import cli


def test_doctor_json_output_with_mocked_checks(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "_run_doctor_checks",
        lambda: [
            cli.DoctorCheck("python", True, "3.10.12", True),
            cli.DoctorCheck("docker_binary", True, "Docker version 26", True),
        ],
    )

    rc = cli.main(["doctor", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["name"] == "python"


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
    captured: list[tuple[object, object]] = []

    def _fake_check_writable_dir(path, name):
        captured.append((path, name))
        return cli.DoctorCheck(name, True, str(path), True)

    monkeypatch.setattr(cli, "_check_python_version", lambda: cli.DoctorCheck("python", True, "3.10.12", True))
    monkeypatch.setattr(cli, "_check_docker_binary", lambda: cli.DoctorCheck("docker_binary", True, "ok", True))
    monkeypatch.setattr(cli, "_check_docker_daemon", lambda: cli.DoctorCheck("docker_daemon", True, "ok", True))
    monkeypatch.setattr(cli, "_check_module_available", lambda module: cli.DoctorCheck(f"module:{module}", True, "ok", True))
    monkeypatch.setattr(cli, "_check_writable_dir", _fake_check_writable_dir)
    monkeypatch.setattr(cli, "_check_submission_template", lambda: cli.DoctorCheck("submission_template", True, "ok", True))
    monkeypatch.setattr(cli, "_check_benchmark_engine", lambda: cli.DoctorCheck("benchmark_engine", True, "ok", True))

    cli._run_doctor_checks()

    assert captured[0] == (cli.REPO_ROOT / "swarm" / "state", "state_dir")
    assert captured[1][1] == "model_dir"
    assert cli._runtime_state_dir() == cli.REPO_ROOT / "swarm" / "state"


def test_benchmark_invokes_engine_directly(monkeypatch, tmp_path):
    model_path = tmp_path / "UID_178.zip"
    model_path.write_bytes(b"zip")
    captured: dict[str, list[str]] = {}

    def _fake_benchmark_main(argv):
        captured["argv"] = list(argv)

    monkeypatch.setattr("swarm.benchmark.engine.main", _fake_benchmark_main)
    rc = cli.main(
        [
            "benchmark",
            "--full",
            "--model",
            str(model_path),
            "--workers",
            "3",
            "--rpc-verbosity",
            "low",
        ]
    )
    assert rc == 0
    assert "--model" in captured["argv"]
    assert "--workers" in captured["argv"]
    assert "3" in captured["argv"]
    assert "--rpc-verbosity" in captured["argv"]
    assert "low" in captured["argv"]


def test_benchmark_fails_if_model_missing(capsys, tmp_path):
    rc = cli.main(["benchmark", "--model", str(tmp_path / "missing.zip")])

    assert rc == 1
    assert "Model not found" in capsys.readouterr().err


def test_model_verify_passes_for_valid_rpc_submission(tmp_path, capsys):
    model_zip = tmp_path / "submission.zip"
    with zipfile.ZipFile(model_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("drone_agent.py", "class DroneFlightController:\n    pass\n")

    rc = cli.main(["model", "verify", "--model", str(model_zip), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["compliant"] is True


def test_model_verify_fails_if_missing_drone_agent(tmp_path):
    model_zip = tmp_path / "bad.zip"
    with zipfile.ZipFile(model_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("something_else.py", "print('x')\n")

    assert cli.main(["model", "verify", "--model", str(model_zip)]) == 1


def test_model_package_creates_zip(tmp_path):
    src = tmp_path / "agent"
    src.mkdir(parents=True, exist_ok=True)
    (src / "drone_agent.py").write_text("class DroneFlightController:\n    pass\n")
    (src / "requirements.txt").write_text("numpy\n")
    (src / "policy.pt").write_bytes(b"weights")
    out_zip = tmp_path / "submission.zip"

    rc = cli.main(
        ["model", "package", "--source", str(src), "--output", str(out_zip)]
    )
    assert rc == 0
    assert out_zip.exists()
    with zipfile.ZipFile(out_zip) as zf:
        names = set(zf.namelist())
    assert "drone_agent.py" in names
    assert "policy.pt" in names


def test_model_package_requires_overwrite_for_existing_output(tmp_path):
    src = tmp_path / "agent"
    src.mkdir(parents=True, exist_ok=True)
    (src / "drone_agent.py").write_text("class DroneFlightController:\n    pass\n")
    out_zip = tmp_path / "submission.zip"
    out_zip.write_bytes(b"existing")

    rc = cli.main(
        ["model", "package", "--source", str(src), "--output", str(out_zip)]
    )

    assert rc == 1


def test_model_package_skips_pycache_files(tmp_path):
    src = tmp_path / "agent"
    pycache_dir = src / "__pycache__"
    pycache_dir.mkdir(parents=True, exist_ok=True)
    (src / "drone_agent.py").write_text("class DroneFlightController:\n    pass\n")
    (src / "weights.pt").write_bytes(b"weights")
    (pycache_dir / "drone_agent.cpython-310.pyc").write_bytes(b"compiled")
    out_zip = tmp_path / "submission.zip"

    rc = cli.main(
        ["model", "package", "--source", str(src), "--output", str(out_zip)]
    )

    assert rc == 0
    with zipfile.ZipFile(out_zip) as zf:
        names = set(zf.namelist())
    assert "__pycache__/drone_agent.cpython-310.pyc" not in names
    assert "weights.pt" in names


def test_model_test_fails_for_invalid_requirements(tmp_path):
    src = tmp_path / "agent"
    src.mkdir(parents=True, exist_ok=True)
    (src / "drone_agent.py").write_text("class DroneFlightController:\n    pass\n")
    (src / "requirements.txt").write_text("-r other.txt\n")

    assert cli.main(["model", "test", "--source", str(src)]) == 1


def test_model_test_json_success_for_valid_source(tmp_path, capsys):
    src = tmp_path / "agent"
    src.mkdir(parents=True, exist_ok=True)
    (src / "drone_agent.py").write_text("class DroneFlightController:\n    pass\n")
    (src / "weights.pt").write_bytes(b"weights")

    rc = cli.main(["model", "test", "--source", str(src), "--json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    checks = {item["name"]: item for item in payload}
    assert checks["drone_agent.py"]["ok"] is True
    assert checks["estimated_package_size"]["ok"] is True


def test_model_test_fails_for_invalid_python(tmp_path):
    src = tmp_path / "agent"
    src.mkdir(parents=True, exist_ok=True)
    (src / "drone_agent.py").write_text("def broken(:\n")

    assert cli.main(["model", "test", "--source", str(src)]) == 1


def test_report_json_parses_benchmark_log(tmp_path, capsys):
    log_path = tmp_path / "bench.log"
    log_path.write_text(
        "\n".join(
            [
                "[17:28:58] === RESULTS ===",
                "",
                "  Run summary:",
                "    Seeds evaluated:           50",
                "    Success rate:              40/50 (80.0%)",
                "    Clean execution rate:      49/50 (98.0%)",
                "    Total wall-clock:          120.0s (2.0 min)",
                "    Avg wall / seed:           2.40s",
                "    Median wall / seed:        2.20s",
                "    P90 wall / seed:           3.10s",
                "    Avg sim time / seed:       1.20s",
                "    Total seed-worker time:    140.0s",
                "    Throughput:                25.00 seeds/min",
                "    Throughput per worker:     8.33 seeds/min/worker",
                "    Effective parallelism:     2.33x (utilization 77.7% of 3 workers)",
                "    Batches run:               50",
                "    Avg seeds / container:     1.00",
                "    Total startup overhead:    20.0s",
                "    Avg startup / container:   0.40s",
                "",
                "    Workers used:              3",
                "    Estimated wall-clock:      900.0s (15.0 min)",
                "    Estimated avg wall / seed: 0.90s",
                "    Estimated throughput:      66.67 seeds/min",
                "",
                "[17:28:58] === BENCHMARK COMPLETE ===",
            ]
        )
    )

    rc = cli.main(["report", "--input", str(log_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["seeds_evaluated"] == 50
    assert payload["workers_used"] == 3
    assert payload["clean_execution_rate_pct"] == 98.0
    assert "=== RESULTS ===" in payload["results_block"]
    assert payload["report_source"] == str(log_path)


def test_report_text_output_contains_results_block(tmp_path, capsys):
    log_path = tmp_path / "bench.log"
    log_path.write_text(
        "\n".join(
            [
                "[17:28:58] === RESULTS ===",
                "",
                "  Group                  Seed   Score   OK?    SimT   WallT",
                "  ------------------ -------- ------- ----- ------- -------",
                "  type2_open           323521  0.9439     Y  13.12s    8.5s",
                "    -> AVG                     0.9439                 8.5s",
                "",
                "  Run summary:",
                "    Seeds evaluated:           20",
                "    Clean execution rate:      20/20 (100.0%)",
                "    Total wall-clock:          50.0s (0.8 min)",
                "    Workers used:              2",
                "    Throughput:                24.00 seeds/min",
                "",
                "[17:28:58] === BENCHMARK COMPLETE ===",
            ]
        )
    )

    rc = cli.main(["report", "--input", str(log_path)])

    assert rc == 0
    output = capsys.readouterr().out
    assert "Report source:" in output
    assert "=== RESULTS ===" in output
    assert "type2_open" in output
    assert "Clean execution rate:      20/20 (100.0%)" in output
    assert "=== BENCHMARK COMPLETE ===" in output


def test_extract_benchmark_results_block_strips_ansi_and_progress_noise():
    raw = (
        "\x1b[34mnoise\x1b[0m\n"
        "\rSeed progress: 100%|####|\n"
        "[17:28:58] === RESULTS ===\n\n"
        "  Run summary:\n"
        "    Seeds evaluated:           5\n"
        "[17:28:58] === BENCHMARK COMPLETE ===\n"
    )

    block = cli.extract_benchmark_results_block(raw)

    assert block is not None
    assert "\x1b" not in block
    assert "Seed progress" not in block
    assert "=== RESULTS ===" in block
    assert "=== BENCHMARK COMPLETE ===" in block


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
    assert "benchmark" in result.stdout


def test_python_module_entrypoint_model_test_runs(tmp_path):
    src = tmp_path / "agent"
    src.mkdir(parents=True, exist_ok=True)
    (src / "drone_agent.py").write_text("class DroneFlightController:\n    pass\n")

    result = subprocess.run(
        [sys.executable, "-m", "swarm", "model", "test", "--source", str(src), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert any(item["name"] == "drone_agent.py" and item["ok"] for item in payload)
