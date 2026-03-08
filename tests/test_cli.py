from __future__ import annotations

import json
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
            cli.DoctorCheck("SWARM_PRIVATE_BENCHMARK_SECRET", False, "not set", False),
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


def test_benchmark_invokes_bench_script(monkeypatch, tmp_path):
    model_path = tmp_path / "UID_178.zip"
    model_path.write_bytes(b"zip")
    captured: dict[str, list[str]] = {}

    def _fake_run(command, check=False):  # noqa: ARG001
        captured["command"] = list(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", _fake_run)
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
    assert "debugging/bench_full_eval.py" in " ".join(captured["command"])
    assert "--workers" in captured["command"]


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


def test_model_test_fails_for_invalid_requirements(tmp_path):
    src = tmp_path / "agent"
    src.mkdir(parents=True, exist_ok=True)
    (src / "drone_agent.py").write_text("class DroneFlightController:\n    pass\n")
    (src / "requirements.txt").write_text("-r other.txt\n")

    assert cli.main(["model", "test", "--source", str(src)]) == 1


def test_report_json_parses_benchmark_log(tmp_path, capsys):
    log_path = tmp_path / "bench.log"
    log_path.write_text(
        "\n".join(
            [
                "  Run summary:",
                "    Seeds evaluated:           50",
                "    Success rate:              40/50 (80.0%)",
                "    Total wall-clock:          120.0s (2.0 min)",
                "    Avg wall / seed:           2.40s",
                "    Median wall / seed:        2.20s",
                "    P90 wall / seed:           3.10s",
                "    Throughput:                25.00 seeds/min",
                "    Throughput per worker:     8.33 seeds/min/worker",
                "",
                "    Workers used:              3",
                "    Estimated wall-clock:      900.0s (15.0 min)",
                "    Estimated throughput:      66.67 seeds/min",
            ]
        )
    )

    rc = cli.main(["report", "--input", str(log_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["seeds_evaluated"] == 50
    assert payload["workers_used"] == 3


def test_report_fails_for_non_report_log(tmp_path):
    log_path = tmp_path / "bad.log"
    log_path.write_text("nothing useful here\n")
    assert cli.main(["report", "--input", str(log_path)]) == 1
