from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from types import SimpleNamespace

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

    rc = cli.main(["doctor"])
    assert rc == 0
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
    captured: list[tuple[object, object]] = []

    def _fake_check_writable_dir(path, name):
        captured.append((path, name))
        return cli.DoctorCheck(name, True, str(path), True)

    monkeypatch.setattr(cli, "_check_python_version", lambda: cli.DoctorCheck("python", True, "3.11.14", True))
    monkeypatch.setattr(cli, "_check_docker_binary", lambda: cli.DoctorCheck("docker_binary", True, "ok", True))
    monkeypatch.setattr(cli, "_check_docker_daemon", lambda: cli.DoctorCheck("docker_daemon", True, "ok", True))
    monkeypatch.setattr(cli, "_check_binary_available", lambda binary: cli.DoctorCheck(f"binary:{binary}", True, "ok", True))
    monkeypatch.setattr(
        cli,
        "_check_sandbox_lockdown_permissions",
        lambda: cli.DoctorCheck("sandbox_lockdown_permissions", True, "ok", False),
    )
    monkeypatch.setattr(cli, "_check_module_available", lambda module: cli.DoctorCheck(f"module:{module}", True, "ok", True))
    monkeypatch.setattr(cli, "_check_writable_dir", _fake_check_writable_dir)
    monkeypatch.setattr(cli, "_check_submission_template", lambda: cli.DoctorCheck("submission_template", True, "ok", True))
    monkeypatch.setattr(cli, "_check_benchmark_engine", lambda: cli.DoctorCheck("benchmark_engine", True, "ok", True))

    cli._run_doctor_checks()

    assert captured[0] == (cli.REPO_ROOT / "swarm" / "state", "state_dir")
    assert captured[1][1] == "model_dir"
    assert cli._runtime_state_dir() == cli.REPO_ROOT / "swarm" / "state"


def test_sandbox_lockdown_permissions_ok_for_root(monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(cli.os, "geteuid", lambda: 0)

    check = cli._check_sandbox_lockdown_permissions()

    assert check.ok is True
    assert check.required is False
    assert "running as root" in check.detail


def test_sandbox_lockdown_permissions_warns_for_non_root_without_caps(monkeypatch):
    monkeypatch.setattr(
        cli.shutil,
        "which",
        lambda name: {
            "nsenter": "/usr/bin/nsenter",
            "iptables": "/usr/sbin/iptables",
            "getcap": "/usr/sbin/getcap",
        }.get(name),
    )
    monkeypatch.setattr(cli.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(cli.os.path, "realpath", lambda path: "/usr/sbin/xtables-nft-multi")
    monkeypatch.setattr(cli, "_binary_capabilities", lambda path: set())

    check = cli._check_sandbox_lockdown_permissions()

    assert check.ok is False
    assert check.required is False
    assert "current user is not root" in check.detail
    assert "sudo -E" in check.detail
    assert "cap_sys_admin" in check.detail
    assert "cap_net_admin" in check.detail


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
            "--model",
            str(model_path),
            "--workers",
            "3",
            "--save-seed-file",
            str(tmp_path / "seeds.json"),
            "--summary-json-out",
            str(tmp_path / "summary.json"),
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
    assert "--save-seed-file" in captured["argv"]
    assert "--summary-json-out" in captured["argv"]


def test_benchmark_fails_if_model_missing(capsys, tmp_path):
    rc = cli.main(["benchmark", "--model", str(tmp_path / "missing.zip")])

    assert rc == 1
    assert "Model not found" in capsys.readouterr().err


def test_visualize_invokes_visualizer_main(monkeypatch):
    captured: dict[str, list[str]] = {}

    def _fake_visualize_main(argv):
        captured["argv"] = list(argv)

    monkeypatch.setattr("scripts.visualize_map.main", _fake_visualize_main)

    rc = cli.main(
        [
            "visualize",
            "--type",
            "1",
            "--width",
            "960",
            "--height",
            "540",
            "--sim-fps",
            "10",
            "--gpu",
        ]
    )

    assert rc == 0
    assert captured["argv"][:2] == ["--type", "1"]
    assert "--seed" not in captured["argv"]
    assert "--width" in captured["argv"]
    assert "--height" in captured["argv"]
    assert "--sim-fps" in captured["argv"]
    assert "--gpu" in captured["argv"]


def test_visualize_inferrs_type_from_seed_when_omitted(monkeypatch):
    captured: dict[str, list[str]] = {}

    monkeypatch.setattr(cli, "_infer_benchmark_type_from_seed", lambda seed: 4)
    monkeypatch.setattr(
        "scripts.visualize_map.main",
        lambda argv: captured.setdefault("argv", list(argv)),
    )

    rc = cli.main(["visualize", "--seed", "657398"])

    assert rc == 0
    assert captured["argv"][:4] == ["--type", "4", "--seed", "657398"]


def test_visualize_lists_failed_seeds_from_summary(tmp_path, capsys):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "group_results": {
                    "type1_city": [{"seed": 101, "score": 0.50, "success": True, "sim_time": 9.0}],
                    "type2_open": [{"seed": 202, "score": 0.01, "success": False, "sim_time": 8.0}],
                    "type3_mountain": [{"seed": 303, "score": 0.02, "success": False, "sim_time": 7.0}],
                    "type4_village": [],
                    "type5_warehouse": [],
                    "type6_forest": [],
                }
            }
        )
    )

    rc = cli.main(["visualize", "--summary-json", str(summary_path), "--failed"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "Failed benchmark seeds" in output
    assert "seed 202" in output
    assert "seed 303" in output
    assert "--failed-index N" in output


def test_visualize_failed_index_opens_matching_failed_seed(monkeypatch, tmp_path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "group_results": {
                    "type1_city": [{"seed": 101, "score": 0.50, "success": True, "sim_time": 9.0}],
                    "type2_open": [{"seed": 202, "score": 0.01, "success": False, "sim_time": 8.0}],
                    "type3_mountain": [{"seed": 303, "score": 0.02, "success": False, "sim_time": 7.0}],
                    "type4_village": [],
                    "type5_warehouse": [],
                    "type6_forest": [],
                }
            }
        )
    )
    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(
        "scripts.visualize_map.main",
        lambda argv: captured.setdefault("argv", list(argv)),
    )

    rc = cli.main(
        ["visualize", "--summary-json", str(summary_path), "--failed-index", "2"]
    )

    assert rc == 0
    assert captured["argv"][:4] == ["--type", "3", "--seed", "303"]


def test_visualize_inferrs_type_from_seed_file(monkeypatch, tmp_path):
    seed_file = tmp_path / "seeds.json"
    seed_file.write_text(
        json.dumps(
            {
                "type1_city": [101],
                "type2_open": [202],
                "type3_mountain": [303],
                "type4_village": [404],
                "type5_warehouse": [505],
                "type6_forest": [606],
            }
        )
    )
    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(
        "scripts.visualize_map.main",
        lambda argv: captured.setdefault("argv", list(argv)),
    )

    rc = cli.main(["visualize", "--seed-file", str(seed_file), "--seed", "404"])

    assert rc == 0
    assert captured["argv"][:4] == ["--type", "4", "--seed", "404"]


def test_visualize_rejects_failed_mode_without_summary_json(capsys):
    rc = cli.main(["visualize", "--failed"])

    assert rc == 1
    assert "--failed" in capsys.readouterr().err


def test_visualize_reports_failure(monkeypatch, capsys):
    monkeypatch.setattr(
        "scripts.visualize_map.main",
        lambda argv: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    rc = cli.main(["visualize", "--type", "1"])

    assert rc == 1
    assert "Visualizer failed: boom" in capsys.readouterr().err


def test_video_invokes_generator_main_for_seed_file(monkeypatch, tmp_path):
    model_path = tmp_path / "UID_178.zip"
    model_path.write_bytes(b"zip")
    seed_file = tmp_path / "seeds.json"
    seed_file.write_text("{}")
    captured: dict[str, list[str]] = {}

    def _fake_video_main(argv):
        captured["argv"] = list(argv)

    monkeypatch.setattr("scripts.generate_video.main", _fake_video_main)

    rc = cli.main(
        [
            "video",
            "--model",
            str(model_path),
            "--seed-file",
            str(seed_file),
            "--mode",
            "all",
            "--out",
            str(tmp_path / "videos"),
            "--skip-existing",
        ]
    )

    assert rc == 0
    assert captured["argv"][:2] == ["--model", str(model_path)]
    assert "--seed-file" in captured["argv"]
    assert "--mode" in captured["argv"]
    assert "all" in captured["argv"]
    assert "--backend" in captured["argv"]
    assert "benchmark" in captured["argv"]
    assert "--skip-existing" in captured["argv"]


def test_video_requires_seed_or_seed_file(tmp_path, capsys):
    model_path = tmp_path / "UID_178.zip"
    model_path.write_bytes(b"zip")

    rc = cli.main(["video", "--model", str(model_path)])

    assert rc == 1
    assert "Provide either --seed-file, or both --seed and --type." in capsys.readouterr().err


def test_model_verify_passes_for_valid_rpc_submission(tmp_path, capsys):
    model_zip = tmp_path / "submission.zip"
    with zipfile.ZipFile(model_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("drone_agent.py", "class DroneFlightController:\n    pass\n")

    rc = cli.main(["model", "verify", "--model", str(model_zip)])
    assert rc == 0
    output = capsys.readouterr().out
    assert "Model:" in output
    assert "Compliant: True" in output


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


def test_model_test_success_for_valid_source(tmp_path, capsys):
    src = tmp_path / "agent"
    src.mkdir(parents=True, exist_ok=True)
    (src / "drone_agent.py").write_text("class DroneFlightController:\n    pass\n")
    (src / "weights.pt").write_bytes(b"weights")

    rc = cli.main(["model", "test", "--source", str(src)])

    assert rc == 0
    output = capsys.readouterr().out
    assert "Model Test" in output
    assert "drone_agent.py: present" in output
    assert "estimated_package_size:" in output


def test_model_test_fails_for_invalid_python(tmp_path):
    src = tmp_path / "agent"
    src.mkdir(parents=True, exist_ok=True)
    (src / "drone_agent.py").write_text("def broken(:\n")

    assert cli.main(["model", "test", "--source", str(src)]) == 1


def test_report_text_output_parses_summary_without_results_block(tmp_path, capsys):
    log_path = tmp_path / "bench.log"
    log_path.write_text(
        "\n".join(
            [
                "    Seeds evaluated:           50",
                "    Success rate:              40/50 (80.0%)",
                "    Clean execution rate:      49/50 (98.0%)",
                "    Total wall-clock:          120.0s (2.0 min)",
                "    Throughput:                25.00 seeds/min",
                "    Workers used:              3",
                "    Estimated wall-clock:      900.0s (15.0 min)",
            ]
        )
    )

    rc = cli.main(["report", "--input", str(log_path)])
    assert rc == 0
    output = capsys.readouterr().out
    assert f"Report source: {log_path}" in output
    assert "Seeds evaluated: 50" in output
    assert "Workers used: 3" in output
    assert "Total wall-clock: 120.0s" in output
    assert "Throughput: 25.00 seeds/min" in output
    assert "Estimated wall-clock for 1000 seeds: 900.0s" in output


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
        [sys.executable, "-m", "swarm", "model", "test", "--source", str(src)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Model Test" in result.stdout
    assert "drone_agent.py: present" in result.stdout
