from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from types import SimpleNamespace

from swarm import cli
from swarm.policy_interface import POLICY_CONTRACT_FILENAME, render_artifact_policy_contract


def _write_smoke_ready_agent(src, *, speed: float = 0.5):
    src.mkdir(parents=True, exist_ok=True)
    (src / "drone_agent.py").write_text(
        "\n".join(
            [
                "import numpy as np",
                "",
                "class DroneFlightController:",
                "    def reset(self):",
                "        return None",
                "",
                "    def act(self, observation):",
                "        n = 6 if isinstance(observation, dict) and 'rgb' in observation else 5",
                "        a = np.zeros(n, dtype=np.float32)",
                f"        a[3] = {speed}",
                "        return a",
                "",
            ]
        )
    )
    (src / "weights.pt").write_bytes(b"weights")


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


def test_model_verify_passes_for_valid_rpc_submission(tmp_path, capsys):
    model_zip = tmp_path / "submission.zip"
    with zipfile.ZipFile(model_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "drone_agent.py",
            "\n".join(
                [
                    "import numpy as np",
                    "",
                    "class DroneFlightController:",
                    "    def reset(self):",
                    "        return None",
                    "",
                    "    def act(self, observation):",
                    "        return np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0], dtype=np.float32)",
                    "",
                ]
            ),
        )
        zf.writestr(
            POLICY_CONTRACT_FILENAME,
            render_artifact_policy_contract(
                "cf_search_and_rescue",
                "submission_zip.v1",
            ),
        )

    rc = cli.main(["model", "verify", "--model", str(model_zip)])
    assert rc == 0
    output = capsys.readouterr().out
    assert "Model:" in output
    assert "Compliant: True" in output
    assert "Runtime smoke: True (ok)" in output


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
        contract = json.loads(zf.read(POLICY_CONTRACT_FILENAME).decode("utf-8"))
    assert "drone_agent.py" in names
    assert "policy.pt" in names
    assert POLICY_CONTRACT_FILENAME in names
    assert contract["family_id"] == "cf_search_and_rescue"
    assert contract["interface_version"] == "submission_zip.v1"


def test_model_package_supports_family_override(tmp_path):
    src = tmp_path / "agent"
    src.mkdir(parents=True, exist_ok=True)
    (src / "drone_agent.py").write_text("class DroneFlightController:\n    pass\n")
    out_zip = tmp_path / "submission.zip"

    rc = cli.main(
        [
            "model",
            "package",
            "--source",
            str(src),
            "--output",
            str(out_zip),
            "--family-id",
            "cf_autopilot",
        ]
    )

    assert rc == 0
    with zipfile.ZipFile(out_zip) as zf:
        contract = json.loads(zf.read(POLICY_CONTRACT_FILENAME).decode("utf-8"))
    assert contract["family_id"] == "cf_autopilot"


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


def test_repo_package_creates_multi_family_manifest_and_artifacts(tmp_path):
    repo_root = tmp_path / "submission_repo"
    sar_src = tmp_path / "sar_agent"
    autopilot_src = tmp_path / "autopilot_agent"
    _write_smoke_ready_agent(sar_src, speed=0.5)
    _write_smoke_ready_agent(autopilot_src, speed=0.4)

    rc = cli.main(
        [
            "repo",
            "package",
            "--repo-root",
            str(repo_root),
            "--family-source",
            f"cf_search_and_rescue={sar_src}",
            "--family-source",
            f"cf_autopilot={autopilot_src}",
        ]
    )

    assert rc == 0
    manifest_path = repo_root / "submission_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert [item["family_id"] for item in manifest["artifacts"]] == [
        "cf_autopilot",
        "cf_search_and_rescue",
    ]
    assert (repo_root / "artifacts" / "cf_autopilot" / "submission.zip").exists()
    assert (repo_root / "artifacts" / "cf_search_and_rescue" / "submission.zip").exists()


def test_repo_package_supports_single_family_incremental_updates(tmp_path):
    repo_root = tmp_path / "submission_repo"
    sar_src = tmp_path / "sar_agent"
    autopilot_src = tmp_path / "autopilot_agent"
    _write_smoke_ready_agent(sar_src, speed=0.5)
    _write_smoke_ready_agent(autopilot_src, speed=0.4)

    assert (
        cli.main(
            [
                "repo",
                "package",
                "--repo-root",
                str(repo_root),
                "--family-source",
                f"cf_search_and_rescue={sar_src}",
            ]
        )
        == 0
    )
    assert (
        cli.main(
            [
                "repo",
                "package",
                "--repo-root",
                str(repo_root),
                "--source",
                str(autopilot_src),
                "--family-id",
                "cf_autopilot",
            ]
        )
        == 0
    )

    manifest = json.loads((repo_root / "submission_manifest.json").read_text())
    assert {item["family_id"] for item in manifest["artifacts"]} == {
        "cf_search_and_rescue",
        "cf_autopilot",
    }


def test_repo_verify_passes_for_packaged_multi_family_repo(tmp_path, capsys):
    repo_root = tmp_path / "submission_repo"
    sar_src = tmp_path / "sar_agent"
    autopilot_src = tmp_path / "autopilot_agent"
    _write_smoke_ready_agent(sar_src, speed=0.5)
    _write_smoke_ready_agent(autopilot_src, speed=0.4)

    assert (
        cli.main(
            [
                "repo",
                "package",
                "--repo-root",
                str(repo_root),
                "--family-source",
                f"cf_search_and_rescue={sar_src}",
                "--family-source",
                f"cf_autopilot={autopilot_src}",
            ]
        )
        == 0
    )

    rc = cli.main(["repo", "verify", "--repo-root", str(repo_root), "--strict-manifest"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "Compliant: True" in output
    assert "Family: cf_autopilot" in output
    assert "Family: cf_search_and_rescue" in output
    assert "Runtime smoke: True (ok)" in output


def test_repo_verify_rejects_bad_artifact_and_surfaces_family_id(tmp_path, capsys):
    repo_root = tmp_path / "submission_repo"
    sar_src = tmp_path / "sar_agent"
    autopilot_src = tmp_path / "autopilot_agent"
    _write_smoke_ready_agent(sar_src, speed=0.5)
    _write_smoke_ready_agent(autopilot_src, speed=0.4)

    assert (
        cli.main(
            [
                "repo",
                "package",
                "--repo-root",
                str(repo_root),
                "--family-source",
                f"cf_search_and_rescue={sar_src}",
                "--family-source",
                f"cf_autopilot={autopilot_src}",
            ]
        )
        == 0
    )

    bad_artifact = repo_root / "artifacts" / "cf_autopilot" / "submission.zip"
    with zipfile.ZipFile(bad_artifact, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("drone_agent.py", "class DroneFlightController:\n    pass\n")

    rc = cli.main(["repo", "verify", "--repo-root", str(repo_root), "--strict-manifest"])

    assert rc == 1
    output = capsys.readouterr().out
    assert "Compliant: False" in output
    assert "Reason: artifact_hash_mismatch:cf_autopilot:artifacts/cf_autopilot/submission.zip" in output


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
